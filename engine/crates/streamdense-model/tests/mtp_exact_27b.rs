//! Per-position exactness of MTP speculative decode on the REAL 27B (models/qwen38_27b_mlx4, MLX 4-bit + mtp/).
//! For each prompt: plain greedy decode (K = 1 passes, `feed_stream`) records per generated position the token, the
//! f32 logits row (to a scratch file), its f32 argmax / top1-top2 margin, and the bf16 final-hidden (hf) row bits.
//! Then, from a fresh reset + re-prefill, MTP deep spec decode (`feed_stream_spec_mode`, the gateway's pipelined
//! loop) at fixed depth 2, 4, 7 and auto: inside `on_tok` every emitted token's verify row (row = position - pass
//! start; accepted drafts + the correction / bonus row) is read from the verify pass's logits / hf scratch and
//! compared BIT FOR BIT with the plain row of the same position. Position 0 (the prefill's token) is compared by
//! token only: its logits row is overwritten by the MTP first draft before `on_tok` sees it.
//! One engine, gateway-default options (MTP, spec_depth auto = 7, gemv mma, fast prefill, draft vocab, draft table in
//! the scratch dir, activity + read mask hooks on, verify_fused / verify_mixers / decode_fused off). SD_VERIFY_FUSED=1
//! / SD_VERIFY_MIXERS=1 force those options at open (QwenEngine::open) for a second run.
//! Run (under sd-watchdog): mtp_exact_27b-<hash> --ignored --nocapture --test-threads 1
//! SD_EXACT_BENCH=1: speed run instead (plain greedy, then MTP auto with the context-blind planner (SD_SPEC_CTX_COST=0)
//! and with the default context-aware one; token ids only, decode tok/s, acceptance, depth histogram, tokens equal).
//! Env: SD_EXACT_MODEL (model dir), SD_EXACT_ORACLE=1 (oracle drafts, harness check), SD_EXACT_N (tokens per run, default 48), SD_EXACT_DIR (scratch dir for the plain logits rows + draft table).
//! Long contexts: SD_EXACT_PROMPT_FILE = a file of u32 token ids (whitespace separated, chat-templated) used instead
//! of the built-in prompts, SD_EXACT_MAX_CTX = max_ctx (default 2048), SD_EXACT_CAP_GB = engine RAM budget cap (default 17). With a prompt file the prompt minus its last
//! `SNAP_TAIL` tokens is prefilled ONCE (MTP engine: main + MTP KV) and saved as a snapshot in SD_EXACT_DIR; every run
//! (plain and each depth) restores it and feeds the tail, so an 8K prompt is not re-prefilled per depth. Each table
//! line counts the compared decode positions by attention path of their kv length N (N < SDPA_TWO_PASS_MIN single
//! pass, up to SDPA_VEC_MAX_N MLX two-pass mirror, above it split-K `attn_split_rows` + `attn_reduce_gate`).
//! KV format: SD_EXACT_KV=q8 opens the engine with 8-bit KV (fast prefill via attn_flash_q8 as bf16; every decode attention row split-K).
//! Quality runs: SD_EXACT_TOKENS_OUT=FILE appends per plain position `prompt pos token argmax top1-top2-margin` (tab
//! separated); SD_EXACT_PLAIN_ONLY=1 runs only the plain greedy part (no MTP depths), e.g. to compare bf16 vs q8 KV;
//! SD_EXACT_PREFILL_EXACT=1 = exact prompt read (bf16 and q8), to separate the two effects.
//! DFlash2 (bench only): SD_EXACT_DFLASH2=DIR (drafter dir, e.g. models/dflash2_27b_q4) adds rows after the MTP ones
//! from a SECOND engine (the engine refuses DFlash2 together with MTP, and two 27B engines do not fit: the MTP engine is
//! dropped first) with the same cap / kv / max_ctx: "dflash2" (`feed_stream_dflash2`) and "plain (dflash2 engine)".
//! The drafter needs context rows for every prompt position and a snapshot restore leaves none, so the DFlash2 engine
//! prefills the prompt prefix FRESH (one prefill per prompt), saves a snapshot, runs dflash2 straight on from that live
//! state (tail + N tokens), then restores the snapshot for its own plain row. Tokens are compared with the MTP engine's
//! plain row and with the DFlash2 engine's plain row. SD_EXACT_DFLASH2_ONLY=1: skip the MTP engine (no MTP rows).
use std::io::{Read, Seek, SeekFrom, Write};
use std::time::Instant;

use streamdense_model::draft_table::DraftTable;
use streamdense_model::spec::DraftOverride;
use streamdense_model::spec_deep::SpecMode;
use streamdense_model::{KvFormat, QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
/// Prompt-file mode: tokens fed after the snapshot restore (the chat-template tail and a bit of body).
const SNAP_TAIL: usize = 16;

// chat-templated prompts (user turn + "<think>\n\n</think>\n\n"), tokenized with the model's tokenizer.json
const PROMPTS: [(&str, &[u32]); 4] = [
    ("prose", &[248045, 846, 198, 7734, 264, 2716, 13901, 883, 264, 11012, 6353, 4088, 539, 264, 21128, 303, 40297, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]),
    ("code", &[248045, 846, 198, 7734, 264, 12654, 709, 421, 4523, 279, 307, 7324, 76938, 1324, 5199, 7638, 11, 440, 264, 4479, 889, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]),
    ("arith", &[248045, 846, 198, 44712, 220, 18, 23, 19, 22, 348, 220, 17, 21, 2923, 539, 2923, 11, 1179, 884, 220, 16, 17, 24, 20, 310, 279, 1067, 13, 6712, 1754, 2923, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]),
    ("repeat", &[248045, 846, 198, 37436, 279, 2614, 1500, 6681, 4097, 2942, 11, 799, 791, 1500, 25, 561, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]),
];

fn fnv(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &x in b { h ^= x as u64; h = h.wrapping_mul(0x100000001b3); }
    h
}
fn top2(row: &[f32]) -> (u32, f32) {
    let (mut i1, mut v1, mut v2) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > v1 { v2 = v1; v1 = v; i1 = i; } else if v > v2 { v2 = v; }
    }
    (i1 as u32, v1 - v2)
}

#[derive(Clone, Debug)]
struct Rec { tok: u32, argmax: u32, margin: f32, hh: u64 }

/// Logits row `r` and hf row `r` of the last pass (f32 [rows][vocab], bf16 [rows][hidden]).
fn rows(e: &QwenEngine, r: usize) -> (&[f32], &[u8]) {
    let io = e.io();
    let v = e.w.lm_head.rows;
    let hb = e.cfg.hidden * 2;
    unsafe {
        let l = std::slice::from_raw_parts(e.arena.base().add(io.logits + r * v * 4) as *const f32, v);
        let h = std::slice::from_raw_parts(e.arena.base().add(io.hf + r * hb), hb);
        (l, h)
    }
}

#[test]
#[ignore]
fn mtp_exact_27b_per_position() {
    // SD_EXACT_MODEL: another model dir (e.g. models/tiny_qwen35 to check the harness; prompt ids taken mod its vocab)
    let d = std::env::var("SD_EXACT_MODEL").unwrap_or_else(|_| format!("{}/models/qwen38_27b_mlx4", ROOT));
    if !std::path::Path::new(&format!("{}/mtp", d)).exists() { eprintln!("SKIP: needs {} (with mtp/)", d); return; }
    let n: usize = std::env::var("SD_EXACT_N").ok().and_then(|s| s.parse().ok()).unwrap_or(48);
    let sdir = std::path::PathBuf::from(std::env::var("SD_EXACT_DIR").unwrap_or_else(|_| std::env::temp_dir().join("sd_mtp_exact").display().to_string()));
    std::fs::create_dir_all(&sdir).unwrap();
    let max_ctx: usize = std::env::var("SD_EXACT_MAX_CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
    // engine budget cap in GB (default: QwenOptions default, 17 GB); keep it at or under the sd-watchdog cap
    let cap_gb: Option<f64> = std::env::var("SD_EXACT_CAP_GB").ok().and_then(|s| s.parse().ok());
    let pfile = std::env::var("SD_EXACT_PROMPT_FILE").ok();
    // SD_EXACT_BENCH=1: speed run (plain + MTP auto only, no logits rows written / read / compared)
    let bench = std::env::var("SD_EXACT_BENCH").is_ok_and(|s| s == "1");
    let kv = match std::env::var("SD_EXACT_KV").as_deref() {
        Ok("q8") => KvFormat::Q8,
        Ok("bf16") | Err(_) => KvFormat::Bf16,
        Ok(k) => panic!("SD_EXACT_KV={}: want bf16 or q8", k),
    };
    let plain_only = std::env::var("SD_EXACT_PLAIN_ONLY").is_ok_and(|s| s == "1");
    let tokens_out = std::env::var("SD_EXACT_TOKENS_OUT").ok();
    let df2_dir: Option<std::path::PathBuf> = std::env::var("SD_EXACT_DFLASH2").ok().filter(|s| !s.is_empty()).map(Into::into);
    let df2_only = std::env::var("SD_EXACT_DFLASH2_ONLY").is_ok_and(|s| s == "1");
    if let Some(p) = &df2_dir {
        assert!(bench, "SD_EXACT_DFLASH2 needs SD_EXACT_BENCH=1");
        assert!(p.join("config.json").exists() && p.join("model.safetensors").exists(), "SD_EXACT_DFLASH2={}: no config.json + model.safetensors", p.display());
    }
    assert!(!df2_only || df2_dir.is_some(), "SD_EXACT_DFLASH2_ONLY=1 needs SD_EXACT_DFLASH2=DIR");
    let prompts: Vec<(String, Vec<u32>)> = match &pfile {
        Some(p) => {
            let ids: Vec<u32> = std::fs::read_to_string(p).unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
            assert!(ids.len() > SNAP_TAIL, "{}: too few tokens", p);
            let name = std::path::Path::new(p).file_stem().unwrap().to_string_lossy().into_owned();
            vec![(name, ids)]
        }
        None => PROMPTS.iter().map(|(n, ids)| (n.to_string(), ids.to_vec())).collect(),
    };
    let cap = cap_gb.map_or(QwenOptions::default().cap, |g| (g * 1e9) as u64);
    let prefill_exact = std::env::var("SD_EXACT_PREFILL_EXACT").is_ok_and(|s| s == "1");
    // the DFlash2 engine: the gateway's --dflash2 options (MTP off, no draft vocab / table), same cap / kv / max_ctx
    let df2_opt = df2_dir.as_ref().map(|p| QwenOptions {
        cap, prefill_exact, max_ctx, kv, mtp: false, spec_depth: 1, activity: true, activity_batch: true, read_mask: true,
        dflash2: Some(p.clone()), ..Default::default()
    });
    if df2_only {
        let (lines, bad, peak) = dflash2_bench(&d, df2_opt.unwrap(), &prompts, n, &sdir, pfile.is_some(), None);
        print_bench(&lines, &format!("peak footprint (DFlash2 engine only) {:.2} GB", peak));
        assert_eq!(bad, 0, "{} DFlash2 rows: tokens differ from plain greedy", bad);
        return;
    }
    let vk = format!("{}/draft_vocab.keep", d);
    let opt = QwenOptions {
        cap, prefill_exact, max_ctx, kv, mtp: true, spec_depth: 7, spec_auto: true, activity: true, activity_batch: true, read_mask: true,
        draft_vocab: std::path::Path::new(&vk).exists().then(|| vk.clone().into()),
        draft_table: Some(sdir.join("draft_table")),
        ..Default::default()
    };
    let t_open = Instant::now();
    let e = QwenEngine::open(&d, opt).unwrap();
    eprintln!("kv {:?}; open {:.1} s; verify_fused {} verify_mixers {} decode_fused {} gemv {} prefill_exact {} draft_vocab {}",
              e.opt.kv, t_open.elapsed().as_secs_f64(), e.opt.verify_fused, e.opt.verify_mixers, e.opt.decode_fused, e.opt.gemv.name(),
              e.opt.prefill_exact, e.opt.draft_vocab.is_some());
    let v = e.w.lm_head.rows;
    let modes = [SpecMode::Fixed(2), SpecMode::Fixed(4), SpecMode::Fixed(7), SpecMode::Auto { max: 7 }];
    let mut table: Vec<String> = vec![];
    let mut bad = 0usize;
    // bench: the MTP engine's plain greedy tokens per prompt (compared with DFlash2's)
    let mut plain_toks: Vec<Vec<u32>> = vec![];
    let (tp_min, vec_max) = (streamdense_model::qwen_layers::sdpa_two_pass_min(), streamdense_model::qwen_layers::sdpa_vec_max_n());
    eprintln!("max_ctx {}; attention paths: single pass N < {}, two-pass {}..={}, split-K N > {}", max_ctx, tp_min, tp_min, vec_max, vec_max);
    for (name, ids) in &prompts {
        let ids: Vec<u32> = ids.iter().map(|&x| x % e.cfg.vocab as u32).collect();
        let ids = &ids[..];
        let pl = ids.len();
        // prompt-file mode: prefill the prefix once, snapshot; each run restores it and feeds only the tail
        let snap = pfile.as_ref().map(|_| {
            let sp = sdir.join(format!("prefix_{}.sdsnap", name));
            let pre = &ids[..pl - SNAP_TAIL];
            e.reset();
            let t = Instant::now();
            e.feed_stream_mtp(pre, 1, &[], &mut |_| true).unwrap();
            assert_eq!(e.pos(), pre.len());
            e.save_snapshot(&sp, pre).unwrap();
            eprintln!("{}: prompt {} tokens; prefix {} prefilled + snapshot in {:.1} s ({:.2} GB file)", name, pl, pre.len(),
                      t.elapsed().as_secs_f64(), std::fs::metadata(&sp).unwrap().len() as f64 / 1e9);
            sp
        });
        // fresh state for a run → the ids to feed
        let start = || -> &[u32] {
            e.reset();
            match &snap {
                Some(sp) => { assert_eq!(e.restore_snapshot(sp).unwrap().len(), pl - SNAP_TAIL); &ids[pl - SNAP_TAIL..] }
                None => ids,
            }
        };
        if bench {
            // speed only: plain greedy then MTP auto; on_tok only collects token ids and a timestamp
            let feed = start();
            let mut tp: Vec<u32> = vec![];
            let (mut t0, mut t1) = (None, None);
            let tw = Instant::now();
            e.feed_stream(feed, n, &[], &mut |t| {
                let now = Instant::now();
                if t0.is_none() { t0 = Some(now); }
                t1 = Some(now);
                tp.push(t);
                true
            }).unwrap();
            let wall_p = tw.elapsed().as_secs_f64();
            let dec_p = t1.unwrap().duration_since(t0.unwrap()).as_secs_f64();
            let line_p = format!("| {} | plain | {} | {:.2} s | {:.2} tok/s | {:.2} s | {:.2} tok/s | - | - | - |", name, tp.len(), dec_p,
                                 (tp.len() - 1) as f64 / dec_p, wall_p, tp.len() as f64 / wall_p);
            eprintln!("{}", line_p);
            table.push(line_p);
            plain_toks.push(tp.clone());
            for (label, ctx_cost) in [("mtp auto (old, context-blind)", "0"), ("mtp auto (context-aware)", "1")] {
                std::env::set_var("SD_SPEC_CTX_COST", ctx_cost);
                // a fresh, empty draft table per run: neither run learns from the other's continuation
                let tdir = sdir.join(format!("draft_table_bench_{}", ctx_cost));
                let _ = std::fs::remove_dir_all(&tdir);
                e.set_draft_source(Some(Box::new(DraftTable::open(&tdir, e.opt.draft_table_bytes).unwrap())));
                let feed = start();
                let mut tm: Vec<u32> = vec![];
                let (mut s0, mut s1) = (None, None);
                let tw = Instant::now();
                let (st, r) = e.feed_stream_spec_mode(feed, n, &[], &mut |t| {
                    let now = Instant::now();
                    if s0.is_none() { s0 = Some(now); }
                    s1 = Some(now);
                    tm.push(t);
                    true
                }, SpecMode::Auto { max: 7 }, &DraftOverride::None).unwrap();
                let wall_m = tw.elapsed().as_secs_f64();
                e.flush_draft_source();
                let dec_m = s1.unwrap().duration_since(s0.unwrap()).as_secs_f64();
                let drafts = r.drafts_mtp + r.drafts_table + r.drafts_pad;
                let line_m = format!("| {} | {} | {} | {:.2} s | {:.2} tok/s | {:.2} s | {:.2} tok/s | {:.3} ({} / {}) | {} ({}) | {:?} |", name, label, tm.len(), dec_m,
                                     (tm.len() - 1) as f64 / dec_m, wall_m, tm.len() as f64 / wall_m,
                                     r.accepted as f64 / drafts.max(1) as f64, r.accepted, drafts, r.passes, st.stop_reason, r.depth_hist);
                eprintln!("{}\ntokens equal plain vs {}: {} ({} vs {} tokens)", line_m, label, tp == tm, tp.len(), tm.len());
                table.push(line_m);
                if tp != tm { bad += 1; }
                e.set_draft_source(None);
                let _ = std::fs::remove_dir_all(&tdir);
            }
            std::env::remove_var("SD_SPEC_CTX_COST");
            if let Some(sp) = &snap { let _ = std::fs::remove_file(sp); }
            continue;
        }
        // ---- (a) plain greedy
        let path = sdir.join(format!("plain_{}.f32", name));
        let mut f = std::fs::File::create(&path).unwrap();
        let mut plain: Vec<Rec> = vec![];
        let feed = start();
        let tp = Instant::now();
        e.feed_stream(feed, n, &[], &mut |t| {
            let (l, h) = rows(&e, 0);
            let (am, mg) = top2(l);
            let lb = unsafe { std::slice::from_raw_parts(l.as_ptr() as *const u8, v * 4) };
            f.write_all(lb).unwrap();
            plain.push(Rec { tok: t, argmax: am, margin: mg, hh: fnv(h) });
            true
        }).unwrap();
        drop(f);
        let am_ok = plain.iter().skip(1).filter(|r| r.argmax == r.tok).count();
        eprintln!("{}: plain {} tokens in {:.1} s; f32 argmax == token at {}/{} decode positions; min margin {:.4}", name, plain.len(),
                  tp.elapsed().as_secs_f64(), am_ok, plain.len() - 1,
                  plain.iter().skip(1).map(|r| r.margin).fold(f32::INFINITY, f32::min));
        if let Some(to) = &tokens_out {
            let mut o = std::fs::OpenOptions::new().create(true).append(true).open(to).unwrap();
            for (i, r) in plain.iter().enumerate() { writeln!(o, "{}\t{}\t{}\t{}\t{:.6}", name, i, r.tok, r.argmax, r.margin).unwrap(); }
        }
        let mut f = std::fs::File::open(&path).unwrap();
        // ---- (b) spec at each depth, (c) compare per position
        for m in modes {
            if plain_only { break; }
            let mut got: Vec<(u32, bool, bool, usize, f32)> = vec![]; // (token, logits bit-identical, hf bit-identical, row, max |diff|)
            let mut cnt = 0usize;
            let mut buf = vec![0u8; v * 4];
            // SD_EXACT_ORACLE=1 (harness check on a model whose MTP drafts are poor): drafts replaced by the plain
            // continuation with p = 0.8, so the verify passes accept rows 1..7 too
            let ov = if std::env::var("SD_EXACT_ORACLE").is_ok() {
                DraftOverride::Oracle { seq: ids.iter().copied().chain(plain.iter().map(|r| r.tok)).collect(), p: 0.8, seed: 7 }
            } else { DraftOverride::None };
            let feed = start();
            let ts = Instant::now();
            let (st, r) = e.feed_stream_spec_mode(feed, n, &[], &mut |t| {
                let k = cnt;
                cnt += 1;
                if k == 0 { got.push((t, true, true, usize::MAX, 0.0)); return true; }
                let row = pl + k - 1 - e.pos();
                assert!(row < 8, "row {} out of the verify pass", row);
                let (l, h) = rows(&e, row);
                let lb = unsafe { std::slice::from_raw_parts(l.as_ptr() as *const u8, v * 4) };
                let (lsame, hsame, md) = if k < plain.len() {
                    f.seek(SeekFrom::Start((k * v * 4) as u64)).unwrap();
                    f.read_exact(&mut buf).unwrap();
                    let same = buf[..] == lb[..];
                    let md = if same { 0.0 } else {
                        let pr = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const f32, v) };
                        pr.iter().zip(l).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
                    };
                    (same, fnv(h) == plain[k].hh, md)
                } else { (false, false, f32::NAN) };
                got.push((t, lsame, hsame, row, md));
                true
            }, m, &ov).unwrap();
            e.flush_draft_source();
            let wall = ts.elapsed().as_secs_f64();
            let cmp = got.len().min(plain.len());
            let tok_same = (0..cmp).filter(|&i| got[i].0 == plain[i].tok).count();
            let lsame = (1..cmp).filter(|&i| got[i].1).count();
            let hsame = (1..cmp).filter(|&i| got[i].2).count();
            let rowhist = { let mut h = [0usize; 8]; for g in &got[1..] { if g.3 < 8 { h[g.3] += 1; } } h };
            // decode position k was produced by the pass row feeding position pl + k - 1: kv length N = pl + k
            let paths = (1..cmp).fold([0usize; 3], |mut a, k| {
                let nk = pl + k;
                a[if kv == KvFormat::Q8 || nk > vec_max { 2 } else if nk < tp_min { 0 } else { 1 }] += 1; // q8: split-K only
                a
            });
            let first = (0..cmp).find(|&i| got[i].0 != plain[i].tok || !got[i].1 || !got[i].2);
            let fm = match first {
                Some(i) => format!("pos {} (N {}, row {}): tok {} vs {}, logits {}, hf {}, max|diff| {:e}, plain margin {:.4}", i, pl + i, got[i].3,
                                   got[i].0, plain[i].tok, got[i].1, got[i].2, got[i].4, plain[i].margin),
                None => "none".into(),
            };
            if first.is_some() || got.len() != plain.len() { bad += 1; }
            let minm = plain.iter().take(cmp).skip(1).map(|r| r.margin).fold(f32::INFINITY, f32::min);
            let line = format!("| {} | {} | {} | {} / {} / {} | {:.4} | {}/{} | {}/{} | {}/{} | {:.3} ({} / {} drafts, {} passes) | rows {:?} | {} | {:.1} s ({}) |",
                               name, m.name(), cmp - 1, paths[0], paths[1], paths[2], minm, lsame, cmp - 1, hsame, cmp - 1, tok_same, cmp, r.accepted as f64 / (r.drafts_mtp + r.drafts_table + r.drafts_pad).max(1) as f64, r.accepted,
                               r.drafts_mtp + r.drafts_table + r.drafts_pad, r.passes, rowhist, fm, wall, st.stop_reason);
            eprintln!("{}", line);
            table.push(line);
        }
        let _ = std::fs::remove_file(&path);
        if let Some(sp) = &snap { let _ = std::fs::remove_file(sp); }
    }
    if bench {
        let mut peak = format!("peak footprint {:.2} GB", e.peak() as f64 / 1e9);
        if let Some(o) = df2_opt {
            drop(e); // two 27B engines do not fit: free the MTP engine before the DFlash2 one opens
            let (lines, b2, pk) = dflash2_bench(&d, o, &prompts, n, &sdir, pfile.is_some(), Some(&plain_toks));
            table.extend(lines);
            bad += b2;
            peak = format!("{} (MTP engine); DFlash2 engine: process peak {:.2} GB (includes the MTP engine's peak if that was higher)", peak, pk);
        }
        print_bench(&table, &peak);
        assert_eq!(bad, 0, "{} rows: MTP / DFlash2 tokens differ from plain greedy", bad);
        return;
    }
    println!("\n| prompt | depth | positions | N paths (single / two-pass / split-K) | min plain margin | logits bit-identical | hf bit-identical | tokens identical | acceptance | verify rows used (row 0..7) | first mismatch | time |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|");
    for l in &table { println!("{}", l); }
    println!("peak footprint {:.2} GB", e.peak() as f64 / 1e9);
    assert_eq!(bad, 0, "{} prompt x depth runs not bit-identical to plain greedy", bad);
}

fn print_bench(table: &[String], peak: &str) {
    println!("\n| prompt | mode | tokens | decode s (first to last token) | decode tok/s (tokens - 1) | wall s incl. {} tail | wall tok/s | acceptance (accepted / drafts) | passes (stop) | real drafts per pass (0..7); dflash2: accepted per pass (0..7) |", SNAP_TAIL);
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for l in table { println!("{}", l); }
    println!("{}", peak);
}

/// Bench rows of a DFlash2 engine (opened here, dropped at the end): per prompt "dflash2" then "plain (dflash2 engine)".
/// Prompt-file mode: the prefix is prefilled fresh (every pass captures the drafter's context rows), saved as a
/// snapshot, dflash2 runs on from that live state; the plain row restores the snapshot (it needs no drafter context).
/// Returns (table lines, rows whose tokens differ from plain, process peak footprint GB).
fn dflash2_bench(d: &str, opt: QwenOptions, prompts: &[(String, Vec<u32>)], n: usize, sdir: &std::path::Path, snap_mode: bool,
                 plain_mtp: Option<&[Vec<u32>]>) -> (Vec<String>, usize, f64) {
    match QwenEngine::budget(d, &opt) {
        Ok(b) => eprintln!("DFlash2 engine budget (cap {:.2} GB): arena {:.3} GB + reserve {:.3} GB = {:.3} GB\n{}", b.cap as f64 / 1e9,
                           b.arena_bytes() as f64 / 1e9, b.reserve as f64 / 1e9, (b.arena_bytes() + b.reserve) as f64 / 1e9, b.report()),
        Err(x) => panic!("DFlash2 engine budget: {}", x),
    }
    let t_open = Instant::now();
    let e = QwenEngine::open(d, opt).unwrap();
    let c = e.dflash2_config().unwrap();
    eprintln!("DFlash2 engine: kv {:?}; open {:.1} s; block {} window {} ring {} rows; taps {:?}", e.opt.kv, t_open.elapsed().as_secs_f64(),
              c.block, c.window, e.dflash2_ring_rows(), c.layer_ids);
    let mut lines = vec![];
    let mut bad = 0usize;
    for (pi, (name, ids)) in prompts.iter().enumerate() {
        let ids: Vec<u32> = ids.iter().map(|&x| x % e.cfg.vocab as u32).collect();
        let pl = ids.len();
        e.reset();
        // fresh prefill of the prefix (context rows captured), snapshot for the plain row
        let snap = snap_mode.then(|| {
            let sp = sdir.join(format!("prefix_df2_{}.sdsnap", name));
            let pre = &ids[..pl - SNAP_TAIL];
            let t = Instant::now();
            e.feed_stream(pre, 1, &[], &mut |_| true).unwrap();
            assert_eq!(e.pos(), pre.len());
            e.save_snapshot(&sp, pre).unwrap();
            eprintln!("{}: DFlash2 engine: prefix {} prefilled fresh (drafter context {} rows, oldest kept {}) + snapshot in {:.1} s", name, pre.len(),
                      e.dflash2_ctx_len(), e.dflash2_ctx_lo(), t.elapsed().as_secs_f64());
            sp
        });
        let feed: &[u32] = if snap.is_some() { &ids[pl - SNAP_TAIL..] } else { &ids };
        // ---- dflash2
        let mut td: Vec<u32> = vec![];
        let (mut s0, mut s1) = (None, None);
        let tw = Instant::now();
        let (st, ds) = e.feed_stream_dflash2(feed, n, &[], &mut |t| {
            let now = Instant::now();
            if s0.is_none() { s0 = Some(now); }
            s1 = Some(now);
            td.push(t);
            true
        }).unwrap();
        let wall_d = tw.elapsed().as_secs_f64();
        let dec_d = s1.unwrap().duration_since(s0.unwrap()).as_secs_f64();
        let line_d = format!("| {} | dflash2 | {} | {:.2} s | {:.2} tok/s | {:.2} s | {:.2} tok/s | {:.3} ({} / {}) | {} ({}) | {:?} |", name, td.len(), dec_d,
                             (td.len() - 1) as f64 / dec_d, wall_d, td.len() as f64 / wall_d, ds.acceptance(), ds.accepted, ds.drafted,
                             ds.passes, st.stop_reason, ds.accept_hist);
        eprintln!("{}\ndflash2 stats {}", line_d, ds.json());
        if ds.short_ctx > 0 { eprintln!("WARNING: {} rounds drafted with a short drafter context (positions without context rows)", ds.short_ctx); }
        lines.push(line_d);
        // ---- plain on the DFlash2 engine (every decode pass also captures the drafter's tap rows)
        e.reset();
        if let Some(sp) = &snap { assert_eq!(e.restore_snapshot(sp).unwrap().len(), pl - SNAP_TAIL); }
        let mut tp: Vec<u32> = vec![];
        let (mut t0, mut t1) = (None, None);
        let tw = Instant::now();
        e.feed_stream(feed, n, &[], &mut |t| {
            let now = Instant::now();
            if t0.is_none() { t0 = Some(now); }
            t1 = Some(now);
            tp.push(t);
            true
        }).unwrap();
        let wall_p = tw.elapsed().as_secs_f64();
        let dec_p = t1.unwrap().duration_since(t0.unwrap()).as_secs_f64();
        let line_p = format!("| {} | plain (dflash2 engine, taps captured) | {} | {:.2} s | {:.2} tok/s | {:.2} s | {:.2} tok/s | - | - | - |", name, tp.len(), dec_p,
                             (tp.len() - 1) as f64 / dec_p, wall_p, tp.len() as f64 / wall_p);
        eprintln!("{}", line_p);
        lines.push(line_p);
        eprintln!("tokens equal plain (dflash2 engine) vs dflash2: {} ({} vs {} tokens)", tp == td, tp.len(), td.len());
        if tp != td { bad += 1; }
        if let Some(pm) = plain_mtp.and_then(|v| v.get(pi)) {
            eprintln!("tokens equal plain (MTP engine) vs dflash2: {} ({} vs {} tokens)", *pm == td, pm.len(), td.len());
            if *pm != td { bad += 1; }
        }
        if let Some(sp) = &snap { let _ = std::fs::remove_file(sp); }
    }
    (lines, bad, e.peak() as f64 / 1e9)
}
