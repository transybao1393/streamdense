//! activity recorder + unload plan on the tiny model (models/tiny_qwen35):
//! - recorder on vs off: identical tokens and margins (bitwise) for plain greedy (pipelined + sync, exact and fast
//!   prefill), MTP draft 1 / verify 2 and deep spec (depth 3).
//! - counters: rows = prompt + decode passes, resid_rows = rows of <= 8-row passes; sums finite, >= 0, non-zero per
//!   layer; down norms > 0; reset zeroes the window; snapshot file round trip; serve / gateway request JSON.
//! - plan byte math vs the real arena tensors (keep 0: every FFN byte of the tiny model).
//! - MTP draft-layer region + counters (on / off identity covers it: MTP and deep spec run with it, tokens and
//!   margins), per-window token counters; per id vocab / embed_rows counts (plain, MTP depth 1, deep 3), MTP stats fields, scripts
//!   classifier, file round trip, reset.
//! Run: cargo test --release --offline -p streamdense-model --test activity -- --nocapture --test-threads 1
use serde_json::{json, Value};
use streamdense_model::activity::{handle_request, script_class, script_classes, ActivityStats, SCRIPT_NAMES};
use streamdense_model::spec::{generate_mtp, DraftOverride};
use streamdense_model::spec_deep::{generate_spec_mode, SpecMode};
use streamdense_model::unload::unload_plan;
use streamdense_model::{QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn open(activity: bool, exact: bool) -> Option<QwenEngine> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&format!("{}/config.json", d)).exists() { eprintln!("SKIP: {} missing", d); return None; }
    let opt = QwenOptions { max_ctx: 2048, mtp: true, spec_depth: 3, prefill_exact: exact, activity, ..Default::default() };
    Some(QwenEngine::open(&d, opt).expect("open tiny"))
}

fn bits(v: &[f32]) -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() }

fn prompts() -> Vec<(String, Vec<u32>)> {
    let s = std::fs::read_to_string(format!("{}/engine/refs/tiny_qwen35_mlxref_prompts.json", ROOT)).expect("tiny prompts");
    let j: Value = serde_json::from_str(&s).unwrap();
    j["prompts"].as_array().unwrap().iter().filter(|r| ["short", "code", "ctx1k"].contains(&r["id"].as_str().unwrap()))
        .map(|r| (r["id"].as_str().unwrap().to_string(), r["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()))
        .collect()
}

#[test]
fn recorder_on_off_identical() {
    for exact in [false, true] {
        let (Some(on), Some(off)) = (open(true, exact), open(false, exact)) else { return };
        assert!(on.activity.is_some() && off.activity.is_none());
        for (id, p) in prompts() {
            for pipe in [true, false] {
                let a = on.generate(&p, 48, pipe).unwrap();
                let b = off.generate(&p, 48, pipe).unwrap();
                assert_eq!(a.tokens, b.tokens, "{} exact {} pipe {}", id, exact, pipe);
                assert_eq!(a.margins.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), b.margins.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
            }
            let a = generate_mtp(&on, &p, 40, DraftOverride::None, true).unwrap();
            let b = generate_mtp(&off, &p, 40, DraftOverride::None, true).unwrap();
            assert_eq!(a.tokens, b.tokens, "{} mtp", id);
            assert_eq!(bits(&a.margins), bits(&b.margins), "{} mtp margins", id);
            let a = generate_spec_mode(&on, &p, 40, SpecMode::Fixed(3), DraftOverride::None).unwrap();
            let b = generate_spec_mode(&off, &p, 40, SpecMode::Fixed(3), DraftOverride::None).unwrap();
            assert_eq!(a.tokens, b.tokens, "{} spec 3", id);
            assert_eq!(bits(&a.margins), bits(&b.margins), "{} spec 3 margins", id);
            assert!(!a.margins.is_empty());
            eprintln!("{} exact {}: identical ({} tokens)", id, exact, a.tokens.len());
        }
        assert!(!off.activity_snapshot().enabled);
    }
}

#[test]
fn counters_and_plan() {
    let Some(eng) = open(true, false) else { return };
    let ps = prompts();
    eng.activity_reset();
    let (mut rows, mut dec) = (0u64, 0u64);
    for (_, p) in &ps {
        let g = eng.generate(p, 32, false).unwrap();
        rows += (p.len() + g.tokens.len() - 1) as u64;
        dec += (g.tokens.len() - 1) as u64;
    }
    let s = eng.activity_snapshot();
    let c = &eng.cfg;
    eprintln!("rows {} passes {} resid_rows {} (decode {})", s.rows, s.passes, s.resid_rows, dec);
    assert!(s.enabled);
    assert_eq!(s.rows, rows);
    assert!(s.resid_rows >= dec && s.resid_rows < rows); // decode + exact prefill tails; fast prefill chunks skip it
    assert_eq!((s.neuron.len(), s.down_norm.len(), s.resid.len()), (c.layers * c.inter, c.layers * c.inter, c.layers));
    assert_eq!((s.dn_head.len(), s.attn_head.len()), (c.n_deltanet() * c.dn_vh, c.n_attention() * c.heads));
    for v in [&s.neuron, &s.down_norm, &s.dn_head, &s.attn_head, &s.resid] { assert!(v.iter().all(|x| x.is_finite() && *x >= 0.0)); }
    assert!(s.down_norm.iter().all(|x| *x > 0.0));
    for l in 0..c.layers {
        assert!(s.neuron[l * c.inter..(l + 1) * c.inter].iter().sum::<f32>() > 0.0, "layer {} FFN", l);
        assert!(s.head_mass(l).iter().all(|x| *x > 0.0), "layer {} heads", l);
        assert!(s.resid[l] > 0.0);
    }
    // mean |a| per row is a sane magnitude
    let mean = s.neuron.iter().map(|x| *x as f64).sum::<f64>() / (s.neuron.len() as f64 * s.rows as f64);
    eprintln!("mean |a| {:.4e}; summary {}", mean, s.summary());
    assert!(mean > 1e-6 && mean < 1e3);

    // file round trip + request JSON
    let path = std::env::temp_dir().join(format!("sd_activity_{}.safetensors", std::process::id())).display().to_string();
    let r = handle_request(eng.activity.as_deref(), &json!({"path": path, "plan": {"keep": 0.995, "granularity": 64, "layers": true}}));
    assert_eq!(r["written"], json!(path));
    assert_eq!(r["activity"]["rows"], json!(rows));
    assert!(r["plan"]["layers"].as_array().unwrap().len() == c.layers);
    let back = ActivityStats::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!((back.rows, back.passes, back.resid_rows, back.kinds.clone()), (s.rows, s.passes, s.resid_rows, s.kinds.clone()));
    assert_eq!(back.neuron, s.neuron);
    assert_eq!(back.down_norm, s.down_norm);
    assert_eq!(back.attn_head, s.attn_head);

    // byte math: keep 0 -> every FFN byte (gate, up, down incl. scales / biases) of the arena
    let p = unload_plan(&s, 0.0, 64);
    let ffn: usize = eng.w.layers.iter().map(|lw| [lw.ffn.gate, lw.ffn.up, lw.ffn.down].iter().map(|m| m.w.len + m.s.len + m.b.len).sum::<usize>()).sum();
    assert_eq!(p.ffn_bytes, ffn as u64);
    assert_eq!(p.cold_neurons, c.layers * c.inter);
    let p1 = unload_plan(&s, 1.0, 64);
    assert_eq!((p1.cold_neurons, p1.unload_bytes, p1.skip_bytes_per_pass), (0, 0, 0));
    let p9 = unload_plan(&s, 0.995, 64);
    eprintln!("plan 0.995/64: {}", p9.to_json(false));

    // reset: new window
    let r = handle_request(eng.activity.as_deref(), &json!({"reset": true}));
    assert_eq!(r["reset"], json!(true));
    let z = eng.activity_snapshot();
    assert_eq!((z.rows, z.passes, z.resid_rows), (0, 0, 0));
    assert!(z.neuron.iter().chain(&z.dn_head).chain(&z.attn_head).chain(&z.resid).all(|x| *x == 0.0));
    let g = eng.generate(&ps[0].1, 4, false).unwrap();
    assert_eq!(eng.activity_snapshot().rows, (ps[0].1.len() + g.tokens.len() - 1) as u64);
}

/// the MTP draft layer has its own region + counters (MTP passes only); per-window token counters of embedding rows read and
/// emitted tokens; `snapshot_lite` = `snapshot` without the down norms; the live view needs no copy.
#[test]
fn mtp_layer_and_token_counts() {
    let Some(eng) = open(true, false) else { return };
    let tap = eng.activity.clone().unwrap();
    let c = &eng.cfg;
    let p = &prompts()[0].1;
    let m0 = tap.mtp_counters();
    let g = eng.generate(p, 8, false).unwrap();
    assert_eq!(tap.mtp_counters(), m0, "plain decode runs no MTP layer");
    {
        let e = &tap.embed_counts;
        assert!(e.total() as usize >= p.len() + g.tokens.len() - 1, "embedding rows read: {} for {} + {}", e.total(), p.len(), g.tokens.len());
        assert!(p.iter().all(|t| e.get(*t) > 0));
    }
    let a = generate_mtp(&eng, p, 24, DraftOverride::None, true).unwrap();
    let (r1, p1) = tap.mtp_counters();
    eprintln!("mtp layer: rows {} passes {} (tokens {})", r1.wrapping_sub(m0.0), p1.wrapping_sub(m0.1), a.tokens.len());
    assert!(p1 > m0.1 && r1 > m0.0);
    tap.with_mtp_live(|v| {
        assert_eq!(v.len(), c.inter + c.heads);
        assert!(v.iter().all(|x| x.is_finite() && *x >= 0.0));
        assert!(v[..c.inter].iter().sum::<f32>() > 0.0 && v[c.inter..].iter().all(|x| *x > 0.0));
    });
    // main counters move with the live view; nothing is copied
    let (r, _, _) = tap.counters();
    let s = tap.snapshot_lite();
    let f = tap.snapshot();
    assert!(s.down_norm.is_empty() && !f.down_norm.is_empty());
    assert!(s.vocab.is_empty() && s.embed_rows.is_empty() && f.embed_rows.len() == c.vocab);
    assert_eq!((s.neuron.clone(), s.dn_head.clone(), s.attn_head.clone(), s.resid.clone(), s.rows), (f.neuron, f.dn_head, f.attn_head, f.resid, f.rows));
    tap.with_live(|v| assert_eq!(v.len(), s.neuron.len() + s.dn_head.len() + s.attn_head.len() + s.resid.len()));
    assert!(r > 0);
    // per-window counters: log_out counts, take_tokens returns them sparse and zeroes both
    tap.reset();
    tap.log_out(&[3, 4, 3]);
    assert_eq!((tap.out_counts.get(3), tap.out_counts.get(4), tap.out_counts.total()), (2, 1, 3));
    let (o, e) = tap.take_tokens();
    assert_eq!(o, vec![(3, 2), (4, 1)]);
    assert!(e.is_empty());
    assert_eq!(tap.out_counts.total(), 0);
    // u32 counters (2026-11-01 heat-on-disk): exact past the old u16 limit; heap-backed without a recorder
    let t = streamdense_model::activity::TokCounts::new(2);
    t.add(&vec![1u32; 65_537]);
    assert_eq!((t.get(1), t.saturated()), (65_537, 0));
    // the gateway's counters live in the recorder's arena region (model memory) and are read in place
    tap.log_out(&[5, 5]);
    assert_eq!(tap.out_counts.with_counts(|c| c[5].load(std::sync::atomic::Ordering::Relaxed)), 2);
    assert!(format!("{:?}", tap.out_counts).contains("arena true"));
    tap.reset();
    assert_eq!(tap.out_counts.get(5), 0);
}

fn hist(v: usize, ids: &[u32]) -> Vec<u32> { let mut h = vec![0u32; v]; for &t in ids { h[t as usize] += 1; } h }

/// coverage: `vocab` (emitted tokens via log_out, as the gateway worker wraps on_tok) for plain, MTP depth 1 and
/// deep spec; `embed_rows` (plain: = prompt + fed tokens); MTP layer fields; scripts; file round trip; reset.
#[test]
fn vocab_embed_mtp_counts() {
    let Some(eng) = open(true, false) else { return };
    let tap = eng.activity.clone().unwrap();
    let c = &eng.cfg;
    let v = c.vocab;
    let ps = prompts();
    let p = &ps[1].1;
    for mode in ["plain", "mtp1", "deep3"] {
        eng.reset();
        tap.reset();
        let mut out: Vec<u32> = Vec::new();
        let mut on_tok = |t: u32| { tap.log_out(&[t]); out.push(t); true };
        match mode {
            "plain" => { eng.feed_stream(p, 40, &[], &mut on_tok).unwrap(); }
            "mtp1" => { eng.feed_stream_mtp_ov(p, 40, &[], &mut on_tok, &DraftOverride::None).unwrap(); }
            _ => { eng.feed_stream_mtp(p, 40, &[], &mut on_tok).unwrap(); } // spec_depth 3 -> deep loop
        }
        let s = tap.snapshot();
        let read: u64 = s.embed_rows.iter().map(|&x| x as u64).sum();
        eprintln!("{}: {} emitted, {} embedding rows read, mtp rows {} passes {}", mode, out.len(), read, s.mtp_rows, s.mtp_passes);
        assert_eq!((s.vocab.len(), s.embed_rows.len()), (v, v));
        assert_eq!(s.vocab, hist(v, &out), "{} vocab", mode);
        if mode == "plain" {
            // prompt + every emitted token except the held-back last one
            let mut fed = p.clone();
            fed.extend_from_slice(&out[..out.len() - 1]);
            assert_eq!(s.embed_rows, hist(v, &fed));
            assert_eq!((s.mtp_rows, s.mtp_passes), (0, 0));
            assert!(s.mtp_neuron.iter().chain(&s.mtp_head).all(|x| *x == 0.0));
        } else {
            assert!(s.embed_rows.iter().map(|&x| x as usize).sum::<usize>() >= p.len() + out.len() - 1);
            assert!(s.mtp_rows > 0 && s.mtp_passes > 0);
            assert_eq!((s.mtp_neuron.len(), s.mtp_head.len(), s.mtp_down_norm.len()), (c.inter, c.heads, c.inter));
            assert!(s.mtp_neuron.iter().chain(&s.mtp_head).all(|x| x.is_finite() && *x >= 0.0));
            assert!(s.mtp_neuron.iter().sum::<f32>() > 0.0 && s.mtp_head.iter().all(|x| *x > 0.0));
            assert!(s.mtp_down_norm.iter().all(|x| *x > 0.0));
        }
    }

    // scripts (the tiny model has no tokenizer: synthetic classes) + by-script totals + file round trip
    let cls: Vec<u8> = (0..v).map(|i| (i % SCRIPT_NAMES.len()) as u8).collect();
    assert!(!tap.set_scripts(vec![0; v - 1]) && tap.set_scripts(cls.clone()) && !tap.set_scripts(cls.clone()));
    let s = tap.snapshot();
    let by = s.vocab_by_script();
    assert_eq!(by.len(), SCRIPT_NAMES.len());
    for (k, (name, n)) in by.iter().enumerate() {
        assert_eq!(name, SCRIPT_NAMES[k]);
        assert_eq!(*n, (0..v).filter(|i| i % SCRIPT_NAMES.len() == k).map(|i| s.vocab[i] as u64).sum::<u64>());
    }
    assert_eq!(by.iter().map(|x| x.1).sum::<u64>(), s.vocab.iter().map(|&x| x as u64).sum::<u64>());
    assert_eq!(ActivityStats { scripts: Vec::new(), ..s.clone() }.vocab_by_script()[0].0, "unknown");
    let path = std::env::temp_dir().join(format!("sd_activity_w25_{}.safetensors", std::process::id())).display().to_string();
    s.write(&path).unwrap();
    let back = ActivityStats::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!((back.vocab.clone(), back.embed_rows.clone(), back.scripts.clone()), (s.vocab.clone(), s.embed_rows.clone(), s.scripts.clone()));
    assert_eq!((back.mtp_rows, back.mtp_passes), (s.mtp_rows, s.mtp_passes));
    assert_eq!((back.mtp_neuron.clone(), back.mtp_head.clone(), back.mtp_down_norm.clone()), (s.mtp_neuron.clone(), s.mtp_head.clone(), s.mtp_down_norm.clone()));
    assert_eq!(back.neuron, s.neuron);
    let j = s.summary();
    assert_eq!(j["out_tokens"], json!(s.vocab.iter().map(|&x| x as u64).sum::<u64>()));

    // reset: every new field back to 0
    tap.reset();
    let z = tap.snapshot_lite();
    assert!(z.vocab.is_empty() && z.embed_rows.is_empty() && z.scripts.is_empty()); // lite: no dense token counts
    assert_eq!((tap.out_counts.total(), tap.embed_counts.total()), (0, 0));
    let zf = tap.snapshot();
    assert!(zf.vocab.iter().chain(&zf.embed_rows).all(|x| *x == 0) && zf.scripts == cls);
    assert!(z.mtp_neuron.iter().chain(&z.mtp_head).all(|x| *x == 0.0) && (z.mtp_rows, z.mtp_passes) == (0, 0));
    assert!(z.mtp_down_norm.is_empty());
}

#[test]
fn script_classifier() {
    let k = |s: &str| SCRIPT_NAMES[script_class(s.as_bytes(), false) as usize];
    for (s, want) in [("hello", "ascii"), ("fn main()", "ascii"), (" Việt", "latin"), ("ệ", "latin"), (" café", "latin"),
                      (" привет", "cyrillic"), ("中文", "han"), ("ひらがな", "kana"), ("カタ", "kana"), ("한국어", "hangul"),
                      ("ไทย", "thai"), ("عربي", "arabic"), ("🙂", "other"), (" \u{0301}", "latin"), ("—", "other")] {
        assert_eq!(k(s), want, "{:?}", s);
    }
    assert_eq!(SCRIPT_NAMES[script_class(&[0xE4, 0xB8], false) as usize], "han"); // truncated Han lead
    assert_eq!(SCRIPT_NAMES[script_class(&[0xE3], false) as usize], "ascii"); // single byte = byte fallback
    assert_eq!(SCRIPT_NAMES[script_class(&[0xE3, 0x81], false) as usize], "kana"); // 3040-307F
    assert_eq!(SCRIPT_NAMES[script_class(&[0x80, 0x81], false) as usize], "other"); // starts mid-character
    assert_eq!(SCRIPT_NAMES[script_class(b"<|im_end|>", true) as usize], "special");
    assert_eq!(SCRIPT_NAMES[script_class(b"", false) as usize], "other");
    let cls = script_classes(3, |id| (["a", "中", "й"][id as usize].as_bytes(), false));
    assert_eq!(cls, vec![0, 3, 2]);
}

/// batch region + recorder gate (2026-11-01): add_to_batch sums whole jobs (f32 sums, MTP, token counts) into a
/// second region in model memory; reset zeroes only the recorder; reset_batch zeroes the batch. A closed gate holds
/// the next pass back until it opens (the gateway's hand-off to sd-heat); generation output is unchanged.
#[test]
fn batch_region_and_gate() {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if !std::path::Path::new(&format!("{}/config.json", d)).exists() { eprintln!("SKIP: {} missing", d); return; }
    let eng = QwenEngine::open(&d, QwenOptions { max_ctx: 2048, activity_batch: true, ..Default::default() }).expect("open tiny");
    let tap = eng.activity_tap().unwrap();
    assert!(tap.has_batch());
    let p = &prompts()[0].1;
    eng.activity_reset();
    tap.reset_batch();
    let g1 = eng.generate(p, 16, false).unwrap();
    tap.log_out(&g1.tokens);
    let j1 = tap.with_live(|v| v.to_vec());
    let (o1, e1) = (tap.out_counts.load(), tap.embed_counts.load());
    tap.add_to_batch();
    tap.reset();
    assert!(tap.with_live(|v| v.iter().all(|x| *x == 0.0)));
    // second job, waiting on a closed gate (opened by another thread after 200 ms)
    let gate = &streamdense_model::activity::GATE;
    let w0 = gate.waits.load(std::sync::atomic::Ordering::Relaxed);
    gate.close();
    let t0 = std::time::Instant::now();
    let h = std::thread::spawn(|| { std::thread::sleep(std::time::Duration::from_millis(200)); streamdense_model::activity::GATE.open(); });
    let g2 = eng.generate(p, 16, false).unwrap();
    h.join().unwrap();
    assert!(t0.elapsed().as_millis() >= 190 && gate.waits.load(std::sync::atomic::Ordering::Relaxed) > w0, "the pass waited for the gate");
    assert_eq!(g1.tokens, g2.tokens, "same output after the wait");
    tap.log_out(&g2.tokens);
    let j2 = tap.with_live(|v| v.to_vec());
    let (o2, e2) = (tap.out_counts.load(), tap.embed_counts.load());
    tap.add_to_batch();
    tap.with_batch(|b| for i in 0..b.len() { assert_eq!(b[i], j1[i] + j2[i], "batch sum at {}", i); });
    tap.with_batch_counts(|o, e| {
        for i in 0..o.len() { assert_eq!((o[i], e[i]), (o1[i] + o2[i], e1[i] + e2[i]), "token counts at {}", i); }
        assert!(o.iter().any(|x| *x > 0) && e.iter().any(|x| *x > 0));
    });
    tap.reset_batch();
    tap.with_batch(|b| assert!(b.iter().all(|x| *x == 0.0)));
}
