//! streamdense qwen budget | check | teacher | dump | bench | run   (Qwen3.8-27B engine, plan sections 9 and 10 U16)
//!   common:  --model DIR (default models/qwen38_27b_mlx4) --cap 17 --reserve GB --max-ctx 8192 --kv bf16|q8 [--mtp]
//!            --prefill-k N --threads N --abort-gb 18 --swap-gb 7 --out FILE
//!            --spec-depth N|auto (with --mtp: MTP drafts verified per pass, default 1 = draft 1 / verify 2; auto = adaptive
//!            0..=7 per pass, spec_deep.rs planner; [--spec-cost ..] [--draft-table DIR] [--draft-table-mb 8]; N >= 2 = chained
//!            drafts, one exact verify pass of N + 1 rows, DeltaNet rollback by replay; spec_deep.rs)
//!            --gemv mma|old (matmul family of every exact pass: decode, verify, MTP head; default mma = gemv_mma_q4.metal,
//!            K-invariant, not bit-exact vs MLX; old = gemv_q4_bf16.metal, bit-exact vs MLX qmv, use it for check / dump)
//!            --prefill-t N (fast prefill chunk rows, default 128) | --prefill-exact (K <= 8 GEMV prefill, bit-exact vs
//!            MLX stepwise; the fast GEMM prefill is not: it is held to the teacher bar, plan U25)
//!            --verify-fused (fused K-row verify matmuls for exact passes of 3..8 rows, bit-identical; default off)
//!            --draft-vocab FILE (with --mtp: pruned MTP draft head over the keep-list of `streamdense vocab keep`;
//!            verify stays full-vocab, output unchanged; default off)
//!   budget:  dry run (headers only): prints the arena table, exit 3 if it does not fit
//!   check:   --ref engine/refs/qwen38_mlx4_greedy.json --prompts engine/refs/qwen38_prompts.json [--ids short,code] [--nnew N] [--nosync]
//!            margin-aware token bar (plan 9.2 a) + pipelined == sync (9.2 c)
//!            with --mtp also (P2 bar 9.2): MTP tokens == engine plain tokens for the real MTP head (pipelined + sync)
//!            and the oracle draft override at --mtp-oracle 0,0.5,0.9,1 ("none" = skip); accept and reject both > 0
//!   teacher: --ref engine/refs/qwen38_teacher_ctx1k.npz,engine/refs/qwen38_teacher_readme2k.npz [--topk 32]
//!            top-1 agreement >= 99.5% and KL(top-32) <= 1e-3 (9.2 b)
//!   dump:    --tokens 1,2,3 | --prompts FILE --id NAME, --layers 0,3 (default all), --out DIR/dump.safetensors [--json FILE]
//!   bench:   --prompts FILE [--ids short,ctx1k,readme2k] [--ngen 64]   (--mtp: + MTP pipelined/sync, acceptance)
//!   run:     --tokens 1,2,3 | --prompts FILE --id NAME   [--nnew 64]
//!   spec-bench (needs --mtp): [--ctx 1024,4096] [--k 1,2,4,8,16] [--depth 4] [--n 200] [--reps 20]
//!            [--warmup 3] [--prompts FILE (27B default engine/refs/qwen38_spec_prompts.json: short, code, math, vietnamese, agent,
//!            ctx1k, readme2k)] [--ids a,b] [--ctx-text FILE,FILE] [--agree] [--skip-verify] [--skip-accept]
//!            [--no-real-mtp] [--out-json PATH]. (a) verify-pass time vs K at each ctx (GEMV and GEMM paths, head on all
//!            K rows); (b) chained-MTP acceptance by teacher forcing over N greedy tokens per prompt; projected tok/s.
//!            --max-ctx defaults to what the run needs.
//!            [--e2e 1,2,3,4,auto] [--e2e-sync] (c) real end-to-end spec decode per policy (d or auto = adaptive 0..=7, with
//!            --draft-table DIR and --spec-cost "t1,..,t8[;mtp=MS][;round=MS]") over --n tokens per prompt, reporting
//!            tok/s, tokens/pass, the drafts-per-pass histogram, table vs MTP draft share, GPU idle ms per round; d >= 2 and
//!            auto run the pipelined loop (--e2e-sync: fixed depths through the old synchronous loop) (d = 1: the
//!            pipelined draft 1 / verify 2 path; d >= 2: spec_deep.rs, one exact verify pass of d + 1 rows), against plain
//!            greedy (pipelined): tok/s, tokens/pass, accepted-drafts histogram, DeltaNet replay GPU ms per row, and the
//!            emitted tokens must equal plain greedy for every prompt (else exit 1). Opens the engine with spec_depth =
//!            max d. Only (c): add --skip-verify --skip-accept. Per prompt and depth also: tokens / pass and the chained-MTP
//!            acceptance per step 1..d (from the accepted-drafts histogram). --gemv sets the family of every pass;
//!            [--plain-old] then reopens the engine with --gemv old (one model at a time) for plain greedy tok/s.
//!   serve:   load once (--max-ctx default 4096, --abort-gb / --swap-gb default [ram] cap_gb / swap_gb of streamdense.toml),
//!            then JSON lines on stdin / stdout:
//!            in  {"op":"gen","tokens":[..],"nnew":N,"stop":[ids],"continue":bool,"mtp":bool (needs --mtp)} | {"op":"stop"} | {"op":"reset"} | {"op":"stats"}
//!            out {"ready":..} once, {"tok":id} per token, then {"done":true,..}; {"error":..}; logs go to stderr only.
//!            continue = feed only the new tokens (after the previous reply's last token) on the kept KV / DeltaNet state.
//!            text in: {"op":"chat","messages":[{"role","content"}..],"think":bool,"nnew":N,"mtp":bool,"continue":bool}
//!            -> {"start":..}, {"text":".."} pieces, {"done":true,..}; tokenizer + chat template in Rust
//!            (streamdense-tokenizer, --tokenizer DIR, default the model dir). Full protocol: docs/SERVE_PROTOCOL.md
//!            gen / chat also take "spec_depth":N|"auto" (1..=--spec-depth, default --spec-depth) for an MTP request.
//!            {"op":"activity","path":FILE?,"plan":true|{"keep":0.995,"granularity":64,"layers":bool}?,"reset":bool?} -> activity usage
//!            summary (+ full snapshot written to FILE, + dry-run unload plan; activity.rs handle_request). --no-activity: recorder off.
//!            --fold-ids (tests only): chat ids fed modulo the model vocab (tiny test model + the 27B tokenizer).
//! Token ids for check/bench/run come from the reference prompt files in engine/refs/ (made with MLX / mlx-lm).
use crate::{conditions, ids, Args};
use serde_json::{json, Value};
use std::collections::HashMap;
use streamdense_core::{alloc, footprint, gb};
use streamdense_model::qwen_engine::{DumpSpec, KvFormat, QwenEngine, QwenOptions};
use streamdense_model::qwen_layers::GemvImpl;
use streamdense_model::qwen_engine::StreamStats;
use streamdense_model::spec::{generate_mtp, DraftOverride, SpecResult};
use streamdense_model::spec_deep::{generate_spec_mode, SpecCost, SpecMode};
use streamdense_tokenizer::{ChatOptions, ChatState, Message, Mode, StreamDecoder, Tokenizer, ENDOFTEXT, IM_END};

pub(crate) const DEFAULT_MODEL: &str = "models/qwen38_27b_mlx4";

fn read_json(p: &str) -> Value {
    let s = std::fs::read_to_string(p).unwrap_or_else(|e| { eprintln!("ERROR: read {}: {}", p, e); std::process::exit(2) });
    serde_json::from_str(&s).unwrap_or_else(|e| { eprintln!("ERROR: parse {}: {}", p, e); std::process::exit(2) })
}
fn die(e: String) -> ! { eprintln!("ERROR: {}", e); std::process::exit(2) }
fn f64s(v: &Value) -> Vec<f64> { v.as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect()).unwrap_or_default() }

/// Ref file naming in engine/refs/: "qwen38" for the default model, "{dir name}_mlxref" otherwise.
fn ref_tag(model: &str) -> (String, bool) {
    let name = std::path::Path::new(model.trim_end_matches('/')).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    if name == "qwen38_27b_mlx4" { ("qwen38".into(), true) } else { (format!("{}_mlxref", name), false) }
}

/// Size of one bf16 ulp at |x| (normal range): 2^(exponent - 7).
fn bf16_ulp(x: f64) -> f64 {
    let a = x.abs();
    if a == 0.0 || !a.is_finite() { return f64::MIN_POSITIVE; }
    2f64.powi(a.log2().floor() as i32 - 7)
}

/// prompts JSON from ref_tokens.py: {"prompts": [{"id": "short", "ids": [...]}, ...]} -> (id, ids) in file order.
fn load_prompts(p: &str) -> Vec<(String, Vec<u32>)> {
    let pj = read_json(p);
    pj["prompts"].as_array().unwrap_or_else(|| die(format!("{}: no \"prompts\" array", p))).iter()
        .map(|r| (r["id"].as_str().map(|s| s.to_string()).unwrap_or_else(|| r["id"].to_string()), ids(&r["ids"]))).collect()
}
fn pick<'a>(all: &'a [(String, Vec<u32>)], want: &Option<String>) -> Vec<&'a (String, Vec<u32>)> {
    match want {
        None => all.iter().collect(),
        Some(w) => w.split(',').filter(|x| !x.is_empty()).map(|x| all.iter().find(|p| p.0 == x)
            .unwrap_or_else(|| die(format!("prompt id {:?} not in the prompts file", x)))).collect(),
    }
}
fn prompt_arg(a: &Args, prompts_default: &str) -> (String, Vec<u32>) {
    match a.s("--tokens") {
        Some(t) => ("tokens".into(), t.split(',').map(|x| x.trim().parse().unwrap()).collect()),
        None => {
            let all = load_prompts(&a.sd("--prompts", prompts_default));
            let id = a.sd("--id", "short");
            all.into_iter().find(|p| p.0 == id).unwrap_or_else(|| die(format!("prompt id {:?} not in the prompts file", id)))
        }
    }
}

// ---------------------------------------------------------------- minimal .npz (np.savez: stored zip, maybe zip64) reader
struct Npy { descr: String, shape: Vec<usize>, data: Vec<u8> }
impl Npy {
    fn u32s(&self) -> Vec<u32> {
        match self.descr.as_str() {
            "<u4" | "<i4" => self.data.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "<u8" | "<i8" => self.data.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap()) as u32).collect(),
            d => die(format!("npy: dtype {} where integers were expected", d)),
        }
    }
    fn f32s(&self) -> Vec<f32> {
        match self.descr.as_str() {
            "<f4" => self.data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "<f8" => self.data.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
            d => die(format!("npy: dtype {} where floats were expected", d)),
        }
    }
}
fn le16(b: &[u8], o: usize) -> usize { u16::from_le_bytes([b[o], b[o + 1]]) as usize }
fn le32(b: &[u8], o: usize) -> u64 { u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as u64 }
fn le64(b: &[u8], o: usize) -> u64 { u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) }

fn parse_npy(b: &[u8]) -> Result<Npy, String> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" { return Err("not a .npy member".into()); }
    let (hl, h0) = if b[6] == 1 { (le16(b, 8), 10) } else { (le32(b, 8) as usize, 12) };
    let h = std::str::from_utf8(&b[h0..h0 + hl]).map_err(|e| e.to_string())?;
    let field = |k: &str| h.find(k).map(|i| &h[i + k.len()..]);
    let descr = field("'descr':").and_then(|r| r.split('\'').nth(1)).ok_or("npy: no descr")?.to_string();
    if field("'fortran_order':").map(|r| r.trim_start().starts_with("True")).unwrap_or(false) { return Err("npy: fortran order".into()); }
    let sh = field("'shape':").and_then(|r| r.find('(').zip(r.find(')')).map(|(i, j)| &r[i + 1..j])).ok_or("npy: no shape")?;
    let shape: Vec<usize> = sh.split(',').map(|x| x.trim()).filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect();
    Ok(Npy { descr, shape, data: b[h0 + hl..].to_vec() })
}

/// Members of an uncompressed .npz, by name without ".npy". Uses the central directory (zip64 aware).
fn read_npz(path: &str) -> Result<HashMap<String, Npy>, String> {
    let b = std::fs::read(path).map_err(|e| format!("read {}: {}", path, e))?;
    let eocd = (0..b.len().saturating_sub(21)).rev().find(|&i| le32(&b, i) == 0x0605_4b50).ok_or("npz: no end of central directory")?;
    let (mut n, mut cd) = (le16(&b, eocd + 10) as u64, le32(&b, eocd + 16));
    if eocd >= 20 && le32(&b, eocd - 20) == 0x0706_4b50 {
        let z = le64(&b, eocd - 12) as usize;
        if le32(&b, z) != 0x0606_4b50 { return Err("npz: bad zip64 end record".into()); }
        n = le64(&b, z + 32); cd = le64(&b, z + 48);
    }
    let mut out = HashMap::new();
    let mut o = cd as usize;
    for _ in 0..n {
        if le32(&b, o) != 0x0201_4b50 { return Err("npz: bad central directory entry".into()); }
        let method = le16(&b, o + 10);
        let (mut csize, mut usize_, mut loff) = (le32(&b, o + 20), le32(&b, o + 24), le32(&b, o + 42));
        let (nl, xl, cl) = (le16(&b, o + 28), le16(&b, o + 30), le16(&b, o + 32));
        let name = String::from_utf8_lossy(&b[o + 46..o + 46 + nl]).to_string();
        let mut x = o + 46 + nl;
        while x + 4 <= o + 46 + nl + xl {
            let (id, sz) = (le16(&b, x), le16(&b, x + 2));
            if id == 1 {
                let mut p = x + 4;
                if usize_ == 0xFFFF_FFFF { usize_ = le64(&b, p); p += 8; }
                if csize == 0xFFFF_FFFF { csize = le64(&b, p); p += 8; }
                if loff == 0xFFFF_FFFF { loff = le64(&b, p); }
            }
            x += 4 + sz;
        }
        if method != 0 { return Err(format!("npz: member {} is compressed (np.savez_compressed is not supported)", name)); }
        let l = loff as usize;
        let d0 = l + 30 + le16(&b, l + 26) + le16(&b, l + 28);
        let _ = usize_;
        let npy = parse_npy(&b[d0..d0 + csize as usize]).map_err(|e| format!("{}: {}", name, e))?;
        out.insert(name.trim_end_matches(".npy").to_string(), npy);
        o += 46 + nl + xl + cl;
    }
    Ok(out)
}

// ---------------------------------------------------------------- main
pub fn main(a: &Args) {
    let cmd = a.0.get(2).cloned().unwrap_or_default();
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
    let model = a.sd("--model", &format!("{}/{}", root, DEFAULT_MODEL));
    let (tag, is_main) = ref_tag(&model);
    let prompts_default = format!("{}/engine/refs/{}_prompts.json", root, tag);
    let greedy_default = format!("{}/engine/refs/{}.json", root, if is_main { "qwen38_mlx4_greedy".to_string() } else { format!("{}_greedy", tag) });

    let mut opt = QwenOptions::default();
    opt.cap = (a.f("--cap", 17.0) * 1e9) as u64;
    if let Some(r) = a.s("--reserve") { opt.reserve = (r.parse::<f64>().unwrap() * 1e9) as u64; }
    opt.max_ctx = a.f("--max-ctx", if cmd == "serve" { 4096.0 } else { 8192.0 }) as usize;
    opt.kv = match a.sd("--kv", "bf16").as_str() { "bf16" => KvFormat::Bf16, "q8" => KvFormat::Q8, k => die(format!("--kv {}: want bf16|q8", k)) };
    opt.mtp = a.has("--mtp");
    opt.activity = !a.has("--no-activity");
    spec_opts(a, &mut opt, None, None);
    opt.gemv = GemvImpl::parse(&a.sd("--gemv", "mma")).unwrap_or_else(|e| die(e));
    if let Some(k) = a.s("--prefill-k") { opt.prefill_k = k.parse().unwrap(); }
    if let Some(t) = a.s("--prefill-t") { opt.prefill_t = t.parse().unwrap(); }
    opt.prefill_exact = a.has("--prefill-exact");
    opt.verify_fused = a.has("--verify-fused");
    if let Some(t) = a.s("--threads") { opt.load_threads = t.parse().unwrap(); }
    opt.draft_vocab = a.s("--draft-vocab").map(|p| p.into());
    let out_path = a.s("--out").or_else(|| a.s("--out-json"));
    // spec-bench: arguments, prompts and the context text (tokenized here, before the engine opens); sets max_ctx.
    // Default prompt set for the 27B: engine/refs/qwen38_spec_prompts.json (`streamdense prompts spec`)
    let spec_prompts = if is_main { format!("{}/engine/refs/qwen38_spec_prompts.json", root) } else { prompts_default.clone() };
    let sb = if cmd == "spec-bench" { Some(spec_bench_prep(a, &model, root, &spec_prompts, &mut opt)) } else { None };
    let settings = json!({"model": model, "cap_gb": gb(opt.cap), "reserve_gb": gb(opt.reserve), "max_ctx": opt.max_ctx,
        "kv": a.sd("--kv", "bf16"), "mtp": opt.mtp, "prefill_k": opt.prefill_k, "prefill_t": opt.prefill_t, "prefill_exact": opt.prefill_exact,
        "load_threads": opt.load_threads, "spec_depth": opt.spec_depth, "spec_auto": opt.spec_auto, "gemv": opt.gemv.name(),
        "draft_table": opt.draft_table.as_ref().map(|p| p.display().to_string()),
        "draft_vocab": opt.draft_vocab.as_ref().map(|p| p.display().to_string())});

    // dry run first: every command prints the arena table, `budget` stops here
    let bud = QwenEngine::budget(&model, &opt).unwrap_or_else(|e| die(e));
    let fits = bud.check();
    eprintln!("budget ({}):\n{}", model, bud.report());
    let mut out = json!({"settings": settings, "budget": bud.report(),
        "budget_items": bud.items.iter().map(|(n, b)| json!({"name": n, "bytes": b})).collect::<Vec<_>>(),
        "arena_bytes": bud.arena_bytes(), "budget_fits": fits.is_ok()});
    if cmd == "budget" {
        match &fits { Ok(b) => println!("fits: arena {:.3} GB + reserve {:.3} GB <= cap {:.3} GB", gb(*b), gb(opt.reserve), gb(opt.cap)),
                      Err(e) => println!("DOES NOT FIT: {}", e.lines().next().unwrap_or("")) }
        if let Some(p) = &out_path { std::fs::write(p, serde_json::to_string_pretty(&out).unwrap()).unwrap(); println!("wrote {}", p); }
        std::process::exit(if fits.is_ok() { 0 } else { 3 });
    }
    if let Err(e) = fits { die(e); }

    // references are read before the engine opens (keeps the heap limit meaningful)
    let refs_check = if cmd == "check" {
        let refj = read_json(&a.sd("--ref", &greedy_default));
        let rows: HashMap<String, Value> = refj["rows"].as_array().unwrap_or_else(|| die("ref JSON has no \"rows\"".into())).iter()
            .map(|r| (r["id"].as_str().unwrap_or("").to_string(), r.clone())).collect();
        Some((refj["prefill"].clone(), rows))
    } else { None };
    let refs_teacher = if cmd == "teacher" {
        let list = a.s("--ref").unwrap_or_else(|| ["ctx1k", "readme2k"].iter().map(|i| format!("{}/engine/refs/{}_teacher_{}.npz", root, tag, i)).collect::<Vec<_>>().join(","));
        Some(list.split(',').filter(|x| !x.is_empty()).map(|p| (p.to_string(), read_npz(p).unwrap_or_else(|e| die(e)))).collect::<Vec<_>>())
    } else { None };
    if cmd == "dump" {
        let layers: Vec<usize> = a.s("--layers").map(|l| l.split(',').map(|x| x.trim().parse().unwrap()).collect()).unwrap_or_default();
        let path = out_path.clone().unwrap_or_else(|| format!("{}/engine/refs/dumps/engine_{}/dump.safetensors", root, tag));
        if let Some(d) = std::path::Path::new(&path).parent() { let _ = std::fs::create_dir_all(d); }
        opt.dump = Some(DumpSpec { layers, path });
    }

    // serve: tokenizer + chat template (loaded before the engine, like the refs above)
    let tok = if cmd == "serve" { Some(load_tokenizer(a, &model, root)) } else { None };

    // serve: --abort-gb / --swap-gb default from streamdense.toml [ram]; the test commands keep their own 18 / 7 GB
    let ram = if cmd == "serve" { let c = crate::load_config(); eprintln!("config: {}", c.source_str()); c.cfg.ram } else { Default::default() };
    let abort = (a.f("--abort-gb", if cmd == "serve" { ram.cap_gb } else { 18.0 }) * 1e9) as u64;
    let mon = footprint::Monitor::start(abort, (a.f("--swap-gb", ram.swap_gb) * 1e9) as u64, 250);
    eprintln!("conditions:\n{}", conditions());
    let t_open = std::time::Instant::now();
    #[allow(unused_mut)] // reset() may be &self or &mut self in U15
    let mut eng = QwenEngine::open(&model, opt.clone()).unwrap_or_else(|e| die(e));
    let open_s = t_open.elapsed().as_secs_f64();
    eprintln!("opened in {:.2} s | footprint {:.3} GB | heap {:.1} MB", open_s, gb(footprint::self_footprint()), alloc::heap_now() as f64 / 1e6);
    // the decode loop may not grow the CPU heap past this (dump merges its per-step parts on the CPU: no limit)
    if cmd != "dump" { alloc::set_limit_extra(256 << 20); }
    out["conditions"] = json!(conditions());
    out["open_s"] = json!(open_s);
    out["footprint_after_open_gb"] = json!(gb(footprint::self_footprint()));
    let mut pass = true;

    match cmd.as_str() {
        "check" => {
            let (ref_prefill, rows) = refs_check.unwrap();
            let all = load_prompts(&a.sd("--prompts", &prompts_default));
            let nnew_cap = a.s("--nnew").map(|x| x.parse::<usize>().unwrap());
            let mut res = vec![];
            let (mut n_ok, mut n_exact, mut n_pipe, mut n_all) = (0, 0, 0, 0);
            let mut mtp_tot = (0usize, 0usize, 0usize, 0usize); // equal runs, accepted, rejected, runs
            let oracle_ps: Vec<f64> = match a.sd("--mtp-oracle", "0,0.5,0.9,1").as_str() {
                "none" | "" => vec![], l => l.split(',').map(|x| x.trim().parse().unwrap()).collect() };
            for (id, prompt) in pick(&all, &a.s("--ids")) {
                let r = rows.get(id).unwrap_or_else(|| die(format!("prompt id {:?} not in the ref", id)));
                let mut rt = ids(&r["tok"]);
                if let Some(n) = nnew_cap { rt.truncate(n); }
                let (rm, rt1) = (f64s(&r["margin"]), f64s(&r["top1"]));
                eng.reset();
                let g = eng.generate(prompt, rt.len(), true).unwrap_or_else(|e| die(e));
                let gs = if a.has("--nosync") { None } else { eng.reset(); Some(eng.generate(prompt, rt.len(), false).unwrap_or_else(|e| die(e))) };
                let em: Vec<f64> = g.margins.iter().map(|&m| m as f64).collect();
                let n_cmp = g.tokens.len().min(rt.len());
                let first = (0..n_cmp).find(|&i| g.tokens[i] != rt[i]);
                let exact = first.is_none() && g.tokens.len() >= rt.len();
                // plan 9.2 (a): a divergence is allowed only where MLX has an exact bf16 tie (margin 0)
                // and the engine's own margin there is <= 1 bf16 ulp (ulp size taken at the MLX top-1 logit)
                let at = first.map(|f| {
                    let ulps = em.get(f).map(|m| m / bf16_ulp(rt1.get(f).copied().unwrap_or(1.0)));
                    (f, rm.get(f).copied(), em.get(f).copied(), ulps)
                });
                let tie_ok = at.map(|(_, r_m, _, ulps)| r_m == Some(0.0) && ulps.map(|u| u <= 1.0 + 1e-9).unwrap_or(false)).unwrap_or(false);
                let ok = exact || tie_ok;
                let short = first.is_none() && g.tokens.len() < rt.len();
                let pipe_eq = gs.as_ref().map(|s| s.tokens == g.tokens);
                let ms = g.gen_wall_s * 1e3 / g.tokens.len().max(1) as f64;
                let ms_sync = gs.as_ref().map(|s| s.gen_wall_s * 1e3 / s.tokens.len().max(1) as f64);
                n_all += 1; if ok { n_ok += 1; } if exact { n_exact += 1; } if pipe_eq != Some(false) { n_pipe += 1; }
                println!("prompt {:>8} (P={:4}): {:3}/{:3} tok, ref bar: {}{}{}, pipelined == sync: {}, {:.2} ms/token pipelined{}",
                    id, prompt.len(), g.tokens.len(), rt.len(), if ok { "PASS" } else { "FAIL" },
                    if exact { " (identical)".to_string() } else if short { " (engine stopped early)".to_string() } else { String::new() },
                    at.map(|(f, r_m, e_m, u)| format!(" (first diff @{}: ref margin {:?}, ours {:?} = {:.2} ulp{})", f, r_m, e_m, u.unwrap_or(f64::NAN),
                        if tie_ok { ", bf16 tie: allowed" } else { "" })).unwrap_or_default(),
                    pipe_eq.map(|x| x.to_string()).unwrap_or("skipped".into()), ms,
                    ms_sync.map(|m| format!(", {:.2} sync", m)).unwrap_or_default());
                res.push(json!({"id": id, "prompt_tokens": prompt.len(), "ref_n": rt.len(), "tokens": g.tokens,
                    "tokens_sync": gs.as_ref().map(|s| s.tokens.clone()), "identical_to_ref": exact, "passes_bar": ok,
                    "engine_short": short, "first_diff": first, "ref_margin_at_diff": at.and_then(|x| x.1),
                    "margin_at_diff": at.and_then(|x| x.2), "margin_ulps_at_diff": at.and_then(|x| x.3), "allowed_tie": tie_ok,
                    "margins": em, "pipelined_equals_sync": pipe_eq, "ms_per_token_wall": ms, "tok_per_s": 1e3 / ms,
                    "ms_per_token_gpu": g.gpu_ms.iter().map(|&x| x as f64).sum::<f64>() / g.gpu_ms.len().max(1) as f64,
                    "sync_ms_per_token_wall": ms_sync}));
                if opt.mtp {
                    let seq: Vec<u32> = prompt.iter().chain(g.tokens.iter()).copied().collect();
                    let mut runs: Vec<(String, DraftOverride, bool)> = vec![("head pipe".into(), DraftOverride::None, true),
                                                                          ("head sync".into(), DraftOverride::None, false)];
                    for &p in &oracle_ps { runs.push((format!("oracle {} pipe", p), DraftOverride::Oracle { seq: seq.clone(), p, seed: 7 }, true)); }
                    let mut mj = vec![];
                    for (name, ov, pipe) in runs {
                        let m = generate_mtp(&eng, prompt, g.tokens.len(), ov, pipe).unwrap_or_else(|e| die(e));
                        let eq = m.tokens == g.tokens;
                        let first = (0..m.tokens.len().min(g.tokens.len())).find(|&i| m.tokens[i] != g.tokens[i]);
                        mtp_tot.3 += 1; if eq { mtp_tot.0 += 1; } mtp_tot.1 += m.accepted; mtp_tot.2 += m.rejects;
                        println!("    mtp {:>16}: == plain {} {}| {} passes, acc {} rej {} (acceptance {:.3}, {:.2} tok/pass, {} overridden), {:.2} ms/token",
                            name, eq, first.map(|f| format!("(first diff @{}) ", f)).unwrap_or_default(), m.passes, m.accepted, m.rejects,
                            m.acceptance, m.tokens_per_pass, m.overridden, m.wall * 1e3 / m.tokens.len().saturating_sub(1).max(1) as f64);
                        mj.push(spec_json(&name, &m, eq, first));
                    }
                    res.last_mut().unwrap()["mtp"] = json!(mj);
                }
            }
            pass = n_ok == n_all && n_pipe == n_all && n_all > 0;
            if opt.mtp {
                let (ok, acc, rej) = (mtp_tot.0, mtp_tot.1, mtp_tot.2);
                println!("MTP == plain on {} / {} runs, accepted {} rejected {} (both paths: {})", ok, mtp_tot.3, acc, rej, acc > 0 && rej > 0);
                out["mtp_equal_runs"] = json!(ok); out["mtp_runs"] = json!(mtp_tot.3);
                out["mtp_accepted_total"] = json!(acc); out["mtp_rejected_total"] = json!(rej);
                pass &= ok == mtp_tot.3 && acc > 0 && rej > 0;
            }
            println!("ref bar (9.2 a) on {} / {} prompts ({} identical), pipelined == sync on {} / {}", n_ok, n_all, n_exact, n_pipe, n_all);
            out["ref_prefill"] = ref_prefill;
            out["results"] = json!(res);
            out["prompts"] = json!(n_all); out["passing_prompts"] = json!(n_ok); out["identical_prompts"] = json!(n_exact);
            out["pipelined_equals_sync_prompts"] = json!(n_pipe);
        }
        "teacher" => {
            let topk = a.f("--topk", 32.0) as usize;
            let mut res = vec![];
            for (path, z) in refs_teacher.unwrap() {
                let get = |k: &str| z.get(k).unwrap_or_else(|| die(format!("{}: no array {:?}", path, k)));
                let seq = get("ids").u32s();
                let n_prompt = get("n_prompt").u32s().first().copied().unwrap_or(0) as usize;
                let (ri, rv, rl, ra) = (get("top_idx").u32s(), get("top_val").f32s(), get("lse").f32s(), get("argmax").u32s());
                let rk = get("top_idx").shape.get(1).copied().unwrap_or(topk);
                eng.reset();
                let t0 = std::time::Instant::now();
                let e = eng.teacher(&seq, topk).unwrap_or_else(|e| die(e));
                let secs = t0.elapsed().as_secs_f64();
                let n = e.len().min(ra.len());
                let (mut agree, mut agree_c, mut n_c, mut kl_sum, mut kl_max, mut missing) = (0usize, 0usize, 0usize, 0f64, 0f64, 0usize);
                let mut bad: Vec<Value> = Vec::new(); // positions with KL > 1e-2 or a top-1 miss (diagnostics)
                for p in 0..n {
                    let (eidx, eval_, e_lse) = (&e[p].idx, &e[p].val, e[p].lse as f64);
                    let top1 = eidx.first().copied();
                    let hit = top1 == Some(ra[p]);
                    if hit { agree += 1; }
                    if p + 1 >= n_prompt { n_c += 1; if hit { agree_c += 1; } }
                    // KL(P_mlx || Q_engine) over the MLX top-k set plus one tail bin (a lower bound of the full KL);
                    // a token missing from the engine's top-k gets the engine's k-th value (upper bound on its q)
                    let emap: HashMap<u32, f64> = eidx.iter().zip(eval_.iter()).map(|(&i, &v)| (i, v as f64)).collect();
                    let emin = eval_.iter().map(|&v| v as f64).fold(f64::INFINITY, f64::min);
                    let rlse = rl[p] as f64;
                    let (mut kl, mut sp, mut sq) = (0f64, 0f64, 0f64);
                    for j in 0..rk {
                        let (ti, tv) = (ri[p * rk + j], rv[p * rk + j] as f64);
                        let lp = tv - rlse;
                        let lq = match emap.get(&ti) { Some(&v) => v - e_lse, None => { missing += 1; emin - e_lse } };
                        let pp = lp.exp();
                        kl += pp * (lp - lq); sp += pp; sq += lq.exp();
                    }
                    let (pt, qt) = ((1.0 - sp).max(0.0), (1.0 - sq).max(1e-30));
                    if pt > 0.0 { kl += pt * (pt / qt).ln(); }
                    let kl = kl.max(0.0);
                    kl_sum += kl; kl_max = kl_max.max(kl);
                    if (kl > 1e-2 || !hit) && bad.len() < 4000 {
                        bad.push(json!([p, kl, hit, &eidx[..4.min(eidx.len())], &eval_[..4.min(eval_.len())], e_lse,
                            &ri[p * rk..p * rk + 4], &rv[p * rk..p * rk + 4], rlse, seq.get(p), seq.get(p + 1)]));
                    }
                }
                let (top1, top1_c, kl_mean) = (agree as f64 / n.max(1) as f64, agree_c as f64 / n_c.max(1) as f64, kl_sum / n.max(1) as f64);
                let ok = n == seq.len() && top1 >= 0.995 && kl_mean <= 1e-3;
                pass &= ok;
                println!("{}: {} positions ({} prompt), top-1 agreement {:.4}% (continuation {:.4}%), KL(top-{}) mean {:.3e} max {:.3e}, {} missing, {:.1} s -> {}",
                    path, n, n_prompt, 100.0 * top1, 100.0 * top1_c, rk, kl_mean, kl_max, missing, secs, if ok { "PASS" } else { "FAIL" });
                res.push(json!({"ref": path, "positions": n, "ref_positions": seq.len(), "n_prompt": n_prompt, "top1_agree": top1,
                    "top1_agree_continuation": top1_c, "kl_mean": kl_mean, "kl_max": kl_max, "missing_in_engine_topk": missing,
                    "topk": rk, "secs": secs, "passes_bar": ok, "bad_positions": bad}));
            }
            out["results"] = json!(res);
        }
        "dump" => {
            let (id, prompt) = prompt_arg(a, &prompts_default);
            eng.reset();
            let first = eng.prefill(&prompt).unwrap_or_else(|e| die(e));
            let path = opt.dump.as_ref().map(|d| d.path.clone()).unwrap();
            println!("dump of {} ({} tokens), first token {} -> {}", id, prompt.len(), first, path);
            out["prompt"] = json!(id); out["tokens"] = json!(prompt); out["first_token"] = json!(first); out["dump"] = json!(path);
        }
        "bench" => {
            let all = load_prompts(&a.sd("--prompts", &prompts_default));
            let ngen = a.f("--ngen", 64.0) as usize;
            let mut rows = vec![];
            println!("{:>9} {:>6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>6}", "prompt", "P", "total s", "sync ms", "GPU ms", "pipe ms", "pipe t/s", "same");
            for (id, prompt) in pick(&all, &Some(a.sd("--ids", "short,ctx1k,readme2k"))) {
                eng.reset();
                let t0 = std::time::Instant::now();
                let gp = eng.generate(prompt, ngen, true).unwrap_or_else(|e| die(e));
                let total = t0.elapsed().as_secs_f64();
                eng.reset();
                let gs = eng.generate(prompt, ngen, false).unwrap_or_else(|e| die(e));
                let p = gp.gen_wall_s * 1e3 / gp.tokens.len().max(1) as f64;
                let w = gs.gen_wall_s * 1e3 / gs.tokens.len().max(1) as f64;
                let gpu = gs.gpu_ms.iter().map(|&x| x as f64).sum::<f64>() / gs.gpu_ms.len().max(1) as f64;
                let same = gp.tokens == gs.tokens;
                println!("{:>9} {:>6} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>9.2} {:>6}", id, prompt.len(), total, w, gpu, p, 1e3 / p, same);
                rows.push(json!({"id": id, "prompt_len": prompt.len(), "n_gen": gp.tokens.len(), "total_s_pipelined": total,
                    "sync_wall_ms": w, "sync_gpu_ms": gpu, "pipelined_ms_per_token": p, "tok_per_s_pipelined": 1e3 / p,
                    "pipelined_tokens_identical": same}));
                pass &= same;
                if opt.mtp {
                    let mp = generate_mtp(&eng, prompt, ngen, DraftOverride::None, true).unwrap_or_else(|e| die(e));
                    let msy = generate_mtp(&eng, prompt, ngen, DraftOverride::None, false).unwrap_or_else(|e| die(e));
                    let per = |m: &SpecResult| m.wall * 1e3 / m.tokens.len().saturating_sub(1).max(1) as f64;
                    let eq = mp.tokens == gp.tokens && msy.tokens == gp.tokens;
                    println!("{:>9} {:>6}   MTP: pipe {:.2} ms/token ({:.2} tok/s, {:.2}x plain), sync {:.2} ms/token, GPU {:.2} ms/step, acceptance {:.3}, {:.2} tok/pass, == plain {}",
                        "", "", per(&mp), 1e3 / per(&mp), p / per(&mp), per(&msy),
                        msy.gpu_ms.iter().sum::<f64>() / msy.gpu_ms.len().max(1) as f64, mp.acceptance, mp.tokens_per_pass, eq);
                    rows.last_mut().unwrap()["mtp_pipe"] = spec_json("pipe", &mp, mp.tokens == gp.tokens, None);
                    rows.last_mut().unwrap()["mtp_sync"] = spec_json("sync", &msy, msy.tokens == gp.tokens, None);
                    rows.last_mut().unwrap()["mtp_speedup_vs_plain_pipe"] = json!(p / per(&mp));
                    pass &= eq;
                }
            }
            out["rows"] = json!(rows);
        }
        "spec-bench" => {
            let sb = sb.unwrap();
            let e2e = sb.e2e.clone();
            let e2e_sync = sb.e2e_sync;
            let (n, prompts) = (sb.n, sb.prompts.clone());
            spec_bench_run(&eng, sb, &mut out, &mut pass);
            let main_toks = if !e2e.is_empty() { spec_bench_e2e(&eng, &e2e, e2e_sync, n, &prompts, &mut out, &mut pass) } else { vec![] };
            if a.has("--plain-old") && opt.gemv != GemvImpl::Old {
                // one model in RAM at a time: close this engine, reopen with the old matmul family
                drop(eng);
                eng = QwenEngine::open(&model, QwenOptions { gemv: GemvImpl::Old, ..opt.clone() }).unwrap_or_else(|e| die(e));
                spec_bench_plain_old(&eng, n, &prompts, &main_toks, &mut out);
            }
        }
        "serve" => {
            serve(&eng, &mon, &opt, &model, open_s, bud.arena_bytes() as u64, tok.unwrap());
            drop(eng);
            std::process::exit(0);
        }
        "run" => {
            let (id, prompt) = prompt_arg(a, &prompts_default);
            eng.reset();
            let g = eng.generate(&prompt, a.f("--nnew", 64.0) as usize, true).unwrap_or_else(|e| die(e));
            println!("{}", g.tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
            println!("{} ({} prompt tokens): {} tokens, {:.2} ms/token ({:.1} tok/s)", id, prompt.len(), g.tokens.len(),
                g.gen_wall_s * 1e3 / g.tokens.len().max(1) as f64, g.tokens.len() as f64 / g.gen_wall_s);
            out["prompt"] = json!(id); out["tokens"] = json!(g.tokens);
            out["ms_per_token_wall"] = json!(g.gen_wall_s * 1e3 / g.tokens.len().max(1) as f64);
        }
        _ => { eprintln!("usage: streamdense qwen budget|check|teacher|dump|bench|run|serve|spec-bench [options] (see src/qwen.rs)"); std::process::exit(1); }
    }
    let peak = mon.peak();
    println!("peak phys_footprint {:.3} GB (cap {:.3} GB, abort {:.3} GB), CPU heap peak {:.1} MB", gb(peak), gb(opt.cap), gb(abort), alloc::heap_peak() as f64 / 1e6);
    out["peak_phys_footprint_gb"] = json!(gb(peak)); out["cpu_heap_peak_mb"] = json!(alloc::heap_peak() as f64 / 1e6);
    out["pass"] = json!(pass);
    if let Some(p) = out_path.filter(|_| cmd != "dump") { alloc::clear_limit(); std::fs::write(&p, serde_json::to_string_pretty(&out).unwrap()).unwrap(); println!("wrote {}", p); }
    else if cmd == "dump" { if let Some(p) = a.s("--json") { alloc::clear_limit(); std::fs::write(&p, serde_json::to_string_pretty(&out).unwrap()).unwrap(); println!("wrote {}", p); } }
    drop(eng);
    if !pass { std::process::exit(1); }
}

fn spec_json(name: &str, m: &SpecResult, eq: bool, first: Option<usize>) -> Value {
    let n = m.tokens.len().saturating_sub(1).max(1) as f64;
    json!({"run": name, "equal_plain": eq, "first_diff": first, "tokens": m.tokens, "passes": m.passes, "accepted": m.accepted,
        "rejects": m.rejects, "acceptance": m.acceptance, "tokens_per_pass": m.tokens_per_pass, "overridden": m.overridden, "drafts": m.drafts, "draft_pos": m.draft_pos,
        "decode_wall_s": m.wall, "ms_per_token_wall": m.wall * 1e3 / n, "tok_per_s": n / m.wall.max(1e-9),
        "gpu_ms_per_step": m.gpu_ms.iter().sum::<f64>() / m.gpu_ms.len().max(1) as f64, "prefill_s": m.prefill_s,
        "handoff_dec_p50_us": m.handoff_dec.map(|h| h.p50_us), "handoff_draft_p50_us": m.handoff_draft.map(|h| h.p50_us)})
}

// ---------------------------------------------------------------- serve (JSON lines; used by the SwiftUI app in app/, see docs/SERVE_PROTOCOL.md)
fn emit(v: Value) {
    use std::io::Write;
    let mut o = std::io::stdout().lock();
    let _ = writeln!(o, "{}", v);
    let _ = o.flush();
}

/// Tokenizer for `serve`: --tokenizer DIR, else the model dir when it has tokenizer.json, else the default 27B dir.
/// A load failure is not fatal: the ready line says chat_available false and why (token-level `gen` still works).
/// `fold`: --fold-ids (tests only): prompt ids are fed modulo the model vocab, so a tiny test model without a
/// tokenizer of its own can run the chat plumbing with the 27B tokenizer (the text is meaningless).
struct TokState { tk: Option<Tokenizer>, info: Value, fold: bool }

fn load_tokenizer(a: &Args, model: &str, root: &str) -> TokState {
    let dir = a.s("--tokenizer").unwrap_or_else(|| {
        if std::path::Path::new(model).join("tokenizer.json").exists() { model.to_string() } else { format!("{}/{}", root, DEFAULT_MODEL) }
    });
    let t0 = std::time::Instant::now();
    match Tokenizer::from_path(&dir) {
        Ok(tk) => {
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            eprintln!("tokenizer: {} ({} ids, {} merges) in {:.0} ms", dir, tk.vocab_size(), tk.merges_len(), ms);
            let path = std::fs::canonicalize(format!("{}/tokenizer.json", dir.trim_end_matches('/'))).map(|p| p.display().to_string()).unwrap_or(dir.clone());
            let info = json!({"tokenizer": path, "vocab_size": tk.vocab_size(), "tokenizer_load_ms": ms});
            TokState { tk: Some(tk), info, fold: a.has("--fold-ids") }
        }
        Err(e) => {
            eprintln!("tokenizer: not loaded: {}", e);
            TokState { tk: None, info: json!({"tokenizer": dir, "tokenizer_error": e}), fold: false }
        }
    }
}

/// Parse a chat request: (messages, template options, nnew, mtp requested, continue).
fn chat_request(v: &Value) -> Result<(Vec<Message>, ChatOptions, usize, bool, bool), String> {
    let arr = v["messages"].as_array().ok_or("chat: \"messages\" must be an array")?;
    let mut msgs = Vec::with_capacity(arr.len());
    for (i, m) in arr.iter().enumerate() {
        let role = m["role"].as_str().ok_or_else(|| format!("chat: messages[{}].role must be a string", i))?;
        let content = match &m["content"] { Value::String(s) => s.clone(), Value::Null => String::new(),
            _ => return Err(format!("chat: messages[{}].content must be a string", i)) };
        let msg = match (role, m["reasoning_content"].as_str()) {
            ("assistant", Some(r)) => Message { role: role.into(), content, reasoning_content: Some(r.into()) },
            ("assistant", None) => Message::assistant_from_raw(&content),
            _ => Message { role: role.into(), content, reasoning_content: None },
        };
        msgs.push(msg);
    }
    let opt = ChatOptions { add_generation_prompt: true, enable_thinking: v["think"].as_bool().unwrap_or(false),
        reasoning_effort: v["reasoning_effort"].as_str().map(|s| s.to_string()) };
    let nnew = v["nnew"].as_u64().unwrap_or(2048).max(1) as usize;
    Ok((msgs, opt, nnew, v["mtp"].as_bool().unwrap_or(false), v["continue"].as_bool().unwrap_or(true)))
}

/// Shared fields of a `done` line (gen and chat).
#[allow(clippy::too_many_arguments)]
fn done_fields(st: &StreamStats, spec: Option<&SpecResult>, n_fed: usize, ttft: Option<f64>, eng: &QwenEngine, pending: Option<u32>,
               opt: &QwenOptions, mon: &footprint::Monitor, use_mtp: bool, mtp_req: bool) -> Value {
    let dec_n = st.gen.saturating_sub(1);
    let ms_per_tok = if dec_n > 0 { st.decode_s * 1e3 / dec_n as f64 } else { f64::NAN };
    json!({"done": true, "prefill_ms": st.prefill_s * 1e3, "prefill_tok_s": n_fed as f64 / st.prefill_s.max(1e-9),
        "ttft_ms": ttft.map(|t| t * 1e3), "gen_tokens": st.gen, "decode_ms": st.decode_s * 1e3,
        "ms_per_tok": if ms_per_tok.is_finite() { json!(ms_per_tok) } else { Value::Null },
        "decode_tok_s": if ms_per_tok.is_finite() { json!(1e3 / ms_per_tok) } else { Value::Null },
        "gpu_ms_per_tok": if dec_n > 0 { json!(st.gpu_ms / dec_n as f64) } else { Value::Null },
        "pos": eng.pos(), "ctx_used": eng.pos() + pending.is_some() as usize, "max_ctx": opt.max_ctx,
        "footprint_gb": gb(footprint::self_footprint()), "peak_footprint_gb": gb(mon.peak()),
        "stop_reason": st.stop_reason, "mtp": use_mtp, "mtp_requested": mtp_req, "mtp_available": opt.mtp,
        "spec": spec.map(|m| json!({"passes": m.passes, "accepted": m.accepted, "rejects": m.rejects,
            "acceptance": m.acceptance, "tokens_per_pass": m.tokens_per_pass, "depth": m.depth, "accept_hist": m.accept_hist}))})
}

fn serve(eng: &QwenEngine, mon: &footprint::Monitor, opt: &QwenOptions, model: &str, open_s: f64, arena: u64, tok: TokState) {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{mpsc, Arc};
    let cancel = Arc::new(AtomicBool::new(false));
    let (tx, rx) = mpsc::channel::<Value>();
    {
        let cancel = cancel.clone();
        std::thread::spawn(move || {
            use std::io::BufRead;
            for line in std::io::stdin().lock().lines() {
                let Ok(line) = line else { break };
                if line.trim().is_empty() { continue; }
                match serde_json::from_str::<Value>(&line) {
                    Ok(v) => {
                        match v["op"].as_str() {
                            Some("stop") => { cancel.store(true, Ordering::SeqCst); continue; }
                            Some("gen") | Some("chat") => cancel.store(false, Ordering::SeqCst),
                            _ => {}
                        }
                        if tx.send(v).is_err() { break; }
                    }
                    Err(e) => emit(json!({"error": format!("bad JSON line: {}", e)})),
                }
            }
            // stdin closed: the main loop ends when the channel closes
        });
    }
    // a tokenizer with ids beyond the model's embedding rows (tiny test models) cannot drive chat
    let mut tok_info = tok.info;
    let fold = if tok.fold { tok_info["fold_ids"] = json!(true); Some(eng.cfg.vocab as u32) } else { None };
    let tk = match tok.tk {
        Some(t) if t.vocab_size() > eng.cfg.vocab && !tok.fold => {
            let e = format!("tokenizer has {} ids but the model vocab is {}: chat disabled (use gen)", t.vocab_size(), eng.cfg.vocab);
            eprintln!("tokenizer: {}", e);
            tok_info["tokenizer_error"] = json!(e);
            None
        }
        t => t,
    };
    let stop_ids = [IM_END, ENDOFTEXT];
    let mut chat = ChatState::new();
    let mut pending: Option<u32> = None; // last emitted token, not yet fed
    let mut ready = json!({"ready": true, "pid": std::process::id(), "model": model, "max_ctx": opt.max_ctx, "open_s": open_s,
        "arena_gb": gb(arena), "footprint_gb": gb(footprint::self_footprint()), "peak_footprint_gb": gb(mon.peak()),
        "prefill_k": opt.prefill_k, "prefill_t": opt.prefill_t, "prefill_exact": opt.prefill_exact, "decode": "sync, greedy",
        "mtp_available": opt.mtp, "spec_depth": opt.spec_depth, "spec_auto": opt.spec_auto, "chat_available": tk.is_some(), "stop_ids": stop_ids});
    for (k, x) in tok_info.as_object().unwrap() { ready[k] = x.clone(); }
    emit(ready);
    eprintln!("serve: ready (max_ctx {}, chat {})", opt.max_ctx, tk.is_some());
    for v in rx {
        match v["op"].as_str().unwrap_or("") {
            "reset" => { eng.reset(); pending = None; chat.invalidate(); emit(json!({"reset": true, "pos": 0})); }
            "activity" => { let mut r = streamdense_model::activity::handle_request(eng.activity.as_deref(), &v); r["op"] = json!("activity"); emit(r); }
            "stats" => emit(json!({"stats": true, "pos": eng.pos(), "pending": pending, "kv_tokens": eng.pos() + pending.is_some() as usize,
                "max_ctx": opt.max_ctx, "footprint_gb": gb(footprint::self_footprint()), "peak_footprint_gb": gb(mon.peak()), "heap_mb": eng.heap_mb()})),
            "chat" => {
                let fail = |e: String| emit(json!({"done": true, "op": "chat", "stop_reason": "error", "error": e, "n_prompt": 0, "n_new": 0, "kv_reused": 0}));
                let Some(tk) = tk.as_ref() else {
                    fail(format!("chat unavailable: {}", tok_info["tokenizer_error"].as_str().unwrap_or("no tokenizer"))); continue };
                let (msgs, copt, nnew, mtp_req, want_cont) = match chat_request(&v) { Ok(x) => x, Err(e) => { fail(e); continue } };
                // the chat bookkeeping must describe the engine state exactly, else nothing is reused
                if chat.is_valid() && chat.seq.len() != eng.pos() + pending.is_some() as usize {
                    eprintln!("serve: chat state ({} tokens) != engine state ({} + pending): full prefill", chat.seq.len(), eng.pos());
                    chat.invalidate();
                }
                let t_tok = std::time::Instant::now();
                let plan = match chat.plan(tk, &msgs, &copt, want_cont) { Ok(p) => p, Err(e) => { fail(e); continue } };
                let tokenize_ms = t_tok.elapsed().as_secs_f64() * 1e3;
                let (mut mode, mut toks) = (plan.mode, plan.tokens);
                let mut feed: Vec<u32> = vec![];
                if mode != Mode::Full {
                    feed.extend(pending);
                    feed.extend_from_slice(&toks);
                    if eng.pos() + feed.len() + 1 > opt.max_ctx { mode = Mode::Full; toks = plan.full; }
                }
                let kv_reused = if mode == Mode::Full { 0 } else { eng.pos() };
                if mode == Mode::Full { feed = toks.clone(); }
                let n_prompt = kv_reused + feed.len();
                let base = json!({"op": "chat", "n_prompt": n_prompt, "kv_reused": kv_reused, "n_feed": feed.len(), "kv_mode": mode.as_str(),
                    "think": copt.enable_thinking, "tokenize_ms": tokenize_ms});
                if mode == Mode::Full && feed.len() + 1 > opt.max_ctx {
                    chat.invalidate();
                    let mut d = base.clone();
                    d["done"] = json!(true); d["stop_reason"] = json!("ctx_full"); d["n_new"] = json!(0); d["gen_tokens"] = json!(0);
                    d["reply"] = json!(""); d["max_ctx"] = json!(opt.max_ctx); d["pos"] = json!(eng.pos());
                    emit(d);
                    continue;
                }
                if mode == Mode::Full { eng.reset(); }
                let mut start = base.clone();
                start["start"] = json!(true);
                emit(start);
                let t0 = std::time::Instant::now();
                let mut ttft: Option<f64> = None;
                let mut gen_ids: Vec<u32> = Vec::with_capacity(nnew.min(8192));
                let mut dec = StreamDecoder::new();
                let cancel_ref = cancel.clone();
                let tap = eng.activity.as_deref();
                let mut on_tok = |t: u32| -> bool {
                    if ttft.is_none() { ttft = Some(t0.elapsed().as_secs_f64()); }
                    if let Some(a) = tap { a.log_out(&[t]); } // output-vocab counts (CPU only)
                    gen_ids.push(t);
                    if !stop_ids.contains(&t) {
                        let piece = dec.push(tk, t);
                        if !piece.is_empty() { emit(json!({"text": piece})); }
                    }
                    !cancel_ref.load(Ordering::SeqCst)
                };
                let use_mtp = mtp_req && opt.mtp;
                if let Some(v) = fold { for t in feed.iter_mut() { *t %= v; } }
                let r = if use_mtp { { let r = eng.feed_stream_spec_mode(&feed, nnew, &stop_ids, &mut on_tok, spec_mode(&v, opt), &DraftOverride::None); eng.flush_draft_source(); r.map(|(s, m)| (s, Some(m))) } }
                        else { eng.feed_stream(&feed, nnew, &stop_ids, &mut on_tok).map(|s| (s, None)) };
                let tail = dec.finish();
                if !tail.is_empty() { emit(json!({"text": tail})); }
                match r {
                    Ok((st, spec)) => {
                        pending = st.last;
                        let ok = st.stop_reason != "ctx_full" && st.gen > 0;
                        chat.commit(mode, &toks, &gen_ids, &msgs, &copt, ok);
                        if chat.is_valid() && chat.seq.len() != eng.pos() + pending.is_some() as usize {
                            eprintln!("serve: chat state ({} tokens) != engine state ({} + pending) after the reply: next chat is a full prefill",
                                      chat.seq.len(), eng.pos());
                            chat.invalidate();
                        }
                        let body = match gen_ids.last() { Some(l) if stop_ids.contains(l) => &gen_ids[..gen_ids.len() - 1], _ => &gen_ids[..] };
                        let mut d = done_fields(&st, spec.as_ref(), feed.len(), ttft, eng, pending, opt, mon, use_mtp, mtp_req);
                        for (k, x) in base.as_object().unwrap() { d[k] = x.clone(); }
                        d["n_new"] = json!(st.gen);
                        d["reply"] = json!(tk.decode(body));
                        d["continue_requested"] = json!(want_cont);
                        emit(d);
                    }
                    Err(e) => {
                        eng.reset(); pending = None; chat.invalidate();
                        let mut d = base.clone();
                        d["done"] = json!(true); d["stop_reason"] = json!("error"); d["error"] = json!(e); d["n_new"] = json!(gen_ids.len());
                        emit(d);
                    }
                }
            }
            "gen" => {
                chat.invalidate();
                let toks = ids_or_empty(&v["tokens"]);
                let nnew = v["nnew"].as_u64().unwrap_or(256).max(1) as usize;
                let stop = ids_or_empty(&v["stop"]);
                let want_cont = v["continue"].as_bool().unwrap_or(false);
                let cont = want_cont && eng.pos() > 0;
                let mut feed: Vec<u32> = Vec::with_capacity(toks.len() + 1);
                if cont { if let Some(p) = pending { feed.push(p); } }
                feed.extend_from_slice(&toks);
                if feed.is_empty() { emit(json!({"error": "gen: no tokens"})); continue; }
                if !cont { eng.reset(); pending = None; }
                if eng.pos() + feed.len() > opt.max_ctx {
                    emit(json!({"done": true, "continued": cont, "prompt_tokens": feed.len(), "gen_tokens": 0, "pos": eng.pos(),
                        "max_ctx": opt.max_ctx, "stop_reason": "ctx_full", "footprint_gb": gb(footprint::self_footprint()),
                        "peak_footprint_gb": gb(mon.peak())}));
                    continue;
                }
                let t0 = std::time::Instant::now();
                let mut ttft: Option<f64> = None;
                let cancel_ref = cancel.clone();
                let tap = eng.activity.as_deref();
                let mut on_tok = |t: u32| -> bool {
                    if ttft.is_none() { ttft = Some(t0.elapsed().as_secs_f64()); }
                    if let Some(a) = tap { a.log_out(&[t]); } // output-vocab counts (CPU only)
                    emit(json!({"tok": t}));
                    !cancel_ref.load(Ordering::SeqCst)
                };
                let mtp_req = v["mtp"].as_bool().unwrap_or(false);
                let use_mtp = mtp_req && opt.mtp;
                let r = if use_mtp { { let r = eng.feed_stream_spec_mode(&feed, nnew, &stop, &mut on_tok, spec_mode(&v, opt), &DraftOverride::None); eng.flush_draft_source(); r.map(|(s, m)| (s, Some(m))) } }
                        else { eng.feed_stream(&feed, nnew, &stop, &mut on_tok).map(|s| (s, None)) };
                match r {
                    Ok((st, spec)) => {
                        pending = st.last;
                        let mut d = done_fields(&st, spec.as_ref(), feed.len(), ttft, eng, pending, opt, mon, use_mtp, mtp_req);
                        d["continued"] = json!(cont); d["continue_requested"] = json!(want_cont); d["prompt_tokens"] = json!(feed.len());
                        emit(d);
                    }
                    Err(e) => { eng.reset(); pending = None; emit(json!({"error": e, "done": true, "stop_reason": "error"})); }
                }
            }
            op => emit(json!({"error": format!("unknown op {:?}", op)})),
        }
    }
    eprintln!("serve: stdin closed, exiting");
}
/// Spec policy of an MTP request: "spec_depth" N (clamped to 1..=opt.spec_depth, the replay buffer sized at open) or
/// "auto" (adaptive up to opt.spec_depth); default the engine's (--spec-depth N | auto).
fn spec_mode(v: &Value, opt: &QwenOptions) -> SpecMode {
    if v["spec_depth"].as_str() == Some("auto") && opt.spec_depth >= 2 { return SpecMode::Auto { max: opt.spec_depth }; }
    match v["spec_depth"].as_u64() {
        Some(d) => SpecMode::Fixed((d as usize).clamp(1, opt.spec_depth.max(1))),
        None => if opt.spec_auto { SpecMode::Auto { max: opt.spec_depth } } else { SpecMode::Fixed(opt.spec_depth) },
    }
}

/// --spec-depth N | auto (auto = adaptive, max SPEC_DEPTH_MAX; the replay buffer is sized for it), --spec-cost
/// "t1,..,t8[;mtp=MS][;round=MS]" (planner cost table), --draft-table DIR ("" = off), --draft-table-mb MB.
/// `depth_default` / `table_default`: config values used when the flag is absent (gateway).
pub(crate) fn spec_opts(a: &Args, opt: &mut QwenOptions, depth_default: Option<String>, table_default: Option<String>) {
    if let Some(d) = a.s("--spec-depth").or(depth_default) {
        if d == "auto" {
            opt.spec_auto = true;
            opt.spec_depth = streamdense_model::spec_deep::SPEC_DEPTH_MAX;
        } else {
            opt.spec_auto = false;
            opt.spec_depth = d.parse().unwrap_or_else(|_| die(format!("--spec-depth {}: want 1..=7 or auto", d)));
        }
    }
    if let Some(c) = a.s("--spec-cost") { opt.spec_cost = SpecCost::parse(&c).unwrap_or_else(|e| die(e)); }
    opt.draft_table = a.s("--draft-table").or(table_default).filter(|s| !s.is_empty()).map(std::path::PathBuf::from);
    if let Some(m) = a.s("--draft-table-mb") {
        opt.draft_table_bytes = (m.parse::<f64>().unwrap_or_else(|_| die(format!("--draft-table-mb {}", m))) * 1e6) as usize;
    }
}

fn ids_or_empty(v: &Value) -> Vec<u32> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect()).unwrap_or_default()
}

// ---------------------------------------------------------------- spec-bench
pub(crate) struct SpecBench {
    ctxs: Vec<usize>,
    ks: Vec<usize>,
    depth: usize,
    n: usize,
    reps: usize,
    warmup: usize,
    prompts: Vec<(String, Vec<u32>)>,
    text: Vec<u32>,
    text_src: String,
    agree: bool,
    verify: bool,
    accept: bool,
    real_mtp: bool,
    e2e: Vec<SpecMode>,
    e2e_sync: bool,
}

fn usizes(s: &str) -> Vec<usize> { s.split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().unwrap_or_else(|_| die(format!("bad number {:?}", x)))).collect() }

fn spec_bench_prep(a: &Args, model: &str, root: &str, prompts_default: &str, opt: &mut QwenOptions) -> SpecBench {
    if !opt.mtp { die("spec-bench needs --mtp".into()); }
    if opt.prefill_exact { die("spec-bench times the GEMM path too: drop --prefill-exact".into()); }
    let ctxs = usizes(&a.sd("--ctx", "1024,4096"));
    let ks = usizes(&a.sd("--k", "1,2,4,8,16"));
    let depth = a.f("--depth", 4.0) as usize;
    if depth == 0 || ctxs.is_empty() || ks.is_empty() || ks.contains(&0) { die("spec-bench: --depth, --ctx and --k must be >= 1".into()); }
    let n = a.f("--n", 200.0) as usize;
    let all = load_prompts(&a.sd("--prompts", prompts_default));
    let prompts: Vec<(String, Vec<u32>)> = pick(&all, &a.s("--ids")).into_iter().cloned().collect();
    // context text: real text (the model README first), tokenized with the Rust tokenizer, folded into a tiny
    // model's vocab when needed
    let files = a.s("--ctx-text").unwrap_or_else(|| format!("{}/{}/README.md,{r}/docs/ARCHITECTURE.md,{r}/docs/GATEWAY.md", root, DEFAULT_MODEL, r = root));
    let mut txt = String::new();
    for f in files.split(',').filter(|x| !x.is_empty()) {
        txt.push_str(&std::fs::read_to_string(f).unwrap_or_else(|e| die(format!("read {}: {}", f, e))));
        txt.push_str("\n\n");
    }
    let tdir = a.s("--tokenizer").unwrap_or_else(|| {
        if std::path::Path::new(model).join("tokenizer.json").exists() { model.to_string() } else { format!("{}/{}", root, DEFAULT_MODEL) }
    });
    let vocab = streamdense_model::QwenConfig::from_model_dir(model).unwrap_or_else(|e| die(e)).vocab as u32;
    let text: Vec<u32> = {
        let tk = Tokenizer::from_path(&tdir).unwrap_or_else(|e| die(format!("tokenizer {}: {}", tdir, e)));
        tk.encode(&txt).into_iter().map(|t| t % vocab).collect()
    };
    // (c) end-to-end depths: the engine is opened with spec_depth = the largest (replay buffer + verify rows)
    let dmax = streamdense_model::spec_deep::SPEC_DEPTH_MAX;
    let e2e: Vec<SpecMode> = a.s("--e2e").map(|s| s.split(',').map(str::trim).filter(|x| !x.is_empty())
        .map(|x| SpecMode::parse(x, dmax).unwrap_or_else(|e| die(e))).collect()).unwrap_or_default();
    if e2e.iter().any(|m| m.max_depth() == 0 || m.max_depth() > dmax) {
        die(format!("spec-bench: --e2e depths must be in 1..={} or auto", dmax));
    }
    let e2e_max = e2e.iter().map(|m| m.max_depth()).max().unwrap_or(1);
    opt.spec_depth = opt.spec_depth.max(e2e_max);
    let (maxc, maxk) = (*ctxs.iter().max().unwrap(), (*ks.iter().max().unwrap()).max(depth + 1));
    if text.len() < maxc + maxk { die(format!("spec-bench: context text has {} tokens, need {} (--ctx-text)", text.len(), maxc + maxk)); }
    if a.s("--max-ctx").is_none() {
        let maxp = prompts.iter().map(|p| p.1.len()).max().unwrap_or(0);
        opt.max_ctx = (maxc + maxk + 1).max(maxp + n + depth.max(e2e_max) + 2);
    }
    eprintln!("spec-bench: ctx {:?}, k {:?}, depth {}, n {}, {} prompts, context text {} tokens ({}), max_ctx {}",
              ctxs, ks, depth, n, prompts.len(), text.len(), files, opt.max_ctx);
    SpecBench { ctxs, ks, depth, n, reps: a.f("--reps", 20.0) as usize, warmup: a.f("--warmup", 3.0) as usize, prompts, text,
                text_src: files, agree: a.has("--agree"), verify: !a.has("--skip-verify"), accept: !a.has("--skip-accept"),
                real_mtp: !a.has("--no-real-mtp"), e2e, e2e_sync: a.has("--e2e-sync") }
}

fn pctl(v: &[f64], q: f64) -> f64 {
    if v.is_empty() { return f64::NAN; }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((q * s.len() as f64).ceil() as usize).clamp(1, s.len()) - 1; // nearest rank
    s[i]
}

/// Acceptance counts of one chain set (pooled by summing).
#[derive(Clone, Default)]
struct Acc {
    num: Vec<usize>,      // [s] positions with the first s+1 drafts all correct
    den: Vec<usize>,      // [s] positions with the first s correct and draft s+1 evaluable
    sum_min: Vec<usize>,  // [d-1] sum over positions of min(d, run)
    npos: usize,
    sim_tok: Vec<usize>,  // [d-1] simulated spec decode: tokens produced
    sim_pass: Vec<usize>, // [d-1] passes
}
impl Acc {
    fn new(depth: usize) -> Acc { Acc { num: vec![0; depth], den: vec![0; depth], sum_min: vec![0; depth], npos: 0, sim_tok: vec![0; depth], sim_pass: vec![0; depth] } }
    fn from_run(gen: &[u32], drafts: &[Vec<u32>], depth: usize) -> Acc {
        let mut a = Acc::new(depth);
        let mut run = Vec::with_capacity(drafts.len());
        for (i, ds) in drafts.iter().enumerate() {
            let avail = depth.min(gen.len().saturating_sub(1 + i));
            if avail == 0 { break; }
            let mut r = 0;
            while r < avail && ds[r] == gen[i + 1 + r] { r += 1; }
            for s in 0..avail { if r >= s { a.den[s] += 1; } if r > s { a.num[s] += 1; } }
            for d in 1..=depth { a.sum_min[d - 1] += r.min(d); }
            a.npos += 1;
            run.push(r);
        }
        for d in 1..=depth {
            let mut i = 0;
            while i < run.len() {
                let acc = run[i].min(d);
                a.sim_pass[d - 1] += 1;
                a.sim_tok[d - 1] += acc + 1;
                i += acc + 1;
            }
        }
        a
    }
    fn add(&mut self, o: &Acc) {
        for i in 0..self.num.len() {
            self.num[i] += o.num[i]; self.den[i] += o.den[i]; self.sum_min[i] += o.sum_min[i];
            self.sim_tok[i] += o.sim_tok[i]; self.sim_pass[i] += o.sim_pass[i];
        }
        self.npos += o.npos;
    }
    fn cond(&self, s: usize) -> f64 { self.num[s] as f64 / self.den[s].max(1) as f64 }
    fn mean_acc(&self, d: usize) -> f64 { self.sum_min[d - 1] as f64 / self.npos.max(1) as f64 }
    fn tpp(&self, d: usize) -> f64 { self.sim_tok[d - 1] as f64 / self.sim_pass[d - 1].max(1) as f64 }
    fn json(&self) -> Value {
        let depth = self.num.len();
        json!({"positions": self.npos, "num": self.num, "den": self.den,
            "cond_accept": (0..depth).map(|s| self.cond(s)).collect::<Vec<_>>(),
            "mean_accepted_per_position": (1..=depth).map(|d| self.mean_acc(d)).collect::<Vec<_>>(),
            "sim_tokens_per_pass": (1..=depth).map(|d| self.tpp(d)).collect::<Vec<_>>(),
            "sim_passes": self.sim_pass, "sim_tokens": self.sim_tok})
    }
}

fn spec_bench_run(eng: &QwenEngine, sb: SpecBench, out: &mut Value, pass: &mut bool) {
    const KS: [usize; 8] = streamdense_model::qwen_layers::EXACT_KS; // exact GEMV pass sizes
    let depth = sb.depth;
    out["spec_bench"] = json!({"ctx": sb.ctxs, "k": sb.ks, "depth": depth, "n": sb.n, "reps": sb.reps, "warmup": sb.warmup,
        "context_text": sb.text_src, "context_text_tokens": sb.text.len(), "max_ctx": eng.opt.max_ctx});
    // ---- (a) verify-pass time vs K
    let mut kall: Vec<usize> = sb.ks.clone();
    kall.extend((1..=depth).map(|d| d + 1));
    kall.push(1);
    kall.sort(); kall.dedup();
    let mut vrows = vec![];
    // per ctx: K -> (best median ms, path) without checkpoint; t(1)
    let mut best: Vec<(usize, HashMap<usize, (f64, String)>, f64)> = vec![];
    if sb.verify {
        if sb.agree {
            // correctness probe (fresh state per pass): GEMM rows vs GEMV rows, smallest ctx
            let c = sb.ctxs[0];
            for &k in kall.iter().filter(|k| KS.contains(k) && **k >= 2) {
                let toks = &sb.text[c..c + k];
                let gv = eng.verify_rows(&sb.text[..c], toks, false).unwrap_or_else(|e| die(e));
                let gm = eng.verify_rows(&sb.text[..c], toks, true).unwrap_or_else(|e| die(e));
                let same = gv.iter().zip(&gm).filter(|(a, b)| a == b).count();
                println!("agree @ctx {} K {}: GEMV vs GEMM argmax rows equal {}/{} (gemv {:?} gemm {:?})", c, k, same, k, gv, gm);
            }
        }
        println!("\n(a) verify pass: one pass over K new rows, head (final norm + lm_head + argmax) on all K rows; {} reps after {} warm-up; wall = commit -> completed",
                 sb.reps, sb.warmup);
        for &c in &sb.ctxs {
            eng.reset();
            let t0 = std::time::Instant::now();
            eng.prefill(&sb.text[..c]).unwrap_or_else(|e| die(e));
            let pf = t0.elapsed().as_secs_f64();
            println!("\nctx {}: prefill {:.2} s ({:.0} tok/s)", c, pf, c as f64 / pf);
            println!("| K | path | ckpt | median ms | p90 ms | GPU median ms | ms/token | bound K*t(1)/t(K) |");
            println!("|---|---|---|---|---|---|---|---|");
            let mut cfgs: Vec<(usize, bool, bool)> = vec![]; // (k, gemm, ck)
            for &k in &kall {
                if KS.contains(&k) { cfgs.push((k, false, false)); if k >= 2 { cfgs.push((k, false, true)); } }
                if k >= 2 { cfgs.push((k, true, false)); }
            }
            let mut t1 = f64::NAN;
            let mut bm: HashMap<usize, (f64, String)> = HashMap::new();
            let mut gemv: HashMap<usize, f64> = HashMap::new(); // K -> GEMV median, no checkpoint
            for (k, gemm, ck) in cfgs {
                let (w, g) = eng.time_verify(c, &sb.text[c..c + k], gemm, ck, sb.warmup, sb.reps).unwrap_or_else(|e| die(e));
                let (med, p90, gmed) = (pctl(&w, 0.5), pctl(&w, 0.9), pctl(&g, 0.5));
                if k == 1 && !gemm && !ck { t1 = med; }
                let path = if gemm { if k >= streamdense_model::qwen_layers::PREFILL_FLASH_MIN_T { "GEMM + flash attn" } else { "GEMM + sdpa_vec attn" } } else { "GEMV (decode kernels)" };
                if !ck && !gemm { gemv.insert(k, med); }
                if !ck {
                    let e = bm.entry(k).or_insert((f64::INFINITY, String::new()));
                    if med < e.0 { *e = (med, path.to_string()); }
                }
                let bound = k as f64 * t1 / med;
                println!("| {} | {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |", k, path, if ck { "row 0" } else { "-" }, med, p90, gmed, med / k as f64, bound);
                vrows.push(json!({"ctx": c, "k": k, "path": path, "gemm": gemm, "checkpoint": ck, "median_ms": med, "p90_ms": p90,
                    "gpu_median_ms": gmed, "gpu_p90_ms": pctl(&g, 0.9), "ms_per_token": med / k as f64, "speedup_bound": bound,
                    "wall_ms": w, "gpu_ms": g}));
            }
            // GEMV for K not in KS = the next GEMV size, padded
            for &k in &kall {
                if let Some(&kp) = KS.iter().find(|&&x| x >= k && x != k) {
                    if let Some(&m) = gemv.get(&kp) {
                        let e = bm.entry(k).or_insert((f64::INFINITY, String::new()));
                        if m < e.0 { *e = (m, format!("GEMV K={} padded", kp)); }
                    }
                }
            }
            println!("best per K (no checkpoint): {}", kall.iter().map(|k| format!("K={} {:.2} ms ({})", k, bm[k].0, bm[k].1)).collect::<Vec<_>>().join("; "));
            out["spec_bench"]["prefill_s"][c.to_string()] = json!(pf);
            out["spec_bench"]["best"][c.to_string()] = json!(kall.iter().map(|k| json!({"k": k, "ms": bm[k].0, "path": bm[k].1})).collect::<Vec<_>>());
            best.push((c, bm, t1));
        }
        out["spec_bench"]["verify"] = json!(vrows);
    }
    // ---- (b) chained MTP acceptance (teacher forcing)
    if !sb.accept { return; }
    println!("\n(b) chained MTP acceptance, teacher forcing: {} greedy tokens per prompt, chains of {} drafts at every position", sb.n, depth);
    let (mut pool_n, mut pool_p) = (Acc::new(depth), Acc::new(depth));
    let mut prow = vec![];
    let mut chain_ms: Vec<f64> = vec![];
    let mut first_ms: Vec<f64> = vec![];
    let mut target_ms: Vec<f64> = vec![];
    let ref_acc: HashMap<&str, f64> = [("short", 0.70), ("code", 0.92), ("math", 0.98), ("chinese", 0.71), ("ctx1k", 0.88), ("readme2k", 0.83)].into_iter().collect(); // earlier measured MTP acceptance (depth 1), for comparison
    for (id, prompt) in &sb.prompts {
        let r = eng.mtp_chain_accept(prompt, sb.n, depth).unwrap_or_else(|e| die(e));
        let an = Acc::from_run(&r.gen, &r.drafts_norm, depth);
        let ap = Acc::from_run(&r.gen, &r.drafts_prenorm, depth);
        pool_n.add(&an); pool_p.add(&ap);
        chain_ms.extend(&r.chain_gpu_ms); first_ms.extend(&r.first_wall_ms); target_ms.extend(&r.target_gpu_ms);
        let real = if sb.real_mtp {
            let m = generate_mtp(eng, prompt, sb.n.min(r.gen.len()), DraftOverride::None, true).unwrap_or_else(|e| die(e));
            let eq = m.tokens.len() <= r.gen.len() && m.tokens[..] == r.gen[..m.tokens.len()];
            if !eq { *pass = false; }
            Some((m, eq))
        } else { None };
        println!("{:>9} P={:4}: {} tokens{}, {} chains | norm: cond {} | tok/pass d1..{} {} | prenorm: cond {} | real MTP (d=1): {}",
            id, prompt.len(), r.gen.len(), if r.gen.last().map(|t| eng.cfg.eos.contains(t)).unwrap_or(false) { " (eos)" } else { "" }, an.npos,
            (0..depth).map(|s| format!("{:.3}", an.cond(s))).collect::<Vec<_>>().join(" "), depth,
            (1..=depth).map(|d| format!("{:.2}", an.tpp(d))).collect::<Vec<_>>().join(" "),
            (0..depth).map(|s| format!("{:.3}", ap.cond(s))).collect::<Vec<_>>().join(" "),
            real.as_ref().map(|(m, eq)| format!("acceptance {:.3} ({} passes, {:.2} tok/s), tokens == teacher-forced greedy {}{}", m.acceptance, m.passes,
                m.tokens.len().saturating_sub(1) as f64 / m.wall.max(1e-9), eq,
                ref_acc.get(id.as_str()).map(|v| format!(", earlier {:.2}", v)).unwrap_or_default())).unwrap_or("skipped".into()));
        prow.push(json!({"id": id, "prompt_len": prompt.len(), "gen_tokens": r.gen.len(), "gen": r.gen, "prefill_s": r.prefill_s, "wall_s": r.wall_s,
            "norm": an.json(), "prenorm": ap.json(), "drafts_norm": r.drafts_norm, "drafts_prenorm": r.drafts_prenorm,
            "real_mtp": real.as_ref().map(|(m, eq)| json!({"acceptance": m.acceptance, "passes": m.passes, "accepted": m.accepted,
                "tokens_per_pass": m.tokens_per_pass, "tokens": m.tokens.len(), "tok_per_s": m.tokens.len().saturating_sub(1) as f64 / m.wall.max(1e-9),
                "equal_teacher_forced_greedy": eq})),
            "ref_acceptance": ref_acc.get(id.as_str())}));
    }
    let (cm, fm, tm) = (pctl(&chain_ms, 0.5), pctl(&first_ms, 0.5), pctl(&target_ms, 0.5));
    println!("\nGPU median: chained MTP step {:.2} ms, k=1 target pass {:.2} ms; step-1 draft wall median {:.2} ms (pread + CB)", cm, tm, fm);
    println!("\nacceptance per depth step, pooled over {} prompts ({} positions): P(step i correct | steps < i correct)", sb.prompts.len(), pool_n.npos);
    println!("| hidden fed back | {} |", (1..=depth).map(|s| format!("step {}", s)).collect::<Vec<_>>().join(" | "));
    println!("|---|{}", "---|".repeat(depth));
    for (name, a) in [("norm (after mtp.norm)", &pool_n), ("prenorm", &pool_p)] {
        println!("| {} | {} |", name, (0..depth).map(|s| format!("{:.3} ({}/{})", a.cond(s), a.num[s], a.den[s])).collect::<Vec<_>>().join(" | "));
    }
    println!("\nmean accepted per pass (simulated passes; per-position mean in brackets), pooled:");
    println!("| hidden | {} |", (1..=depth).map(|d| format!("d={}", d)).collect::<Vec<_>>().join(" | "));
    println!("|---|{}", "---|".repeat(depth));
    for (name, a) in [("norm", &pool_n), ("prenorm", &pool_p)] {
        println!("| {} | {} |", name, (1..=depth).map(|d| format!("{:.3} [{:.3}]", a.tpp(d) - 1.0, a.mean_acc(d))).collect::<Vec<_>>().join(" | "));
    }
    // ---- projection
    let use_p = (1..=depth).map(|d| pool_p.tpp(d)).sum::<f64>() > (1..=depth).map(|d| pool_n.tpp(d)).sum::<f64>();
    let pool = if use_p { &pool_p } else { &pool_n };
    let mut proj = vec![];
    if !best.is_empty() {
        println!("\nprojected tok/s = (1 + mean accepted) / t(K = d + 1), chain = {} hidden; + drafts = with d chained MTP steps of {:.2} ms each",
                 if use_p { "prenorm" } else { "norm" }, cm);
        println!("| ctx | plain 1/t(1) | {} |", (1..=depth).map(|d| format!("d={}", d)).collect::<Vec<_>>().join(" | "));
        println!("|---|---|{}", "---|".repeat(depth));
        for (c, bm, t1) in &best {
            let cells: Vec<String> = (1..=depth).map(|d| {
                let (t, p) = &bm[&(d + 1)];
                let tpp = pool.tpp(d);
                let (ts, ts_d) = (tpp * 1e3 / t, tpp * 1e3 / (t + d as f64 * cm));
                proj.push(json!({"ctx": c, "d": d, "k": d + 1, "t_k_ms": t, "path": p, "tokens_per_pass": tpp, "tok_s": ts,
                    "draft_ms_each": cm, "tok_s_with_drafts": ts_d}));
                format!("{:.1} ({:.1} w/ drafts; {:.2} tok/pass, {:.1} ms)", ts, ts_d, tpp, t)
            }).collect();
            println!("| {} | {:.1} | {} |", c, 1e3 / t1, cells.join(" | "));
        }
    }
    out["spec_bench"]["accept"] = json!({"prompts": prow, "pooled_norm": pool_n.json(), "pooled_prenorm": pool_p.json(),
        "projection_hidden": if use_p { "prenorm" } else { "norm" }, "chain_step_gpu_median_ms": cm, "first_draft_wall_median_ms": fm,
        "target_pass_gpu_median_ms": tm});
    out["spec_bench"]["projection"] = json!(proj);
}

/// Chained-MTP acceptance per step s = 1..=d from a pass histogram (hist[j] = passes that accepted exactly j drafts):
/// P(draft s accepted | drafts 1..s-1 accepted) = sum_{j >= s} hist[j] / sum_{j >= s-1} hist[j] (NaN when no pass got
/// that far). d = 1 without a histogram: accepted / passes.
fn accept_per_step(hist: &[usize], d: usize, accepted: usize, passes: usize) -> Vec<f64> {
    if hist.iter().sum::<usize>() == 0 {
        return if d == 1 && passes > 0 { vec![accepted as f64 / passes as f64] } else { vec![f64::NAN; d] };
    }
    let tail = |s: usize| hist.iter().skip(s).sum::<usize>();
    (1..=d).map(|s| { let den = tail(s - 1); if den > 0 { tail(s) as f64 / den as f64 } else { f64::NAN } }).collect()
}
fn fmt_steps(v: &[f64]) -> String { v.iter().map(|x| format!("{:.2}", x)).collect::<Vec<_>>().join(" ") }

/// (c) real end-to-end spec decode per policy (see the header): plain greedy (pipelined) is the reference for tokens
/// and tok/s; d = 1 runs the pipelined draft 1 / verify 2 path, d >= 2 and auto the pipelined deep loop
/// (`generate_spec_mode`; `sync`: fixed depths through the synchronous reference loop instead). Any token difference
/// fails the run. Returns the plain greedy tokens per prompt.
fn spec_bench_e2e(eng: &QwenEngine, modes: &[SpecMode], sync: bool, n: usize, prompts: &[(String, Vec<u32>)], out: &mut Value, pass: &mut bool) -> Vec<Vec<u32>> {
    let names: Vec<String> = modes.iter().map(|m| m.name()).collect();
    println!("\n(c) end-to-end spec decode, {} tokens per prompt, policies {:?} (engine spec_depth {}, draft table {}{})", n, names,
             eng.opt.spec_depth, eng.opt.draft_table.as_ref().map_or("off".to_string(), |p| p.display().to_string()), if sync { ", fixed depths sync" } else { "" });
    let tps = |toks: usize, s: f64| toks.saturating_sub(1) as f64 / s.max(1e-9);
    let mut rows = vec![];
    // per policy: summed results (tokens after the first, wall, passes, accepted, equal, runs, ...)
    #[derive(Clone, Default)]
    struct Agg { tok: usize, s: f64, passes: usize, acc: usize, eq: usize, runs: usize, replay: Vec<(usize, f64)>, hist: Vec<usize>,
                 dhist: Vec<usize>, dm: usize, dt: usize, dp: usize, am: usize, at: usize, ap: usize, idle: Vec<f64>, round: Vec<f64>,
                 reenc: usize, pre: usize }
    let mut agg: Vec<Agg> = modes.iter().map(|m| Agg { hist: vec![0; m.max_depth() + 1], dhist: vec![0; m.max_depth() + 1], ..Default::default() }).collect();
    let (mut plain_tok, mut plain_s) = (0usize, 0.0f64);
    let mut plain_toks = vec![];
    for (id, prompt) in prompts {
        let g = eng.generate(prompt, n, true).unwrap_or_else(|e| die(e));
        plain_tok += g.tokens.len().saturating_sub(1);
        plain_s += g.gen_wall_s;
        let mut cells = vec![format!("plain {:.2} tok/s", tps(g.tokens.len(), g.gen_wall_s))];
        let mut dj = vec![];
        for (i, &m) in modes.iter().enumerate() {
            let d = m.max_depth();
            let r = match m {
                SpecMode::Fixed(1) => generate_mtp(eng, prompt, n, DraftOverride::None, true),
                SpecMode::Fixed(d) if sync => streamdense_model::spec_deep::generate_spec_sync(eng, prompt, n, d, DraftOverride::None),
                _ => generate_spec_mode(eng, prompt, n, m, DraftOverride::None),
            }.unwrap_or_else(|e| die(format!("{} {}: {}", id, m.name(), e)));
            let first = (0..r.tokens.len().min(g.tokens.len())).find(|&k| r.tokens[k] != g.tokens[k]);
            let eq = first.is_none() && r.tokens.len() == g.tokens.len();
            if !eq { *pass = false; }
            let a = &mut agg[i];
            a.tok += r.tokens.len().saturating_sub(1); a.s += r.wall; a.passes += r.passes; a.acc += r.accepted; a.eq += eq as usize; a.runs += 1;
            a.replay.extend(&r.replay);
            for (j, c) in r.accept_hist.iter().enumerate() { if j < a.hist.len() { a.hist[j] += c; } }
            for (j, c) in r.depth_hist.iter().enumerate() { if j < a.dhist.len() { a.dhist[j] += c; } }
            a.dm += r.drafts_mtp; a.dt += r.drafts_table; a.dp += r.drafts_pad; a.am += r.acc_mtp; a.at += r.acc_table; a.ap += r.acc_pad;
            a.idle.extend(&r.idle_ms); a.round.extend(&r.round_ms); a.reenc += r.reencode_ms.len(); a.pre += r.preencoded;
            let steps = accept_per_step(&r.accept_hist, d, r.accepted, r.passes);
            let nd = (r.drafts_mtp + r.drafts_table).max(1) as f64;
            let idle = if r.idle_ms.is_empty() { f64::NAN } else { r.idle_ms.iter().sum::<f64>() / r.idle_ms.len() as f64 };
            cells.push(format!("{} {:.2} tok/s {:.2} tok/pass{}{}", m.name(), tps(r.tokens.len(), r.wall), r.tokens_per_pass,
                if r.depth_hist.iter().sum::<usize>() > 0 { format!(" depth {:?} table {:.0}% idle {:.2} ms/round", r.depth_hist, 100.0 * r.drafts_table as f64 / nd, idle) }
                else { format!(" acc/step [{}]", fmt_steps(&steps)) },
                if eq { String::new() } else { format!(" DIFFERS at {:?}", first) }));
            dj.push(json!({"policy": m.name(), "depth": d, "equal_plain": eq, "first_diff": first, "tokens": r.tokens.len(), "tok_s": tps(r.tokens.len(), r.wall),
                "passes": r.passes, "accepted": r.accepted, "tokens_per_pass": r.tokens_per_pass, "accept_hist": r.accept_hist,
                "depth_hist": r.depth_hist, "drafts": {"mtp": r.drafts_mtp, "table": r.drafts_table, "pad": r.drafts_pad},
                "accepted_by_src": {"mtp": r.acc_mtp, "table": r.acc_table, "pad": r.acc_pad},
                "gpu_idle_ms_per_round": idle, "round_ms_median": pctl(&r.round_ms, 0.5), "reencodes": r.reencode_ms.len(), "preencoded": r.preencoded,
                "reencode_ms_median": pctl(&r.reencode_ms, 0.5), "overlap_encode_ms_median": pctl(&r.overlap_encode_ms, 0.5),
                "accept_per_step": steps, "decode_wall_s": r.wall, "gpu_ms_per_step": r.gpu_ms.iter().sum::<f64>() / r.gpu_ms.len().max(1) as f64,
                "verify_gpu_median_ms": pctl(&r.verify_ms, 0.5), "draft_gpu_median_ms": pctl(&r.draft_ms, 0.5),
                "replay": r.replay.iter().map(|x| json!([x.0, x.1])).collect::<Vec<_>>()}));
        }
        println!("{:>9} P={:4}: {} tokens | {}", id, prompt.len(), g.tokens.len(), cells.join(" | "));
        rows.push(json!({"id": id, "prompt_len": prompt.len(), "plain_tokens": g.tokens.len(), "plain_tok_s": tps(g.tokens.len(), g.gen_wall_s),
            "policies": dj}));
        plain_toks.push(g.tokens);
    }
    let plain = plain_tok as f64 / plain_s.max(1e-9);
    println!("\npooled over {} prompts (plain {:.2} tok/s):", prompts.len(), plain);
    println!("| policy | tok/s | x plain | tok/pass | accepted/pass | == plain | drafts/pass by depth (hist) | table share (acc) | MTP acc | pad (acc) | GPU idle ms/round | round ms (median) | re-encodes / passes |");
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    let mut pooled = vec![];
    for (i, m) in modes.iter().enumerate() {
        let a = &agg[i];
        let ts = a.tok as f64 / a.s.max(1e-9);
        let nd = (a.dm + a.dt).max(1) as f64;
        let idle = if a.idle.is_empty() { f64::NAN } else { a.idle.iter().sum::<f64>() / a.idle.len() as f64 };
        let rate = |x: usize, n: usize| if n > 0 { x as f64 / n as f64 } else { f64::NAN };
        let steps = accept_per_step(&a.hist, m.max_depth(), a.acc, a.passes);
        println!("| {} | {:.2} | {:.2} | {:.2} | {:.3} | {}/{} | {:?} | {:.0}% ({:.2}) | {:.2} | {} ({:.2}) | {:.2} | {:.1} | {} / {} |", m.name(), ts, ts / plain,
            a.tok as f64 / a.passes.max(1) as f64, a.acc as f64 / a.passes.max(1) as f64, a.eq, a.runs, a.dhist, 100.0 * a.dt as f64 / nd, rate(a.at, a.dt),
            rate(a.am, a.dm), a.dp, rate(a.ap, a.dp), idle, pctl(&a.round, 0.5), a.reenc, a.passes);
        pooled.push(json!({"policy": m.name(), "depth": m.max_depth(), "tok_s": ts, "speedup_vs_plain": ts / plain,
            "tokens_per_pass": a.tok as f64 / a.passes.max(1) as f64, "accepted_per_pass": a.acc as f64 / a.passes.max(1) as f64,
            "equal_runs": a.eq, "runs": a.runs, "accept_hist": a.hist, "accept_per_step": steps, "depth_hist": a.dhist,
            "drafts": {"mtp": a.dm, "table": a.dt, "pad": a.dp}, "accepted_by_src": {"mtp": a.am, "table": a.at, "pad": a.ap},
            "table_share": a.dt as f64 / nd, "gpu_idle_ms_per_round": idle, "round_ms_median": pctl(&a.round, 0.5),
            "reencodes": a.reenc, "preencoded": a.pre, "replays": a.replay.len(),
            "replay_mtp_rows_gpu_ms_median": pctl(&a.replay.iter().map(|x| x.1).collect::<Vec<_>>(), 0.5)}));
    }
    out["spec_bench"]["e2e"] = json!({"n": n, "policies": names, "sync": sync, "gemv": eng.opt.gemv.name(), "plain_tok_s": plain,
        "draft_table": eng.opt.draft_table.as_ref().map(|p| p.display().to_string()), "spec_cost": format!("{:?}", eng.opt.spec_cost),
        "prompts": rows, "pooled": pooled});
    plain_toks
}

/// --plain-old: plain greedy (pipelined) per prompt on an engine opened with `--gemv old` (the previous matmul
/// family), against the main run's plain tokens (informational: the two families differ in the last bits, so a token
/// may flip at a near-tie).
fn spec_bench_plain_old(eng: &QwenEngine, n: usize, prompts: &[(String, Vec<u32>)], main_toks: &[Vec<u32>], out: &mut Value) {
    println!("\nplain greedy with --gemv old ({} tokens per prompt):", n);
    let (mut tok, mut s, mut rows) = (0usize, 0.0f64, vec![]);
    for (i, (id, prompt)) in prompts.iter().enumerate() {
        let g = eng.generate(prompt, n, true).unwrap_or_else(|e| die(e));
        let t = g.tokens.len().saturating_sub(1);
        tok += t;
        s += g.gen_wall_s;
        let first = main_toks.get(i).and_then(|m| (0..m.len().min(g.tokens.len())).find(|&k| m[k] != g.tokens[k]));
        println!("{:>9} P={:4}: old {:.2} tok/s, tokens == mma plain: {}", id, prompt.len(), t as f64 / g.gen_wall_s.max(1e-9),
                 if first.is_none() { "yes".to_string() } else { format!("no (first diff {:?})", first) });
        rows.push(json!({"id": id, "tok_s": t as f64 / g.gen_wall_s.max(1e-9), "equal_mma_plain": first.is_none(), "first_diff": first}));
    }
    let ts = tok as f64 / s.max(1e-9);
    println!("pooled plain old {:.2} tok/s", ts);
    out["spec_bench"]["plain_old"] = json!({"tok_s": ts, "prompts": rows});
}
