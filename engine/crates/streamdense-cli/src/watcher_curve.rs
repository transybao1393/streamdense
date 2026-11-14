//! `streamdense watcher curve [--dir ~/.streamdense/watcher] [--model DIR] [--granularity 64] [--out FILE.json]
//! [--min-tokens 200000]`
//! CPU only, read only, over the disk store (`store.rs`: full f32 on disk, 0 host-heap
//! RAM for the numbers). The store of the model's dims is opened read only; every number is streamed from disk through
//! fixed stack buffers (one `CH` f32 chunk + one record + one radix histogram), no Vec / HashMap in the compute path.
//! - per label (domains, mixed, unassigned) and pooled (all labels all-time, now, session): cold FFN share / bytes, cold
//!   heads, A ms per pass saved (bytes / 240 GB/s) and B GB freed for keep levels 0.999 .. 0.80 at granularity 1 (exact
//!   per neuron) and 64 (blocks), via `Store::plan`;
//! - lm_head / embedding rows (emitted-token / rows-read counts, all labels pooled) and the MTP draft layer (FFN + heads),
//!   via `store::hot` (exact: the fewest largest rows holding `keep` of the mass; the rest is cold).
//! Only the model's config.json (dims + q4 layout) and the tokenizer (per-script split of the lm_head rows; optional) are
//! read, never weights. Progress on stderr while it runs (~150 ms throttle).
use std::io::Write;
use std::time::Instant;

use streamdense_model::activity::{script_classes, ActivityStats, SCRIPT_NAMES};
use streamdense_model::unload::{q4_row_bytes, READ_BW};

use crate::store::{self, Buf, Hist, Rec, Sel, Store, CH, NOW, SESS};
use crate::watcher::{self, Dims, ND, NP, UNASSIGNED};
use crate::{config_root, load_config, Args};

pub const KEEPS: [f64; 7] = [0.999, 0.995, 0.99, 0.98, 0.95, 0.90, 0.80];
/// granularities of the FFN curve: exact per neuron, 64-blocks (stage A)
pub const GS: [usize; 2] = [1, 64];
const NK: usize = KEEPS.len();
const NR: usize = NK * GS.len();
/// labels + pooled all-time + now + session
const NE: usize = NP + 3;
const NS: usize = SCRIPT_NAMES.len();

/// Store label names by index: the domains (mixed included), then unassigned.
pub fn names() -> [&'static str; NP] {
    let mut n = [""; NP];
    n[..ND].copy_from_slice(&watcher::DOMAINS);
    n[UNASSIGNED] = "unassigned";
    n
}

fn now_s() -> f64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0) }

pub fn main(a: &Args) {
    if matches!(a.0.get(2).map(String::as_str), None | Some("mask")) { mask_main(a); return; }
    if a.0.get(2).map(String::as_str) != Some("curve") {
        eprintln!("usage: streamdense watcher [mask] [--dir ~/.streamdense/watcher] [--model DIR] [--window all] [--keep 0.995] [--sweep] [--json]");
        eprintln!("       streamdense watcher curve [--dir ~/.streamdense/watcher] [--model DIR] [--granularity 64] [--out FILE.json] [--min-tokens 200000]");
        eprintln!("       streamdense watcher heat [--dir ~/.streamdense/watcher] [--layers]");
        std::process::exit(2);
    }
    let conf = load_config();
    let dir = a.s("--dir").map(|d| watcher::expand_home(&d)).unwrap_or_else(|| watcher::config_dir(&conf.cfg.watcher.dir));
    let model = a.s("--model").map(std::path::PathBuf::from).unwrap_or_else(|| conf.model_path(&config_root(&conf)));
    let die = |e: String| -> ! { eprintln!("ERROR: {}", e); std::process::exit(2) };
    let cfg = streamdense_model::QwenConfig::from_model_dir(&model.display().to_string()).unwrap_or_else(|e| die(e));
    let min_tokens = a.f("--min-tokens", watcher::REPRESENTATIVE_TOKENS as f64) as u64;
    let gran = a.f("--granularity", conf.cfg.watcher.granularity as f64) as usize;
    let mut d = Dims::from_stats(&ActivityStats::dims(&cfg), gran);
    d.vocab = cfg.vocab;
    let mut st = Store::open(&dir, d, names(), watcher::MIXED, true).unwrap_or_else(|e| die(format!("store: {}", e)));
    st.refresh();
    st.t = now_s();
    // per-script classes of the token ids (tokenizer, CPU; built before the compute path)
    let script = streamdense_tokenizer::Tokenizer::from_path(&model.display().to_string())
        .map(|tk| script_classes(cfg.vocab, |id| (tk.id_to_bytes(id), tk.is_special(id)))).unwrap_or_default();
    let mut prog = Prog::new(true);
    let c = compute(&st, &script, &mut prog);
    prog.finish();
    let disk = st.disk_bytes();
    let mut out = std::io::stdout().lock();
    let _ = text(&c, &st, disk, min_tokens, &mut out);
    let _ = out.flush();
    if let Some(o) = a.s("--out") {
        let mut f = std::fs::File::create(&o).unwrap_or_else(|e| die(format!("{}: {}", o, e)));
        json(&c, &st, disk, min_tokens, &mut f).unwrap_or_else(|e| die(format!("{}: {}", o, e)));
        eprintln!("wrote {}", o);
    }
}

// ---- results (fixed size, stack) ------------------------------------------------------------------------------------

/// One curve row (keep x granularity).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Row {
    pub keep: f64,
    pub g: usize,
    pub cold_neurons: usize,
    pub neurons: usize,
    pub ffn_bytes: u64,
    pub cold_heads: usize,
    pub heads: usize,
    /// A: g = 64 the stage-A 64-block mask (bytes not read per pass); g = 1 a gather kernel skipping single rows
    pub a_ms: f64,
    pub b_bytes: u64,
}

/// One label or pooled set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Entry {
    pub name: &'static str,
    pub sel: Sel,
    /// now / session: decayed (tau 5 min / 4 h) at the store clock
    pub decayed: bool,
    pub tokens: u64,
    pub turns: u64,
    /// decayed rows (now / session; 0 otherwise)
    pub rows_d: f64,
    pub rows: [Row; NR],
}

/// lm_head / embedding row counts cold at one keep level.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VRow { pub keep: f64, pub lm_cold: usize, pub by_script: [usize; NS], pub emb_cold: usize }

/// MTP draft layer cold shares at one keep level (None = no mass yet).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MRow { pub keep: f64, pub cold_ffn: Option<f64>, pub cold_heads: Option<f64> }

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Extra {
    /// labels with any emitted-token / embedding counts
    pub labels: [bool; NP],
    pub out_tok: u64,
    pub emb_rows: u64,
    pub mtp_rows: u64,
    pub rows_seen: usize,
    pub has_script: bool,
    pub v: [VRow; NK],
    pub m: [MRow; NK],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Curve {
    pub n: usize,
    pub e: [Option<Entry>; NE],
    pub ffn_bytes_total: u64,
    pub x: Extra,
}

/// Progress on stderr (throttled to one line per 150 ms; nothing for runs shorter than that).
pub struct Prog { on: bool, t0: Instant, last: Instant, shown: bool, pub done: usize, pub total: usize }
impl Prog {
    pub fn new(on: bool) -> Prog { let t = Instant::now(); Prog { on, t0: t, last: t, shown: false, done: 0, total: 0 } }
    fn tick(&mut self, what: &str, keep: f64, g: usize) {
        self.done += 1;
        if !self.on { return; }
        let now = Instant::now();
        if now.duration_since(self.last).as_millis() < 150 { return; }
        self.last = now;
        self.shown = true;
        eprint!("\rcurve: {}/{} {} keep {} g {}            ", self.done, self.total, what, keep, g);
    }
    pub fn finish(&mut self) {
        if self.on && self.shown { eprintln!("\rcurve: {} steps in {:.2} s                              ", self.done, self.t0.elapsed().as_secs_f64()); }
    }
}

/// All FFN bytes of the model (every neuron cold; the `Store::plan` byte formula).
/// `streamdense watcher [mask] [--dir DIR] [--model DIR] [--window W] [--keep K] [--sweep] [--json]`: the read
/// mask (dry numbers) built read only from the store exactly as the gateway builds it (`predict::build_profile` + the
/// joined units from predict.bin), the CPU prompt predictor's scores and the exploration bookkeeping. CPU only; reads
/// the model's config.json only; never writes.
pub fn mask_main(a: &Args) {
    use crate::watcher::predict::{self, Cost, Predictor, ReadMask, Report, Window};
    let conf = load_config();
    let wc = &conf.cfg.watcher;
    let dir = a.s("--dir").map(|d| watcher::expand_home(&d)).unwrap_or_else(|| watcher::config_dir(&wc.dir));
    let model = a.s("--model").map(std::path::PathBuf::from).unwrap_or_else(|| conf.model_path(&config_root(&conf)));
    let die = |e: String| -> ! { eprintln!("ERROR: {}", e); std::process::exit(2) };
    let cfg = streamdense_model::QwenConfig::from_model_dir(&model.display().to_string()).unwrap_or_else(|e| die(e));
    let mut d = Dims::from_stats(&ActivityStats::dims(&cfg), wc.granularity);
    d.vocab = cfg.vocab;
    let mut st = Store::open(&dir, d.clone(), names(), watcher::MIXED, true).unwrap_or_else(|e| die(format!("store: {}", e)));
    st.refresh();
    let win = Window::parse(&a.s("--window").unwrap_or_else(|| wc.profile_window.clone()));
    let keeps: Vec<f64> = if a.has("--sweep") { KEEPS.to_vec() } else { vec![a.f("--keep", wc.keep_frac)] };
    let pf = Predictor::open_ro(&st.dir);
    let mut joined = ReadMask::boxed(&st);
    if let Some(p) = &pf { let _ = p.load_joined(&mut joined); }
    let (cost, t) = (Cost::of(&d), now_s());
    let mut m = ReadMask::boxed(&st);
    let mut rows: Vec<(f64, Report)> = vec![];
    for &k in &keeps {
        let t0 = Instant::now();
        let tok = predict::build_profile(&mut st, win, k, t, &mut m);
        m.or_from(&joined);
        let mut r = Report::default();
        predict::report(&m, &cost, &mut r);
        (r.ts, r.build_ms, r.window, r.keep, r.joined) = (t, t0.elapsed().as_secs_f64() * 1e3, win.name(), k, joined.count().0 + joined.count().1);
        (r.tokens_now, r.tokens_session, r.tokens_all) = tok;
        r.ready = tok.2 >= watcher::PLAN_MIN_TOKENS;
        eprintln!("mask: keep {} built in {:.0} ms", k, r.build_ms);
        rows.push((k, r));
    }
    if a.has("--json") {
        let v = serde_json::json!({"store": st.path().display().to_string(), "config_mask": wc.mask, "profiles": rows.iter().map(|(_, r)| r.json()).collect::<Vec<_>>(),
            "predictor": pf.as_ref().map(|p| p.json())});
        println!("{}", v);
        return;
    }
    let total_ffn = ffn_bytes_total(&d) as f64;
    println!("read mask (dry, read only): {} | config mask = {:?} | window {} | pooled tokens: now {:.0}, session {:.0}, all {}{}", st.path().display(), wc.mask, win.name(),
             rows[0].1.tokens_now, rows[0].1.tokens_session, rows[0].1.tokens_all, if rows[0].1.ready { "" } else { " (few tokens)" });
    println!("| keep | masked 64-blocks | share | masked heads | joined | skip / pass | of FFN bytes | ms / pass saved (A) | build ms |");
    println!("|---|---|---|---|---|---|---|---|---|");
    for (k, r) in &rows {
        println!("| {} | {} / {} | {:.2}% | {} / {} | {} | {:.3} GB | {:.2}% | {:.2} | {:.0} |", k, r.masked_blocks, r.blocks, 100.0 * r.masked_blocks as f64 / r.blocks.max(1) as f64,
                 r.masked_heads, r.heads, r.joined, r.skip_bytes() as f64 / 1e9, 100.0 * r.ffn_skip as f64 / total_ffn.max(1.0), r.ms(), r.build_ms);
    }
    match &pf {
        Some(p) => {
            let v = p.json();
            println!("predictor: {} ({} B, {} buckets x {} units): {} requests learned, {} scored; precision {} recall {}; masked precision {} recall {}; out-of-profile active {} / request; {} B predicted, {} B missed / request",
                     p.path.display(), p.disk_bytes(), p.nb, p.units, p.h.jobs, p.h.scored, v["precision"], v["recall"], v["masked_precision"], v["masked_recall"],
                     v["out_of_profile_active_per_request"], v["bytes_predicted_per_request"], v["bytes_missed_per_request"]);
            println!("exploration: {} rounds, {} scheduled, {} read by the engine, {} found active, {} expired, {} joined now",
                     p.h.rounds, p.h.sched, p.h.read, p.h.found, p.h.expired, joined.count().0 + joined.count().1);
        }
        None => println!("predictor: none yet (predict.bin is created by the gateway with [watcher] mask = \"dry\" or \"on\")"),
    }
}

fn ffn_bytes_total(d: &Dims) -> u64 {
    let t = &d.tmpl;
    let row = q4_row_bytes(t.hidden, t.group, t.bits);
    let per_neuron = 2 * row + (t.hidden * t.bits / 8) as u64;
    let per_group = (t.hidden * 4) as u64;
    let groups = d.inter / t.group.max(1);
    d.layers as u64 * (d.inter as u64 * per_neuron + groups as u64 * per_group)
}

// ---- compute (stack only) -------------------------------------------------------------------------------------------

/// The whole curve, streamed from the store (no heap: fixed arrays on this stack + the store's own stack buffers).
pub fn compute(st: &Store, script: &[u8], prog: &mut Prog) -> Curve {
    let mut c = Curve { n: 0, e: [None; NE], ffn_bytes_total: ffn_bytes_total(&st.d), x: Extra::default() };
    let row0 = [Row::default(); NR];
    let (mut tok, mut turns) = (0u64, 0u64);
    for s in 0..NP {
        if !st.has(s) { continue; }
        let h = st.hdr(s);
        if h.tokens == 0 && h.turns == 0 { continue; }
        tok += h.tokens;
        turns += h.turns;
        c.e[c.n] = Some(Entry { name: st.name(s), sel: Sel::Set(s), decayed: false, tokens: h.tokens, turns: h.turns, rows_d: 0.0, rows: row0 });
        c.n += 1;
    }
    if c.n > 0 {
        c.e[c.n] = Some(Entry { name: "pooled", sel: Sel::Labels, decayed: false, tokens: tok, turns, rows_d: 0.0, rows: row0 });
        c.n += 1;
    }
    for s in [NOW, SESS] {
        if !st.has(s) { continue; }
        let h = st.hdr(s);
        if h.turns == 0 { continue; }
        let f = (-(st.t - h.t0).max(0.0) / store::TAUS_S[s - NOW]).exp();
        c.e[c.n] = Some(Entry { name: st.name(s), sel: Sel::Set(s), decayed: true, tokens: h.tokens, turns: h.turns, rows_d: h.rows_d * f, rows: row0 });
        c.n += 1;
    }
    prog.total = c.n * NR + NK;
    for e in c.e[..c.n].iter_mut().flatten() {
        for (ki, &keep) in KEEPS.iter().enumerate() {
            for (gi, &g) in GS.iter().enumerate() {
                let p = st.plan(e.sel, keep, g);
                e.rows[ki * GS.len() + gi] = Row { keep, g, cold_neurons: p.cold_neurons, neurons: p.neurons, ffn_bytes: p.ffn_bytes,
                    cold_heads: p.cold_heads, heads: p.heads, a_ms: if g == 1 { p.ms_b() } else { p.ms_a() }, b_bytes: p.unload_bytes() };
                prog.tick(e.name, keep, g);
            }
        }
    }
    c.x = extra(st, script, prog);
    c
}

/// lm_head / embedding rows and the MTP layer (all labels pooled, all-time).
fn extra(st: &Store, script: &[u8], prog: &mut Prog) -> Extra {
    let (v, inter) = (st.d.vocab, st.d.inter);
    let (o_out, o_emb, o_mtp, n_mtp) = (st.o_out(), st.o_emb(), st.o_mtp(), st.n_mtp());
    let mut x = Extra { has_script: v > 0 && script.len() == v, ..Default::default() };
    for s in 0..NP {
        if !st.has(s) { continue; }
        let h = st.hdr(s);
        x.out_tok += h.out_tok;
        x.emb_rows += h.emb_rows;
        x.mtp_rows += h.mtp_rows;
        x.labels[s] = h.out_tok > 0 || h.emb_rows > 0;
    }
    let mut buf: Buf = [0.0; CH];
    let mut rec: Rec = [0u8; std::mem::size_of::<Rec>()];
    let mut hist = Hist::default();
    for (ki, &keep) in KEEPS.iter().enumerate() {
        let mut r = VRow { keep, ..Default::default() };
        {
            let mut rd = |o: usize, b: &mut [f32]| st.read_sel(Sel::Labels, o_out + o, b, &mut rec);
            let p = store::hot(v, keep, &mut rd, &mut buf, &mut hist);
            r.lm_cold = v - p.n.min(v);
            if ki == 0 { x.rows_seen = p.seen; }
            if x.has_script {
                let (mut i, mut ties) = (0usize, 0usize);
                let by = &mut r.by_script;
                store::each(v, &mut rd, &mut buf, &mut |y| {
                    if !p.hot(y, &mut ties) { by[(script[i] as usize).min(NS - 1)] += 1; }
                    i += 1;
                });
            }
        }
        {
            let mut rd = |o: usize, b: &mut [f32]| st.read_sel(Sel::Labels, o_emb + o, b, &mut rec);
            let p = store::hot(v, keep, &mut rd, &mut buf, &mut hist);
            r.emb_cold = v - p.n.min(v);
        }
        x.v[ki] = r;
        let mut frac = |off: usize, len: usize| -> Option<f64> {
            let mut rd = |o: usize, b: &mut [f32]| st.read_sel(Sel::Labels, off + o, b, &mut rec);
            let p = store::hot(len, keep, &mut rd, &mut buf, &mut hist);
            if p.total > 0.0 && len > 0 { Some((len - p.n.min(len)) as f64 / len as f64) } else { None }
        };
        let cold_ffn = frac(o_mtp, inter);
        let cold_heads = frac(o_mtp + inter, n_mtp - inter);
        x.m[ki] = MRow { keep, cold_ffn, cold_heads };
        prog.tick("lm_head / embed / MTP", keep, 1);
    }
    x
}

// ---- output ---------------------------------------------------------------------------------------------------------

fn tau_name(e: &Entry) -> &'static str { if e.name == "now" { "tau 5 min" } else { "tau 4 h" } }

/// Text tables.
pub fn text(c: &Curve, st: &Store, disk: u64, min_tokens: u64, w: &mut dyn Write) -> std::io::Result<()> {
    writeln!(w, "store {} ({:.2} MB of {:.2} MB on disk; FFN total {:.2} GB)", st.dir.display(), disk as f64 / 1e6, st.cap as f64 / 1e6,
             c.ffn_bytes_total as f64 / 1e9)?;
    if c.n == 0 { writeln!(w, "no watcher data yet (the gateway folds each job into the store)")?; }
    for e in c.e[..c.n].iter().flatten() {
        write!(w, "\n{}: {} tokens, {} turns", e.name, e.tokens, e.turns)?;
        if e.decayed { write!(w, ", {:.0} decayed rows ({})", e.rows_d, tau_name(e))?; }
        writeln!(w, ", representative: {}", if e.tokens >= min_tokens { "yes" } else { "not yet" })?;
        writeln!(w, "  keep  |  g  | cold FFN | FFN GB | cold heads | A ms/pass | B GB")?;
        for r in &e.rows {
            writeln!(w, "  {:<5} | {:>3} | {:>7.2}% | {:>6.3} | {:>4} / {:<4} | {:>9.3} | {:.3}", r.keep, r.g,
                     r.cold_neurons as f64 / r.neurons.max(1) as f64 * 100.0, r.ffn_bytes as f64 / 1e9, r.cold_heads, r.heads, r.a_ms, r.b_bytes as f64 / 1e9)?;
        }
    }
    let x = &c.x;
    let row_b = q4_row_bytes(st.d.tmpl.hidden, st.d.tmpl.group, st.d.tmpl.bits);
    write!(w, "\nlm_head / embedding rows ({} out tokens, {} embedding rows read, {} rows seen of {}, labels", x.out_tok, x.emb_rows, x.rows_seen, st.d.vocab)?;
    let mut any = false;
    for s in (0..NP).filter(|&s| x.labels[s]) { write!(w, "{} {}", if any { "," } else { "" }, st.name(s))?; any = true; }
    if !any { write!(w, " none")?; }
    writeln!(w, "):\n  keep  | lm_head cold rows | GB    | by script | embed cold rows (SSD GB)")?;
    for r in &x.v {
        write!(w, "  {:<5} | {:>17} | {:.3} | ", r.keep, r.lm_cold, (r.lm_cold as u64 * row_b) as f64 / 1e9)?;
        let mut first = true;
        for (i, &n) in r.by_script.iter().enumerate().filter(|(_, n)| **n > 0) {
            write!(w, "{}{} {}", if first { "" } else { ", " }, SCRIPT_NAMES[i], n)?;
            first = false;
        }
        if !x.has_script { write!(w, "n/a")?; }
        writeln!(w, " | {} ({:.3})", r.emb_cold, (r.emb_cold as u64 * row_b) as f64 / 1e9)?;
    }
    writeln!(w, "MTP draft layer ({} rows, all-time):\n  keep  | cold FFN | cold heads", x.mtp_rows)?;
    for r in &x.m {
        writeln!(w, "  {:<5} | {} | {}", r.keep, Pct(r.cold_ffn), Pct(r.cold_heads))?;
    }
    Ok(())
}

struct Pct(Option<f64>);
impl std::fmt::Display for Pct {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.0 { Some(x) => write!(f, "{:>7.2}%", x * 100.0), None => write!(f, "{:>8}", "no data") }
    }
}
/// JSON number (null when not finite)
struct JF(f64);
impl std::fmt::Display for JF {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { if self.0.is_finite() { write!(f, "{}", self.0) } else { f.write_str("null") } }
}
/// JSON string
struct JS<'a>(&'a str);
impl std::fmt::Display for JS<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("\"")?;
        for ch in self.0.chars() {
            match ch { '"' => f.write_str("\\\"")?, '\\' => f.write_str("\\\\")?, c if (c as u32) < 0x20 => write!(f, "\\u{:04x}", c as u32)?, c => write!(f, "{}", c)? }
        }
        f.write_str("\"")
    }
}
fn jopt(x: Option<f64>) -> JF { JF(x.unwrap_or(f64::NAN)) }

/// The same as JSON, streamed to `w` (no document tree in memory).
pub fn json(c: &Curve, st: &Store, disk: u64, min_tokens: u64, w: &mut dyn Write) -> std::io::Result<()> {
    write!(w, "{{\"store\": {}, \"disk_bytes\": {}, \"cap_bytes\": {}, \"read_bw_gbs\": {}, \"min_tokens_representative\": {}, \"ffn_gb_total\": {}, \"keeps\": [",
           JS(&st.dir.display().to_string()), disk, st.cap, JF(READ_BW / 1e9), min_tokens, JF(c.ffn_bytes_total as f64 / 1e9))?;
    for (i, k) in KEEPS.iter().enumerate() { write!(w, "{}{}", if i > 0 { ", " } else { "" }, JF(*k))?; }
    write!(w, "],\n \"labels\": [")?;
    for (i, e) in c.e[..c.n].iter().flatten().enumerate() {
        let set = match e.sel { Sel::Labels => "labels", Sel::Set(s) if s >= NOW => "pooled-decayed", Sel::Set(_) => "label" };
        write!(w, "{}\n  {{\"label\": {}, \"set\": {}, \"tokens\": {}, \"turns\": {}, \"decayed_rows\": {}, \"per_neuron\": true, \"representative\": {}, \"rows\": [",
               if i > 0 { "," } else { "" }, JS(e.name), JS(set), e.tokens, e.turns, if e.decayed { JF(e.rows_d) } else { JF(f64::NAN) }, e.tokens >= min_tokens)?;
        for (j, r) in e.rows.iter().enumerate() {
            write!(w, "{}\n   {{\"keep\": {}, \"g\": {}, \"available\": true, \"cold_ffn_frac\": {}, \"cold_neurons\": {}, \"ffn_gb\": {}, \"cold_heads\": {}, \"heads\": {}, \"a_ms_per_pass\": {}, \"b_gb\": {}}}",
                   if j > 0 { "," } else { "" }, JF(r.keep), r.g, JF(r.cold_neurons as f64 / r.neurons.max(1) as f64), r.cold_neurons, JF(r.ffn_bytes as f64 / 1e9),
                   r.cold_heads, r.heads, JF(r.a_ms), JF(r.b_bytes as f64 / 1e9))?;
        }
        write!(w, "]}}")?;
    }
    let x = &c.x;
    let row_b = q4_row_bytes(st.d.tmpl.hidden, st.d.tmpl.group, st.d.tmpl.bits);
    write!(w, "],\n \"extra\": {{\"vocab\": {{\"labels\": [")?;
    let mut first = true;
    for s in (0..NP).filter(|&s| x.labels[s]) { write!(w, "{}{}", if first { "" } else { ", " }, JS(st.name(s)))?; first = false; }
    write!(w, "], \"out_tokens\": {}, \"embed_rows_read\": {}, \"rows_seen\": {}, \"vocab\": {}, \"row_bytes\": {}, \"rows\": [", x.out_tok, x.emb_rows, x.rows_seen, st.d.vocab, row_b)?;
    for (i, r) in x.v.iter().enumerate() {
        write!(w, "{}\n   {{\"keep\": {}, \"lm_head_cold_rows\": {}, \"lm_head_cold_gb\": {}, \"lm_head_cold_by_script\": {{", if i > 0 { "," } else { "" }, JF(r.keep), r.lm_cold,
               JF((r.lm_cold as u64 * row_b) as f64 / 1e9))?;
        let mut first = true;
        for (k, &n) in r.by_script.iter().enumerate().filter(|(_, n)| **n > 0) { write!(w, "{}{}: {}", if first { "" } else { ", " }, JS(SCRIPT_NAMES[k]), n)?; first = false; }
        write!(w, "}}, \"embed_cold_rows\": {}, \"embed_cold_gb_ssd\": {}}}", r.emb_cold, JF((r.emb_cold as u64 * row_b) as f64 / 1e9))?;
    }
    write!(w, "],\n  \"note\": \"cold = rows outside the top rows holding `keep` of the emitted-token (lm_head) / rows-read (embedding) counts; unseen rows are cold\"}},\n  \"mtp\": {{\"tokens\": {}, \"rows\": [", x.mtp_rows)?;
    for (i, r) in x.m.iter().enumerate() {
        write!(w, "{}{{\"keep\": {}, \"cold_ffn_frac\": {}, \"cold_head_frac\": {}}}", if i > 0 { ", " } else { "" }, JF(r.keep), jopt(r.cold_ffn), jopt(r.cold_heads))?;
    }
    writeln!(w, "]}}}},\n \"note\": \"A = weight bytes not read per pass / 240 GB/s (g = 64: the stage-A 64-block mask; g = 1: a gather kernel); B = RAM freed by a real unload. Streamed from the disk store (full f32).\"}}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::JobSrc;
    use streamdense_model::LayerKind;

    fn tmpl() -> ActivityStats {
        // inter 1280 > MAXU: g = 1 goes through the exact radix path, g = 64 through the unit sort
        let kinds = (0..4).map(|l| if l % 4 == 3 { LayerKind::Attention } else { LayerKind::DeltaNet }).collect();
        ActivityStats { enabled: true, layers: 4, hidden: 128, inter: 1280, kinds, dn_kh: 1, dn_dk: 128, dn_vh: 2, dn_dv: 128, heads: 3, kv_heads: 1,
            head_dim: 128, conv_k: 4, group: 64, bits: 4, ..Default::default() }
    }

    struct Job { v: Vec<f32>, rows: (u64, u64) }
    impl JobSrc for Job {
        fn rows(&self) -> (u64, u64) { self.rows }
        fn read(&self, off: usize, out: &mut [f32]) { out.copy_from_slice(&self.v[off..off + out.len()]); }
    }

    /// A job over the unified layout: random FFN / heads / MTP (some zeros), integer token counts.
    fn job(st: &Store, seed: u64) -> Job {
        let mut r = seed;
        let mut rnd = || { r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((r >> 40) as f32) / (1u64 << 24) as f32 };
        let mut v = vec![0f32; st.n()];
        for (i, x) in v[..st.o_out()].iter_mut().enumerate() { let y = rnd(); *x = if i % 7 == 3 { 0.0 } else { y * y * y }; }
        for x in &mut v[st.o_out()..] { let y = rnd(); *x = if y < 0.4 { 0.0 } else { (y * 40.0).floor() }; }
        Job { v, rows: (1000 + seed, 10 + seed) }
    }

    /// greedy ascending cold count (`unload::cold_mask`): x <= 0 always cold, then smallest first while mass <= budget
    fn bf_cold(m: &[f64], keep: f64) -> usize {
        let total: f64 = m.iter().filter(|x| **x > 0.0).sum();
        if keep >= 1.0 || !(total > 0.0) { return if keep >= 1.0 { 0 } else { m.len() }; }
        let budget = (1.0 - keep) * total * (1.0 + 1e-9);
        let mut idx: Vec<usize> = (0..m.len()).collect();
        idx.sort_by(|&a, &b| m[a].total_cmp(&m[b]).then(a.cmp(&b)));
        let (mut c, mut n) = (0f64, 0);
        for &i in &idx { if m[i] <= 0.0 { n += 1; continue; } if c + m[i] > budget { break; } c += m[i]; n += 1; }
        n
    }
    /// hot set by brute-force sort: the fewest largest positive rows holding `keep` of the mass
    fn bf_hot(v: &[f32], keep: f64) -> Vec<bool> {
        let mut idx: Vec<usize> = (0..v.len()).filter(|&i| v[i] > 0.0).collect();
        idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
        let total: f64 = idx.iter().map(|&i| v[i] as f64).sum();
        let mut hot = vec![false; v.len()];
        let mut c = 0f64;
        for &i in &idx { if c >= keep * total { break; } c += v[i] as f64; hot[i] = true; }
        hot
    }
    /// expected (cold neurons, cold heads) of a selection's values at keep / g
    fn bf_plan(st: &Store, v: &[f32], keep: f64, g: usize) -> (usize, usize) {
        let (layers, inter) = (st.d.layers, st.d.inter);
        let (mut cn, mut ch) = (0, 0);
        for l in 0..layers {
            let row = &v[l * inter..(l + 1) * inter];
            if g == 1 { cn += bf_cold(&row.iter().map(|x| *x as f64).collect::<Vec<_>>(), keep); } else {
                let units: Vec<f64> = row.chunks(g).map(|c| c.iter().map(|x| *x as f64).sum()).collect();
                // unit greedy: every unit's mass counts, zeros included (cold_units semantics)
                let total: f64 = units.iter().sum();
                let budget = (1.0 - keep) * total * (1.0 + 1e-9);
                let mut idx: Vec<usize> = (0..units.len()).collect();
                idx.sort_by(|&a, &b| units[a].total_cmp(&units[b]).then(a.cmp(&b)));
                let mut c = 0f64;
                for &u in &idx { if c + units[u] > budget { break; } c += units[u]; cn += row.chunks(g).nth(u).unwrap().len(); }
            }
            let hr = st.heads_of(l);
            let hm: Vec<f64> = v[hr].iter().map(|x| *x as f64).collect();
            let total: f64 = hm.iter().sum();
            let budget = (1.0 - keep) * total * (1.0 + 1e-9);
            let mut idx: Vec<usize> = (0..hm.len()).collect();
            idx.sort_by(|&a, &b| hm[a].total_cmp(&hm[b]).then(a.cmp(&b)));
            let mut c = 0f64;
            if total > 0.0 { for &u in &idx { if c + hm[u] > budget { break; } c += hm[u]; ch += 1; } }
        }
        (cn, ch)
    }

    #[test]
    fn curve_matches_brute_force() {
        let dir = std::env::temp_dir().join(format!("sd_curve_store_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut d = Dims::from_stats(&tmpl(), 64);
        d.vocab = 300;
        // empty store (read only, nothing on disk): no entries, no data
        {
            let st = Store::open(&dir, d.clone(), names(), watcher::MIXED, true).unwrap();
            let c = compute(&st, &[], &mut Prog::new(false));
            assert_eq!(c.n, 0);
            assert!(c.x.m.iter().all(|r| r.cold_ffn.is_none()));
        }
        let (a, b) = (0usize, 3usize); // code-rust, agent-tools
        let t = 1000.0;
        let (ja, jb) = {
            let mut w = Store::open(&dir, d.clone(), names(), watcher::MIXED, false).unwrap();
            let (ja, jb) = (job(&w, 1), job(&w, 2));
            w.fold_job(a, &ja, 1, t).unwrap();
            w.fold_job(b, &jb, 2, t).unwrap();
            (ja, jb)
        };
        let mut st = Store::open(&dir, d.clone(), names(), watcher::MIXED, true).unwrap();
        st.refresh();
        st.t = t;
        let script: Vec<u8> = (0..d.vocab).map(|i| (i % 4) as u8).collect();
        let a0 = streamdense_core::alloc::thread_allocs();
        let c = compute(&st, &script, &mut Prog::new(false));
        assert_eq!(streamdense_core::alloc::thread_allocs(), a0, "compute path allocated on the heap");
        let pooled: Vec<f32> = ja.v.iter().zip(&jb.v).map(|(x, y)| (0.0f32 + x) + y).collect();
        let names_got: Vec<&str> = c.e[..c.n].iter().flatten().map(|e| e.name).collect();
        assert_eq!(names_got, vec!["code-rust", "agent-tools", "pooled", "now", "session"]);
        for e in c.e[..c.n].iter().flatten() {
            let v: &[f32] = match e.name { "code-rust" => &ja.v, "agent-tools" => &jb.v, _ => &pooled };
            for r in &e.rows {
                let (cn, ch) = bf_plan(&st, v, r.keep, r.g);
                assert_eq!((r.cold_neurons, r.cold_heads), (cn, ch), "{} keep {} g {}", e.name, r.keep, r.g);
                assert_eq!(r.neurons, 4 * 1280);
            }
            // fewer cold neurons the more is kept
            for g in 0..GS.len() { assert!((1..NK).all(|k| e.rows[k * 2 + g].cold_neurons >= e.rows[(k - 1) * 2 + g].cold_neurons)); }
        }
        let e0 = c.e[0].unwrap();
        assert_eq!((e0.tokens, e0.turns), (1001, 1));
        assert_eq!(c.e[2].unwrap().tokens, 1001 + 1002);
        // lm_head / embedding / MTP vs a brute-force sort of the pooled rows
        let (o_out, o_emb, o_mtp, v) = (st.o_out(), st.o_emb(), st.o_mtp(), d.vocab);
        for r in &c.x.v {
            let hot = bf_hot(&pooled[o_out..o_out + v], r.keep);
            assert_eq!(r.lm_cold, hot.iter().filter(|h| !**h).count(), "lm_head keep {}", r.keep);
            let mut by = [0usize; NS];
            for i in (0..v).filter(|&i| !hot[i]) { by[script[i] as usize] += 1; }
            assert_eq!(r.by_script, by);
            let he = bf_hot(&pooled[o_emb..o_emb + v], r.keep);
            assert_eq!(r.emb_cold, he.iter().filter(|h| !**h).count(), "embed keep {}", r.keep);
        }
        for r in &c.x.m {
            let hf = bf_hot(&pooled[o_mtp..o_mtp + 1280], r.keep);
            assert_eq!(r.cold_ffn, Some(hf.iter().filter(|h| !**h).count() as f64 / 1280.0));
            let hh = bf_hot(&pooled[o_mtp + 1280..o_out], r.keep);
            assert_eq!(r.cold_heads, Some(hh.iter().filter(|h| !**h).count() as f64 / 3.0));
        }
        assert_eq!(c.x.mtp_rows, 11 + 12);
        assert_eq!(c.x.rows_seen, pooled[o_out..o_out + v].iter().filter(|x| **x > 0.0).count());
        assert!(c.x.labels[a] && c.x.labels[b] && !c.x.labels[1]);
        // outputs: text table + valid JSON with the same numbers
        let mut t = Vec::new();
        text(&c, &st, st.disk_bytes(), 200_000, &mut t).unwrap();
        let t = String::from_utf8(t).unwrap();
        assert!(t.contains("code-rust: 1001 tokens, 1 turns, representative: not yet") && t.contains("MTP draft layer (23 rows"), "{}", t);
        let mut j = Vec::new();
        json(&c, &st, st.disk_bytes(), 200_000, &mut j).unwrap();
        let j: serde_json::Value = serde_json::from_slice(&j).unwrap();
        assert_eq!(j["labels"][2]["label"], "pooled");
        assert_eq!(j["labels"][0]["rows"][3]["cold_neurons"].as_u64(), Some(c.e[0].unwrap().rows[3].cold_neurons as u64));
        assert_eq!(j["extra"]["vocab"]["rows"][0]["lm_head_cold_rows"].as_u64(), Some(c.x.v[0].lm_cold as u64));
        assert!(j["labels"][3]["decayed_rows"].as_f64().is_some() && j["labels"][0]["decayed_rows"].is_null());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
