//! streamdense config [--json]
//!   Prints the resolved streamdense.toml values and the file they came from (or "built-in defaults").
//!   --json: one JSON object {"source", "ram": {..}, "gateway": {.., "model_path"}, "watcher": {..}, "forheader": {..}} (read by the Swift app).
//! streamdense up [gateway args...]
//!   exec: sd-watchdog --cap-gb C --margin-gb M --swap-gb S -- nice -n N streamdense gateway --model .. --port ..
//!         --max-ctx .. --abort-gb C --swap-gb S [--mtp] [--spec-depth N|auto] [--draft-table DIR] [gateway args...]
//!   All values from streamdense.toml; a flag given in the extra args replaces the config value (--no-mtp drops --mtp).
//!   One-run state elsewhere: --watcher-dir DIR (or $SD_WATCHER_DIR) and --snapshot-dir DIR pass through to the gateway.
//!   Stop: SIGTERM to this pid (sd-watchdog after the exec) is forwarded to the gateway (SIGINT after 3 s), or kill the process group.
use crate::{config_root, load_config, Args};
use serde_json::json;
use std::os::unix::process::CommandExt;

pub fn config_main(a: &Args) {
    let c = load_config();
    let model = config_root(&c).join(&c.cfg.gateway.model);
    let model = model.canonicalize().unwrap_or(model).display().to_string();
    if a.has("--json") {
        let mut g = serde_json::to_value(&c.cfg.gateway).unwrap();
        g["model_path"] = json!(model);
        let o = json!({"source": c.source.as_ref().map(|p| p.display().to_string()), "ram": c.cfg.ram, "gateway": g, "watcher": c.cfg.watcher, "forheader": c.cfg.forheader});
        println!("{}", o);
        return;
    }
    let (r, g) = (&c.cfg.ram, &c.cfg.gateway);
    println!("source: {}", c.source_str());
    println!("[ram]");
    println!("  cap_gb    = {}   (sd-watchdog --cap-gb, engine --abort-gb)", r.cap_gb);
    println!("  swap_gb   = {}   (sd-watchdog + engine --swap-gb)", r.swap_gb);
    println!("  margin_gb = {}   (sd-watchdog --margin-gb)", r.margin_gb);
    println!("  nice      = {}", r.nice);
    println!("[gateway]");
    println!("  model     = {}   ({})", g.model, model);
    println!("  port      = {}", g.port);
    println!("  max_ctx   = {}", g.max_ctx);
    println!("  mtp       = {}", g.mtp);
    println!("  spec_depth = {}   (MTP drafts verified per pass, or auto; gateway --spec-depth)", g.spec_depth.arg());
    println!("  draft_table = {:?}   (learning draft table for spec_depth auto, \"\" = off; gateway --draft-table)", g.draft_table);
    println!("  gemv      = {}   (matmul family of the exact passes: mma | old; gateway --gemv)", g.gemv);
    let w = &c.cfg.watcher;
    println!("[watcher]   (always on; GET /v1/streamdense/watcher)");
    println!("  mode      = {}   (observe | unload (reserved, observes until then))", w.mode);
    println!("  decision window = {} folds + >= {} turns, margin = {}, hysteresis = {} window(s)", w.decide_window_folds, w.decide_window_min_turns, w.margin, w.hysteresis);
    println!("  go-live   = {} clean audits in a row (every reply audited in idle time) · heat sample {} ms", w.go_live_points, w.sample_ms);
    println!("  languages = keep {}, cut {:?} (never domains; cut = unload-first)", w.keep_languages, w.cut_languages);
    if w.decide_every_turns.is_some() || w.confidence.is_some() { println!("  (decide_every_turns / confidence are retired and ignored)"); }
    println!("  keep_frac = {}, granularity = {}", w.keep_frac, w.granularity);
    println!("  dir       = {}   (gateway --watcher-dir DIR or ${} for one run{})", w.dir, crate::watcher::WATCHER_DIR_ENV,
             std::env::var(crate::watcher::WATCHER_DIR_ENV).ok().filter(|d| !d.is_empty()).map(|d| format!("; set now: {}", d)).unwrap_or_default());
    let f = &c.cfg.forheader;
    println!("[forheader]   (12 GB mode, research; gateway --forheader off|12gb)");
    println!("  mode      = {}   (off | 12gb)", f.mode);
    println!("  cap_gb    = {}   (RAM the resident set is sized for, by formula; the hard cap stays ram.cap_gb = {})", f.cap_gb, c.cfg.ram.cap_gb);
    println!("  mtp       = {}   (MTP decode in this mode; costs resident neurons)", f.mtp);
    println!("  slots     = {}, host = {}, stream_tokens = {}, fast = {}, fused = {}   (fast / fused: gateway --no-fast / --no-fused)", f.slots, f.host, f.stream_tokens, f.fast, f.fused);
    println!("  ctx_draft = {}, ctx_draft_max = {}   (context drafts, lossless; gateway --ctx-draft / --no-ctx-draft)", f.ctx_draft, f.ctx_draft_max);
    let t1 = crate::watcher::expand_home(&f.t1);
    println!("  t1        = {}   ({})", f.t1, if t1.is_file() { "present" } else { "MISSING: streamdense forheader export --data DIR" });
    if f.on() { println!("  (12 GB mode: activity recorder and answer library are off)"); }
}

pub fn up_main(a: &Args) {
    let c = load_config();
    let (r, g) = (&c.cfg.ram, &c.cfg.gateway);
    let extra: Vec<String> = a.0.iter().skip(2).cloned().collect();
    let ex = Args(extra.clone());
    let exe = std::env::current_exe().unwrap_or_else(|e| { eprintln!("ERROR: current_exe: {}", e); std::process::exit(2) });
    let watchdog = exe.with_file_name("sd-watchdog");
    if !watchdog.is_file() { eprintln!("ERROR: {} not found (cargo build --release in engine/)", watchdog.display()); std::process::exit(2); }

    let model = config_root(&c).join(&g.model).display().to_string();
    // effective values (extra args win) for the summary line and the watchdog
    let cap: f64 = ex.s("--abort-gb").and_then(|v| v.parse().ok()).unwrap_or(r.cap_gb);
    let swap: f64 = ex.s("--swap-gb").and_then(|v| v.parse().ok()).unwrap_or(r.swap_gb);
    let mut gw: Vec<String> = vec!["gateway".into()];
    let mut put = |k: &str, v: String| if !ex.has(k) { gw.push(k.into()); gw.push(v); };
    put("--model", model);
    put("--port", g.port.to_string());
    put("--max-ctx", g.max_ctx.to_string());
    put("--abort-gb", cap.to_string());
    put("--swap-gb", swap.to_string());
    if g.mtp && !ex.has("--mtp") && !ex.has("--no-mtp") { gw.push("--mtp".into()); }
    if g.mtp && g.spec_depth.deep() && !ex.has("--spec-depth") { gw.push("--spec-depth".into()); gw.push(g.spec_depth.arg()); }
    if g.mtp && !g.draft_table.is_empty() && !ex.has("--draft-table") { gw.push("--draft-table".into()); gw.push(g.draft_table.clone()); }
    if !ex.has("--gemv") { gw.push("--gemv".into()); gw.push(g.gemv.clone()); }
    gw.extend(extra);
    let show = |k: &str| { let i = gw.iter().position(|x| x == k).unwrap(); gw[i + 1].clone() };
    let mtp = gw.iter().any(|x| x == "--mtp") && !gw.iter().any(|x| x == "--no-mtp");
    let fh = ex.s("--forheader").unwrap_or_else(|| c.cfg.forheader.mode.clone()) == "12gb";
    eprintln!("cap {} GB · swap {} GB · ctx {} · model {} · port {}{}{} · config {}",
        cap, swap, show("--max-ctx"), show("--model"), show("--port"), if mtp && !fh { " · mtp" } else { "" },
        if fh { format!(" · 12 GB mode (research, cap {} GB{})", c.cfg.forheader.cap_gb, if c.cfg.forheader.mtp { ", mtp" } else { "" }) } else { String::new() }, c.source_str());
    // the watcher directory of this run (--watcher-dir > $SD_WATCHER_DIR > [watcher] dir): test / smoke runs keep the user's state untouched
    let wdir = ex.s("--watcher-dir").map(std::path::PathBuf::from).unwrap_or_else(|| crate::watcher::config_dir(&c.cfg.watcher.dir));
    eprintln!("watcher dir {}{}", wdir.display(), if ex.has("--watcher-dir") { " (--watcher-dir)" }
              else if std::env::var(crate::watcher::WATCHER_DIR_ENV).is_ok_and(|d| !d.is_empty()) { " ($SD_WATCHER_DIR)" } else { " ([watcher] dir)" });

    let mut cmd = std::process::Command::new(&watchdog);
    cmd.args(["--cap-gb", &cap.to_string(), "--margin-gb", &r.margin_gb.to_string(), "--swap-gb", &swap.to_string(), "--"])
        .args(["nice", "-n", &r.nice.to_string()]).arg(&exe).args(&gw);
    let e = cmd.exec(); // replaces this process: signals reach sd-watchdog directly
    eprintln!("ERROR: exec {}: {}", watchdog.display(), e);
    std::process::exit(2);
}
