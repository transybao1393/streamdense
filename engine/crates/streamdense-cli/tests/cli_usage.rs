//! Unknown subcommands and help must print usage and exit without touching any model (regression: the CLI once
//! loaded a model for `streamdense down` / `--help`). Runs the real binary; the model path prints "conditions:" before Engine::open,
//! so its absence (plus a short wall time) proves no load was started.
use std::process::Command;
use std::time::{Duration, Instant};

fn run(args: &[&str]) -> (Option<i32>, String, String, Duration) {
    let t = Instant::now();
    let o = Command::new(env!("CARGO_BIN_EXE_streamdense")).args(args).output().unwrap();
    (o.status.code(), String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned(), t.elapsed())
}

#[test]
fn unknown_subcommand_prints_usage_and_exits_nonzero() {
    for args in [&["down"][..], &["--export", "x"][..], &[][..]] {
        let (code, out, err, dt) = run(args);
        assert_eq!(code, Some(2), "{:?}: {}", args, err);
        assert!(err.contains("usage: streamdense"), "{:?}: {}", args, err);
        assert!(!err.contains("conditions:") && !out.contains("conditions:"), "{:?} started a model load", args);
        assert!(dt < Duration::from_secs(5), "{:?} took {:?}", args, dt);
    }
}

#[test]
fn help_prints_usage_and_exits_zero() {
    for h in ["help", "--help", "-h"] {
        let (code, out, err, dt) = run(&[h]);
        assert_eq!(code, Some(0), "{}: {}", h, err);
        assert!(out.contains("usage: streamdense"), "{}: {}", h, out);
        assert!(!err.contains("conditions:"), "{} started a model load", h);
        assert!(dt < Duration::from_secs(5), "{} took {:?}", h, dt);
    }
}
