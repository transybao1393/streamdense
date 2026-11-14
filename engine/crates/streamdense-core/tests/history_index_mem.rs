//! Real heap bound of the history index (Counting allocator installed; one test so nothing else allocates beside it):
//! open + multi-threaded project build + flood of appends + queries + save stay under the configured budget.
use std::fs;
use std::sync::Arc;
use streamdense_core::alloc;
use streamdense_core::history_index::{Config, Control, Drafts, Encoder, HistoryIndex};

#[global_allocator]
static A: alloc::Counting = alloc::Counting;

#[test]
fn heap_stays_under_budget() {
    let d = std::env::temp_dir().join(format!("sd_history_mem_{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    let proj = d.join("proj");
    fs::create_dir_all(&proj).unwrap();
    // 200 files × ~40 KB of word text (8 MB) — far more n-grams than a 4 MB table holds
    let mut x = 88172645463325252u64;
    for f in 0..200 {
        let mut s = String::with_capacity(40 << 10);
        while s.len() < 40 << 10 { x ^= x << 13; x ^= x >> 7; x ^= x << 17; s.push_str(&format!("w{} ", x % 20_000)); if x % 11 == 0 { s.push('\n') } }
        fs::write(proj.join(format!("f{:03}.txt", f)), s).unwrap();
    }
    let enc: Encoder = Arc::new(|s: &str| s.split_whitespace().map(|w| w.bytes().fold(7u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32)) % 100_000).collect());
    let budget = 4 << 20;
    let base = alloc::heap_now();
    let cfg = Config { budget_bytes: budget, threads: 4, max_file_bytes: 64 << 10, ..Config::with_dir(&d.join("idx")) };
    let idx = HistoryIndex::open(cfg).unwrap().0;
    let r = idx.index_dir(&proj, &enc, &Control::default());
    assert_eq!(r.files_indexed, 200, "{:?}", r);
    let mut y = 1u64;
    for _ in 0..200 {
        let t: Vec<u32> = (0..2000).map(|_| { y ^= y << 13; y ^= y >> 7; y ^= y << 17; (y % 50_000) as u32 }).collect();
        idx.ingest(&t);
    }
    let mut out = Drafts::default();
    for i in 0..1000u32 { idx.query_into(&[i, i + 1, i + 2, i + 3], 16, 8, &mut out) }
    idx.save().unwrap();
    let st = idx.stats();
    let peak = alloc::heap_peak() - base;
    let now = alloc::heap_now() - base;
    println!("budget {} B, table {} B, ram_bytes {} B, heap now {} B, heap peak {} B (incl. test corpus strings), evictions {}", budget, st.table_bytes, st.ram_bytes, now, peak, st.evictions);
    assert!(st.evictions > 0);
    assert!(st.ram_bytes <= budget);
    assert!(now <= budget, "heap now {} > budget {}", now, budget);
    assert!(peak <= budget, "heap peak {} > budget {}", peak, budget);
    drop(idx);
    let _ = fs::remove_dir_all(&d);
}
