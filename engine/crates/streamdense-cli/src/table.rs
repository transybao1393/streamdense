//! streamdense table stats DIR [--budget-mb 32]   (speculative draft table: entries/bytes, see streamdense-model/src/draft_table.rs)
use std::path::Path;
use streamdense_model::draft_table::DraftTable;
use crate::Args;

pub fn main(a: &Args) {
    let (sub, dir) = (a.0.get(2).map(|s| s.as_str()), a.0.get(3));
    let (Some("stats"), Some(dir)) = (sub, dir) else { eprintln!("usage: streamdense table stats DIR [--budget-mb 32]"); std::process::exit(1) };
    if !Path::new(dir).is_dir() { eprintln!("ERROR: {} is not a directory", dir); std::process::exit(2) }
    let t = DraftTable::open(Path::new(dir), (a.f("--budget-mb", 32.0) * 1048576.0) as usize).unwrap_or_else(|e| { eprintln!("ERROR: {}", e); std::process::exit(2) });
    let s = t.stats();
    let mb = |b: f64| b / 1048576.0;
    println!("dir            {}", dir);
    println!("history tokens {} on disk | {} in RAM ring (cap {}) | torn tail on open: {}", s.disk_tokens, s.ram_hist_tokens, s.hist_cap, s.torn_tail);
    println!("RAM entries    hist {} | seq {} (cap {}) | alt {}", s.hist_entries, s.seq_entries, s.seq_cap, s.alt_entries);
    println!("disk index     {} entries", t.disk_index_entries());
    println!("bytes          RAM {:.2} MB of {:.2} MB budget | disk {:.2} MB", mb(s.ram_bytes as f64), mb(s.budget as f64), mb(s.disk_bytes as f64));
    for (n, (acc, tot)) in ["seq", "hist", "disk", "alt"].iter().zip(s.accept) {
        println!("accept {:<5}   {}/{} ({:.1}%)", n, acc, tot, if tot > 0 { 100.0 * acc as f64 / tot as f64 } else { 0.0 });
    }
}
