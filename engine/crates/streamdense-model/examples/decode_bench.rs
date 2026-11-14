//! Per-token decode time budget at a model's real shapes, random weights (see streamdense_model::decode_bench).
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = streamdense_model::decode_bench::run_cli(&args) {
        eprintln!("decode_bench: {}", e);
        std::process::exit(1);
    }
}
