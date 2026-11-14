//! Fused K-row verify kernels (verify_fused_q4.metal) vs the current exact verify kernels at the 27B shapes, random
//! weights (see streamdense_model::verify_bench).
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = streamdense_model::verify_bench::run_cli(&args) {
        eprintln!("verify_bench: {}", e);
        std::process::exit(1);
    }
}
