//! Build stamp for the gateway's session line (streamdense-cli/src/gateway.rs): `git describe` of the tree at build
//! time and the build time (Unix seconds). Re-runs whenever a file of this package changes (no rerun-if-changed).
fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let git = std::process::Command::new("git").args(["--no-optional-locks", "describe", "--always", "--dirty", "--tags"]).current_dir(&dir).output().ok()
        .filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    println!("cargo:rustc-env=SD_GIT_DESCRIBE={}", git);
    println!("cargo:rustc-env=SD_BUILD_TS={}", ts);
}
