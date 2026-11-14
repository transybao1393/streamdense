//! U2 self-test: safetensors writer/reader round trip, real Qwen3.8-27B header counts (header bytes only),
//! bf16 RNE edge cases.
//! Run: cargo test --release --offline -p streamdense-core --test safetensors -- --nocapture
use streamdense_core::{bf16_to_f32, f32_to_bf16, write_safetensors, DType, SafeIndex};

const REAL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models/qwen38_27b_mlx4");

fn scratch(name: &str) -> String {
    let d = std::env::temp_dir().join("sd_u2_test");
    std::fs::create_dir_all(&d).unwrap();
    d.join(name).to_string_lossy().into_owned()
}

#[test]
fn write_read_round_trip() {
    let a: Vec<u8> = (0..4 * 3 * 4).map(|i| i as u8).collect();         // U32 [4,3]
    let b: Vec<u8> = (0..2 * 5).map(|i| (i * 3) as u8).collect();         // BF16 [5]
    let c: Vec<u8> = vec![7u8; 4];                                        // F32 [1]
    let d: Vec<u8> = vec![1, 2, 3];                                       // U8 [3] (header padding with odd sizes)
    let e: Vec<u8> = vec![];                                              // F16 [0]
    let tens: Vec<(&str, DType, &[usize], &[u8])> = vec![
        ("w.weight", DType::U32, &[4, 3], &a), ("L0.dn.\"q\"\\", DType::BF16, &[5], &b),
        ("f", DType::F32, &[1], &c), ("u8", DType::U8, &[3], &d), ("empty", DType::F16, &[0], &e),
    ];
    let p = scratch("rt.safetensors");
    write_safetensors(&p, &tens).unwrap();
    let raw = std::fs::read(&p).unwrap();
    let hl = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    assert_eq!(hl % 8, 0, "header padded to 8");
    let idx = SafeIndex::open_file(&p).unwrap();
    assert_eq!(idx.len(), tens.len());
    assert_eq!(idx.names().collect::<Vec<_>>(), tens.iter().map(|t| t.0).collect::<Vec<_>>());
    for (n, dt, sh, data) in &tens {
        let t = idx.get(n).unwrap();
        assert_eq!((t.dtype, &t.shape[..], t.nbytes), (*dt, *sh, data.len()), "{}", n);
        assert_eq!(&raw[t.off as usize..t.off as usize + t.nbytes], *data, "{}", n);
    }
    // wrong byte count is refused
    assert!(write_safetensors(&scratch("bad.safetensors"), &[("x", DType::F32, &[2], &[0u8; 4])]).is_err());
    // truncated file is refused
    std::fs::write(scratch("trunc.safetensors"), &raw[..raw.len() - 1]).unwrap();
    assert!(SafeIndex::open_file(&scratch("trunc.safetensors")).is_err());
    println!("  round trip: {} tensors, header {} B, offsets/bytes/shape/dtype exact; bad size + truncation refused", tens.len(), hl);
}

#[test]
fn bf16_rne_edges() {
    let c = |f: f32| f32_to_bf16(f);
    // exact round trip of every non-NaN bf16
    for h in 0..=0xffffu16 {
        let f = bf16_to_f32(h);
        if f.is_nan() { assert!(bf16_to_f32(c(f)).is_nan()); continue; }
        assert_eq!(c(f), h);
    }
    let one = 1.0f32.to_bits();
    assert_eq!(c(f32::from_bits(one | 0x8000)), 0x3f80);          // tie -> even (down)
    assert_eq!(c(f32::from_bits(0x3f818000)), 0x3f82);             // tie -> even (up)
    assert_eq!(c(f32::from_bits(one | 0x8001)), 0x3f81);           // just above the tie
    assert_eq!(c(f32::from_bits(one | 0x7fff)), 0x3f80);           // just below
    assert_eq!(c(f32::MAX), 0x7f80);                                // overflow -> +inf
    assert_eq!(c(-f32::MAX), 0xff80);
    assert_eq!(c(f32::from_bits(0x7f7f7fff)), 0x7f7f);             // largest finite stays finite
    assert_eq!(c(f32::INFINITY), 0x7f80);
    assert_eq!(c(-0.0), 0x8000);
    assert_eq!(c(f32::from_bits(0x00018000)), 0x0002);             // subnormal tie -> even
    let n = c(f32::from_bits(0xffc00001));
    assert!(bf16_to_f32(n).is_nan() && n & 0x8000 != 0, "NaN keeps sign");
    assert!(bf16_to_f32(c(f32::from_bits(0x7f800001))).is_nan(), "signalling NaN with low payload stays NaN");
    println!("  bf16: all 65536 round trip, ties/overflow/subnormal/NaN ok");
}

#[test]
fn real_headers_counts() {
    if !std::path::Path::new(&format!("{}/model.safetensors.index.json", REAL)).exists() {
        eprintln!("SKIP: {} missing", REAL);
        return;
    }
    let idx = SafeIndex::open_mlx_dir(REAL).unwrap();
    let mtp = idx.names().filter(|n| n.starts_with("mtp.")).count();
    let main = idx.len() - mtp;
    let skipped: usize = idx.skipped.iter().map(|s| s.1).sum();
    let non_embed: u64 = idx.iter().filter(|(n, _)| !n.contains("embed_tokens")).map(|(_, t)| t.nbytes as u64).sum();
    let all: u64 = idx.iter().map(|(_, t)| t.nbytes as u64).sum();
    println!("  real: {} shard tensors + {} MTP, {} files, skipped {:?} ({} entries); bytes all {} / without embed {}",
             main, mtp, idx.files().len(), idx.skipped, skipped, all, non_embed);
    assert_eq!((main, mtp), (1847, 29));
    assert_eq!(skipped, 333);
    assert_eq!(non_embed, 14_731_937_792, "text weights without embed (incl. MTP)");
    let t = idx.get("language_model.model.layers.0.mlp.gate_proj.weight").unwrap();
    assert_eq!((t.dtype, &t.shape[..]), (DType::U32, &[17408usize, 640][..]));
}
