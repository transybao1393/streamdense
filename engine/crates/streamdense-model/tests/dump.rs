//! U13 self-test: streamdense-model/src/dump.rs. Capture known arena buffers with `Dumper::capture` (GPU copy_bytes),
//! `write` them, and read the file back with `SafeIndex::open_file`: names, dtypes, shapes and bytes must match.
//! Also: layer filter, duplicate name, region overflow, empty shape, reset.
//! Run: cargo test --offline -p streamdense-model --test dump -- --nocapture
use streamdense_core::arena::Arena;
use streamdense_core::{DType, SafeIndex};
use streamdense_metal::{commit, contents, wait, Enc, Gpu};
use streamdense_model::dump::Dumper;

fn bytes_of<T: Copy>(v: &[T]) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }.to_vec()
}

#[test]
fn dump_roundtrip() {
    let mut gpu = Gpu::from_shader_files(&["common_bf16", "head_sample"]).unwrap();
    gpu.pipeline("copy_bytes").unwrap();
    let size = 1 << 20;
    let buf = gpu.buffer(size).unwrap();
    let mut arena = Arena::new(contents(&buf), size);
    let src = arena.alloc("src", 64 << 10).unwrap();
    let region = arena.alloc("dump", 8 << 10).unwrap();

    // known source data at three offsets inside `src`
    let a_bf: Vec<u16> = (0..15u16).map(|i| 0x3f80 + i * 7).collect(); // bf16 [3, 5]
    let b_f32: Vec<f32> = (0..8).map(|i| i as f32 * -1.25 + 0.5).collect(); // f32 [2, 4]
    let c_u32: Vec<u32> = vec![248046, 7, 0, u32::MAX]; // u32 [4] (indices)
    let d_u8: Vec<u8> = (0..7u8).map(|i| i * 37).collect(); // u8 [7] (odd size: next slot must stay aligned)
    let e_bf: Vec<u16> = (0..256u16).map(|i| i ^ 0x5a5a).collect(); // bf16 [1, 256]
    let parts: [(usize, Vec<u8>); 5] =
        [(0, bytes_of(&a_bf)), (4096, bytes_of(&b_f32)), (8192, bytes_of(&c_u32)), (12288, bytes_of(&d_u8)), (16384, bytes_of(&e_bf))];
    for (o, b) in &parts {
        unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), arena.base().add(src.at(*o)), b.len()) };
    }
    // fill the dump region with garbage so a missing copy is visible
    unsafe { std::ptr::write_bytes(arena.base().add(region.off), 0xcd, region.len) };

    let dumper = Dumper::new(region, vec![3]);
    let cb = gpu.command_buffer();
    {
        let e = Enc::new(&gpu, &cb, &buf);
        dumper.capture(&e, "emb", src.at(0), DType::BF16, &[3, 5]).unwrap();
        dumper.capture(&e, "L3.dn.g", src.at(4096), DType::F32, &[2, 4]).unwrap();
        dumper.capture(&e, "logits_top32.idx", src.at(8192), DType::U32, &[4]).unwrap();
        dumper.capture(&e, "L3.raw", src.at(12288), DType::U8, &[7]).unwrap();
        dumper.capture(&e, "L3.dn.y", src.at(16384), DType::BF16, &[1, 256]).unwrap();
        // layer 5 is not enabled -> skipped (Ok, nothing recorded)
        dumper.capture(&e, "L5.dn.y", src.at(16384), DType::BF16, &[1, 256]).unwrap();
        // errors
        assert!(dumper.capture(&e, "emb", src.at(0), DType::BF16, &[3, 5]).is_err(), "duplicate name");
        assert!(dumper.capture(&e, "hf", src.at(0), DType::BF16, &[0, 5]).is_err(), "empty shape");
        assert!(dumper.capture(&e, "big", src.at(0), DType::F32, &[4096]).is_err(), "region overflow");
        e.end();
    }
    commit(&cb);
    wait(&cb).unwrap();
    assert_eq!(dumper.names(), vec!["emb", "L3.dn.g", "logits_top32.idx", "L3.raw", "L3.dn.y"]);

    let dir = std::env::temp_dir().join(format!("sd_dump_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("dump.safetensors");
    let path_s = path.to_str().unwrap();
    dumper.write(&arena, path_s).unwrap();

    let idx = SafeIndex::open_file(path_s).unwrap();
    let file = std::fs::read(&path).unwrap();
    let want: [(&str, DType, Vec<usize>, &Vec<u8>); 5] = [
        ("emb", DType::BF16, vec![3, 5], &parts[0].1),
        ("L3.dn.g", DType::F32, vec![2, 4], &parts[1].1),
        ("logits_top32.idx", DType::U32, vec![4], &parts[2].1),
        ("L3.raw", DType::U8, vec![7], &parts[3].1),
        ("L3.dn.y", DType::BF16, vec![1, 256], &parts[4].1),
    ];
    for (name, dt, shape, data) in want.iter() {
        let ti = idx.get(name).unwrap_or_else(|| panic!("{} missing from the dump", name));
        assert_eq!(ti.dtype, *dt, "{} dtype", name);
        assert_eq!(&ti.shape, shape, "{} shape", name);
        assert_eq!(ti.nbytes, data.len(), "{} nbytes", name);
        let got = &file[ti.off as usize..ti.off as usize + ti.nbytes];
        assert_eq!(got, data.as_slice(), "{} bytes", name);
        println!("dump {:18} {:?} {:?}: {} B round trip ok", name, dt, shape, ti.nbytes);
    }
    assert!(idx.get("L5.dn.y").is_none(), "disabled layer must not be dumped");
    assert_eq!(idx.names().count(), 5);

    // reset: region reusable, names cleared
    dumper.reset();
    assert!(dumper.is_empty());
    let cb = gpu.command_buffer();
    {
        let e = Enc::new(&gpu, &cb, &buf);
        dumper.capture(&e, "emb", src.at(16384), DType::BF16, &[1, 256]).unwrap();
        e.end();
    }
    commit(&cb);
    wait(&cb).unwrap();
    dumper.write(&arena, path_s).unwrap();
    let idx = SafeIndex::open_file(path_s).unwrap();
    let ti = idx.get("emb").unwrap();
    let file = std::fs::read(&path).unwrap();
    assert_eq!(&file[ti.off as usize..ti.off as usize + ti.nbytes], parts[4].1.as_slice());
    println!("dump reset + recapture ok");

    // empty-enabled list = every layer
    let all = Dumper::new(region, vec![]);
    assert!(all.enabled("L63.h_out") && all.enabled("mtp.h_out"));
    assert!(!dumper.enabled("L4.h_out") && dumper.enabled("L3.h_out") && dumper.enabled("hf"));
    std::fs::remove_dir_all(&dir).ok();
}
