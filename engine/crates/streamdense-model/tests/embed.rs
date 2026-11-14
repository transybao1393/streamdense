//! U10 self-test: EmbedReader rows of the tiny model (staged bytes == file rows; embed_q4 on them == the MLX `emb`
//! dump), and the Plain hand-off driven by real GPU event signals from a dummy "GPU" loop.
//! Run: cargo test --release --offline -p streamdense-model --test embed -- --nocapture --test-threads 1
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;
use streamdense_core::SafeIndex;
use streamdense_metal::{cb_signal, commit, contents, wait, Enc, Gpu, SdCtrl};
use streamdense_model::embed::{row_bytes, EmbedReader, Handoff, HandoffKind, HandoffPolicy, StopReason};
use streamdense_model::weights::load;
use streamdense_model::{QwenConfig, Weights};

const MODELS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models");
const REFS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../refs");

fn tiny() -> Option<(QwenConfig, SafeIndex, Weights, Gpu)> {
    let dir = format!("{}/tiny_qwen35", MODELS);
    if !std::path::Path::new(&format!("{}/config.json", dir)).exists() { eprintln!("SKIP: {} missing", dir); return None; }
    let cfg = QwenConfig::from_model_dir(&dir).unwrap();
    let idx = SafeIndex::open_mlx_dir(&dir).unwrap();
    let mut gpu = Gpu::from_shader_files(&["common_bf16", "head_sample"]).unwrap();
    gpu.pipeline("embed_q4").unwrap();
    let size = 64 << 20;
    let buf = gpu.buffer(size).unwrap();
    let mut arena = streamdense_core::arena::Arena::new(contents(&buf), size);
    let (w, _) = load(&idx, &cfg, &mut arena, false, 2).unwrap();
    Some((cfg, idx, w, gpu))
}

fn file_row(file: &[u8], idx: &SafeIndex, h: usize, tok: usize) -> Vec<u8> {
    let g = |s: &str| idx.get(&format!("language_model.model.embed_tokens.{}", s)).unwrap().off as usize;
    let (wb, sb) = (h / 2, h / 64 * 2);
    let mut r = Vec::new();
    r.extend_from_slice(&file[g("weight") + tok * wb..g("weight") + (tok + 1) * wb]);
    r.extend_from_slice(&file[g("scales") + tok * sb..g("scales") + (tok + 1) * sb]);
    r.extend_from_slice(&file[g("biases") + tok * sb..g("biases") + (tok + 1) * sb]);
    r
}

#[test]
fn tiny_rows_vs_mlx_emb() {
    let Some((cfg, idx, w, gpu)) = tiny() else { return; };
    let h = cfg.hidden;
    let rb = row_bytes(h);
    let rd = EmbedReader::open(&w.embed, &cfg, 4).unwrap();
    assert_eq!(rd.row_bytes(), rb);
    let file = std::fs::read(&w.embed.file).unwrap();
    // 1) staged bytes == file rows; small (1 thread) and large (threaded) batches; out-of-range refused
    let toks: Vec<u32> = vec![0, 1, 2047, 5, 5, 1024, 77, 3, 2046, 9, 11];
    let mut stage = vec![0u8; toks.len() * rb];
    for chunk in [&toks[..2], &toks[..]] {
        stage.iter_mut().for_each(|b| *b = 0);
        rd.read_rows(chunk, stage.as_mut_ptr()).unwrap();
        for (i, &t) in chunk.iter().enumerate() { assert_eq!(&stage[i * rb..(i + 1) * rb], &file_row(&file, &idx, h, t as usize)[..], "row {}", i); }
    }
    assert!(rd.read_rows(&[cfg.vocab as u32], stage.as_mut_ptr()).is_err());
    // 2) embed_q4 on staged rows == MLX `emb` of the tiny dump (bf16 values widened to f32)
    let dump = format!("{}/dumps/tiny/dump.safetensors", REFS);
    let di = SafeIndex::open_file(&dump).unwrap();
    let dbytes = std::fs::read(&dump).unwrap();
    let get = |n: &str| { let t = di.get(n).unwrap(); (t.dtype, t.shape.clone(), dbytes[t.off as usize..t.off as usize + t.nbytes].to_vec()) };
    let (tdt, _, tb) = get("tokens");
    assert_eq!(tdt, streamdense_core::DType::U32, "U32 tokens (dump-name decision)");
    let seq: Vec<u32> = tb.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let (_, esh, eb) = get("emb");
    let emb: Vec<f32> = eb.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(esh, vec![seq.len(), h]);
    let t = seq.len();
    let buf = gpu.buffer(t * rb + t * h * 2 + 4096).unwrap();
    rd.read_rows(&seq, contents(&buf)).unwrap();
    let out = (t * rb + 255) / 256 * 256;
    let cb = gpu.command_buffer();
    let e = Enc::new(&gpu, &cb, &buf);
    e.pso("embed_q4").b(0, 0).b(1, out).u(2, t as u32);
    e.threads((h / 8, t), 64);
    e.end();
    commit(&cb);
    wait(&cb).unwrap();
    let got: Vec<f32> = unsafe { std::slice::from_raw_parts(contents(&buf).add(out) as *const u16, t * h) }
        .iter().map(|&x| streamdense_core::bf16_to_f32(x)).collect();
    let ndiff = got.iter().zip(&emb).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    println!("  tiny: {} staged rows == file bytes; embed_q4 on {} SSD rows (tokens {:?}) vs MLX emb: {} / {} differ",
             toks.len(), t, seq, ndiff, t * h);
    assert_eq!(ndiff, 0);
}

#[test]
fn plain_handoff_dummy_gpu() {
    let Some((cfg, idx, w, gpu)) = tiny() else { return; };
    let h = cfg.hidden;
    let rb = row_bytes(h);
    let rd = Arc::new(EmbedReader::open(&w.embed, &cfg, 1).unwrap());
    let file = std::fs::read(&w.embed.file).unwrap();
    let eos = cfg.eos[0];
    // (sequence of tokens the "GPU" produces, max_emit, expected tokens, expected stop)
    let cases: Vec<(Vec<u32>, usize, Vec<u32>, StopReason)> = vec![
        (vec![10, 20, 30, 40, 50, 60, 70], 5, vec![10, 20, 30, 40, 50], StopReason::MaxTokens),
        (vec![7, 8, eos, 9, 9], 50, vec![7, 8, eos], StopReason::Eos),
    ];
    for (seq, max_emit, want, why) in cases {
        let ctrl_buf = gpu.buffer(4096).unwrap();
        let ctrl = contents(&ctrl_buf) as *mut SdCtrl;
        let mut stage = vec![0u8; 2 * rb];
        let ev = gpu.shared_event();
        let (tx, rx) = mpsc::channel::<u64>();
        let commit_fn: Box<dyn FnMut(u64) + Send> = Box::new(move |n| { tx.send(n).unwrap(); });
        let mut pol = HandoffPolicy::new(HandoffKind::Plain, 1, 1, cfg.eos.clone(), max_emit);
        pol.timeout = Duration::from_secs(5);
        // step 1: "GPU" writes tok[0] = seq[0], then a real command buffer signals ev = 1
        unsafe { std::ptr::write_volatile(ctrl, SdCtrl { tok: [seq[0], 0], ..Default::default() }) };
        let ho = Handoff::spawn(rd.clone(), ev.clone(), ctrl, stage.as_mut_ptr(), pol, Some(commit_fn));
        let signal = |v: u64| { let cb = gpu.command_buffer(); cb_signal(&cb, &ev, v); commit(&cb); wait(&cb).unwrap(); };
        signal(1);
        let mut commits = Vec::new();
        let mut stage_ok = true;
        while let Ok(n) = rx.recv_timeout(Duration::from_secs(2)) {
            // hand-off of step n-1 staged token seq[n-2] in slot n % 2 and asked for CB(n)
            let slot = (n % 2) as usize;
            let row = unsafe { std::slice::from_raw_parts(stage.as_ptr().add(slot * rb), rb) };
            stage_ok &= row == &file_row(&file, &idx, h, seq[(n - 2) as usize] as usize)[..];
            commits.push(n);
            let next = seq[(n - 1) as usize];
            unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*ctrl).tok[0]), next) };
            signal(n);
        }
        let out = ho.join().unwrap();
        println!("  plain max_emit {}: tokens {:?} stop {:?} steps {} commits {:?} staged rows ok {}; hand-off p50 {:.1} us p99 {:.1} us, pread p50 {:.1} us",
                 max_emit, out.tokens, out.stop, out.steps, commits, stage_ok, out.handoff.p50_us, out.handoff.p99_us, out.pread.p50_us);
        assert_eq!(out.tokens, want);
        assert_eq!(out.stop, why);
        assert!(stage_ok);
        assert_eq!(commits, (2..=want.len() as u64).collect::<Vec<_>>(), "commit(n+1) after each non-final step");
    }
    // abort while waiting: no signal ever comes
    let ctrl_buf = gpu.buffer(4096).unwrap();
    let mut stage = vec![0u8; 2 * rb];
    let ev = gpu.shared_event();
    let ho = Handoff::spawn(rd.clone(), ev, contents(&ctrl_buf) as *mut SdCtrl, stage.as_mut_ptr(),
                            HandoffPolicy::new(HandoffKind::Plain, 1, 1, cfg.eos.clone(), 10), None);
    std::thread::sleep(Duration::from_millis(50));
    ho.abort();
    let out = ho.join().unwrap();
    println!("  abort while waiting -> stop {:?}, tokens {:?}", out.stop, out.tokens);
    assert_eq!(out.stop, StopReason::Abort);
}
