//! U9 self-test: weights::plan on the real headers (bytes per section 2 row, total 14,731,937,792 B), tiny load vs
//! checksums.json + raw file bytes, +1 only on the MTP norms.
//! Run: cargo test --release --offline -p streamdense-model --test weights -- --nocapture --test-threads 1
use serde_json::Value;
use streamdense_core::arena::{Arena, Region};
use streamdense_core::budget::Budget;
use streamdense_core::{bf16_to_f32, f32_to_bf16, SafeIndex};
use streamdense_metal::{contents, Gpu};
use streamdense_model::weights::{load, plan};
use streamdense_model::{LayerW, Mixer, Q4b, QwenConfig};

const MODELS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models");

#[test]
fn plan_real_headers() {
    let dir = format!("{}/qwen38_27b_mlx4", MODELS);
    if !std::path::Path::new(&format!("{}/config.json", dir)).exists() { eprintln!("SKIP: {} missing", dir); return; }
    let cfg = QwenConfig::from_model_dir(&dir).unwrap();
    cfg.validate().unwrap();
    let idx = SafeIndex::open_mlx_dir(&dir).unwrap();
    let mut bud = Budget::new(u64::MAX, 0);
    let total = plan(&idx, &cfg, true, &mut bud).unwrap();
    // section 2 formulas (0.5625 * R * C = R*C/2 + 4*R*C/64)
    let q4 = |r: u64, c: u64| r * c / 2 + 4 * r * (c / 64);
    let (h, i, v) = (5120u64, 17408u64, 248320u64);
    let attn = q4(14336, h) + q4(h, 6144);
    let ffn = q4(i, h) * 2 + q4(h, i);
    let want = [
        ("FFN gate/up/down (4-bit)", 64 * ffn),
        ("DeltaNet proj (4-bit)", 48 * (q4(16480, h) + q4(h, 6144))),
        ("DeltaNet small (conv, A, dt, norm)", 48 * (10240 * 4 + 48 + 48 + 128) * 2),
        ("attention q/k/v/o (4-bit)", 16 * attn),
        ("norms", (64 * 2 + 1) * h * 2 + 16 * 2 * 256 * 2),
        ("lm_head (4-bit)", q4(v, h)),
        ("MTP layer (fc bf16 + 4-bit)", h * 2 * h * 2 + attn + ffn + (5 * h + 2 * 256) * 2),
    ];
    println!("{}", bud.report());
    for (name, b) in &want {
        let got = bud.items.iter().find(|x| x.0 == *name).map(|x| x.1);
        println!("  {:<36} plan {:>14?}  formula {:>14}", name, got, b);
        assert_eq!(got, Some(*b), "{}", name);
    }
    println!("  total {} (want 14,731,937,792)", total);
    assert_eq!(total, 14_731_937_792);
    // without MTP
    let mut b2 = Budget::new(u64::MAX, 0);
    let t2 = plan(&idx, &cfg, false, &mut b2).unwrap();
    assert_eq!(t2, total - want[6].1);
}

fn sum_u32(b: &[u8]) -> u32 {
    let mut s = 0u32;
    for c in b.chunks(4) { let mut w = [0u8; 4]; w[..c.len()].copy_from_slice(c); s = s.wrapping_add(u32::from_le_bytes(w)); }
    s
}

/// name -> (region, byte offset inside it, nbytes) for every tensor `load` places, by the documented layout.
fn placements(cfg: &QwenConfig, w: &streamdense_model::Weights) -> Vec<(String, Region, usize, bool /*mtp norm*/)> {
    let mut out = Vec::new();
    fn q4(out: &mut Vec<(String, Region, usize, bool)>, q: &Q4b, parts: &[(String, usize)]) {
        let (wc, gc) = (q.cols / 8 * 4, q.cols / 64 * 2);
        let mut r0 = 0;
        for (p, r) in parts {
            out.push((format!("{}.weight", p), q.w, r0 * wc, false));
            out.push((format!("{}.scales", p), q.s, r0 * gc, false));
            out.push((format!("{}.biases", p), q.b, r0 * gc, false));
            r0 += r;
        }
        assert_eq!(r0, q.rows);
    }
    let layer = |out: &mut Vec<(String, Region, usize, bool)>, pre: &str, lw: &LayerW, mtp: bool| {
        out.push((format!("{}input_layernorm.weight", pre), lw.in_norm, 0, mtp));
        out.push((format!("{}post_attention_layernorm.weight", pre), lw.post_norm, 0, mtp));
        match &lw.mixer {
            Mixer::Dn(d) => {
                let p = format!("{}linear_attn.", pre);
                q4(out, &d.in_proj, &[(format!("{}in_proj_qkv", p), cfg.conv_dim()), (format!("{}in_proj_z", p), cfg.dn_v_dim()),
                                      (format!("{}in_proj_b", p), cfg.dn_vh), (format!("{}in_proj_a", p), cfg.dn_vh)]);
                q4(out, &d.out_proj, &[(format!("{}out_proj", p), cfg.hidden)]);
                out.push((format!("{}conv1d.weight", p), d.conv_w, 0, false));
                out.push((format!("{}A_log", p), d.a_log, 0, false));
                out.push((format!("{}dt_bias", p), d.dt_bias, 0, false));
                out.push((format!("{}norm.weight", p), d.norm_w, 0, false));
            }
            Mixer::Attn(a) => {
                let p = format!("{}self_attn.", pre);
                q4(out, &a.in_proj, &[(format!("{}q_proj", p), 2 * cfg.q_dim()), (format!("{}k_proj", p), cfg.kv_dim()), (format!("{}v_proj", p), cfg.kv_dim())]);
                q4(out, &a.o_proj, &[(format!("{}o_proj", p), cfg.hidden)]);
                out.push((format!("{}q_norm.weight", p), a.q_norm, 0, mtp));
                out.push((format!("{}k_norm.weight", p), a.k_norm, 0, mtp));
            }
        }
        let m = format!("{}mlp.", pre);
        q4(out, &lw.ffn.gate, &[(format!("{}gate_proj", m), cfg.inter)]);
        q4(out, &lw.ffn.up, &[(format!("{}up_proj", m), cfg.inter)]);
        q4(out, &lw.ffn.down, &[(format!("{}down_proj", m), cfg.hidden)]);
    };
    for (l, lw) in w.layers.iter().enumerate() { layer(&mut out, &format!("language_model.model.layers.{}.", l), lw, false); }
    out.push(("language_model.model.norm.weight".into(), w.final_norm, 0, false));
    q4(&mut out, &w.lm_head, &[("language_model.lm_head".into(), cfg.vocab)]);
    if let Some(m) = &w.mtp {
        out.push(("mtp.fc.weight".into(), m.fc, 0, false));
        out.push(("mtp.pre_fc_norm_embedding.weight".into(), m.pre_e, 0, true));
        out.push(("mtp.pre_fc_norm_hidden.weight".into(), m.pre_h, 0, true));
        out.push(("mtp.norm.weight".into(), m.norm, 0, true));
        layer(&mut out, "mtp.layers.0.", &m.layer, true);
    }
    out
}

#[test]
fn tiny_load_checksums_and_plus_one() {
    let dir = format!("{}/tiny_qwen35", MODELS);
    if !std::path::Path::new(&format!("{}/checksums.json", dir)).exists() { eprintln!("SKIP: {} missing", dir); return; }
    let cfg = QwenConfig::from_model_dir(&dir).unwrap();
    cfg.validate().unwrap();
    let idx = SafeIndex::open_mlx_dir(&dir).unwrap();
    let cs: Value = serde_json::from_str(&std::fs::read_to_string(format!("{}/checksums.json", dir)).unwrap()).unwrap();
    let cs = cs.as_object().unwrap();
    let mut bud = Budget::new(u64::MAX, 0);
    let planned = plan(&idx, &cfg, true, &mut bud).unwrap();
    let gpu = Gpu::from_shader_files(&["common_bf16"]).unwrap();
    let size = planned as usize + 2000 * 16384;
    let buf = gpu.buffer(size).unwrap();
    let mut arena = Arena::new(contents(&buf), size);
    let (w, loaded) = load(&idx, &cfg, &mut arena, true, 4).unwrap();
    assert_eq!(loaded, planned, "bytes read == planned");
    let pl = placements(&cfg, &w);
    let files: Vec<Vec<u8>> = idx.files().iter().map(|f| std::fs::read(f).unwrap()).collect();
    let (mut n_ok, mut n_plus, mut bad) = (0, 0, Vec::new());
    for (name, r, sub, plus) in &pl {
        let t = idx.get(name).unwrap_or_else(|| panic!("{} not in index", name));
        let c = cs.get(name).unwrap_or_else(|| panic!("{} not in checksums.json", name));
        let raw = &files[t.file][t.off as usize..t.off as usize + t.nbytes];
        assert_eq!(sum_u32(raw) as u64, c["sum_u32"].as_u64().unwrap(), "{}: file bytes vs checksums.json", name);
        assert_eq!(t.nbytes as u64, c["nbytes"].as_u64().unwrap(), "{}", name);
        let a: &[u8] = &arena.slice_mut::<u8>(*r)[*sub..*sub + t.nbytes];
        if *plus {
            let want: Vec<u8> = raw.chunks(2).flat_map(|b| f32_to_bf16(bf16_to_f32(u16::from_le_bytes([b[0], b[1]])) + 1.0).to_le_bytes()).collect();
            if a != &want[..] { bad.push(format!("{} (+1)", name)); } else { n_plus += 1; }
        } else if sum_u32(a) != sum_u32(raw) || a != raw {
            bad.push(name.clone());
        } else { n_ok += 1; }
    }
    // every checksums.json tensor accounted for (placed, or the SSD embedding)
    let placed: std::collections::HashSet<&str> = pl.iter().map(|p| p.0.as_str()).collect();
    let missing: Vec<&String> = cs.keys().filter(|k| !placed.contains(k.as_str()) && !k.contains("embed_tokens")).collect();
    // embedding source offsets
    let e = |s: &str| idx.get(&format!("language_model.model.embed_tokens.{}", s)).unwrap().off;
    let emb_ok = w.embed.w_off == e("weight") && w.embed.s_off == e("scales") && w.embed.b_off == e("biases") && w.embed.file.ends_with("model-00001-of-00001.safetensors");
    let off_rel = cs["language_model.model.embed_tokens.weight"]["offset"].as_u64().unwrap();
    println!("  tiny load: {} B read (== plan), {} tensors bit-exact, {} MTP norms == bf(raw + 1), bad {:?}, unplaced {:?}",
             loaded, n_ok, n_plus, bad, missing);
    println!("  embed src: offsets == index {} (checksums.json 'offset' {} vs absolute {})", emb_ok, off_rel, w.embed.w_off);
    assert!(bad.is_empty() && missing.is_empty() && emb_ok);
    assert_eq!(n_plus, 7, "+1 on exactly the 7 MTP norms");
    // main norms: means > 0.5 (stored +1), and the gated linear_attn.norm raw
    let m0 = arena.slice_mut::<u16>(w.layers[0].in_norm).iter().map(|&x| bf16_to_f32(x)).sum::<f32>() / cfg.hidden as f32;
    println!("  layer0 input_layernorm mean {:.3} (stored +1, untouched)", m0);
    // mtp=false: no MTP regions, fewer bytes
    let mut arena2 = Arena::new(contents(&buf), size);
    let (w2, l2) = load(&idx, &cfg, &mut arena2, false, 2).unwrap();
    assert!(w2.mtp.is_none() && l2 < loaded);
}
