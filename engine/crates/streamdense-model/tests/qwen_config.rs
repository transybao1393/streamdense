//! U3 self-test: QwenConfig on the real Qwen3.8-27B config, the tiny config, and rejected configs.
//! Run: cargo test --release --offline -p streamdense-model --test qwen_config -- --nocapture
use serde_json::Value;
use streamdense_model::{LayerKind, QwenConfig};

const MODELS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models");

#[test]
fn real_config() {
    let dir = format!("{}/qwen38_27b_mlx4", MODELS);
    if !std::path::Path::new(&format!("{}/config.json", dir)).exists() { eprintln!("SKIP: {} missing", dir); return; }
    let c = QwenConfig::from_model_dir(&dir).unwrap();
    c.validate().unwrap();
    println!("  real: {:?}", c);
    assert_eq!((c.hidden, c.inter, c.layers, c.vocab), (5120, 17408, 64, 248320));
    assert_eq!((c.heads, c.kv_heads, c.head_dim, c.rope_dims), (24, 4, 256, 64));
    assert_eq!(c.rope_theta, 1e7);
    assert_eq!((c.dn_kh, c.dn_vh, c.dn_dk, c.dn_dv, c.conv_k), (16, 48, 128, 128, 4));
    assert_eq!(c.eps, 1e-6);
    assert!(c.eos.contains(&248046) && c.eos.contains(&248044), "eos {:?}", c.eos);
    assert_eq!(c.mtp_layers, 1);
    for l in 0..64 {
        let want = if (l + 1) % 4 == 0 { LayerKind::Attention } else { LayerKind::DeltaNet };
        assert_eq!(c.kinds[l], want, "layer {}", l);
    }
    assert_eq!((c.n_attention(), c.n_deltanet()), (16, 48));
    assert_eq!((c.conv_dim(), c.dn_in_rows(), c.attn_in_rows()), (10240, 16480, 14336));
    assert_eq!((c.q_dim(), c.kv_dim(), c.dn_k_dim(), c.dn_v_dim()), (6144, 1024, 2048, 6144));
}

#[test]
fn tiny_config() {
    let dir = format!("{}/tiny_qwen35", MODELS);
    if !std::path::Path::new(&format!("{}/config.json", dir)).exists() { eprintln!("SKIP: {} missing", dir); return; }
    let c = QwenConfig::from_model_dir(&dir).unwrap();
    c.validate().unwrap();
    println!("  tiny: {:?}", c);
    assert_eq!((c.hidden, c.inter, c.layers, c.vocab), (512, 1024, 8, 2048));
    assert_eq!((c.heads, c.kv_heads, c.head_dim, c.rope_dims), (4, 2, 256, 64));
    assert_eq!((c.dn_kh, c.dn_vh, c.dn_dk, c.dn_dv, c.conv_k), (2, 6, 128, 128, 4));
    assert_eq!((c.n_attention(), c.n_deltanet()), (2, 6));
    assert_eq!(c.kinds[3], LayerKind::Attention);
    assert_eq!(c.kinds[7], LayerKind::Attention);
    assert_eq!((c.conv_dim(), c.dn_in_rows(), c.attn_in_rows()), (1280, 1280 + 768 + 12, 2 * 1024 + 2 * 512));
    assert_eq!(c.mtp_layers, 1);
    assert!(c.eos.contains(&2046) && c.eos.contains(&2044), "eos {:?}", c.eos);
}

fn base() -> Value {
    let dir = format!("{}/tiny_qwen35", MODELS);
    serde_json::from_str(&std::fs::read_to_string(format!("{}/config.json", dir)).unwrap()).unwrap()
}

#[test]
fn rejects_bad_configs() {
    if !std::path::Path::new(&format!("{}/tiny_qwen35/config.json", MODELS)).exists() { eprintln!("SKIP: needs models/tiny_qwen35/config.json (missing)"); return; }
    // (label, edit of text_config, expect parse error (true) or validate error (false))
    let cases: Vec<(&str, Box<dyn Fn(&mut Value)>, bool)> = vec![
        ("head_dim 128", Box::new(|t| t["head_dim"] = 128.into()), false),
        ("rope factor 0.5", Box::new(|t| { t["partial_rotary_factor"] = 0.5.into(); t["rope_parameters"]["partial_rotary_factor"] = 0.5.into(); }), false),
        ("dk 64", Box::new(|t| t["linear_key_head_dim"] = 64.into()), false),
        ("vh % kh", Box::new(|t| t["linear_num_value_heads"] = 5.into()), false),
        ("heads % kv", Box::new(|t| t["num_key_value_heads"] = 3.into()), false),
        ("conv 3", Box::new(|t| t["linear_conv_kernel_dim"] = 3.into()), false),
        ("hidden % 64", Box::new(|t| t["hidden_size"] = 520.into()), false),
        ("inter % 64", Box::new(|t| t["intermediate_size"] = 1000.into()), false),
        ("missing vocab", Box::new(|t| { t.as_object_mut().unwrap().remove("vocab_size"); }), true),
        ("layer_types length", Box::new(|t| t["num_hidden_layers"] = 9.into()), true),
        ("unknown layer type", Box::new(|t| t["layer_types"][2] = "sliding_attention".into()), true),
    ];
    for (label, edit, parse_err) in cases {
        let mut v = base();
        edit(&mut v["text_config"]);
        let r = QwenConfig::from_json(&v, None);
        let got = match &r { Err(e) => format!("parse: {}", e), Ok(c) => match c.validate() { Err(e) => format!("validate: {}", e), Ok(()) => "accepted".into() } };
        println!("  {:<20} -> {}", label, got);
        if parse_err { assert!(r.is_err(), "{} should fail to parse", label); } else { assert!(r.unwrap().validate().is_err(), "{} should fail validate", label); }
    }
    // no eos anywhere -> error
    let mut v = base();
    v.as_object_mut().unwrap().remove("eos_token_id");
    v["text_config"].as_object_mut().unwrap().remove("eos_token_id");
    assert!(QwenConfig::from_json(&v, None).is_err());
}
