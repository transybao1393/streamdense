//! Config fixtures and chat strings; the local-model tests read config.json, tokenizer.json and safetensors HEADERS only
//! (no weight data, no Metal) and skip when the folder is missing. Folder: `$SD_GRANITE_DIR`, else
//! `<repo>/models/granite-4.2-3b-MLX-4bit`.
use super::*;
use dense_engine::weights::{layout, SafeSource};
use serde_json::json;
use std::path::{Path, PathBuf};
use streamdense_tokenizer::Tokenizer;

/// granite-4.2-3b-MLX-4bit/config.json, values as on disk.
fn granite_4_2_3b() -> Value {
    json!({
        "architectures": ["GraniteForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
        "attention_multiplier": 0.015625, "bos_token_id": 100283, "dtype": "bfloat16", "embedding_multiplier": 1.0,
        "eos_token_id": 100257, "hidden_act": "silu", "hidden_size": 2560, "initializer_range": 0.1,
        "intermediate_size": 8192, "logits_scaling": 1.0, "max_position_embeddings": 131072, "mlp_bias": false,
        "model_type": "granite", "num_attention_heads": 40, "num_hidden_layers": 40, "num_key_value_heads": 8,
        "pad_token_id": 100257, "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
        "quantization_config": {"group_size": 64, "bits": 4, "mode": "affine"}, "residual_multiplier": 1.0,
        "rms_norm_eps": 1e-05, "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"}, "rope_scaling": null,
        "rope_theta": 10000000, "tie_word_embeddings": false, "torch_dtype": "bfloat16",
        "transformers_version": "4.57.1", "use_cache": true, "vocab_size": 100352
    })
}

/// A folder with no generation_config.json.
fn no_dir() -> &'static Path { Path::new("/nonexistent-granite-test-dir") }
fn with(mut v: Value, k: &str, x: Value) -> Value { v[k] = x; v }
fn without(mut v: Value, k: &str) -> Value { v.as_object_mut().unwrap().remove(k); v }
fn has(b: &[String], s: &str) -> bool { b.iter().any(|m| m.contains(s)) }

fn local_dir() -> Option<PathBuf> {
    let d = match std::env::var("SD_GRANITE_DIR") {
        Ok(p) => PathBuf::from(p),
        Err(_) => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/granite-4.2-3b-MLX-4bit"),
    };
    if d.join("config.json").exists() { Some(d) } else { println!("  SKIP: {} missing", d.display()); None }
}

#[test]
fn model_types() {
    assert_eq!(MODEL_TYPES, &["granite"]);
}

#[test]
fn fixture_maps_to_dense_config() {
    let d = dense_config(&granite_4_2_3b(), no_dir()).unwrap();
    assert_eq!(d.family, Family::Granite);
    assert_eq!(d.model_type, "granite");
    assert_eq!((d.hidden, d.inter, d.layers, d.heads, d.kv_heads, d.head_dim, d.vocab), (2560, 8192, 40, 40, 8, 64, 100352));
    assert!(!d.head_dim_explicit);
    assert_eq!((d.rope_dims, d.rope_theta, d.rope_scale, d.rope_scaling.clone()), (64, 1e7, RopeScaling::None, None));
    assert_eq!(d.attn_scale, Some(0.015625));
    assert_eq!(d.attn_scale_f32(), 1.0 / 64.0);
    assert_eq!((d.embed_scale, d.residual_multiplier, d.logits_scaling), (1.0, 1.0, 1.0));
    assert!(!d.attention_bias && !d.o_bias && !d.mlp_bias && !d.tie_word_embeddings && !d.qk_norm);
    assert!(!d.norm_plus_one && !d.four_norms && d.attn_softcap.is_none() && d.final_softcap.is_none());
    assert_eq!((d.activation.clone(), d.hidden_act.as_str()), (Activation::Silu, "silu"));
    assert_eq!((d.eps, d.max_pos), (1e-5, 131072));
    assert!(d.kinds.len() == 40 && d.kinds.iter().all(|k| *k == LayerKind::Full) && d.sliding_window.is_none());
    assert_eq!((d.bos, d.eos.clone(), d.quant), (Some(100283), vec![100257], Some((64, 4))));
    assert_eq!((d.q_dim(), d.kv_dim(), d.grp()), (2560, 512, 5));
    d.validate().unwrap();
    assert_eq!(blockers(&granite_4_2_3b()), Vec::<String>::new());
}

#[test]
fn multipliers_map_one_to_one() {
    let mut v = granite_4_2_3b();
    for (k, x) in [("attention_multiplier", 0.0078125), ("embedding_multiplier", 12.0), ("residual_multiplier", 0.22),
                   ("logits_scaling", 8.0)] {
        v[k] = json!(x);
    }
    let d = dense_config(&v, no_dir()).unwrap();
    assert_eq!(d.attn_scale, Some(0.0078125));
    assert_eq!(d.embed_scale, 12.0);
    assert_eq!(d.residual_multiplier, 0.22);
    // the engine divides by logits_scaling, as HF / mlx_lm Granite do: passed through, not inverted
    assert_eq!(d.logits_scaling, 8.0);
}

#[test]
fn biases_and_tie() {
    let v = with(with(with(granite_4_2_3b(), "attention_bias", json!(true)), "mlp_bias", json!(true)), "tie_word_embeddings", json!(true));
    let d = dense_config(&v, no_dir()).unwrap();
    assert!(d.attention_bias && d.o_bias && d.mlp_bias && d.tie_word_embeddings);
}

#[test]
fn rope_forms() {
    // rope_parameters only
    let v = without(granite_4_2_3b(), "rope_theta");
    assert_eq!(dense_config(&v, no_dir()).unwrap().rope_theta, 1e7);
    // rope_theta only
    let v = without(granite_4_2_3b(), "rope_parameters");
    assert_eq!(dense_config(&v, no_dir()).unwrap().rope_theta, 1e7);
    // disagreement and absence refused
    assert!(dense_config(&with(granite_4_2_3b(), "rope_theta", json!(10000)), no_dir()).unwrap_err().contains("differ"));
    let e = dense_config(&without(without(granite_4_2_3b(), "rope_theta"), "rope_parameters"), no_dir()).unwrap_err();
    assert!(e.contains("rope_theta") && e.contains("not guessed"), "{}", e);
    // implemented scalings parse into rope_scale; others are named by validate()
    let d = dense_config(&with(granite_4_2_3b(), "rope_scaling", json!({"type": "linear", "factor": 4.0})), no_dir()).unwrap();
    assert_eq!(d.rope_scale, RopeScaling::Linear { factor: 4.0 });
    let e = dense_config(&with(granite_4_2_3b(), "rope_scaling", json!({"rope_type": "yarn", "factor": 2.0})), no_dir()).unwrap_err();
    assert!(e.contains("rope_scaling yarn factor 2"), "{}", e);
    assert!(has(&blockers(&with(granite_4_2_3b(), "rope_scaling", json!({"rope_type": "yarn", "factor": 2.0}))), "yarn"));
}

#[test]
fn head_dim_key() {
    let d = dense_config(&with(granite_4_2_3b(), "head_dim", json!(64)), no_dir()).unwrap();
    assert!(d.head_dim_explicit && d.head_dim == 64);
    assert!(dense_config(&with(granite_4_2_3b(), "head_dim", json!(128)), no_dir()).unwrap_err().contains("head_dim 128"));
    let e = dense_config(&with(granite_4_2_3b(), "num_attention_heads", json!(30)), no_dir()).unwrap_err();
    assert!(e.contains("not a multiple"), "{}", e);
    // head dim 80 parses, validate() names it
    let v = with(with(granite_4_2_3b(), "num_attention_heads", json!(32)), "num_key_value_heads", json!(8));
    let e = dense_config(&v, no_dir()).unwrap_err();
    assert!(e.contains("head_dim must be 64, 96, 128, 160 or 256 (got 80)"), "{}", e);
    assert!(has(&blockers(&v), "got 80"));
}

#[test]
fn math_fields_not_guessed() {
    for k in ["attention_multiplier", "embedding_multiplier", "residual_multiplier", "logits_scaling", "tie_word_embeddings",
              "attention_bias", "mlp_bias", "rms_norm_eps", "hidden_act"] {
        let e = dense_config(&without(granite_4_2_3b(), k), no_dir()).unwrap_err();
        assert!(e.contains(k) && e.contains("not guessed"), "{}: {}", k, e);
        assert!(has(&blockers(&without(granite_4_2_3b(), k)), k));
    }
    for k in ["hidden_size", "num_hidden_layers", "intermediate_size", "vocab_size"] {
        assert!(dense_config(&without(granite_4_2_3b(), k), no_dir()).unwrap_err().contains(k), "{}", k);
    }
    // absent num_key_value_heads = plain multi-head (transformers default)
    assert_eq!(dense_config(&without(granite_4_2_3b(), "num_key_value_heads"), no_dir()).unwrap().kv_heads, 40);
}

#[test]
fn activation_named() {
    let e = dense_config(&with(granite_4_2_3b(), "hidden_act", json!("gelu")), no_dir()).unwrap_err();
    assert!(e.contains("hidden_act must be silu or gelu_pytorch_tanh (got gelu)"), "{}", e);
}

#[test]
fn quantization_blockers() {
    let q = |x: Value| { let mut v = granite_4_2_3b(); v["quantization"] = x.clone(); v["quantization_config"] = x; v };
    for (x, want) in [(json!({"group_size": 64, "bits": 8, "mode": "affine"}), "group_size 64 bits 8"),
                      (json!({"group_size": 32, "bits": 4}), "group_size 32 bits 4"),
                      (json!({"group_size": 64, "bits": 4, "mode": "mxfp4"}), "quantization mode \"mxfp4\""),
                      (json!({"group_size": 64, "bits": 4, "model.layers.0.self_attn.o_proj": {"group_size": 64, "bits": 8}}), "per-layer")] {
        let v = q(x);
        assert!(dense_config(&v, no_dir()).unwrap_err().contains(want), "{}", want);
        assert!(has(&blockers(&v), want), "{}", want);
    }
    let v = without(without(granite_4_2_3b(), "quantization"), "quantization_config");
    assert!(dense_config(&v, no_dir()).unwrap_err().contains("no quantization block"));
    // quantization_config alone is read
    let d = dense_config(&without(granite_4_2_3b(), "quantization"), no_dir()).unwrap();
    assert_eq!(d.quant, Some((64, 4)));
}

#[test]
fn eos_from_generation_config() {
    let dir = std::env::temp_dir().join(format!("sd_granite_gen_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id": [100257, 100284]}"#).unwrap();
    let d = dense_config(&granite_4_2_3b(), &dir).unwrap();
    assert_eq!(d.eos, vec![100257, 100284]);
    // eos only in generation_config.json
    let d = dense_config(&without(granite_4_2_3b(), "eos_token_id"), &dir).unwrap();
    assert_eq!(d.eos, vec![100257, 100284]);
    std::fs::write(dir.join("generation_config.json"), "{not json").unwrap();
    assert!(dense_config(&granite_4_2_3b(), &dir).unwrap_err().contains("generation_config.json"));
    std::fs::remove_dir_all(&dir).unwrap();
    // nowhere: refused by dense_config, not a blocker of config.json alone
    assert!(dense_config(&without(granite_4_2_3b(), "eos_token_id"), no_dir()).unwrap_err().contains("eos_token_id"));
    assert!(blockers(&without(granite_4_2_3b(), "eos_token_id")).is_empty());
}

#[test]
fn moe_rejected_dense_only() {
    for v in [json!({"model_type": "granitemoe", "num_local_experts": 40}),
              json!({"model_type": "granitemoe"}),
              json!({"model_type": "granitemoeshared", "num_local_experts": 32}),
              json!({"model_type": "granitemoehybrid", "num_local_experts": 64, "layer_types": ["mamba", "attention"]}),
              with(granite_4_2_3b(), "num_local_experts", json!(8))] {
        let e = dense_config(&v, no_dir()).unwrap_err();
        assert!(e.contains("dense models only"), "{}", e);
        assert!(has(&blockers(&v), "dense models only"));
    }
}

#[test]
fn hybrid_layers_not_supported_yet() {
    let v = json!({"model_type": "granitemoehybrid", "num_local_experts": 0, "layer_types": ["mamba", "mamba", "attention"]});
    let e = dense_config(&v, no_dir()).unwrap_err();
    assert!(e.contains("layer type mamba not supported yet"), "{}", e);
    let e = dense_config(&json!({"model_type": "granitemoehybrid", "num_local_experts": 0, "layer_types": ["attention"]}), no_dir()).unwrap_err();
    assert!(e.contains("shared_mlp") && e.contains("not supported yet"), "{}", e);
}

#[test]
fn other_configs_refused_by_name() {
    let e = dense_config(&json!({"model_type": "granite_speech", "text_config": {"model_type": "granite"}}), no_dir()).unwrap_err();
    assert!(e.contains("text_config") && e.contains("not supported yet"), "{}", e);
    let e = dense_config(&with(granite_4_2_3b(), "model_type", json!("llama")), no_dir()).unwrap_err();
    assert!(e.contains("\"llama\"") && e.contains("not a Granite model"), "{}", e);
    assert!(dense_config(&without(granite_4_2_3b(), "model_type"), no_dir()).unwrap_err().contains("no model_type"));
}

// ---- chat prompt: expected strings rendered by jinja2 from the local chat_template.jinja (transformers settings) ----

fn m(v: &[(&str, &str)]) -> Vec<(String, String)> { v.iter().map(|(r, c)| (r.to_string(), c.to_string())).collect() }

#[test]
fn chat_default_empty_system() {
    assert_eq!(chat_prompt(&m(&[("user", "Hi")]), true).unwrap(),
               "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n");
    assert_eq!(chat_prompt(&m(&[("user", "Hi")]), false).unwrap(),
               "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n");
}

#[test]
fn chat_with_system() {
    assert_eq!(chat_prompt(&m(&[("system", "You are X."), ("user", "Hi")]), true).unwrap(),
               "<|im_start|>system\nYou are X.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n");
}

#[test]
fn chat_multi_turn() {
    assert_eq!(chat_prompt(&m(&[("system", "S"), ("user", "a"), ("assistant", "b"), ("user", "c")]), true).unwrap(),
               "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n<think></think>b<|im_end|>\n\
                <|im_start|>user\nc<|im_end|>\n<|im_start|>assistant\n<think>\n");
    // history thinking dropped, trailing one kept; trim
    assert_eq!(chat_prompt(&m(&[("user", "a"), ("assistant", "<think>r</think>  b  "), ("user", "c"), ("assistant", "<think>r2</think>d")]), false).unwrap(),
               "<|im_start|>system\n<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n<think></think>  b<|im_end|>\n\
                <|im_start|>user\nc<|im_end|>\n<|im_start|>assistant\n<think>r2</think>d<|im_end|>\n");
    assert_eq!(chat_prompt(&m(&[("user", "a"), ("assistant", ""), ("user", "c")]), true).unwrap(),
               "<|im_start|>system\n<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n<think></think><|im_end|>\n\
                <|im_start|>user\nc<|im_end|>\n<|im_start|>assistant\n<think>\n");
    assert_eq!(chat_prompt(&m(&[("user", "a"), ("assistant", "\n x \n")]), false).unwrap(),
               "<|im_start|>system\n<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n<think></think>\n x<|im_end|>\n");
    assert_eq!(chat_prompt(&m(&[("user", "a"), ("assistant", "<think>half"), ("user", "b")]), false).unwrap(),
               "<|im_start|>system\n<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n<think>half<|im_end|>\n\
                <|im_start|>user\nb<|im_end|>\n");
}

#[test]
fn chat_tool_and_other_roles() {
    assert_eq!(chat_prompt(&m(&[("user", "a"), ("tool", "r1"), ("tool", "r2"), ("user", "c")]), true).unwrap(),
               "<|im_start|>system\n<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>user\n<tool_response>\nr1\n</tool_response>\n\
                <tool_response>\nr2\n</tool_response>\n<|im_end|>\n<|im_start|>user\nc<|im_end|>\n<|im_start|>assistant\n<think>\n");
    assert_eq!(chat_prompt(&m(&[("user", "a"), ("system", "s2")]), true).unwrap(),
               "<|im_start|>system\n<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>system\ns2<|im_end|>\n<|im_start|>assistant\n<think>\n");
    assert_eq!(chat_prompt(&m(&[("user", "a"), ("developer", "d")]), false).unwrap(),
               "<|im_start|>system\n<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>developer\nd<|im_end|>\n");
    assert!(chat_prompt(&[], true).unwrap_err().contains("no messages"));
}

// ---- local model files (headers / json only) ----

#[test]
fn local_config_matches_fixture() {
    let Some(dir) = local_dir() else { return };
    let root: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    assert_eq!(root, granite_4_2_3b(), "local config.json differs from the fixture");
    let d = dense_config(&root, &dir).unwrap();
    assert_eq!((d.head_dim, d.attn_scale, d.eos.clone()), (64, Some(0.015625), vec![100257]));
    assert!(blockers(&root).is_empty());
}

#[test]
fn local_weight_headers_fit_the_layout() {
    let Some(dir) = local_dir() else { return };
    let root: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json")).unwrap()).unwrap();
    let d = dense_config(&root, &dir).unwrap();
    let src = SafeSource::open(dir.to_str().unwrap()).unwrap();
    let (w, jobs) = layout(&d, &src).unwrap();
    // every tensor of the files is placed exactly once: no Granite tensor the loader does not know
    assert_eq!(jobs.len(), src.idx.len());
    let mut names: Vec<&str> = jobs.iter().map(|j| j.name.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), src.idx.len());
    assert!(!w.tied && w.layers.len() == 40);
    println!("  granite-4.2-3b: {} tensors, weight region {:.2} GB", jobs.len(), w.bytes as f64 / 1e9);
}

#[test]
fn local_tokenizer_reads_the_chat_prompt() {
    let Some(dir) = local_dir() else { return };
    let tk = Tokenizer::from_path(dir.to_str().unwrap()).unwrap();
    assert_eq!((tk.token_id("<|im_start|>"), tk.token_id("<|im_end|>")), (Some(100256), Some(100257)));
    // no post-processor tokens: the template adds no BOS either
    assert!(tk.special_prefix().is_empty() && tk.special_suffix().is_empty());
    let text = chat_prompt(&m(&[("user", "Hello, world!")]), true).unwrap();
    let ids = tk.encode_with_special(&text);
    assert_eq!(ids[0], 100256);
    assert_eq!(ids.iter().filter(|&&i| i == 100257).count(), 2);
    assert_eq!(tk.decode(&ids), text);
    let plain = tk.decode_skip_special(&ids);
    assert!(!plain.contains("<|im_start|>") && plain.contains("Hello, world!"), "{:?}", plain);
}
