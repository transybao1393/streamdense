//! Config fixture only: no model, no Metal, no weights. The fixture is the real config.json of
//! mlx-community/starcoder2-3b-4bit (fetched 2026-11-15, metadata only; HF generation defaults such as top_k /
//! num_beams and the converter's local `_name_or_path` left out).
use std::path::{Path, PathBuf};

use dense_engine::config::{Activation, Block, Family, LayerKind, Mlp, NormKind, RopeScaling};
use serde_json::{json, Value};

use super::config::map_config;
use super::*;

/// mlx-community/starcoder2-3b-4bit config.json.
fn starcoder2_3b() -> Value {
    json!({
        "architectures": ["Starcoder2ForCausalLM"], "attention_dropout": 0.0, "bos_token_id": 0,
        "embedding_dropout": 0.0, "eos_token_id": 0, "hidden_act": "gelu_pytorch_tanh", "hidden_size": 3072,
        "initializer_range": 0.018042, "intermediate_size": 12288, "max_position_embeddings": 16384,
        "mlp_type": "default", "model_type": "starcoder2", "norm_epsilon": 1e-05, "norm_type": "layer_norm",
        "num_attention_heads": 24, "num_hidden_layers": 30, "num_key_value_heads": 2, "pad_token_id": null,
        "quantization": {"group_size": 64, "bits": 4}, "residual_dropout": 0.0, "rope_theta": 999999.4420358813,
        "sliding_window": 4096, "tie_word_embeddings": true, "torch_dtype": null, "transformers_version": "4.39.0.dev0",
        "use_bias": true, "use_cache": true, "vocab_size": 49152
    })
}

fn no_dir() -> &'static Path { Path::new("/nonexistent-starcoder2-test-dir") }
fn edit(mut v: Value, f: impl FnOnce(&mut Value)) -> Value { f(&mut v); v }
fn remove(v: Value, k: &str) -> Value { edit(v, |v| { v.as_object_mut().unwrap().remove(k); }) }
fn map(v: &Value) -> DenseConfig { map_config(v, None).unwrap().0 }
fn err(v: &Value) -> String { map_config(v, None).unwrap_err() }
fn msgs(m: &[(&str, &str)]) -> Vec<(String, String)> { m.iter().map(|(r, c)| (r.to_string(), c.to_string())).collect() }

/// Blockers the engine named for StarCoder2 before its features landed (SPEC.md): every blocker of the real config
/// must be one of these (today: none).
const PENDING: [&str; 3] = ["norm LayerNorm", "mlp Plain", "gelu (erf)"];
fn only_pending(b: &[String]) -> bool { b.iter().all(|m| PENDING.iter().any(|p| m.contains(p))) }

#[test]
fn model_types() {
    assert_eq!(MODEL_TYPES, &["starcoder2"]);
}

#[test]
fn starcoder2_3b_mapping() {
    let d = map(&starcoder2_3b());
    assert_eq!((d.family, d.model_type.as_str()), (Family::Starcoder2, "starcoder2"));
    assert_eq!((d.hidden, d.inter, d.layers, d.heads, d.kv_heads, d.head_dim, d.vocab), (3072, 12288, 30, 24, 2, 128, 49152));
    assert_eq!((d.grp(), d.kv_dim(), d.head_dim_explicit), (12, 256, false));
    assert_eq!((d.rope_dims, d.rope_theta, d.rope_traditional, d.rope_scale, d.rope_scaling.clone()),
               (128, 999999.4420358813, false, RopeScaling::None, None));
    assert_eq!((d.norm, d.block, d.mlp, d.activation.clone()),
               (NormKind::LayerNorm { bias: true }, Block::Sequential, Mlp::Plain, Activation::GeluErf));
    assert!(d.hidden_act.contains("gelu_pytorch_tanh"));
    assert!(d.attention_bias && d.o_bias && d.mlp_bias && d.tie_word_embeddings && !d.qk_norm && !d.four_norms);
    assert_eq!((d.names.up.as_str(), d.names.down.as_str()), ("mlp.c_fc", "mlp.c_proj"));
    assert!(d.names.gate_up.is_none());
    // sliding_window 4096 in the file, ignored as mlx_lm does
    assert!(d.kinds.len() == 30 && d.kinds.iter().all(|k| *k == LayerKind::Full) && d.sliding_window.is_none());
    assert_eq!((d.eps, d.max_pos, d.attn_scale, d.logits_scaling), (1e-5, 16384, None, 1.0));
    assert_eq!((d.bos, d.eos.clone(), d.quant), (Some(0), vec![0], Some((64, 4))));
}

#[test]
fn starcoder2_3b_blockers_are_engine_features_only() {
    let b = blockers(&starcoder2_3b());
    assert!(only_pending(&b), "{:?}", b);
    match dense_config(&starcoder2_3b(), no_dir()) {
        Ok(d) => assert!(b.is_empty() && d.family == Family::Starcoder2),
        Err(e) => assert!(PENDING.iter().any(|p| e.contains(p)) && !b.is_empty(), "{}", e),
    }
}

#[test]
fn defaults_and_refusals() {
    // defaults: vocab 49152, norm_epsilon 1e-5, tied
    let d = map(&remove(remove(remove(starcoder2_3b(), "vocab_size"), "norm_epsilon"), "tie_word_embeddings"));
    assert_eq!((d.vocab, d.eps, d.tie_word_embeddings), (49152, 1e-5, true));
    assert!(!map(&edit(starcoder2_3b(), |v| v["tie_word_embeddings"] = json!(false))).tie_word_embeddings);
    for k in ["hidden_size", "intermediate_size", "num_key_value_heads", "rope_theta"] {
        assert!(err(&remove(starcoder2_3b(), k)).contains(k), "{}", k);
    }
    assert!(err(&edit(starcoder2_3b(), |v| v["num_attention_heads"] = json!(25))).contains("multiple"));
    assert!(err(&edit(starcoder2_3b(), |v| v["head_dim"] = json!(64))).contains("head_dim 64"));
    assert!(err(&edit(starcoder2_3b(), |v| v["model_type"] = json!("gpt_bigcode"))).contains("StarCoder 1"));
    assert!(err(&edit(starcoder2_3b(), |v| v["architectures"] = json!(["GPTBigCodeForCausalLM"]))).contains("architecture"));
    let has = |v: &Value, s: &str| blockers(v).iter().any(|m| m.contains(s));
    assert!(has(&edit(starcoder2_3b(), |v| v["use_bias"] = json!(false)), "use_bias"));
    assert!(has(&edit(starcoder2_3b(), |v| v["mlp_type"] = json!("gated")), "mlp_type"));
    assert!(has(&edit(starcoder2_3b(), |v| v["norm_type"] = json!("rms_norm")), "norm_type"));
    assert!(has(&edit(starcoder2_3b(), |v| v["rope_scaling"] = json!({"type": "dynamic", "factor": 2.0})), "dynamic"));
    assert!(has(&remove(starcoder2_3b(), "quantization"), "no MLX quantization"));
    let (d, _) = map_config(&starcoder2_3b(), Some(&json!({"eos_token_id": 0}))).unwrap();
    assert_eq!(d.eos, vec![0]);
    assert!(dense_config(&remove(starcoder2_3b(), "eos_token_id"), no_dir()).unwrap_err().contains("eos"));
}

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sd-starcoder2-test-{}-{}", tag, std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

#[test]
fn chat_refused_by_name() {
    let m = msgs(&[("user", "write fizzbuzz")]);
    assert!(chat_prompt(&m, true).unwrap_err().contains("base code model"));
    // the real tokenizer_config.json has no chat_template: base-model error
    let d = tmpdir("base");
    std::fs::write(d.join("tokenizer_config.json"), json!({"bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>"}).to_string()).unwrap();
    assert!(chat_prompt_dir(&d, &m, true).unwrap_err().contains("base code model"));
    // an instruction tune's template: not ported
    std::fs::write(d.join("tokenizer_config.json"), json!({"chat_template": "{{ x }}"}).to_string()).unwrap();
    assert!(chat_prompt_dir(&d, &m, true).unwrap_err().contains("not ported"));
    let _ = std::fs::remove_dir_all(&d);
}
