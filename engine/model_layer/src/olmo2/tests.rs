//! Config fixtures and chat strings only: no model, no Metal, no weights.
//!
//! Fixtures (verbatim, fetched from huggingface.co 2026-11-15, metadata only): config.json of
//! mlx-community/OLMo-2-1124-7B-Instruct-4bit and allenai/OLMo-2-0425-1B-Instruct (bf16 original: no MLX 4-bit 1B
//! repo exists), and their chat template (identical text). Expected chat strings: transformers 5.21
//! `apply_chat_template(tokenize=False)` on those repos' tokenizer files.
use dense_engine::config::{Block, Family, LayerKind, QkNorm, RopeScaling};
use serde_json::{json, Value};

use super::chat::check_template;
use super::*;

const OLMO2_7B: &str = r#"{"architectures": ["Olmo2ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0, "eos_token_id": 100257, "hidden_act": "silu", "hidden_size": 4096, "initializer_range": 0.02, "intermediate_size": 11008, "max_position_embeddings": 4096, "model_type": "olmo2", "num_attention_heads": 32, "num_hidden_layers": 32, "num_key_value_heads": 32, "pad_token_id": 100277, "quantization": {"group_size": 64, "bits": 4}, "quantization_config": {"group_size": 64, "bits": 4}, "rms_norm_eps": 1e-06, "rope_scaling": null, "rope_theta": 500000, "tie_word_embeddings": false, "torch_dtype": "bfloat16", "transformers_version": "4.47.0.dev0", "use_cache": false, "vocab_size": 100352}"#;
const OLMO2_1B: &str = r#"{"architectures": ["Olmo2ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0, "eos_token_id": 100257, "hidden_act": "silu", "hidden_size": 2048, "initializer_range": 0.02, "intermediate_size": 8192, "max_position_embeddings": 4096, "model_type": "olmo2", "num_attention_heads": 16, "num_hidden_layers": 16, "num_key_value_heads": 16, "pad_token_id": 100277, "rms_norm_eps": 1e-06, "rope_scaling": null, "rope_theta": 500000, "tie_word_embeddings": false, "torch_dtype": "bfloat16", "transformers_version": "4.50.0", "use_cache": false, "vocab_size": 100352}"#;
const OLMO2_TPL: &str = "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}{% endfor %}";

fn j(s: &str) -> Value { serde_json::from_str(s).unwrap() }
fn edit(mut v: Value, f: impl FnOnce(&mut Value)) -> Value { f(&mut v); v }
fn m(r: &str, c: &str) -> (String, String) { (r.to_string(), c.to_string()) }

#[test]
fn olmo2_7b_mapping() {
    let d = config::parse(&j(OLMO2_7B), None, Some(100257)).unwrap();
    assert_eq!(d.family, Family::Olmo2);
    assert_eq!((d.hidden, d.inter, d.layers, d.heads, d.kv_heads, d.head_dim, d.vocab), (4096, 11008, 32, 32, 32, 128, 100352));
    assert_eq!((d.block, d.qk_norm, d.qk_norm_kind), (Block::PostNorm, true, QkNorm::FullRms));
    assert_eq!((d.rope_dims, d.rope_theta, d.rope_scale, d.rope_traditional), (128, 500000.0, RopeScaling::None, false));
    assert!(!d.tie_word_embeddings && !d.attention_bias && !d.o_bias && !d.mlp_bias);
    assert_eq!((d.eps, d.max_pos, d.bos, d.eos.clone(), d.quant), (1e-6, 4096, None, vec![100257], Some((64, 4))));
    assert!(d.kinds.iter().all(|k| *k == LayerKind::Full));
    assert_eq!((d.names.post_attn_norm.as_str(), d.names.post_ffn_norm.as_str()), ("post_attention_layernorm", "post_feedforward_layernorm"));
    assert_eq!((d.names.q_norm.as_str(), d.names.k_norm.as_str()), ("self_attn.q_norm", "self_attn.k_norm"));
    assert!(blockers(&j(OLMO2_7B)).is_empty(), "{:?}", blockers(&j(OLMO2_7B)));
    assert!(config::dense_config_with(&j(OLMO2_7B), None, Some(100257)).is_ok());
}

#[test]
fn olmo2_1b_mapping() {
    let d = config::parse(&j(OLMO2_1B), None, None).unwrap();
    assert_eq!((d.hidden, d.inter, d.layers, d.heads, d.kv_heads, d.head_dim, d.vocab), (2048, 8192, 16, 16, 16, 128, 100352));
    assert_eq!(d.quant, None);
    assert!(blockers(&j(OLMO2_1B)).iter().any(|b| b.contains("no MLX quantization")));
}

#[test]
fn olmo2_semantics_and_refusals() {
    let c = j(OLMO2_7B);
    // attention_bias -> q / k / v and o (mlx olmo2)
    let d = config::parse(&edit(c.clone(), |v| v["attention_bias"] = json!(true)), None, None).unwrap();
    assert!(d.attention_bias && d.o_bias);
    // tie default true (mlx olmo2 ModelArgs)
    assert!(config::parse(&edit(c.clone(), |v| { v.as_object_mut().unwrap().remove("tie_word_embeddings"); }), None, None).unwrap().tie_word_embeddings);
    // layer_types / sliding_window are not read by mlx olmo2
    let d = config::parse(&edit(c.clone(), |v| { v["layer_types"] = json!(["sliding_attention"]); v["sliding_window"] = json!(4096); }), None, None).unwrap();
    assert!(d.kinds.iter().all(|k| *k == LayerKind::Full) && d.sliding_window.is_none());
    let e = |v: Value| config::parse(&v, None, None).unwrap_err();
    assert!(e(edit(c.clone(), |v| v["model_type"] = json!("olmoe"))).contains("mixture-of-experts"));
    assert!(e(edit(c.clone(), |v| v["model_type"] = json!("olmo"))).contains("OLMo 1"));
    assert!(e(edit(c.clone(), |v| v["model_type"] = json!("olmo3"))).contains("OLMo 3"));
    assert!(e(edit(c.clone(), |v| v["num_experts"] = json!(64))).contains("mixture-of-experts"));
    assert!(e(edit(c.clone(), |v| { v.as_object_mut().unwrap().remove("rms_norm_eps"); })).contains("rms_norm_eps"));
    let e = config::dense_config_with(&edit(c, |v| v["rope_scaling"] = json!({"rope_type": "yarn", "factor": 8.0})), None, None).unwrap_err();
    assert!(e.contains("rope_scaling yarn"), "{}", e);
}

#[test]
fn olmo2_template() {
    assert!(check_template(Some(OLMO2_TPL), Some(&json!({"bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>"}))).is_ok());
    assert!(check_template(Some(OLMO2_TPL), Some(&json!({"bos_token": "<s>", "eos_token": "<|endoftext|>"}))).unwrap_err().contains("bos_token"));
    assert!(check_template(Some("<|user|>"), None).unwrap_err().contains("not the OLMo 2"));
    assert!(check_template(None, None).is_err());
}

#[test]
fn olmo2_strings() {
    assert_eq!(chat_prompt(&[m("user", "Hello")], true).unwrap(), "<|endoftext|><|user|>\nHello\n<|assistant|>\n");
    assert_eq!(chat_prompt(&[m("user", "Hello")], false).unwrap(), "<|endoftext|><|user|>\nHello\n");
    assert_eq!(chat_prompt(&[m("system", "You are terse."), m("user", "Hi there")], true).unwrap(),
               "<|endoftext|><|system|>\nYou are terse.\n<|user|>\nHi there\n<|assistant|>\n");
    assert_eq!(chat_prompt(&[m("user", "What is 2+2?"), m("assistant", "4"), m("user", "And 3+3?")], true).unwrap(),
               "<|endoftext|><|user|>\nWhat is 2+2?\n<|assistant|>\n4<|endoftext|>\n<|user|>\nAnd 3+3?\n<|assistant|>\n");
    assert_eq!(chat_prompt(&[m("system", "Be brief."), m("user", "A"), m("assistant", "B"), m("user", "C")], false).unwrap(),
               "<|endoftext|><|system|>\nBe brief.\n<|user|>\nA\n<|assistant|>\nB<|endoftext|>\n<|user|>\nC\n");
    assert_eq!(chat_prompt(&[m("user", "  padded \n")], true).unwrap(), "<|endoftext|><|user|>\n  padded \n\n<|assistant|>\n");
    // last message an assistant turn: no trailing newline after eos, generation prompt still added
    assert_eq!(chat_prompt(&[m("user", "a"), m("assistant", "b")], true).unwrap(),
               "<|endoftext|><|user|>\na\n<|assistant|>\nb<|endoftext|><|assistant|>\n");
    assert!(chat_prompt(&[m("tool", "x")], true).unwrap_err().contains("role"));
}

#[test]
fn model_type_list() {
    assert_eq!(MODEL_TYPES, &["olmo2"]);
}
