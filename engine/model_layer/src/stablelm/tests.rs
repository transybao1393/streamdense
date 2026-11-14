//! Config fixtures and chat strings: no model, no Metal, no weights. Fixtures are the real config.json files of
//! mlx-community/stablelm-2-1_6b-chat-4bit and stablelm-2-12b-chat-4bit (fetched 2026-11-15, metadata only; HF
//! generation defaults such as top_k / num_beams left out); the expected chat strings are HF `apply_chat_template`
//! renders of the real templates (generated below the hand-written tests).
use std::path::{Path, PathBuf};

/// Which message lists the template renders lossily (skipped).
type Lossy = dyn Fn(&[(&str, &str)]) -> bool;
/// One chat-render case: messages, add_generation_prompt, expected render or error.
type Render = (&'static [(&'static str, &'static str)], bool, Result<&'static str, &'static str>);

use dense_engine::config::{Activation, Block, Family, LayerKind, Mlp, NormKind, QkNorm, RopeScaling};
use serde_json::{json, Value};

use super::config::map_config;
use super::template::{kind_from_text, TemplateKind};
use super::*;

/// mlx-community/stablelm-2-1_6b-chat-4bit config.json (zephyr-1_6b: same values).
fn stablelm2_1_6b() -> Value {
    json!({
        "architectures": ["StableLmForCausalLM"], "attention_dropout": 0.0, "bos_token_id": 100257,
        "eos_token_id": 100257, "hidden_act": "silu", "hidden_dropout": 0.0, "hidden_size": 2048,
        "initializer_range": 0.02, "intermediate_size": 5632, "layer_norm_eps": 1e-05, "max_position_embeddings": 4096,
        "model_type": "stablelm", "num_attention_heads": 32, "num_hidden_layers": 24, "num_key_value_heads": 32,
        "pad_token_id": null, "partial_rotary_factor": 0.25, "quantization": {"group_size": 64, "bits": 4},
        "rope_scaling": null, "rope_theta": 10000, "tie_word_embeddings": false, "torch_dtype": "float32",
        "transformers_version": "4.40.0.dev0", "use_cache": true, "use_qkv_bias": true, "vocab_size": 100352
    })
}

/// mlx-community/stablelm-2-12b-chat-4bit config.json (the keys mlx_lm reads, + identification).
fn stablelm2_12b() -> Value {
    json!({
        "architectures": ["StableLmForCausalLM"], "bos_token_id": 100257, "eos_token_id": 100257, "hidden_act": "silu",
        "hidden_size": 5120, "intermediate_size": 13824, "layer_norm_eps": 1e-05, "max_position_embeddings": 4096,
        "model_type": "stablelm", "num_attention_heads": 32, "num_hidden_layers": 40, "num_key_value_heads": 8,
        "partial_rotary_factor": 0.25, "qk_layernorm": true, "quantization": {"group_size": 64, "bits": 4},
        "rope_scaling": null, "rope_theta": 10000, "tie_word_embeddings": false, "use_parallel_residual": true,
        "use_qkv_bias": false, "vocab_size": 100352, "torch_dtype": "bfloat16", "transformers_version": "4.40.0.dev0"
    })
}

fn no_dir() -> &'static Path { Path::new("/nonexistent-stablelm-test-dir") }
fn edit(mut v: Value, f: impl FnOnce(&mut Value)) -> Value { f(&mut v); v }
fn remove(v: Value, k: &str) -> Value { edit(v, |v| { v.as_object_mut().unwrap().remove(k); }) }
fn map(v: &Value) -> DenseConfig { map_config(v, None).unwrap().0 }
fn err(v: &Value) -> String { map_config(v, None).unwrap_err() }
fn msgs(m: &[(&str, &str)]) -> Vec<(String, String)> { m.iter().map(|(r, c)| (r.to_string(), c.to_string())).collect() }

/// Blockers the engine names today for StableLM-2 1.6B and that go away as the engine features land (SPEC.md).
const PENDING: [&str; 3] = ["partial rotary", "norm LayerNorm", "block Parallel"];
fn only_pending(b: &[String]) -> bool { b.iter().all(|m| PENDING.iter().any(|p| m.contains(p))) }

#[test]
fn model_types() {
    assert_eq!(MODEL_TYPES, &["stablelm"]);
}

#[test]
fn stablelm2_1_6b_mapping() {
    let d = map(&stablelm2_1_6b());
    assert_eq!((d.family, d.model_type.as_str()), (Family::StableLm, "stablelm"));
    assert_eq!((d.hidden, d.inter, d.layers, d.heads, d.kv_heads, d.head_dim, d.vocab), (2048, 5632, 24, 32, 32, 64, 100352));
    assert!(!d.head_dim_explicit);
    assert_eq!((d.rope_dims, d.rope_theta, d.rope_traditional, d.rope_scale, d.rope_scaling.clone()),
               (16, 10000.0, false, RopeScaling::None, None));
    assert!(d.attention_bias && !d.o_bias && !d.mlp_bias && !d.tie_word_embeddings && !d.qk_norm && !d.four_norms);
    assert_eq!((d.norm, d.block, d.mlp, d.activation.clone()),
               (NormKind::LayerNorm { bias: true }, Block::Sequential, Mlp::Gated, Activation::Silu));
    assert_eq!((d.names.input_norm.as_str(), d.names.post_attn_norm.as_str()), ("input_layernorm", "post_attention_layernorm"));
    assert!(d.names.gate_up.is_none() && d.names.qkv.is_none() && d.no_rope_layers.is_empty());
    assert!(d.kinds.len() == 24 && d.kinds.iter().all(|k| *k == LayerKind::Full) && d.sliding_window.is_none());
    assert_eq!((d.eps, d.max_pos, d.attn_scale), (1e-5, 4096, None));
    assert_eq!((d.bos, d.eos.clone(), d.quant), (Some(100257), vec![100257], Some((64, 4))));
}

#[test]
fn stablelm2_1_6b_blockers_are_engine_features_only() {
    let b = blockers(&stablelm2_1_6b());
    assert!(only_pending(&b), "{:?}", b);
    match dense_config(&stablelm2_1_6b(), no_dir()) {
        Ok(d) => assert!(b.is_empty() && d.family == Family::StableLm),
        Err(e) => assert!(PENDING.iter().any(|p| e.contains(p)) && !b.is_empty(), "{}", e),
    }
}

#[test]
fn stablelm2_12b_mapping() {
    let d = map(&stablelm2_12b());
    assert_eq!((d.hidden, d.heads, d.kv_heads, d.head_dim, d.rope_dims, d.layers), (5120, 32, 8, 160, 40, 40));
    assert!(!d.attention_bias);
    assert_eq!(d.block, Block::Parallel);
    assert!(d.qk_norm);
    assert_eq!(d.qk_norm_kind, QkNorm::HeadLayerNorm);
    assert_eq!((d.names.q_norm.as_str(), d.names.k_norm.as_str()), ("self_attn.q_layernorm", "self_attn.k_layernorm"));
    // head_dim 160: supported since 2026-11-15
    assert!(!blockers(&stablelm2_12b()).iter().any(|m| m.contains("head_dim")));
}

#[test]
fn defaults_and_refusals() {
    // mlx_lm defaults: no parallel residual, no qk layernorm
    let d = map(&remove(remove(stablelm2_12b(), "use_parallel_residual"), "qk_layernorm"));
    assert!(d.block == Block::Sequential && !d.qk_norm);
    let (d, _) = map_config(&stablelm2_1_6b(), Some(&json!({"eos_token_id": [100278, 100257]}))).unwrap();
    assert_eq!(d.eos, vec![100257, 100278]);
    for k in ["hidden_size", "rope_theta", "partial_rotary_factor", "use_qkv_bias", "layer_norm_eps", "num_key_value_heads"] {
        assert!(err(&remove(stablelm2_1_6b(), k)).contains(k), "{}", k);
    }
    assert!(err(&edit(stablelm2_1_6b(), |v| v["head_dim"] = json!(128))).contains("head_dim 128"));
    assert!(err(&edit(stablelm2_1_6b(), |v| v["num_attention_heads"] = json!(30))).contains("multiple"));
    assert!(err(&edit(stablelm2_1_6b(), |v| v["model_type"] = json!("stablelm_epoch"))).contains("remote-code"));
    assert!(err(&edit(stablelm2_1_6b(), |v| v["model_type"] = json!("llama"))).contains("not StableLM"));
    assert!(err(&edit(stablelm2_1_6b(), |v| v["architectures"] = json!(["StableLMEpochForCausalLM"]))).contains("architecture"));
    assert!(err(&edit(stablelm2_1_6b(), |v| v["num_local_experts"] = json!(8))).contains("mixture-of-experts"));
    let has = |v: &Value, s: &str| blockers(v).iter().any(|m| m.contains(s));
    assert!(has(&edit(stablelm2_1_6b(), |v| v["rope_scaling"] = json!({"type": "linear", "factor": 2.0})), "linear factor 2"));
    assert!(has(&edit(stablelm2_1_6b(), |v| v["tie_word_embeddings"] = json!(true)), "tie_word_embeddings"));
    assert!(has(&edit(stablelm2_1_6b(), |v| v["hidden_act"] = json!("gelu")), "hidden_act"));
    assert!(has(&remove(stablelm2_1_6b(), "quantization"), "no MLX quantization"));
    assert!(dense_config(&remove(remove(stablelm2_1_6b(), "eos_token_id"), "bos_token_id"), no_dir()).unwrap_err().contains("eos"));
}

// ---------- chat ----------

/// Real renders: equal, or the port refuses where the template silently loses input (`lossy`) or raises.
fn check(kind: &TemplateKind, cases: &[Render], lossy: &Lossy) {
    for (m, agp, want) in cases {
        let got = kind.render(&msgs(m), *agp);
        match want {
            Ok(_) if lossy(m) => assert!(got.is_err(), "{:?}: port should refuse, rendered {:?}", m, got),
            Ok(s) => assert_eq!(got.as_deref(), Ok(*s), "{:?} agp {}", m, agp),
            Err(e) => assert!(got.is_err(), "template raised {:?}, port rendered {:?}", e, got),
        }
    }
}

/// Roles the port takes; the templates render or drop the others.
fn other_role(m: &[(&str, &str)]) -> bool { m.iter().any(|(r, _)| !matches!(*r, "system" | "user" | "assistant")) }

#[test]
fn chat_strings_equal_hf_renders() {
    let chat = TemplateKind::ChatMl { default_system: Some(CHAT_DEFAULT_SYSTEM.into()) };
    check(&chat, CHAT_RENDERS, &|m| other_role(m) || (m.len() == 1 && m[0].0 == "system"));
    // chat_prompt = the stablelm-2-chat template
    assert_eq!(chat_prompt(&msgs(&[("user", "Hello!")]), true).unwrap(),
               "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n");
}

#[test]
fn zephyr_strings_equal_hf_renders() {
    check(&TemplateKind::Zephyr { eos: "<|endoftext|>".into() }, ZEPHYR_RENDERS, &other_role);
}

#[test]
fn chat_refusals() {
    assert!(chat_prompt(&[], true).unwrap_err().contains("no messages"));
    assert!(chat_prompt(&msgs(&[("system", "s")]), true).unwrap_err().contains("system message"));
    assert!(chat_prompt(&msgs(&[("user", "a"), ("tool", "r")]), true).unwrap_err().contains("\"tool\""));
}

#[test]
fn template_kinds_from_real_templates() {
    let all = |_: &str| true;
    let eos = Some("<|endoftext|>");
    assert_eq!(kind_from_text(Some(CHAT_TPL), &all, eos).unwrap(),
               TemplateKind::ChatMl { default_system: Some("You are a helpful assistant.".into()) });
    assert_eq!(kind_from_text(Some(ZEPHYR_TPL), &all, eos).unwrap(), TemplateKind::Zephyr { eos: "<|endoftext|>".into() });
    assert!(kind_from_text(None, &all, eos).unwrap_err().contains("base model"));
    assert!(kind_from_text(Some(ZEPHYR_TPL), &all, None).unwrap_err().contains("eos_token"));
    assert!(kind_from_text(Some(CHAT_TPL), &|t: &str| t != "<|im_end|>", eos).unwrap_err().contains("lacks <|im_end|>"));
    assert!(kind_from_text(Some("<|im_start|>{{ m }}<|im_end|>"), &all, eos).unwrap_err().contains("ChatML variant"));
    assert!(kind_from_text(Some("[INST] {{ m }}"), &all, eos).unwrap_err().contains("unknown template"));
}

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sd-stablelm-test-{}-{}", tag, std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

#[test]
fn template_from_folder() {
    let d = tmpdir("zephyr");
    std::fs::write(d.join("tokenizer.json"), json!({"added_tokens": [{"id": 100257, "content": "<|endoftext|>"},
        {"id": 100277, "content": "<|im_start|>"}, {"id": 100278, "content": "<|im_end|>"}]}).to_string()).unwrap();
    std::fs::write(d.join("tokenizer_config.json"), json!({"chat_template": ZEPHYR_TPL, "eos_token": "<|endoftext|>"}).to_string()).unwrap();
    assert_eq!(template_kind(&d).unwrap(), TemplateKind::Zephyr { eos: "<|endoftext|>".into() });
    assert_eq!(chat_prompt_dir(&d, &msgs(&[("user", "Hello!")]), true).unwrap(), "<|user|>\nHello!<|endoftext|>\n<|assistant|>\n");
    std::fs::write(d.join("tokenizer_config.json"), json!({"chat_template": CHAT_TPL, "eos_token": "<|endoftext|>"}).to_string()).unwrap();
    assert!(matches!(template_kind(&d).unwrap(), TemplateKind::ChatMl { .. }));
    let _ = std::fs::remove_dir_all(&d);
}

/// HF `apply_chat_template` renders of the real template (mlx-community/stablelm-2-1_6b-chat-4bit), transformers 5.21, jinja2 3.1.6.
const CHAT_RENDERS: &[Render] = &[
    (&[("user", "Hello!")], true, Ok("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n")),
    (&[("user", "Hello!")], false, Ok("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n")),
    (&[("system", "You are terse."), ("user", "Hi")], true, Ok("<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n")),
    (&[("system", "You are terse."), ("user", "Hi")], false, Ok("<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n")),
    (&[("user", "  What is 2+2?  "), ("assistant", " 4 "), ("user", "Thanks")], true, Ok("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n  What is 2+2?  <|im_end|>\n<|im_start|>assistant\n 4 <|im_end|>\n<|im_start|>user\nThanks<|im_end|>\n<|im_start|>assistant\n")),
    (&[("user", "  What is 2+2?  "), ("assistant", " 4 "), ("user", "Thanks")], false, Ok("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n  What is 2+2?  <|im_end|>\n<|im_start|>assistant\n 4 <|im_end|>\n<|im_start|>user\nThanks<|im_end|>\n")),
    (&[("system", "S"), ("user", "a"), ("assistant", "b"), ("user", "c")], true, Ok("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\nb<|im_end|>\n<|im_start|>user\nc<|im_end|>\n<|im_start|>assistant\n")),
    (&[("system", "S"), ("user", "a"), ("assistant", "b"), ("user", "c")], false, Ok("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\nb<|im_end|>\n<|im_start|>user\nc<|im_end|>\n")),
    (&[("user", "a"), ("user", "b")], true, Ok("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>user\nb<|im_end|>\n<|im_start|>assistant\n")),
    (&[("user", "a"), ("user", "b")], false, Ok("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>user\nb<|im_end|>\n")),
    (&[("user", "x"), ("system", "late"), ("user", "y")], true, Ok("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nx<|im_end|>\n<|im_start|>system\nlate<|im_end|>\n<|im_start|>user\ny<|im_end|>\n<|im_start|>assistant\n")),
    (&[("user", "x"), ("system", "late"), ("user", "y")], false, Ok("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nx<|im_end|>\n<|im_start|>system\nlate<|im_end|>\n<|im_start|>user\ny<|im_end|>\n")),
    (&[("user", "q"), ("assistant", "<think>t</think> ans"), ("user", "q2")], true, Ok("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>t</think> ans<|im_end|>\n<|im_start|>user\nq2<|im_end|>\n<|im_start|>assistant\n")),
    (&[("user", "q"), ("assistant", "<think>t</think> ans"), ("user", "q2")], false, Ok("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>t</think> ans<|im_end|>\n<|im_start|>user\nq2<|im_end|>\n")),
    (&[("system", "A"), ("user", "u"), ("system", "B"), ("user", "v")], true, Ok("<|im_start|>system\nA<|im_end|>\n<|im_start|>user\nu<|im_end|>\n<|im_start|>system\nB<|im_end|>\n<|im_start|>user\nv<|im_end|>\n<|im_start|>assistant\n")),
    (&[("system", "A"), ("user", "u"), ("system", "B"), ("user", "v")], false, Ok("<|im_start|>system\nA<|im_end|>\n<|im_start|>user\nu<|im_end|>\n<|im_start|>system\nB<|im_end|>\n<|im_start|>user\nv<|im_end|>\n")),
    (&[("user", "line1\nline2\n"), ("assistant", "\n\nout\n"), ("user", "tab\there")], true, Ok("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nline1\nline2\n<|im_end|>\n<|im_start|>assistant\n\n\nout\n<|im_end|>\n<|im_start|>user\ntab\there<|im_end|>\n<|im_start|>assistant\n")),
    (&[("user", "line1\nline2\n"), ("assistant", "\n\nout\n"), ("user", "tab\there")], false, Ok("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nline1\nline2\n<|im_end|>\n<|im_start|>assistant\n\n\nout\n<|im_end|>\n<|im_start|>user\ntab\there<|im_end|>\n")),
    (&[("system", "only")], true, Ok("<|im_start|>assistant\n")),
    (&[("system", "only")], false, Ok("")),
    (&[("user", "call it"), ("observation", " {\"r\": 1} "), ("user", "ok")], true, Ok("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\ncall it<|im_end|>\n<|im_start|>observation\n {\"r\": 1} <|im_end|>\n<|im_start|>user\nok<|im_end|>\n<|im_start|>assistant\n")),
    (&[("user", "call it"), ("observation", " {\"r\": 1} "), ("user", "ok")], false, Ok("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\ncall it<|im_end|>\n<|im_start|>observation\n {\"r\": 1} <|im_end|>\n<|im_start|>user\nok<|im_end|>\n")),
];
/// HF `apply_chat_template` renders of the real template (mlx-community/stablelm-2-zephyr-1_6b-4bit), transformers 5.21, jinja2 3.1.6.
const ZEPHYR_RENDERS: &[Render] = &[
    (&[("user", "Hello!")], true, Ok("<|user|>\nHello!<|endoftext|>\n<|assistant|>\n")),
    (&[("user", "Hello!")], false, Ok("<|user|>\nHello!<|endoftext|>\n")),
    (&[("system", "You are terse."), ("user", "Hi")], true, Ok("<|system|>\nYou are terse.<|endoftext|>\n<|user|>\nHi<|endoftext|>\n<|assistant|>\n")),
    (&[("system", "You are terse."), ("user", "Hi")], false, Ok("<|system|>\nYou are terse.<|endoftext|>\n<|user|>\nHi<|endoftext|>\n")),
    (&[("user", "  What is 2+2?  "), ("assistant", " 4 "), ("user", "Thanks")], true, Ok("<|user|>\n  What is 2+2?  <|endoftext|>\n<|assistant|>\n 4 <|endoftext|>\n<|user|>\nThanks<|endoftext|>\n<|assistant|>\n")),
    (&[("user", "  What is 2+2?  "), ("assistant", " 4 "), ("user", "Thanks")], false, Ok("<|user|>\n  What is 2+2?  <|endoftext|>\n<|assistant|>\n 4 <|endoftext|>\n<|user|>\nThanks<|endoftext|>\n")),
    (&[("system", "S"), ("user", "a"), ("assistant", "b"), ("user", "c")], true, Ok("<|system|>\nS<|endoftext|>\n<|user|>\na<|endoftext|>\n<|assistant|>\nb<|endoftext|>\n<|user|>\nc<|endoftext|>\n<|assistant|>\n")),
    (&[("system", "S"), ("user", "a"), ("assistant", "b"), ("user", "c")], false, Ok("<|system|>\nS<|endoftext|>\n<|user|>\na<|endoftext|>\n<|assistant|>\nb<|endoftext|>\n<|user|>\nc<|endoftext|>\n")),
    (&[("user", "a"), ("user", "b")], true, Ok("<|user|>\na<|endoftext|>\n<|user|>\nb<|endoftext|>\n<|assistant|>\n")),
    (&[("user", "a"), ("user", "b")], false, Ok("<|user|>\na<|endoftext|>\n<|user|>\nb<|endoftext|>\n")),
    (&[("user", "x"), ("system", "late"), ("user", "y")], true, Ok("<|user|>\nx<|endoftext|>\n<|system|>\nlate<|endoftext|>\n<|user|>\ny<|endoftext|>\n<|assistant|>\n")),
    (&[("user", "x"), ("system", "late"), ("user", "y")], false, Ok("<|user|>\nx<|endoftext|>\n<|system|>\nlate<|endoftext|>\n<|user|>\ny<|endoftext|>\n")),
    (&[("user", "q"), ("assistant", "<think>t</think> ans"), ("user", "q2")], true, Ok("<|user|>\nq<|endoftext|>\n<|assistant|>\n<think>t</think> ans<|endoftext|>\n<|user|>\nq2<|endoftext|>\n<|assistant|>\n")),
    (&[("user", "q"), ("assistant", "<think>t</think> ans"), ("user", "q2")], false, Ok("<|user|>\nq<|endoftext|>\n<|assistant|>\n<think>t</think> ans<|endoftext|>\n<|user|>\nq2<|endoftext|>\n")),
    (&[("system", "A"), ("user", "u"), ("system", "B"), ("user", "v")], true, Ok("<|system|>\nA<|endoftext|>\n<|user|>\nu<|endoftext|>\n<|system|>\nB<|endoftext|>\n<|user|>\nv<|endoftext|>\n<|assistant|>\n")),
    (&[("system", "A"), ("user", "u"), ("system", "B"), ("user", "v")], false, Ok("<|system|>\nA<|endoftext|>\n<|user|>\nu<|endoftext|>\n<|system|>\nB<|endoftext|>\n<|user|>\nv<|endoftext|>\n")),
    (&[("user", "line1\nline2\n"), ("assistant", "\n\nout\n"), ("user", "tab\there")], true, Ok("<|user|>\nline1\nline2\n<|endoftext|>\n<|assistant|>\n\n\nout\n<|endoftext|>\n<|user|>\ntab\there<|endoftext|>\n<|assistant|>\n")),
    (&[("user", "line1\nline2\n"), ("assistant", "\n\nout\n"), ("user", "tab\there")], false, Ok("<|user|>\nline1\nline2\n<|endoftext|>\n<|assistant|>\n\n\nout\n<|endoftext|>\n<|user|>\ntab\there<|endoftext|>\n")),
    (&[("system", "only")], true, Ok("<|system|>\nonly<|endoftext|>\n<|assistant|>\n")),
    (&[("system", "only")], false, Ok("<|system|>\nonly<|endoftext|>\n")),
    (&[("user", "call it"), ("observation", " {\"r\": 1} "), ("user", "ok")], true, Ok("<|user|>\ncall it<|endoftext|>\n<|user|>\nok<|endoftext|>\n<|assistant|>\n")),
    (&[("user", "call it"), ("observation", " {\"r\": 1} "), ("user", "ok")], false, Ok("<|user|>\ncall it<|endoftext|>\n<|user|>\nok<|endoftext|>\n")),
];
/// chat_template of mlx-community/stablelm-2-1_6b-chat-4bit tokenizer_config.json, verbatim.
const CHAT_TPL: &str = r##"{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'You are a helpful assistant.' %}{% endif %}{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{{'<|im_start|>system
' + system_message + '<|im_end|>
'}}{% endif %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}"##;
/// chat_template of mlx-community/stablelm-2-zephyr-1_6b-4bit tokenizer_config.json, verbatim.
const ZEPHYR_TPL: &str = r##"{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<|user|>
' + message['content'] + eos_token }}
{% elif message['role'] == 'system' %}
{{ '<|system|>
' + message['content'] + eos_token }}
{% elif message['role'] == 'assistant' %}
{{ '<|assistant|>
'  + message['content'] + eos_token }}
{% endif %}
{% if loop.last and add_generation_prompt %}
{{ '<|assistant|>' }}
{% endif %}
{% endfor %}"##;
