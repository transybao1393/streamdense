//! Config fixture and chat strings: no model, no Metal, no weights. The fixture is the real config.json of
//! mlx-community/c4ai-command-r7b-12-2024-4bit (fetched 2026-11-15, metadata only); the expected chat strings are HF
//! `apply_chat_template` renders of the "default" chat template of its tokenizer_config.json (generated below the
//! hand-written tests; the template file itself is not copied here).
use std::path::{Path, PathBuf};

/// One chat-render case: messages, add_generation_prompt, expected render or error.
type Render = (&'static [(&'static str, &'static str)], bool, Result<&'static str, &'static str>);

use dense_engine::config::{Activation, Block, Family, LayerKind, Mlp, NormKind, RopeScaling};
use serde_json::{json, Value};

use super::chat::check_template;
use super::config::map_config;
use super::*;

/// Stand-in for the real R7B template: only the marker lines `check_template` looks for (the real file is not copied).
const R7B_TPL: &str = "{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|>' }}\n\
    {{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}\n\
    {{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>'  + content.strip() + '<|END_RESPONSE|><|END_OF_TURN_TOKEN|>' }}\n\
    {{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>' }}\n\
    {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}\n";

/// mlx-community/c4ai-command-r7b-12-2024-4bit config.json, as on disk.
fn r7b() -> Value {
    json!({
        "architectures": ["Cohere2ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0, "bos_token_id": 5,
        "cache_implementation": "hybrid", "eos_token_id": 255001, "head_dim": 128, "hidden_act": "silu",
        "hidden_size": 4096, "initializer_range": 0.02, "intermediate_size": 14336, "layer_norm_eps": 1e-05,
        "layer_switch": 4, "logit_scale": 0.25, "max_position_embeddings": 8192, "model_type": "cohere2",
        "num_attention_heads": 32, "num_hidden_layers": 32, "num_key_value_heads": 8,
        "order_of_interleaved_layers": "local_attn_first", "pad_token_id": 0, "position_embedding_type": "rope_gptj",
        "quantization": {"group_size": 64, "bits": 4}, "quantization_config": {"group_size": 64, "bits": 4},
        "rope_scaling": null, "rope_theta": 50000, "rotary_pct": 1.0, "sliding_window": 4096,
        "sliding_window_pattern": 4, "torch_dtype": "bfloat16", "transformers_version": "4.48.0.dev0", "use_cache": true,
        "use_embedding_sharing": true, "use_gated_activation": true, "use_parallel_block": true,
        "use_parallel_embedding": true, "vocab_size": 256000
    })
}

fn no_dir() -> &'static Path { Path::new("/nonexistent-cohere2-test-dir") }
fn edit(mut v: Value, f: impl FnOnce(&mut Value)) -> Value { f(&mut v); v }
fn remove(v: Value, k: &str) -> Value { edit(v, |v| { v.as_object_mut().unwrap().remove(k); }) }
fn map(v: &Value) -> DenseConfig { map_config(v, None).unwrap().0 }
fn err(v: &Value) -> String { map_config(v, None).unwrap_err() }
fn msgs(m: &[(&str, &str)]) -> Vec<(String, String)> { m.iter().map(|(r, c)| (r.to_string(), c.to_string())).collect() }

/// Blockers the engine named for Cohere2 before its features landed (SPEC.md): every blocker of the real config must
/// be one of these (today: none).
const PENDING: [&str; 4] = ["norm LayerNorm", "block ParallelBranchesFirst", "traditional (interleaved) RoPE", "no-RoPE layers"];
fn only_pending(b: &[String]) -> bool { b.iter().all(|m| PENDING.iter().any(|p| m.contains(p))) }

#[test]
fn model_types() {
    assert_eq!(MODEL_TYPES, &["cohere2"]);
}

#[test]
fn r7b_mapping() {
    let d = map(&r7b());
    assert_eq!((d.family, d.model_type.as_str()), (Family::Cohere2, "cohere2"));
    assert_eq!((d.hidden, d.inter, d.layers, d.heads, d.kv_heads, d.head_dim, d.vocab), (4096, 14336, 32, 32, 8, 128, 256000));
    assert_eq!((d.rope_dims, d.rope_theta, d.rope_traditional, d.rope_scale, d.rope_scaling.clone()),
               (128, 50000.0, true, RopeScaling::None, None));
    // layers 3, 7, ..., 31 global (full, no RoPE); the others sliding 4096 with RoPE
    for (i, k) in d.kinds.iter().enumerate() {
        let global = i % 4 == 3;
        assert_eq!(*k, if global { LayerKind::Full } else { LayerKind::Sliding }, "layer {}", i);
        assert_eq!(d.no_rope_layers[i], global);
        assert_eq!(d.rope_on(i), !global);
    }
    assert_eq!((d.kinds.len(), d.no_rope_layers.len(), d.n_sliding(), d.sliding_window), (32, 32, 24, Some(4096)));
    assert_eq!((d.norm, d.block, d.mlp, d.activation.clone()),
               (NormKind::LayerNorm { bias: false }, Block::ParallelBranchesFirst, Mlp::Gated, Activation::Silu));
    assert!(!d.attention_bias && !d.o_bias && !d.mlp_bias && d.tie_word_embeddings && !d.qk_norm && !d.four_norms);
    assert_eq!((d.logits_scaling, d.eps, d.max_pos, d.attn_scale), (4.0, 1e-5, 8192, None));
    assert_eq!((d.bos, d.eos.clone(), d.quant), (Some(5), vec![255001], Some((64, 4))));
    assert_eq!(d.names.input_norm, "input_layernorm");
}

#[test]
fn r7b_blockers_are_engine_features_only() {
    let b = blockers(&r7b());
    assert!(only_pending(&b), "{:?}", b);
    match dense_config(&r7b(), no_dir()) {
        Ok(d) => assert!(b.is_empty() && d.family == Family::Cohere2),
        Err(e) => assert!(PENDING.iter().any(|p| e.contains(p)) && !b.is_empty(), "{}", e),
    }
}

#[test]
fn defaults_and_refusals() {
    // mlx_lm defaults: logit_scale 0.0625, pattern 4, window 4096, eps 1e-5, no biases
    let v = ["logit_scale", "sliding_window_pattern", "sliding_window", "layer_norm_eps", "attention_bias"]
        .iter().fold(r7b(), |v, k| remove(v, k));
    let d = map(&v);
    assert_eq!((d.logits_scaling, d.sliding_window, d.eps, d.attention_bias), (16.0, Some(4096), 1e-5, false));
    assert_eq!(d.kinds[3], LayerKind::Full);
    let d = map(&edit(r7b(), |v| { v["attention_bias"] = json!(true); v["layer_norm_bias"] = json!(true); }));
    assert!(d.attention_bias && d.o_bias && d.norm == NormKind::LayerNorm { bias: true });
    // a matching layer_types list is accepted, a different one refused
    let lt: Vec<&str> = (0..32).map(|i| if i % 4 == 3 { "full_attention" } else { "sliding_attention" }).collect();
    assert_eq!(map(&edit(r7b(), |v| v["layer_types"] = json!(lt))).n_sliding(), 24);
    assert!(err(&edit(r7b(), |v| v["layer_types"] = json!(vec!["full_attention"; 32]))).contains("layer_types"));
    for k in ["hidden_size", "head_dim", "num_key_value_heads", "rope_theta"] {
        assert!(err(&remove(r7b(), k)).contains(k), "{}", k);
    }
    assert!(err(&edit(r7b(), |v| v["head_dim"] = json!(64))).contains("must equal hidden_size"));
    assert!(err(&edit(r7b(), |v| v["model_type"] = json!("cohere"))).contains("Command-R v1"));
    assert!(err(&edit(r7b(), |v| v["vision_config"] = json!({}))).contains("text models only"));
    assert!(err(&edit(r7b(), |v| v["architectures"] = json!(["AyaVisionForConditionalGeneration"]))).contains("architecture"));
    assert!(err(&edit(r7b(), |v| v["logit_scale"] = json!(0.0))).contains("logit_scale"));
    let has = |v: &Value, s: &str| blockers(v).iter().any(|m| m.contains(s));
    assert!(has(&edit(r7b(), |v| v["rotary_pct"] = json!(0.5)), "rotary_pct 0.5"));
    assert!(has(&edit(r7b(), |v| v["rope_scaling"] = json!({"rope_type": "yarn", "factor": 8.0})), "yarn"));
    assert!(has(&edit(r7b(), |v| v["tie_word_embeddings"] = json!(false)), "tie_word_embeddings"));
    assert!(has(&edit(r7b(), |v| v["use_parallel_block"] = json!(false)), "use_parallel_block"));
    assert!(has(&edit(r7b(), |v| v["hidden_act"] = json!("gelu")), "hidden_act"));
    assert!(has(&edit(r7b(), |v| v["quantization"] = json!({"group_size": 32, "bits": 4})), "group_size 32"));
    assert!(dense_config(&remove(r7b(), "eos_token_id"), no_dir()).unwrap_err().contains("eos"));
}

// ---------- chat ----------

#[test]
fn chat_strings_equal_hf_renders() {
    // the port refuses where the template silently drops input: a later system message, other roles
    let lossy = |m: &[(&str, &str)]| m.iter().enumerate().any(|(i, (r, _))| (*r == "system" && i > 0) || !matches!(*r, "system" | "user" | "assistant"));
    for (m, agp, want) in R7B_RENDERS {
        let got = chat_prompt(&msgs(m), *agp);
        match want {
            Ok(_) if lossy(m) => assert!(got.is_err(), "{:?}: port should refuse, rendered {:?}", m, got),
            Ok(s) => assert_eq!(got.as_deref(), Ok(*s), "{:?} agp {}", m, agp),
            Err(e) => assert!(got.unwrap_err().contains(e)),
        }
    }
}

#[test]
fn chat_refusals() {
    assert!(chat_prompt(&[], true).unwrap_err().contains("no messages"));
    assert!(chat_prompt(&msgs(&[("assistant", "a")]), true).unwrap_err().contains("alternate"));
    assert!(chat_prompt(&msgs(&[("user", "a"), ("tool", "r")]), true).unwrap_err().contains("\"tool\""));
}

#[test]
fn template_check() {
    let all = |_: &str| true;
    check_template(Some(R7B_TPL), &all).unwrap();
    assert!(check_template(None, &all).unwrap_err().contains("base model"));
    assert!(check_template(Some("<|START_OF_TURN_TOKEN|>{{ m }}"), &all).unwrap_err().contains("not the Command R7B template"));
    assert!(check_template(Some(R7B_TPL), &|t: &str| t != "<|START_RESPONSE|>").unwrap_err().contains("lacks <|START_RESPONSE|>"));
}

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sd-cohere2-test-{}-{}", tag, std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

#[test]
fn template_from_folder() {
    let d = tmpdir("r7b");
    let added: Vec<Value> = ["<|START_OF_TURN_TOKEN|>", "<|END_OF_TURN_TOKEN|>", "<|SYSTEM_TOKEN|>", "<|USER_TOKEN|>",
        "<|CHATBOT_TOKEN|>", "<|START_RESPONSE|>", "<|END_RESPONSE|>"].iter().map(|t| json!({"content": t})).collect();
    std::fs::write(d.join("tokenizer.json"), json!({"added_tokens": added}).to_string()).unwrap();
    // the real file's list form: tool_use / rag entries next to "default"
    std::fs::write(d.join("tokenizer_config.json"), json!({"chat_template": [{"name": "default", "template": R7B_TPL},
        {"name": "tool_use", "template": "x"}, {"name": "rag", "template": "y"}]}).to_string()).unwrap();
    assert_eq!(chat_prompt_dir(&d, &msgs(&[("user", "Hello!")]), true).unwrap(),
               "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello!\
                <|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>");
    let _ = std::fs::remove_dir_all(&d);
}

/// HF `apply_chat_template` renders of the real template (mlx-community/c4ai-command-r7b-12-2024-4bit), transformers 5.21, jinja2 3.1.6.
const R7B_RENDERS: &[Render] = &[
    (&[("user", "Hello!")], true, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello!<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>")),
    (&[("user", "Hello!")], false, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello!<|END_OF_TURN_TOKEN|>")),
    (&[("system", "You are terse."), ("user", "Hi")], true, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>You are terse.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hi<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>")),
    (&[("system", "You are terse."), ("user", "Hi")], false, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>You are terse.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hi<|END_OF_TURN_TOKEN|>")),
    (&[("user", "  What is 2+2?  "), ("assistant", " 4 "), ("user", "Thanks")], true, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>What is 2+2?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>4<|END_RESPONSE|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Thanks<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>")),
    (&[("user", "  What is 2+2?  "), ("assistant", " 4 "), ("user", "Thanks")], false, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>What is 2+2?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>4<|END_RESPONSE|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Thanks<|END_OF_TURN_TOKEN|>")),
    (&[("system", "S"), ("user", "a"), ("assistant", "b"), ("user", "c")], true, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>S<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>a<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>b<|END_RESPONSE|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>c<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>")),
    (&[("system", "S"), ("user", "a"), ("assistant", "b"), ("user", "c")], false, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>S<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>a<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>b<|END_RESPONSE|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>c<|END_OF_TURN_TOKEN|>")),
    (&[("user", "a"), ("user", "b")], true, Err("Conversation roles must alternate user/assistant/user/assistant/...")),
    (&[("user", "a"), ("user", "b")], false, Err("Conversation roles must alternate user/assistant/user/assistant/...")),
    (&[("user", "x"), ("system", "late"), ("user", "y")], true, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>x<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>y<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>")),
    (&[("user", "x"), ("system", "late"), ("user", "y")], false, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>x<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>y<|END_OF_TURN_TOKEN|>")),
    (&[("user", "q"), ("assistant", "<think>t</think> ans"), ("user", "q2")], true, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>q<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|><think>t</think> ans<|END_RESPONSE|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>q2<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>")),
    (&[("user", "q"), ("assistant", "<think>t</think> ans"), ("user", "q2")], false, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>q<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|><think>t</think> ans<|END_RESPONSE|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>q2<|END_OF_TURN_TOKEN|>")),
    (&[("system", "A"), ("user", "u"), ("system", "B"), ("user", "v")], true, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>A<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>u<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>v<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>")),
    (&[("system", "A"), ("user", "u"), ("system", "B"), ("user", "v")], false, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>A<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>u<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>v<|END_OF_TURN_TOKEN|>")),
    (&[("user", "line1\nline2\n"), ("assistant", "\n\nout\n"), ("user", "tab\there")], true, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>line1\nline2<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>out<|END_RESPONSE|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>tab\there<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>")),
    (&[("user", "line1\nline2\n"), ("assistant", "\n\nout\n"), ("user", "tab\there")], false, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>line1\nline2<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>out<|END_RESPONSE|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>tab\there<|END_OF_TURN_TOKEN|>")),
    (&[("system", "only")], true, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>only<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>")),
    (&[("system", "only")], false, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>only<|END_OF_TURN_TOKEN|>")),
    (&[("user", "call it"), ("observation", " {\"r\": 1} "), ("user", "ok")], true, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>call it<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>ok<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>")),
    (&[("user", "call it"), ("observation", " {\"r\": 1} "), ("user", "ok")], false, Ok("<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>call it<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>ok<|END_OF_TURN_TOKEN|>")),
];
