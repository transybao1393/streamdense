//! Config fixtures and chat strings only: no model, no Metal, no weights.
//!
//! Fixtures: config.json of mlx-community/Mistral-7B-Instruct-v0.3-4bit, Mistral-Nemo-Instruct-2407-4bit,
//! Ministral-8B-Instruct-2410-4bit and (refused) Ministral-3-8B-Instruct-2512-4bit, fetched from huggingface.co
//! 2026-11-15 (metadata only). Expected chat strings: transformers 5.21 `apply_chat_template(tokenize=False)` on those
//! repos' own tokenizer files.
use dense_engine::config::{Family, LayerKind, RopeScaling};
use serde_json::{json, Value};

use super::chat::{kind_from_text, TemplateKind};
use super::*;

fn dc(v: &Value) -> Result<DenseConfig, String> { config::dense_config_with(v, None, None) }
fn edit(mut v: Value, f: impl FnOnce(&mut Value)) -> Value { f(&mut v); v }
fn m(r: &str, c: &str) -> (String, String) { (r.to_string(), c.to_string()) }

fn mistral_7b_v03() -> Value {
    json!({"architectures": ["MistralForCausalLM"], "attention_dropout": 0.0, "bos_token_id": 1, "eos_token_id": 2,
           "hidden_act": "silu", "hidden_size": 4096, "initializer_range": 0.02, "intermediate_size": 14336,
           "max_position_embeddings": 32768, "model_type": "mistral", "num_attention_heads": 32, "num_hidden_layers": 32,
           "num_key_value_heads": 8, "quantization": {"group_size": 64, "bits": 4}, "rms_norm_eps": 1e-05,
           "rope_theta": 1000000.0, "sliding_window": null, "tie_word_embeddings": false, "torch_dtype": "bfloat16",
           "transformers_version": "4.42.0.dev0", "use_cache": true, "vocab_size": 32768})
}

fn nemo_2407() -> Value {
    json!({"architectures": ["MistralForCausalLM"], "attention_dropout": 0.0, "bos_token_id": 1, "eos_token_id": 2,
           "head_dim": 128, "hidden_act": "silu", "hidden_size": 5120, "initializer_range": 0.02,
           "intermediate_size": 14336, "max_position_embeddings": 1024000, "model_type": "mistral",
           "num_attention_heads": 32, "num_hidden_layers": 40, "num_key_value_heads": 8,
           "quantization": {"group_size": 64, "bits": 4}, "rms_norm_eps": 1e-05, "rope_theta": 1000000.0,
           "sliding_window": null, "tie_word_embeddings": false, "torch_dtype": "bfloat16",
           "transformers_version": "4.43.0.dev0", "use_cache": true, "vocab_size": 131072})
}

fn ministral_8b_2410() -> Value {
    json!({"architectures": ["MistralForCausalLM"], "attention_dropout": 0.0, "bos_token_id": 1, "eos_token_id": 2,
           "head_dim": 128, "hidden_act": "silu", "hidden_size": 4096, "initializer_range": 0.02,
           "intermediate_size": 12288, "max_position_embeddings": 32768, "model_type": "mistral",
           "num_attention_heads": 32, "num_hidden_layers": 36, "num_key_value_heads": 8,
           "quantization": {"group_size": 64, "bits": 4}, "quantization_config": {"group_size": 64, "bits": 4},
           "rms_norm_eps": 1e-05, "rope_theta": 100000000.0, "sliding_window": null, "tie_word_embeddings": false,
           "torch_dtype": "float32", "transformers_version": "4.44.0", "use_cache": true, "vocab_size": 131072})
}

/// Ministral-3-8B-Instruct-2512 (vision_config and text_config shortened to the keys that matter here).
fn ministral3_8b() -> Value {
    json!({"architectures": ["Mistral3ForConditionalGeneration"], "model_type": "mistral3",
           "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
           "text_config": {"head_dim": 128, "hidden_size": 4096, "model_type": "ministral3", "num_hidden_layers": 34,
                           "rope_parameters": {"factor": 16.0, "rope_theta": 1000000.0, "rope_type": "yarn", "type": "yarn"}},
           "vision_config": {"model_type": "pixtral"}})
}

#[test]
fn mistral_7b_v03_config() {
    let d = dc(&mistral_7b_v03()).unwrap();
    assert_eq!(d.family, Family::Mistral);
    assert_eq!((d.hidden, d.inter, d.layers, d.heads, d.kv_heads, d.head_dim, d.vocab), (4096, 14336, 32, 32, 8, 128, 32768));
    assert_eq!((d.rope_dims, d.rope_theta, d.rope_scale, d.rope_traditional), (128, 1e6, RopeScaling::None, false));
    assert!(!d.head_dim_explicit && !d.tie_word_embeddings && !d.attention_bias && !d.o_bias && !d.mlp_bias && !d.qk_norm);
    assert_eq!((d.eps, d.max_pos, d.bos, d.eos.clone(), d.quant), (1e-5, 32768, Some(1), vec![2], Some((64, 4))));
    assert!(d.kinds.iter().all(|k| *k == LayerKind::Full) && d.sliding_window.is_none());
    assert!(blockers(&mistral_7b_v03()).is_empty());
}

#[test]
fn nemo_and_ministral_8b_config() {
    let d = dc(&nemo_2407()).unwrap();
    // Nemo: head_dim 128 explicit (32 x 128 = 4096 != hidden 5120)
    assert_eq!((d.hidden, d.q_dim(), d.kv_dim(), d.head_dim_explicit, d.layers, d.vocab), (5120, 4096, 1024, true, 40, 131072));
    assert_eq!(d.max_pos, 1024000);
    let d = dc(&ministral_8b_2410()).unwrap();
    assert_eq!((d.hidden, d.inter, d.layers, d.rope_theta), (4096, 12288, 36, 1e8));
    assert!(blockers(&nemo_2407()).is_empty() && blockers(&ministral_8b_2410()).is_empty());
}

#[test]
fn mlx_semantics() {
    // sliding_window without layer_types: ignored (mlx llama), all layers full
    let d = dc(&edit(mistral_7b_v03(), |v| v["sliding_window"] = json!(4096))).unwrap();
    assert!(d.kinds.iter().all(|k| *k == LayerKind::Full) && d.sliding_window.is_none());
    // layer_types with sliding layers: windowed
    let lt: Vec<&str> = (0..32).map(|i| if i % 2 == 0 { "sliding_attention" } else { "full_attention" }).collect();
    let d = dc(&edit(mistral_7b_v03(), |v| { v["layer_types"] = json!(lt); v["sliding_window"] = json!(4096); })).unwrap();
    assert_eq!((d.n_sliding(), d.sliding_window), (16, Some(4096)));
    // tie default true (mlx llama ModelArgs)
    let d = dc(&edit(mistral_7b_v03(), |v| { v.as_object_mut().unwrap().remove("tie_word_embeddings"); })).unwrap();
    assert!(d.tie_word_embeddings);
    // attention_bias -> q/k/v and o
    let d = dc(&edit(mistral_7b_v03(), |v| v["attention_bias"] = json!(true))).unwrap();
    assert!(d.attention_bias && d.o_bias);
    // rope_traditional read
    assert!(config::parse(&edit(mistral_7b_v03(), |v| v["rope_traditional"] = json!(true)), None, None).unwrap().rope_traditional);
    // generation_config eos replaces config eos; tokenizer eos added
    let d = config::dense_config_with(&mistral_7b_v03(), Some(&json!({"eos_token_id": [2, 5]})), Some(7)).unwrap();
    assert_eq!(d.eos, vec![2, 5, 7]);
    let d = config::dense_config_with(&mistral_7b_v03(), None, Some(2)).unwrap();
    assert_eq!(d.eos, vec![2]);
}

#[test]
fn refusals() {
    let e = dc(&ministral3_8b()).unwrap_err();
    assert!(e.contains("mistral3") && e.contains("multimodal"), "{}", e);
    assert!(dc(&edit(mistral_7b_v03(), |v| v["model_type"] = json!("mixtral"))).unwrap_err().contains("mixture-of-experts"));
    assert!(dc(&edit(mistral_7b_v03(), |v| v["num_local_experts"] = json!(8))).unwrap_err().contains("mixture-of-experts"));
    assert!(dc(&edit(mistral_7b_v03(), |v| v["model_type"] = json!("llama"))).unwrap_err().contains("not \"mistral\""));
    let e = dc(&edit(mistral_7b_v03(), |v| v["rope_scaling"] = json!({"type": "yarn", "factor": 4.0}))).unwrap_err();
    assert!(e.contains("rope_scaling yarn"), "{}", e);
    let e = dc(&edit(mistral_7b_v03(), |v| v["hidden_act"] = json!("gelu"))).unwrap_err();
    assert!(e.contains("hidden_act"), "{}", e);
    let e = dc(&edit(mistral_7b_v03(), |v| { v.as_object_mut().unwrap().remove("eos_token_id"); })).unwrap_err();
    assert!(e.contains("eos"), "{}", e);
    let b = blockers(&edit(mistral_7b_v03(), |v| { v.as_object_mut().unwrap().remove("quantization"); }));
    assert!(b.iter().any(|x| x.contains("no MLX quantization")), "{:?}", b);
    assert!(blockers(&ministral3_8b())[0].contains("mistral3"));
}

const V03_TPL: &str = "{{bos_token}}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ ' [INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + ' ' + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}";
const NEMO_TPL: &str = "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.last and system_message is defined %}\n            {{- '[INST] ' + system_message + '\\n\\n' + message['content'] + '[/INST]' }}\n        {%- else %}\n            {{- '[INST] ' + message['content'] + '[/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + eos_token}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n";
/// The parts of the Ministral-8B-2410 template the kind is read from (full template: tools, tool calls, results).
const TEKKEN_TPL: &str = "{%- if loop.last and system_message is defined %}\n            {{- \"[INST]\" + system_message + \"\\n\\n\" + message[\"content\"] + \"[/INST]\" }}\n        {%- else %}\n            {{- \"[INST]\" + message[\"content\"] + \"[/INST]\" }}\n        {%- endif %}";

#[test]
fn template_kinds() {
    assert_eq!(kind_from_text(Some(V03_TPL), None).unwrap(), TemplateKind::V3);
    assert_eq!(kind_from_text(Some(NEMO_TPL), None).unwrap(), TemplateKind::Nemo);
    assert_eq!(kind_from_text(Some(TEKKEN_TPL), None).unwrap(), TemplateKind::Tekken);
    assert!(kind_from_text(None, None).unwrap_err().contains("none"));
    assert!(kind_from_text(Some("{{ '[SYSTEM_PROMPT]' }}[INST]"), None).unwrap_err().contains("SYSTEM_PROMPT"));
    assert!(kind_from_text(Some("[INST] something else"), None).unwrap_err().contains("another release"));
    let e = kind_from_text(Some(V03_TPL), Some(&json!({"bos_token": "<|begin|>", "eos_token": "</s>"}))).unwrap_err();
    assert!(e.contains("bos_token"), "{}", e);
    assert!(kind_from_text(Some(V03_TPL), Some(&json!({"bos_token": "<s>", "eos_token": {"content": "</s>"}}))).is_ok());
}

#[test]
fn v03_strings() {
    let k = TemplateKind::V3;
    assert_eq!(k.render(&[m("user", "Hello")], true).unwrap(), "<s> [INST] Hello [/INST]");
    assert_eq!(k.render(&[m("user", "What is 2+2?"), m("assistant", "4"), m("user", "And 3+3?")], true).unwrap(),
               "<s> [INST] What is 2+2? [/INST] 4 </s> [INST] And 3+3? [/INST]");
    assert_eq!(k.render(&[m("user", "  padded \n")], false).unwrap(), "<s> [INST]   padded \n [/INST]");
    assert!(k.render(&[m("system", "You are terse."), m("user", "Hi there")], true).unwrap_err().contains("alternate"));
    assert!(k.render(&[m("user", "a"), m("user", "b")], true).unwrap_err().contains("alternate"));
    assert!(k.render(&[m("user", "a"), m("system", "b")], true).is_err());
    assert_eq!(chat_prompt(&[m("user", "Hello")], true).unwrap(), "<s> [INST] Hello [/INST]");
}

#[test]
fn nemo_strings() {
    let k = TemplateKind::Nemo;
    assert_eq!(k.render(&[m("user", "Hello")], true).unwrap(), "<s>[INST] Hello[/INST]");
    assert_eq!(k.render(&[m("system", "You are terse."), m("user", "Hi there")], true).unwrap(),
               "<s>[INST] You are terse.\n\nHi there[/INST]");
    assert_eq!(k.render(&[m("user", "What is 2+2?"), m("assistant", "4"), m("user", "And 3+3?")], true).unwrap(),
               "<s>[INST] What is 2+2?[/INST] 4</s>[INST] And 3+3?[/INST]");
    assert_eq!(k.render(&[m("system", "Be brief."), m("user", "A"), m("assistant", "B"), m("user", "C")], false).unwrap(),
               "<s>[INST] A[/INST] B</s>[INST] Be brief.\n\nC[/INST]");
    assert_eq!(k.render(&[m("user", "  padded \n")], true).unwrap(), "<s>[INST]   padded \n[/INST]");
    assert!(k.render(&[m("system", "s"), m("user", "a"), m("assistant", "b")], true).unwrap_err().contains("last message"));
}

#[test]
fn tekken_strings() {
    let k = TemplateKind::Tekken;
    assert_eq!(k.render(&[m("user", "Hello")], true).unwrap(), "<s>[INST]Hello[/INST]");
    assert_eq!(k.render(&[m("system", "You are terse."), m("user", "Hi there")], true).unwrap(),
               "<s>[INST]You are terse.\n\nHi there[/INST]");
    assert_eq!(k.render(&[m("user", "What is 2+2?"), m("assistant", "4"), m("user", "And 3+3?")], true).unwrap(),
               "<s>[INST]What is 2+2?[/INST] 4</s>[INST]And 3+3?[/INST]");
    assert_eq!(k.render(&[m("system", "Be brief."), m("user", "A"), m("assistant", "B"), m("user", "C")], true).unwrap(),
               "<s>[INST]A[/INST] B</s>[INST]Be brief.\n\nC[/INST]");
    assert_eq!(k.render(&[m("user", "  padded \n")], true).unwrap(), "<s>[INST]  padded \n[/INST]");
    // assistant content trimmed (template `|trim`)
    assert_eq!(k.render(&[m("user", "a"), m("assistant", " b \n"), m("user", "c")], true).unwrap(),
               "<s>[INST]a[/INST] b</s>[INST]c[/INST]");
    assert!(k.render(&[m("user", "a"), m("tool", "b")], true).unwrap_err().contains("role"));
}

#[test]
fn model_type_list() {
    assert_eq!(MODEL_TYPES, &["mistral"]);
}
