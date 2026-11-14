//! Config fixtures and chat strings only: no model, no Metal, no weights.
//!
//! Fixtures: config.json and chat template (tokenizer_config.json) of mlx-community/Phi-3.5-mini-instruct-4bit,
//! Phi-4-mini-instruct-4bit and Phi-3-mini-4k-instruct-4bit, fetched from huggingface.co 2026-11-15 (metadata only;
//! verbatim). Expected chat strings: transformers 5.21 `apply_chat_template(tokenize=False)` on those repos' own
//! tokenizer files.
use dense_engine::config::{Family, LayerKind, RopeScaling};
use serde_json::{json, Value};

use super::chat::{kind_from_text, TemplateKind};
use super::*;

const PHI35_MINI: &str = r#"{"architectures": ["Phi3ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0, "auto_map": {"AutoConfig": "configuration_phi3.Phi3Config", "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM"}, "bos_token_id": 1, "embd_pdrop": 0.0, "eos_token_id": 32000, "hidden_act": "silu", "hidden_size": 3072, "initializer_range": 0.02, "intermediate_size": 8192, "max_position_embeddings": 131072, "model_type": "phi3", "num_attention_heads": 32, "num_hidden_layers": 32, "num_key_value_heads": 32, "original_max_position_embeddings": 4096, "pad_token_id": 32000, "quantization": {"group_size": 64, "bits": 4}, "resid_pdrop": 0.0, "rms_norm_eps": 1e-05, "rope_scaling": {"long_factor": [1.0800000429153442, 1.1100000143051147, 1.1399999856948853, 1.340000033378601, 1.5899999141693115, 1.600000023841858, 1.6200000047683716, 2.620000123977661, 3.2300000190734863, 3.2300000190734863, 4.789999961853027, 7.400000095367432, 7.700000286102295, 9.09000015258789, 12.199999809265137, 17.670000076293945, 24.46000099182129, 28.57000160217285, 30.420001983642578, 30.840002059936523, 32.590003967285156, 32.93000411987305, 42.320003509521484, 44.96000289916992, 50.340003967285156, 50.45000457763672, 57.55000305175781, 57.93000411987305, 58.21000289916992, 60.1400032043457, 62.61000442504883, 62.62000274658203, 62.71000289916992, 63.1400032043457, 63.1400032043457, 63.77000427246094, 63.93000411987305, 63.96000289916992, 63.970001220703125, 64.02999877929688, 64.06999969482422, 64.08000183105469, 64.12000274658203, 64.41000366210938, 64.4800033569336, 64.51000213623047, 64.52999877929688, 64.83999633789062], "short_factor": [1.0, 1.0199999809265137, 1.0299999713897705, 1.0299999713897705, 1.0499999523162842, 1.0499999523162842, 1.0499999523162842, 1.0499999523162842, 1.0499999523162842, 1.0699999332427979, 1.0999999046325684, 1.1099998950958252, 1.1599998474121094, 1.1599998474121094, 1.1699998378753662, 1.2899998426437378, 1.339999794960022, 1.679999828338623, 1.7899998426437378, 1.8199998140335083, 1.8499997854232788, 1.8799997568130493, 1.9099997282028198, 1.9399996995925903, 1.9899996519088745, 2.0199997425079346, 2.0199997425079346, 2.0199997425079346, 2.0199997425079346, 2.0199997425079346, 2.0199997425079346, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0299997329711914, 2.0799996852874756, 2.0899996757507324, 2.189999580383301, 2.2199995517730713, 2.5899994373321533, 2.729999542236328, 2.749999523162842, 2.8399994373321533], "type": "longrope"}, "rope_theta": 10000.0, "sliding_window": 262144, "tie_word_embeddings": false, "torch_dtype": "bfloat16", "transformers_version": "4.43.3", "use_cache": true, "vocab_size": 32064}"#;
const PHI4_MINI: &str = r#"{"architectures": ["Phi3ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0, "auto_map": {"AutoConfig": "configuration_phi3.Phi3Config", "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM", "AutoTokenizer": "Xenova/gpt-4o"}, "bos_token_id": 199999, "embd_pdrop": 0.0, "eos_token_id": 200020, "full_attn_mod": 1, "hidden_act": "silu", "hidden_size": 3072, "initializer_range": 0.02, "intermediate_size": 8192, "interpolate_factor": 1, "lm_head_bias": false, "max_position_embeddings": 131072, "mlp_bias": false, "model_type": "phi3", "num_attention_heads": 24, "num_hidden_layers": 32, "num_key_value_heads": 8, "original_max_position_embeddings": 4096, "pad_token_id": 199999, "partial_rotary_factor": 0.75, "quantization": {"group_size": 64, "bits": 4}, "quantization_config": {"group_size": 64, "bits": 4}, "resid_pdrop": 0.0, "rms_norm_eps": 1e-05, "rope_scaling": {"long_factor": [1, 1.118320672, 1.250641126, 1.398617824, 1.564103225, 1.74916897, 1.956131817, 2.187582649, 2.446418898, 2.735880826, 3.059592084, 3.421605075, 3.826451687, 4.279200023, 4.785517845, 5.351743533, 5.984965424, 6.693110555, 7.485043894, 8.370679318, 9.36110372, 10.4687158, 11.70738129, 13.09260651, 14.64173252, 16.37415215, 18.31155283, 20.47818807, 22.90118105, 25.61086418, 28.64115884, 32.03, 32.1, 32.13, 32.23, 32.6, 32.61, 32.64, 32.66, 32.7, 32.71, 32.93, 32.97, 33.28, 33.49, 33.5, 44.16, 47.77], "short_factor": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "type": "longrope"}, "rope_theta": 10000.0, "sliding_window": 262144, "tie_word_embeddings": true, "torch_dtype": "bfloat16", "transformers_version": "4.45.0", "use_cache": true, "vocab_size": 200064}"#;
const PHI3_MINI_4K: &str = r#"{"architectures": ["Phi3ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0, "auto_map": {"AutoConfig": "configuration_phi3.Phi3Config", "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM"}, "bos_token_id": 1, "embd_pdrop": 0.0, "eos_token_id": 32000, "hidden_act": "silu", "hidden_size": 3072, "initializer_range": 0.02, "intermediate_size": 8192, "max_position_embeddings": 4096, "model_type": "phi3", "num_attention_heads": 32, "num_hidden_layers": 32, "num_key_value_heads": 32, "original_max_position_embeddings": 4096, "pad_token_id": 32000, "quantization": {"group_size": 64, "bits": 4}, "resid_pdrop": 0.0, "rms_norm_eps": 1e-05, "rope_scaling": null, "rope_theta": 10000.0, "sliding_window": 2047, "tie_word_embeddings": false, "torch_dtype": "bfloat16", "transformers_version": "4.40.2", "use_cache": true, "vocab_size": 32064}"#;
const PHI35_TPL: &str = "{% for message in messages %}{% if message['role'] == 'system' and message['content'] %}{{'<|system|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}";
const PHI4_TPL: &str = "{% for message in messages %}{% if message['role'] == 'system' and 'tools' in message and message['tools'] is not none %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|tool|>' + message['tools'] + '<|/tool|>' + '<|end|>' }}{% else %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %}";
const PHI3_TPL: &str = "{% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}";

fn j(s: &str) -> Value { serde_json::from_str(s).unwrap() }
fn parse(v: &Value) -> DenseConfig { config::parse(v, None, None).unwrap() }
fn edit(mut v: Value, f: impl FnOnce(&mut Value)) -> Value { f(&mut v); v }
fn m(r: &str, c: &str) -> (String, String) { (r.to_string(), c.to_string()) }

#[test]
fn phi35_mini_mapping() {
    let d = parse(&j(PHI35_MINI));
    assert_eq!(d.family, Family::Phi3);
    assert_eq!((d.hidden, d.inter, d.layers, d.heads, d.kv_heads, d.head_dim, d.vocab), (3072, 8192, 32, 32, 32, 96, 32064));
    assert_eq!((d.rope_dims, d.rope_theta, d.rope_scale, d.rope_scaling.clone()), (96, 10000.0, RopeScaling::None, None));
    let su = d.rope_su.clone().unwrap();
    assert_eq!(su.long_factor.len(), 48);
    assert_eq!((su.long_factor[0], su.long_factor[47]), (1.0800000429153442, 64.83999633789062));
    // mlx_lm SuScaledRoPE: factor 131072 / 4096 = 32 -> sqrt(1 + ln 32 / ln 4096)
    assert!((su.mscale - (1.0f64 + 32f64.ln() / 4096f64.ln()).sqrt()).abs() < 1e-12);
    assert_eq!(d.names.qkv.as_deref(), Some("self_attn.qkv_proj"));
    assert_eq!(d.names.gate_up.as_deref(), Some("mlp.gate_up_proj"));
    assert!(!d.tie_word_embeddings && !d.attention_bias && !d.o_bias && !d.mlp_bias && !d.qk_norm && !d.rope_traditional);
    // sliding_window 262144 ignored (mlx phi3)
    assert!(d.kinds.iter().all(|k| *k == LayerKind::Full) && d.sliding_window.is_none());
    assert_eq!((d.eps, d.max_pos, d.bos, d.eos.clone(), d.quant), (1e-5, 131072, Some(1), vec![32000], Some((64, 4))));
    // eos as mlx_lm: config 32000 + tokenizer <|end|> 32007
    assert_eq!(config::parse(&j(PHI35_MINI), None, Some(32007)).unwrap().eos, vec![32000, 32007]);
    let b = blockers(&j(PHI35_MINI));
    assert!(b.is_empty(), "{:?}", b); // head dim 96 supported since 2026-11-15
}

#[test]
fn phi4_mini_mapping() {
    let d = parse(&j(PHI4_MINI));
    assert_eq!((d.hidden, d.inter, d.layers, d.heads, d.kv_heads, d.head_dim, d.vocab), (3072, 8192, 32, 24, 8, 128, 200064));
    // partial_rotary_factor 0.75
    assert_eq!(d.rope_dims, 96);
    let su = d.rope_su.clone().unwrap();
    assert_eq!((su.long_factor.len(), su.long_factor[0], su.long_factor[47]), (48, 1.0, 47.77));
    assert!((su.mscale - (1.0f64 + 32f64.ln() / 4096f64.ln()).sqrt()).abs() < 1e-12);
    assert!(d.tie_word_embeddings);
    assert_eq!((d.bos, d.eos.clone()), (Some(199999), vec![200020]));
    assert_eq!(config::parse(&j(PHI4_MINI), None, Some(199999)).unwrap().eos, vec![200020, 199999]);
    let b = blockers(&j(PHI4_MINI));
    assert!(b.is_empty(), "{:?}", b);
    assert!(config::dense_config_with(&j(PHI4_MINI), None, Some(199999)).is_ok());
}

#[test]
fn phi3_mini_4k_mapping() {
    let d = parse(&j(PHI3_MINI_4K));
    assert_eq!((d.head_dim, d.rope_dims, d.max_pos), (96, 96, 4096));
    // no rope_scaling: plain RoPE; sliding_window 2047 ignored as in mlx_lm
    assert!(d.rope_su.is_none() && d.rope_scale == RopeScaling::None && d.sliding_window.is_none());
}

#[test]
fn refusals_and_variants() {
    let p = j(PHI35_MINI);
    let e = |v: &Value| config::parse(v, None, None).unwrap_err();
    assert!(e(&edit(p.clone(), |v| v["model_type"] = json!("phimoe"))).contains("mixture-of-experts"));
    assert!(e(&edit(p.clone(), |v| v["model_type"] = json!("phi3_v"))).contains("multimodal"));
    assert!(e(&edit(p.clone(), |v| v["attention_bias"] = json!(true))).contains("bias"));
    assert!(e(&edit(p.clone(), |v| v["head_dim"] = json!(128))).contains("head_dim"));
    assert!(e(&edit(p.clone(), |v| v["rope_scaling"] = json!({"type": "yarn", "long_factor": [1.0], "factor": 2.0}))).contains("yarn"));
    assert!(e(&edit(p.clone(), |v| v["rope_scaling"] = json!({"type": "longrope"}))).contains("long_factor"));
    assert!(e(&edit(p.clone(), |v| v["rope_scaling"]["long_factor"] = json!([1.0, 2.0]))).contains("rope dims / 2"));
    assert!(e(&edit(p.clone(), |v| v["hidden_size"] = json!(3000))).contains("multiple"));
    let d = parse(&edit(p.clone(), |v| v["rope_scaling"] = json!({"type": "linear", "factor": 2.0, "long_factor": [1.0]})));
    assert_eq!((d.rope_scale, d.rope_su.is_none()), (RopeScaling::Linear { factor: 2.0 }, true));
    let d = parse(&edit(p.clone(), |v| v["rope_scaling"] = json!({})));
    assert!(d.rope_su.is_none());
    // generation_config eos replaces config eos (mlx_lm)
    let d = config::parse(&p, Some(&json!({"eos_token_id": [32007, 32001, 32000]})), Some(32007)).unwrap();
    assert_eq!(d.eos, vec![32007, 32001, 32000]);
    let b = blockers(&edit(p, |v| { v.as_object_mut().unwrap().remove("quantization"); }));
    assert!(b.iter().any(|x| x.contains("no MLX quantization")));
}

#[test]
fn template_kinds() {
    assert_eq!(kind_from_text(Some(PHI35_TPL), Some("<|end|>")).unwrap(),
               TemplateKind::Phi3 { skip_empty_system: true, eos: "<|end|>".into() });
    assert_eq!(kind_from_text(Some(PHI3_TPL), Some("<|end|>")).unwrap(),
               TemplateKind::Phi3 { skip_empty_system: false, eos: "<|end|>".into() });
    assert_eq!(kind_from_text(Some(PHI4_TPL), Some("<|endoftext|>")).unwrap(), TemplateKind::Phi4 { eos: "<|endoftext|>".into() });
    assert!(kind_from_text(Some(PHI4_TPL), None).unwrap_err().contains("eos_token"));
    assert!(kind_from_text(None, None).unwrap_err().contains("none"));
    assert!(kind_from_text(Some("{{ bos_token }}<|user|>"), Some("<|end|>")).unwrap_err().contains("bos_token"));
    assert!(kind_from_text(Some("<|user|> other"), Some("<|end|>")).unwrap_err().contains("another release"));
}

#[test]
fn phi3_strings() {
    let k = kind_from_text(Some(PHI35_TPL), Some("<|end|>")).unwrap();
    assert_eq!(k.render(&[m("user", "Hello")], true).unwrap(), "<|user|>\nHello<|end|>\n<|assistant|>\n");
    assert_eq!(k.render(&[m("user", "Hello")], false).unwrap(), "<|user|>\nHello<|end|>\n<|end|>");
    assert_eq!(k.render(&[m("system", "You are terse."), m("user", "Hi there")], true).unwrap(),
               "<|system|>\nYou are terse.<|end|>\n<|user|>\nHi there<|end|>\n<|assistant|>\n");
    assert_eq!(k.render(&[m("user", "What is 2+2?"), m("assistant", "4"), m("user", "And 3+3?")], true).unwrap(),
               "<|user|>\nWhat is 2+2?<|end|>\n<|assistant|>\n4<|end|>\n<|user|>\nAnd 3+3?<|end|>\n<|assistant|>\n");
    assert_eq!(k.render(&[m("user", "  padded \n")], true).unwrap(), "<|user|>\n  padded \n<|end|>\n<|assistant|>\n");
    // empty system: skipped by Phi-3.5, written by Phi-3-mini-4k
    assert_eq!(k.render(&[m("system", ""), m("user", "Hi")], true).unwrap(), "<|user|>\nHi<|end|>\n<|assistant|>\n");
    let k3 = kind_from_text(Some(PHI3_TPL), Some("<|end|>")).unwrap();
    assert_eq!(k3.render(&[m("system", ""), m("user", "Hi")], true).unwrap(), "<|system|>\n<|end|>\n<|user|>\nHi<|end|>\n<|assistant|>\n");
    assert!(k.render(&[m("tool", "x")], true).unwrap_err().contains("role"));
    assert_eq!(chat_prompt(&[m("user", "Hello")], true).unwrap(), "<|user|>\nHello<|end|>\n<|assistant|>\n");
}

#[test]
fn phi4_strings() {
    let k = kind_from_text(Some(PHI4_TPL), Some("<|endoftext|>")).unwrap();
    assert_eq!(k.render(&[m("user", "Hello")], true).unwrap(), "<|user|>Hello<|end|><|assistant|>");
    assert_eq!(k.render(&[m("user", "Hello")], false).unwrap(), "<|user|>Hello<|end|><|endoftext|>");
    assert_eq!(k.render(&[m("system", "You are terse."), m("user", "Hi there")], true).unwrap(),
               "<|system|>You are terse.<|end|><|user|>Hi there<|end|><|assistant|>");
    assert_eq!(k.render(&[m("system", "Be brief."), m("user", "A"), m("assistant", "B"), m("user", "C")], true).unwrap(),
               "<|system|>Be brief.<|end|><|user|>A<|end|><|assistant|>B<|end|><|user|>C<|end|><|assistant|>");
    assert_eq!(k.render(&[m("user", "  padded \n")], true).unwrap(), "<|user|>  padded \n<|end|><|assistant|>");
    assert!(k.render(&[m("tool", "x")], true).is_err());
}

#[test]
fn model_type_list() {
    assert_eq!(MODEL_TYPES, &["phi3"]);
}
