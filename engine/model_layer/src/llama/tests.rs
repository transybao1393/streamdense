//! Config fixtures and template strings only: no model, no Metal, no weights.
//!
//! Fixture sources: Llama-2-7B shape = the config.json of a Llama-2-7B derivative in the local HF cache (same shape;
//! its model_type / architectures / hidden_act changed back to the Llama ones). SmolLM2-135M, TinyLlama-1.1B and
//! Llama-3.2-1B values are from the published configs as remembered; values not checked against a file are marked
//! `unverified`.
use std::path::PathBuf;

use dense_engine::config::{Family, RopeScaling};
use serde_json::{json, Value};

use super::template::{kind_from_text, TemplateKind};
use super::*;

fn mlx(mut v: Value) -> Value { v["quantization"] = json!({"group_size": 64, "bits": 4}); v }
fn edit(mut v: Value, f: impl Fn(&mut Value)) -> Value { f(&mut v); v }
fn dc(v: &Value) -> Result<DenseConfig, String> { config::dense_config_with_gen(v, None) }

fn smollm2_135m() -> Value {
    json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 576, "intermediate_size": 1536,
           "num_hidden_layers": 30, "num_attention_heads": 9, "num_key_value_heads": 3, "hidden_act": "silu",
           "rms_norm_eps": 1e-5, // unverified
           "vocab_size": 49152, "rope_theta": 100000, "rope_scaling": null, "tie_word_embeddings": true,
           "max_position_embeddings": 8192, // unverified
           "bos_token_id": 0, "eos_token_id": 0, // unverified (base model ids)
           "attention_bias": false, "mlp_bias": false})
}

fn tinyllama_1_1b() -> Value {
    json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 2048, "intermediate_size": 5632,
           "num_hidden_layers": 22, "num_attention_heads": 32, "num_key_value_heads": 4, "hidden_act": "silu",
           "rms_norm_eps": 1e-5, "vocab_size": 32000, "max_position_embeddings": 2048, "rope_scaling": null,
           "tie_word_embeddings": false, "bos_token_id": 1, "eos_token_id": 2}) // rope_theta absent: unverified (10000 either way)
}

fn llama32_1b() -> Value {
    json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 2048, "intermediate_size": 8192,
           "num_hidden_layers": 16, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 64,
           "hidden_act": "silu", "rms_norm_eps": 1e-5, "vocab_size": 128256, "rope_theta": 500000.0,
           "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                            "original_max_position_embeddings": 8192, "rope_type": "llama3"},
           "max_position_embeddings": 131072, "tie_word_embeddings": true, "attention_bias": false, "mlp_bias": false,
           "bos_token_id": 128000, "eos_token_id": 128001})
}

fn llama2_7b() -> Value {
    json!({"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 4096, "intermediate_size": 11008,
           "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 32, "hidden_act": "silu",
           "rms_norm_eps": 1e-5, "vocab_size": 32000, "max_position_embeddings": 4096, "rope_scaling": null,
           "tie_word_embeddings": false, "bos_token_id": 1, "eos_token_id": 2, "pretraining_tp": 1})
}

#[test]
fn smollm2_135m_config() {
    let d = dc(&mlx(smollm2_135m())).unwrap();
    assert_eq!(d.family, Family::Llama);
    assert_eq!((d.hidden, d.heads, d.kv_heads, d.head_dim, d.grp(), d.rope_dims), (576, 9, 3, 64, 3, 64));
    assert!(!d.head_dim_explicit && d.tie_word_embeddings && !d.qk_norm && !d.attention_bias && !d.o_bias && !d.mlp_bias);
    assert_eq!((d.rope_theta, d.rope_scale), (100000.0, RopeScaling::None));
    assert_eq!((d.q_dim(), d.kv_dim(), d.eps), (576, 192, 1e-5));
    assert!(blockers(&mlx(smollm2_135m())).is_empty());
}

#[test]
fn tinyllama_config() {
    let d = dc(&mlx(tinyllama_1_1b())).unwrap();
    assert_eq!((d.head_dim, d.grp(), d.rope_theta, d.layers, d.inter), (64, 8, 10000.0, 22, 5632));
    assert!(!d.tie_word_embeddings);
    assert_eq!((d.bos, d.eos.clone()), (Some(1), vec![2]));
}

#[test]
fn llama32_1b_config() {
    let d = dc(&mlx(llama32_1b())).unwrap();
    assert_eq!((d.head_dim, d.grp(), d.tie_word_embeddings, d.head_dim_explicit), (64, 4, true, true));
    assert_eq!(d.rope_scale, RopeScaling::Llama3 { factor: 32.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max_pos: 8192 });
    assert_eq!(d.rope_scaling, None);
    // eos merged from generation_config (Instruct style list)
    let d = config::dense_config_with_gen(&mlx(llama32_1b()), Some(&json!({"eos_token_id": [128001, 128008, 128009]}))).unwrap();
    assert_eq!(d.eos, vec![128001, 128008, 128009]);
    // same from a folder
    let dir = tmpdir("gen");
    std::fs::write(dir.join("generation_config.json"), r#"{"bos_token_id": 128000, "eos_token_id": [128001, 128009]}"#).unwrap();
    let d = dense_config(&mlx(edit(llama32_1b(), |v| { v["eos_token_id"] = json!([128001, 128008]); })), &dir).unwrap();
    assert_eq!(d.eos, vec![128001, 128008, 128009]);
    let _ = std::fs::remove_dir_all(&dir);
    // newer rope_parameters form
    let v = edit(llama32_1b(), |v| {
        let o = v.as_object_mut().unwrap();
        let rs = o.remove("rope_scaling").unwrap();
        o.remove("rope_theta");
        let mut rp = rs.clone();
        rp["rope_theta"] = json!(500000.0);
        o.insert("rope_parameters".into(), rp);
    });
    let d = dc(&mlx(v)).unwrap();
    assert_eq!((d.rope_theta, matches!(d.rope_scale, RopeScaling::Llama3 { .. })), (500000.0, true));
}

#[test]
fn llama2_7b_config() {
    let d = dc(&mlx(llama2_7b())).unwrap();
    assert_eq!((d.head_dim, d.grp(), d.attn_in_rows(), d.vocab), (128, 1, 3 * 4096, 32000));
    // HF original (no quantization block): passes dense_config, named by blockers (the loader refuses it)
    dc(&llama2_7b()).unwrap();
    let b = blockers(&llama2_7b());
    assert!(b.len() == 1 && b[0].contains("no MLX quantization block"), "{:?}", b);
    let e = dc(&edit(llama2_7b(), |v| v["quantization_config"] = json!({"quant_method": "gptq", "bits": 4, "group_size": 128}))).unwrap_err();
    assert!(e.contains("gptq"), "{}", e);
}

#[test]
fn bias_mapping() {
    let d = dc(&mlx(edit(llama2_7b(), |v| { v["attention_bias"] = true.into(); v["mlp_bias"] = true.into(); }))).unwrap();
    assert!(d.attention_bias && d.o_bias && d.mlp_bias);
}

#[test]
fn refusals() {
    let e = dc(&mlx(edit(llama32_1b(), |v| v["rope_scaling"] = json!({"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768})))).unwrap_err();
    assert!(e.contains("rope_scaling yarn factor 4 original_max_position_embeddings 32768"), "{}", e);
    assert!(blockers(&mlx(edit(llama2_7b(), |v| v["rope_scaling"] = json!({"type": "dynamic", "factor": 2.0}))))[0].contains("dynamic"));
    let dense_only = |v: Value| {
        let e = dc(&v).unwrap_err();
        assert!(e.contains("dense models only"), "{}", e);
        assert!(blockers(&v)[0].contains("dense models only"));
    };
    dense_only(edit(llama32_1b(), |v| v["model_type"] = "llama4".into()));
    dense_only(edit(llama32_1b(), |v| v["model_type"] = "llama4_text".into()));
    dense_only(edit(llama32_1b(), |v| v["num_local_experts"] = 16.into()));
    dense_only(edit(llama32_1b(), |v| v["num_experts"] = 8.into()));
    dense_only(edit(llama32_1b(), |v| v["text_config"] = json!({"num_local_experts": 16})));
    dense_only(edit(llama32_1b(), |v| v["text_config"] = json!({"hidden_size": 2048})));
    dense_only(edit(llama32_1b(), |v| v["linear_conv_kernel_dim"] = 4.into()));
    assert!(dc(&edit(llama32_1b(), |v| v["model_type"] = "mistral".into())).unwrap_err().contains("not a Llama dense config"));
    assert!(dc(&edit(llama32_1b(), |v| v["architectures"] = json!(["SparseLlamaForCausalLM"]))).unwrap_err().contains("LlamaForCausalLM"));
    assert!(dc(&edit(llama32_1b(), |v| { v.as_object_mut().unwrap().remove("eos_token_id"); })).unwrap_err().contains("eos"));
    assert!(dc(&mlx(edit(llama2_7b(), |v| v["rope_traditional"] = true.into()))).unwrap_err().contains("rope_traditional"));
    assert!(blockers(&mlx(edit(llama2_7b(), |v| v["rope_traditional"] = true.into())))[0].contains("rope_traditional"));
    // engine limits come through validate()
    assert!(dc(&mlx(edit(llama2_7b(), |v| v["head_dim"] = 96.into()))).unwrap_err().contains("head_dim must be 64, 128 or 256"));
    assert!(dc(&mlx(edit(llama2_7b(), |v| v["hidden_act"] = "relu".into()))).unwrap_err().contains("hidden_act"));
    assert!(dc(&edit(llama2_7b(), |v| v["quantization"] = json!({"group_size": 32, "bits": 3}))).unwrap_err().contains("group_size 32 bits 3"));
    assert!(dc(&edit(llama2_7b(), |v| v["layer_types"] = json!(["sliding_attention"]))).unwrap_err().contains("full attention"));
}

#[test]
fn qwen_parser_still_refuses_llama() {
    let e = DenseConfig::from_json(&llama32_1b(), None).unwrap_err();
    assert!(e.contains("not a Qwen dense model"), "{}", e);
}

// ---------- chat templates ----------

fn msgs(m: &[(&str, &str)]) -> Vec<(String, String)> { m.iter().map(|(r, c)| (r.to_string(), c.to_string())).collect() }
fn all_tokens(_: &str) -> bool { true }

/// Meta Llama 3 Instruct template (as published for Meta-Llama-3-8B-Instruct; unverified against the file).
const LLAMA3_TPL: &str = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}";

/// Abridged Llama 3.1 / 3.2 template: the lines the kind reads (unverified against the file).
const LLAMA31_TPL: &str = "{{- bos_token }}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}";

/// SmolLM2 Instruct template (unverified against the file).
const SMOLLM2_TPL: &str = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

#[test]
fn llama3_strings() {
    let k = kind_from_text(Some(LLAMA3_TPL), &all_tokens).unwrap();
    assert_eq!(k, TemplateKind::Llama3);
    let m = msgs(&[("system", "You are helpful."), ("user", " Hi \n")]);
    let want = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>\
                <|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    assert_eq!(k.render(&m, true).unwrap(), want);
    assert_eq!(chat_prompt(&m, true).unwrap(), want);
    assert_eq!(chat_prompt(&msgs(&[("user", "Hi"), ("assistant", "Hello!")]), false).unwrap(),
               "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>\
                <|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>");
    assert!(chat_prompt(&msgs(&[("tool", "{}")]), true).unwrap_err().contains("role \"tool\""));
    assert!(chat_prompt(&[], true).is_err());
}

#[test]
fn llama3_dated_strings() {
    let k = kind_from_text(Some(LLAMA31_TPL), &all_tokens).unwrap();
    assert_eq!(k, TemplateKind::Llama3Dated { knowledge: "December 2023".into(), date: "26 Jul 2024".into() });
    assert_eq!(k.render(&msgs(&[("user", "Hi")]), true).unwrap(),
               "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n\
                Today Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>\
                <|start_header_id|>assistant<|end_header_id|>\n\n");
    assert_eq!(k.render(&msgs(&[("system", " Be brief. "), ("user", "Hi")]), false).unwrap(),
               "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n\
                Today Date: 26 Jul 2024\n\nBe brief.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>");
}

#[test]
fn chatml_strings() {
    let k = kind_from_text(Some(SMOLLM2_TPL), &all_tokens).unwrap();
    let sys = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";
    assert_eq!(k, TemplateKind::ChatMl { default_system: Some(sys.into()) });
    assert_eq!(k.render(&msgs(&[("user", "Hi ")]), true).unwrap(),
               format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nHi <|im_end|>\n<|im_start|>assistant\n", sys));
    assert_eq!(k.render(&msgs(&[("system", "S"), ("user", "Hi")]), false).unwrap(),
               "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n");
    // plain ChatML (no default system)
    let plain = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}";
    assert_eq!(kind_from_text(Some(plain), &all_tokens).unwrap(), TemplateKind::ChatMl { default_system: None });
}

#[test]
fn template_refusals() {
    let e = |t: Option<&str>| kind_from_text(t, &all_tokens).unwrap_err();
    assert!(e(None).contains("none in this model folder"));
    assert!(e(Some("{{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}")).contains("[INST]"));
    assert!(e(Some("{{ '<|user|>\n' + message['content'] + eos_token }}")).contains("Zephyr"));
    assert!(e(Some("{{ message['content'] }}")).contains("unknown template"));
    assert!(e(Some("{%- if tools %}<|im_start|>system<|im_end|>{% endif %}")).contains("tool"));
    let no_hdr = kind_from_text(Some(LLAMA3_TPL), &|t| t != "<|eot_id|>").unwrap_err();
    assert!(no_hdr.contains("tokenizer lacks <|eot_id|>"), "{}", no_hdr);
}

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sd_llama_test_{}_{}", tag, std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn template_from_folder() {
    // tokenizer_config.json chat_template (string) + tokens in tokenizer.json
    let d = tmpdir("tc");
    std::fs::write(d.join("tokenizer_config.json"), json!({"chat_template": SMOLLM2_TPL, "bos_token": "<|im_start|>"}).to_string()).unwrap();
    std::fs::write(d.join("tokenizer.json"), json!({"added_tokens": [{"id": 1, "content": "<|im_start|>"}, {"id": 2, "content": "<|im_end|>"}]}).to_string()).unwrap();
    assert!(matches!(template_kind(&d).unwrap(), TemplateKind::ChatMl { default_system: Some(_) }));
    assert!(chat_prompt_dir(&d, &msgs(&[("user", "x")]), true).unwrap().ends_with("<|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n"));
    // chat_template.jinja wins; tokenizer lacks the Llama 3 tokens -> named error
    std::fs::write(d.join("chat_template.jinja"), LLAMA3_TPL).unwrap();
    assert!(template_kind(&d).unwrap_err().contains("tokenizer lacks"));
    // list form
    let d2 = tmpdir("list");
    std::fs::write(d2.join("tokenizer_config.json"), json!({"chat_template": [{"name": "tool_use", "template": "x"},
        {"name": "default", "template": LLAMA3_TPL}], "added_tokens_decoder": {"128000": {"content": "<|begin_of_text|>"},
        "128006": {"content": "<|start_header_id|>"}, "128007": {"content": "<|end_header_id|>"}, "128009": {"content": "<|eot_id|>"}}}).to_string()).unwrap();
    assert_eq!(template_kind(&d2).unwrap(), TemplateKind::Llama3);
    let _ = std::fs::remove_dir_all(&d);
    let _ = std::fs::remove_dir_all(&d2);
    assert!(template_kind(&tmpdir("empty")).unwrap_err().contains("no tokenizer.json"));
    let _ = std::fs::remove_dir_all(tmpdir("empty"));
}
