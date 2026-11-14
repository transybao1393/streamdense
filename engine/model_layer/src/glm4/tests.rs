//! Config fixtures and chat strings: no model, no Metal, no weights. Fixtures are the real files of
//! mlx-community/GLM-4-9B-0414-4bit and GLM-Z1-9B-0414-4bit (fetched 2026-11-15, metadata only); the expected chat
//! strings are HF `apply_chat_template` renders of their templates (generated below the hand-written tests).
use std::path::{Path, PathBuf};

/// One chat-render case: messages, add_generation_prompt, expected render or error.
type Render = (&'static [(&'static str, &'static str)], bool, Result<&'static str, &'static str>);

use dense_engine::config::{Activation, Block, Family, LayerKind, Mlp, NormKind, RopeScaling};
use serde_json::{json, Value};

use super::config::map_config;
use super::template::{kind_from_text, TemplateKind};
use super::*;

/// mlx-community/GLM-4-9B-0414-4bit config.json, as on disk (GLM-Z1-9B-0414-4bit has the identical file).
fn glm4_9b() -> Value {
    json!({
        "architectures": ["Glm4ForCausalLM"], "attention_bias": true, "attention_dropout": 0.0,
        "eos_token_id": [151329, 151336, 151338], "head_dim": 128, "hidden_act": "silu", "hidden_size": 4096,
        "initializer_range": 0.02, "intermediate_size": 13696, "max_position_embeddings": 32768, "model_type": "glm4",
        "num_attention_heads": 32, "num_hidden_layers": 40, "num_key_value_heads": 2, "pad_token_id": 151329,
        "partial_rotary_factor": 0.5, "quantization": {"group_size": 64, "bits": 4},
        "quantization_config": {"group_size": 64, "bits": 4}, "rms_norm_eps": 1e-05, "rope_theta": 10000.0,
        "tie_word_embeddings": false, "torch_dtype": "bfloat16", "transformers_version": "4.52.0.dev0",
        "use_cache": true, "vocab_size": 151552
    })
}

fn no_dir() -> &'static Path { Path::new("/nonexistent-glm4-test-dir") }
fn edit(mut v: Value, f: impl FnOnce(&mut Value)) -> Value { f(&mut v); v }
fn remove(v: Value, k: &str) -> Value { edit(v, |v| { v.as_object_mut().unwrap().remove(k); }) }
fn map(v: &Value) -> DenseConfig { map_config(v, None).unwrap().0 }
fn err(v: &Value) -> String { map_config(v, None).unwrap_err() }
fn msgs(m: &[(&str, &str)]) -> Vec<(String, String)> { m.iter().map(|(r, c)| (r.to_string(), c.to_string())).collect() }

/// Blockers the engine names today for GLM-4 and that go away as the engine features land (SPEC.md): every blocker of
/// the real config must be one of these.
const PENDING: [&str; 3] = ["partial rotary", "traditional (interleaved) RoPE", "fused qkv / gate_up"];
fn only_pending(b: &[String]) -> bool { b.iter().all(|m| PENDING.iter().any(|p| m.contains(p))) }

#[test]
fn model_types() {
    assert_eq!(MODEL_TYPES, &["glm4"]);
}

#[test]
fn glm4_9b_mapping() {
    let d = map(&glm4_9b());
    assert_eq!((d.family, d.model_type.as_str()), (Family::Glm4, "glm4"));
    assert_eq!((d.hidden, d.inter, d.layers, d.heads, d.kv_heads, d.head_dim, d.vocab), (4096, 13696, 40, 32, 2, 128, 151552));
    assert_eq!((d.grp(), d.q_dim(), d.kv_dim()), (16, 4096, 256));
    assert_eq!((d.rope_dims, d.rope_theta, d.rope_traditional, d.rope_scale, d.rope_scaling.clone()),
               (64, 10000.0, true, RopeScaling::None, None));
    assert!(d.attention_bias && !d.o_bias && !d.mlp_bias && !d.tie_word_embeddings && !d.qk_norm);
    assert!(d.four_norms && !d.norm_plus_one);
    assert_eq!((d.norm, d.block, d.mlp, d.activation.clone()), (NormKind::Rms, Block::Sequential, Mlp::Gated, Activation::Silu));
    assert_eq!(d.names.input_norm, "input_layernorm");
    assert_eq!(d.names.post_attn_norm, "post_self_attn_layernorm");
    assert_eq!(d.names.pre_ffn_norm, "post_attention_layernorm");
    assert_eq!(d.names.post_ffn_norm, "post_mlp_layernorm");
    assert_eq!(d.names.gate_up.as_deref(), Some("mlp.gate_up_proj"));
    assert_eq!((d.names.q.as_str(), d.names.o.as_str(), d.names.down.as_str(), d.names.lm_head.as_str()),
               ("self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj", "lm_head"));
    assert!(d.names.qkv.is_none() && d.no_rope_layers.is_empty() && d.rope_su.is_none());
    assert!(d.kinds.len() == 40 && d.kinds.iter().all(|k| *k == LayerKind::Full) && d.sliding_window.is_none());
    assert_eq!((d.eps, d.max_pos, d.attn_scale, d.embed_scale, d.logits_scaling), (1e-5, 32768, None, 1.0, 1.0));
    assert_eq!((d.bos, d.eos.clone(), d.quant), (None, vec![151329, 151336, 151338], Some((64, 4))));
}

#[test]
fn glm4_9b_blockers_are_engine_features_only() {
    let b = blockers(&glm4_9b());
    assert!(only_pending(&b), "{:?}", b);
    match dense_config(&glm4_9b(), no_dir()) {
        Ok(d) => assert!(b.is_empty() && d.family == Family::Glm4),
        Err(e) => assert!(PENDING.iter().any(|p| e.contains(p)) && !b.is_empty(), "{}", e),
    }
}

#[test]
fn defaults_and_alternatives() {
    // mlx_lm defaults: rope_traditional true, max_position_embeddings 32768
    let d = map(&remove(glm4_9b(), "max_position_embeddings"));
    assert_eq!(d.max_pos, 32768);
    let d = map(&edit(glm4_9b(), |v| v["rope_traditional"] = json!(false)));
    assert!(!d.rope_traditional);
    // rope_parameters form (transformers >= 4.56)
    let v = edit(remove(remove(glm4_9b(), "rope_theta"), "partial_rotary_factor"),
                 |v| v["rope_parameters"] = json!({"rope_theta": 10000.0, "partial_rotary_factor": 0.5, "rope_type": "default"}));
    let d = map(&v);
    assert_eq!((d.rope_theta, d.rope_dims, d.rope_scaling.clone()), (10000.0, 64, None));
    assert!(err(&edit(glm4_9b(), |v| v["rope_parameters"] = json!({"rope_theta": 500.0}))).contains("differ"));
    // generation_config eos merged after config eos, no duplicates
    let (d, _) = map_config(&glm4_9b(), Some(&json!({"eos_token_id": [151336, 151329, 7]}))).unwrap();
    assert_eq!(d.eos, vec![151329, 151336, 151338, 7]);
    // attention_bias false: no q/k/v bias
    assert!(!map(&edit(glm4_9b(), |v| v["attention_bias"] = json!(false))).attention_bias);
}

#[test]
fn refusals() {
    for k in ["hidden_size", "head_dim", "partial_rotary_factor", "rope_theta", "attention_bias", "rms_norm_eps", "num_key_value_heads"] {
        assert!(err(&remove(glm4_9b(), k)).contains(k), "{}", k);
    }
    for (mt, why) in [("glm4_moe", "mixture-of-experts"), ("glm4v", "multimodal"), ("chatglm", "ChatGLM"), ("llama", "not GLM-4")] {
        assert!(err(&edit(glm4_9b(), |v| v["model_type"] = json!(mt))).contains(why), "{}", mt);
    }
    assert!(err(&edit(glm4_9b(), |v| v["n_routed_experts"] = json!(128))).contains("mixture-of-experts"));
    assert!(err(&edit(glm4_9b(), |v| v["architectures"] = json!(["Glm4vForConditionalGeneration"]))).contains("architecture"));
    assert!(err(&edit(glm4_9b(), |v| v["text_config"] = json!({}))).contains("multimodal"));
    assert!(err(&edit(glm4_9b(), |v| v["quantization"] = json!({"quant_method": "gptq", "bits": 4}))).contains("gptq"));
    let has = |v: &Value, s: &str| blockers(v).iter().any(|m| m.contains(s));
    assert!(has(&edit(glm4_9b(), |v| v["rope_scaling"] = json!({"rope_type": "yarn", "factor": 4.0})), "yarn"));
    assert!(has(&edit(glm4_9b(), |v| v["tie_word_embeddings"] = json!(true)), "tie_word_embeddings"));
    assert!(has(&edit(glm4_9b(), |v| v["hidden_act"] = json!("gelu")), "hidden_act"));
    assert!(has(&edit(glm4_9b(), |v| v["quantization"] = json!({"group_size": 64, "bits": 3})), "bits 3"));
    assert!(has(&remove(remove(glm4_9b(), "quantization"), "quantization_config"), "no MLX quantization"));
    assert!(has(&edit(glm4_9b(), |v| v["quantization"]["model.layers.0.mlp.down_proj"] = json!({"bits": 8, "group_size": 64})), "per-layer"));
    assert!(dense_config(&remove(glm4_9b(), "eos_token_id"), no_dir()).unwrap_err().contains("eos"));
}

// ---------- chat ----------

/// Every real render: equal, or both refuse.
fn check(kind: &TemplateKind, cases: &[Render]) {
    for (m, agp, want) in cases {
        let got = kind.render(&msgs(m), *agp);
        match want {
            Ok(s) => assert_eq!(got.as_deref(), Ok(*s), "{:?} agp {}", m, agp),
            Err(e) => assert!(got.is_err(), "template raised {:?}, port rendered {:?}", e, got),
        }
    }
}

#[test]
fn glm4_strings_equal_hf_renders() {
    check(&TemplateKind::Glm4, GLM4_RENDERS);
    // chat_prompt = GLM-4-0414 template
    assert_eq!(chat_prompt(&msgs(&[("user", "Hello!")]), true).unwrap(), "[gMASK]<sop><|user|>\nHello!<|assistant|>");
}

#[test]
fn z1_strings_equal_hf_renders() {
    check(&TemplateKind::Glm4Z1, Z1_RENDERS);
}

#[test]
fn chat_refusals() {
    assert!(chat_prompt(&[], true).unwrap_err().contains("no messages"));
    assert!(chat_prompt(&msgs(&[("user", "a"), ("tool", "r")]), true).unwrap_err().contains("\"tool\""));
}

#[test]
fn template_kinds_from_real_templates() {
    let all = |_: &str| true;
    assert_eq!(kind_from_text(Some(GLM4_TPL), &all).unwrap(), TemplateKind::Glm4);
    assert_eq!(kind_from_text(Some(GLM4_32B_TPL), &all).unwrap(), TemplateKind::Glm4);
    assert_eq!(kind_from_text(Some(Z1_TPL), &all).unwrap(), TemplateKind::Glm4Z1);
    assert!(kind_from_text(None, &all).unwrap_err().contains("base model"));
    assert!(kind_from_text(Some("{{ messages }}"), &all).unwrap_err().contains("not ported"));
    let think = GLM4_TPL.replace("add_generation_prompt %}<|assistant|>{%", "add_generation_prompt %}<|assistant|>\n<think>{%");
    assert!(kind_from_text(Some(&think), &all).unwrap_err().contains("generation prompt"));
    assert!(kind_from_text(Some(GLM4_TPL), &|t: &str| t != "<sop>").unwrap_err().contains("lacks <sop>"));
}

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sd-glm4-test-{}-{}", tag, std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    d
}

#[test]
fn template_from_folder() {
    let d = tmpdir("z1");
    let added: Vec<Value> = ["[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"]
        .iter().enumerate().map(|(i, t)| json!({"id": 151331 + i, "content": t})).collect();
    std::fs::write(d.join("tokenizer.json"), json!({"added_tokens": added}).to_string()).unwrap();
    std::fs::write(d.join("tokenizer_config.json"), json!({"chat_template": Z1_TPL}).to_string()).unwrap();
    assert_eq!(template_kind(&d).unwrap(), TemplateKind::Glm4Z1);
    let m = msgs(&[("user", "q"), ("assistant", "<think>t</think> ans"), ("user", "q2")]);
    assert_eq!(chat_prompt_dir(&d, &m, true).unwrap(), "[gMASK]<sop><|user|>\nq<|assistant|>\nans<|user|>\nq2<|assistant|>");
    // chat_template.jinja wins over tokenizer_config.json
    std::fs::write(d.join("chat_template.jinja"), GLM4_TPL).unwrap();
    assert_eq!(template_kind(&d).unwrap(), TemplateKind::Glm4);
    let _ = std::fs::remove_dir_all(&d);
    assert!(template_kind(&tmpdir("empty")).unwrap_err().contains("no tokenizer.json"));
}

/// HF `apply_chat_template` renders of the real template (mlx-community/GLM-4-9B-0414-4bit), transformers 5.21, jinja2 3.1.6.
const GLM4_RENDERS: &[Render] = &[
    (&[("user", "Hello!")], true, Ok("[gMASK]<sop><|user|>\nHello!<|assistant|>")),
    (&[("user", "Hello!")], false, Ok("[gMASK]<sop><|user|>\nHello!")),
    (&[("system", "You are terse."), ("user", "Hi")], true, Ok("[gMASK]<sop><|system|>\nYou are terse.<|user|>\nHi<|assistant|>")),
    (&[("system", "You are terse."), ("user", "Hi")], false, Ok("[gMASK]<sop><|system|>\nYou are terse.<|user|>\nHi")),
    (&[("user", "  What is 2+2?  "), ("assistant", " 4 "), ("user", "Thanks")], true, Ok("[gMASK]<sop><|user|>\n  What is 2+2?  <|assistant|>\n 4 <|user|>\nThanks<|assistant|>")),
    (&[("user", "  What is 2+2?  "), ("assistant", " 4 "), ("user", "Thanks")], false, Ok("[gMASK]<sop><|user|>\n  What is 2+2?  <|assistant|>\n 4 <|user|>\nThanks")),
    (&[("system", "S"), ("user", "a"), ("assistant", "b"), ("user", "c")], true, Ok("[gMASK]<sop><|system|>\nS<|user|>\na<|assistant|>\nb<|user|>\nc<|assistant|>")),
    (&[("system", "S"), ("user", "a"), ("assistant", "b"), ("user", "c")], false, Ok("[gMASK]<sop><|system|>\nS<|user|>\na<|assistant|>\nb<|user|>\nc")),
    (&[("user", "a"), ("user", "b")], true, Ok("[gMASK]<sop><|user|>\na<|user|>\nb<|assistant|>")),
    (&[("user", "a"), ("user", "b")], false, Ok("[gMASK]<sop><|user|>\na<|user|>\nb")),
    (&[("user", "x"), ("system", "late"), ("user", "y")], true, Ok("[gMASK]<sop><|system|>\nlate<|user|>\nx<|user|>\ny<|assistant|>")),
    (&[("user", "x"), ("system", "late"), ("user", "y")], false, Ok("[gMASK]<sop><|system|>\nlate<|user|>\nx<|user|>\ny")),
    (&[("user", "q"), ("assistant", "<think>t</think> ans"), ("user", "q2")], true, Ok("[gMASK]<sop><|user|>\nq<|assistant|>\n<think>t</think> ans<|user|>\nq2<|assistant|>")),
    (&[("user", "q"), ("assistant", "<think>t</think> ans"), ("user", "q2")], false, Ok("[gMASK]<sop><|user|>\nq<|assistant|>\n<think>t</think> ans<|user|>\nq2")),
    (&[("system", "A"), ("user", "u"), ("system", "B"), ("user", "v")], true, Ok("[gMASK]<sop><|system|>\nA<|system|>\nB<|user|>\nu<|user|>\nv<|assistant|>")),
    (&[("system", "A"), ("user", "u"), ("system", "B"), ("user", "v")], false, Ok("[gMASK]<sop><|system|>\nA<|system|>\nB<|user|>\nu<|user|>\nv")),
    (&[("user", "line1\nline2\n"), ("assistant", "\n\nout\n"), ("user", "tab\there")], true, Ok("[gMASK]<sop><|user|>\nline1\nline2\n<|assistant|>\n\n\nout\n<|user|>\ntab\there<|assistant|>")),
    (&[("user", "line1\nline2\n"), ("assistant", "\n\nout\n"), ("user", "tab\there")], false, Ok("[gMASK]<sop><|user|>\nline1\nline2\n<|assistant|>\n\n\nout\n<|user|>\ntab\there")),
    (&[("system", "only")], true, Ok("[gMASK]<sop><|system|>\nonly<|assistant|>")),
    (&[("system", "only")], false, Ok("[gMASK]<sop><|system|>\nonly")),
    (&[("user", "call it"), ("observation", " {\"r\": 1} "), ("user", "ok")], true, Ok("[gMASK]<sop><|user|>\ncall it<|observation|>\n {\"r\": 1} <|user|>\nok<|assistant|>")),
    (&[("user", "call it"), ("observation", " {\"r\": 1} "), ("user", "ok")], false, Ok("[gMASK]<sop><|user|>\ncall it<|observation|>\n {\"r\": 1} <|user|>\nok")),
];
/// HF `apply_chat_template` renders of the real template (mlx-community/GLM-Z1-9B-0414-4bit), transformers 5.21, jinja2 3.1.6.
const Z1_RENDERS: &[Render] = &[
    (&[("user", "Hello!")], true, Ok("[gMASK]<sop><|user|>\nHello!<|assistant|>")),
    (&[("user", "Hello!")], false, Ok("[gMASK]<sop><|user|>\nHello!")),
    (&[("system", "You are terse."), ("user", "Hi")], true, Ok("[gMASK]<sop><|system|>\nYou are terse.<|user|>\nHi<|assistant|>")),
    (&[("system", "You are terse."), ("user", "Hi")], false, Ok("[gMASK]<sop><|system|>\nYou are terse.<|user|>\nHi")),
    (&[("user", "  What is 2+2?  "), ("assistant", " 4 "), ("user", "Thanks")], true, Ok("[gMASK]<sop><|user|>\nWhat is 2+2?<|assistant|>\n4<|user|>\nThanks<|assistant|>")),
    (&[("user", "  What is 2+2?  "), ("assistant", " 4 "), ("user", "Thanks")], false, Ok("[gMASK]<sop><|user|>\nWhat is 2+2?<|assistant|>\n4<|user|>\nThanks")),
    (&[("system", "S"), ("user", "a"), ("assistant", "b"), ("user", "c")], true, Ok("[gMASK]<sop><|system|>\nS<|user|>\na<|assistant|>\nb<|user|>\nc<|assistant|>")),
    (&[("system", "S"), ("user", "a"), ("assistant", "b"), ("user", "c")], false, Ok("[gMASK]<sop><|system|>\nS<|user|>\na<|assistant|>\nb<|user|>\nc")),
    (&[("user", "a"), ("user", "b")], true, Ok("[gMASK]<sop><|user|>\na<|user|>\nb<|assistant|>")),
    (&[("user", "a"), ("user", "b")], false, Ok("[gMASK]<sop><|user|>\na<|user|>\nb")),
    (&[("user", "x"), ("system", "late"), ("user", "y")], true, Ok("[gMASK]<sop><|system|>\nlate<|user|>\nx<|user|>\ny<|assistant|>")),
    (&[("user", "x"), ("system", "late"), ("user", "y")], false, Ok("[gMASK]<sop><|system|>\nlate<|user|>\nx<|user|>\ny")),
    (&[("user", "q"), ("assistant", "<think>t</think> ans"), ("user", "q2")], true, Ok("[gMASK]<sop><|user|>\nq<|assistant|>\nans<|user|>\nq2<|assistant|>")),
    (&[("user", "q"), ("assistant", "<think>t</think> ans"), ("user", "q2")], false, Ok("[gMASK]<sop><|user|>\nq<|assistant|>\nans<|user|>\nq2")),
    (&[("system", "A"), ("user", "u"), ("system", "B"), ("user", "v")], true, Ok("[gMASK]<sop><|system|>\nA<|system|>\nB<|user|>\nu<|user|>\nv<|assistant|>")),
    (&[("system", "A"), ("user", "u"), ("system", "B"), ("user", "v")], false, Ok("[gMASK]<sop><|system|>\nA<|system|>\nB<|user|>\nu<|user|>\nv")),
    (&[("user", "line1\nline2\n"), ("assistant", "\n\nout\n"), ("user", "tab\there")], true, Ok("[gMASK]<sop><|user|>\nline1\nline2<|assistant|>\nout<|user|>\ntab\there<|assistant|>")),
    (&[("user", "line1\nline2\n"), ("assistant", "\n\nout\n"), ("user", "tab\there")], false, Ok("[gMASK]<sop><|user|>\nline1\nline2<|assistant|>\nout<|user|>\ntab\there")),
    (&[("system", "only")], true, Ok("[gMASK]<sop><|system|>\nonly<|assistant|>")),
    (&[("system", "only")], false, Ok("[gMASK]<sop><|system|>\nonly")),
    (&[("user", "call it"), ("observation", " {\"r\": 1} "), ("user", "ok")], true, Ok("[gMASK]<sop><|user|>\ncall it<|observation|>\n{\"r\": 1}<|user|>\nok<|assistant|>")),
    (&[("user", "call it"), ("observation", " {\"r\": 1} "), ("user", "ok")], false, Ok("[gMASK]<sop><|user|>\ncall it<|observation|>\n{\"r\": 1}<|user|>\nok")),
];
/// chat_template of mlx-community/GLM-4-9B-0414-4bit tokenizer_config.json, verbatim.
const GLM4_TPL: &str = r##"[gMASK]<sop>{%- if tools -%}<|system|>
# 可用工具
{% for tool in tools %}{%- set function = tool.function if tool.get("function") else tool %}

## {{ function.name }}

{{ function | tojson(indent=4, ensure_ascii=False) }}
在调用上述函数时，请使用 Json 格式表示调用的参数。{%- endfor %}{%- endif -%}{%- for msg in messages %}{%- if msg.role == 'system' %}<|system|>
{{ msg.content }}{%- endif %}{%- endfor %}{%- for message in messages if message.role != 'system' %}{%- set role = message['role'] %}{%- set content = message['content'] %}{%- set meta = message.get("metadata", "") %}{%- if role == 'user' %}<|user|>
{{ content }}{%- elif role == 'assistant' and not meta %}<|assistant|>
{{ content }}{%- elif role == 'assistant' and meta %}<|assistant|>{{ meta }} 
{{ content }}{%- elif role == 'observation' %}<|observation|>
{{ content }}{%- endif %}{%- endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"##;
/// chat_template of mlx-community/GLM-Z1-9B-0414-4bit tokenizer_config.json, verbatim.
const Z1_TPL: &str = r##"[gMASK]<sop>{%- if tools -%}<|system|>你是一个名为 ChatGLM 的人工智能助手。你是基于智谱 AI 公司训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。

# 可用工具

{% for tool in tools %}{%- set function = tool.function if tool.get("function") else tool %}

## {{ function.name }}

{{ function | tojson(indent=4, ensure_ascii=False) }}
在调用上述函数时，请使用 Json 格式表示调用的参数。{%- endfor %}{%- endif -%}{%- for msg in messages %}{%- if msg.role == 'system' %}<|system|>
{{ msg.content }}{%- endif %}{%- endfor %}{%- for message in messages if message.role != 'system' %}{%- set role = message['role'] %}{%- set content = message['content'] %}{%- set visible = content.split('</think>')[-1].strip() %}{%- set meta = message.get("metadata", "") %}{%- if role == 'user' %}<|user|>
{{ visible }}{%- elif role == 'assistant' and not meta %}<|assistant|>
{{ visible }}{%- elif role == 'assistant' and meta %}<|assistant|>{{ meta }} 
{{ visible }}{%- elif role == 'observation' %}<|observation|>
{{ visible }}{%- endif %}{%- endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"##;
/// chat_template of mlx-community/GLM-4-32B-0414-4bit tokenizer_config.json, verbatim.
const GLM4_32B_TPL: &str = r##"[gMASK]<sop>
{%- if tools -%}
<|system|>
# 可用工具
{% for tool in tools %}
    {%- set function = tool.function if tool.get("function") else tool %}

## {{ function.name }}

{{ function | tojson(indent=4, ensure_ascii=False) }}
在调用上述函数时，请使用 Json 格式表示调用的参数。
{%- endfor %}
{%- endif -%}

{%- for msg in messages %}
    {%- if msg.role == 'system' %}
<|system|>
{{ msg.content }}
    {%- endif %}
{%- endfor %}

{%- for message in messages if message.role != 'system' %}
    {%- set role = message['role'] %}
    {%- set content = message['content'] %}
    {%- set meta = message.get("metadata", "") %}

    {%- if role == 'user' %}
<|user|>
{{ content }}
    {%- elif role == 'assistant' and not meta %}
<|assistant|>
{{ content }}
    {%- elif role == 'assistant' and meta %}
<|assistant|>{{ meta }}
{{ content }}
    {%- elif role == 'observation' %}
<|observation|>
{{ content }}
    {%- endif %}
{%- endfor %}
{% if add_generation_prompt %}<|assistant|>{% endif %}"##;
