//! Qwen chat strings against fixed expected strings, and the moved config helpers. The local-model tests read only
//! the template / tokenizer text files and skip when the folder is missing: `<repo>/models/Qwen2.5-Coder-7B-Instruct-4bit`
//! (Qwen2.5 template in tokenizer_config.json) and `<repo>/models/qwen38_27b_mlx4` (chat_template.jinja).
use super::*;
use std::path::PathBuf;

fn m(v: &[(&str, &str)]) -> Vec<(String, String)> { v.iter().map(|(r, c)| (r.to_string(), c.to_string())).collect() }

fn local(name: &str) -> Option<PathBuf> {
    let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models").join(name);
    if d.join("tokenizer_config.json").exists() { Some(d) } else { println!("  SKIP: {} missing", d.display()); None }
}

const QWEN25_SYS: &str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
const XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";

/// The no-tools system part and the generation block of the Qwen2.5 template, as in
/// models/Qwen2.5-Coder-7B-Instruct-4bit/tokenizer_config.json (Jinja `\n` escapes kept as two characters).
const QWEN25_TPL: &str = r#"{%- if tools %}{{- '<|im_start|>system\n' }}{%- else %}{%- if messages[0]['role'] == 'system' %}{{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}{%- else %}{{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}{%- endif %}{%- endif %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"#;

/// The generation block of the Qwen3 (hybrid thinking, dense) template of Qwen/Qwen3-8B. Not in a local folder: written
/// from the published template text; it only selects the kind (the rendered text comes from the ChatML renderer).
const QWEN3_TPL: &str = r#"{%- if messages[0].role == 'system' %}{{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}{%- endif %}{%- set content = content.split('</think>')[-1] %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}{%- endif %}"#;

#[test]
fn qwen25_kind_and_strings() {
    let k = kind_from_text(Some(QWEN25_TPL)).unwrap();
    assert_eq!(k, TemplateKind::ChatMl { default_system: Some(QWEN25_SYS.into()), assistant_history: true });
    assert_eq!(k.render(&m(&[("user", "Hi")]), true).unwrap(),
               format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", QWEN25_SYS));
    // a leading system message replaces the default; content is not trimmed
    assert_eq!(k.render(&m(&[("system", "S "), ("user", " Hi\n")]), true).unwrap(),
               "<|im_start|>system\nS <|im_end|>\n<|im_start|>user\n Hi\n<|im_end|>\n<|im_start|>assistant\n");
    assert_eq!(k.render(&m(&[("user", "a"), ("assistant", "b"), ("user", "c")]), false).unwrap(),
               format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\nb<|im_end|>\n<|im_start|>user\nc<|im_end|>\n", QWEN25_SYS));
    assert!(k.render(&m(&[("tool", "r")]), true).is_err());
}

#[test]
fn qwen3_kind_and_strings() {
    let k = kind_from_text(Some(QWEN3_TPL)).unwrap();
    assert_eq!(k, TemplateKind::Qwen3);
    // enable_thinking undefined (HF default): nothing after the assistant header
    assert_eq!(k.render(&m(&[("user", "Hi")]), true).unwrap(), "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    assert_eq!(k.render(&m(&[("system", "S"), ("user", "Hi")]), true).unwrap(),
               "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    assert!(k.render(&m(&[("user", "a"), ("assistant", "b"), ("user", "c")]), true).unwrap_err().contains("not ported"));
}

#[test]
fn qwen38_kind_and_strings() {
    let tpl = "<|im_start|><|im_end|>{%- if enable_thinking is undefined or enable_thinking is true %}'Reasoning effort is set to xhigh.'";
    let k = kind_from_text(Some(tpl)).unwrap();
    assert_eq!(k, TemplateKind::Qwen38);
    assert_eq!(k.render(&m(&[("user", " Hi ")]), true).unwrap(),
               format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n", XHIGH));
    assert_eq!(k.render(&m(&[("system", "S"), ("user", "Hi")]), true).unwrap(),
               format!("<|im_start|>system\n{}\n\nS<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n", XHIGH));
}

/// The chat template of mlx-community/Qwen3.5-0.8B-4bit (chat_template.jinja, copied unchanged 2026-11-14).
const QWEN35_TPL: &str = include_str!("qwen35_chat_template.jinja");

/// expected strings are HF `apply_chat_template(.., add_generation_prompt)` output (transformers, default kwargs)
/// on that template, 2026-11-14.
#[test]
fn qwen35_kind_and_strings() {
    let k = kind_from_text(Some(QWEN35_TPL)).unwrap();
    assert_eq!(k, TemplateKind::Qwen35);
    assert_eq!(k.render(&m(&[("user", "Name three primary colors.")]), true).unwrap(),
               "<|im_start|>user\nName three primary colors.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    assert_eq!(k.render(&m(&[("user", "Name three primary colors.")]), false).unwrap(),
               "<|im_start|>user\nName three primary colors.<|im_end|>\n");
    // system and user content trimmed (render_content(..)|trim)
    assert_eq!(k.render(&m(&[("system", "  Be brief. \n"), ("user", "\n Hi there \t")]), true).unwrap(),
               "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    assert_eq!(k.render(&m(&[("user", "a"), ("user", "b")]), true).unwrap(),
               "<|im_start|>user\na<|im_end|>\n<|im_start|>user\nb<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    assert!(k.render(&m(&[("user", "a"), ("assistant", "b"), ("user", "c")]), true).unwrap_err().contains("not ported"));
    assert!(k.render(&m(&[("user", "a"), ("system", "s")]), true).unwrap_err().contains("at the beginning"));
    assert!(k.render(&m(&[("system", "s")]), true).unwrap_err().contains("no user query"));
    assert!(k.render(&m(&[("user", "<tool_response>x</tool_response>")]), true).unwrap_err().contains("no user query"));
    assert!(k.render(&m(&[("tool", "r"), ("user", "a")]), true).unwrap_err().contains("not ported"));
}

#[test]
fn refusals() {
    assert!(kind_from_text(None).unwrap_err().contains("none in this model folder"));
    assert!(kind_from_text(Some("[INST]")).unwrap_err().contains("ChatML"));
    // a generation prompt that always opens a think block (e.g. Qwen3 Thinking-2507)
    let t = "<|im_start|><|im_end|>{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n' }}{%- endif %}";
    assert!(kind_from_text(Some(t)).unwrap_err().contains("not ported"));
    // ChatML template that rewrites assistant turns: assistant history refused, single user message fine
    let t = "<|im_start|><|im_end|>{%- set c = c.split('</think>')[-1] %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}";
    let k = kind_from_text(Some(t)).unwrap();
    assert_eq!(k, TemplateKind::ChatMl { default_system: None, assistant_history: false });
    assert_eq!(k.render(&m(&[("user", "Hi")]), true).unwrap(), "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    assert!(k.render(&m(&[("user", "a"), ("assistant", "b")]), false).is_err());
}

#[test]
fn local_qwen25_template() {
    let Some(d) = local("Qwen2.5-Coder-7B-Instruct-4bit") else { return };
    assert_eq!(template_kind(&d).unwrap(), TemplateKind::ChatMl { default_system: Some(QWEN25_SYS.into()), assistant_history: true });
    assert_eq!(chat_prompt(&d, &m(&[("user", "Hello, world!")]), true).unwrap(),
               format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nHello, world!<|im_end|>\n<|im_start|>assistant\n", QWEN25_SYS));
}

#[test]
fn local_qwen38_template() {
    let Some(d) = local("qwen38_27b_mlx4") else { return };
    assert_eq!(template_kind(&d).unwrap(), TemplateKind::Qwen38);
    assert_eq!(chat_prompt(&d, &m(&[("user", "Hello, world!")]), true).unwrap(),
               format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\nHello, world!<|im_end|>\n<|im_start|>assistant\n<think>\n", XHIGH));
}

#[test]
fn model_types_cover_the_entries() {
    let all: Vec<&str> = [QWEN2_TYPES, QWEN3_TYPES, HYBRID_TYPES].concat();
    assert_eq!(all, MODEL_TYPES);
}
