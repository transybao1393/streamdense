//! SmolLM3 chat template (mlx-community/SmolLM3-3B-4bit `chat_template.jinja`, HuggingFaceTB/SmolLM3-3B), ported for
//! plain (role, content) messages with the template defaults: `enable_thinking` undefined (= true), no `xml_tools`,
//! no `python_tools`. What it writes:
//!
//! ```text
//! reasoning_mode = "/think"; when the first message is system: "/no_think" in it -> "/no_think", elif "/think" -> "/think";
//!   custom_instructions = system.replace("/no_think", "").replace("/think", "").rstrip()
//! "<|im_start|>system\n"
//! if "/system_override" in the system message: custom_instructions.replace("/system_override", "").rstrip() + "<|im_end|>\n"
//! else: "## Metadata\n\nKnowledge Cutoff Date: {cutoff}\nToday Date: {strftime_now('%d %B %Y')}\nReasoning Mode: {mode}\n\n"
//!       "## Custom Instructions\n\n" + (custom_instructions + "\n\n" | the long /think default | the short /no_think default)
//!       (NO "<|im_end|>" here: the template writes "\n\n<|im_end|>\n" only inside the tools block)
//! per message (the system message too is NOT written again; it only feeds the header):
//!   user:      "<|im_start|>user\n" + content + "<|im_end|>\n"
//!   assistant: "/think": "<|im_start|>assistant\n" + content.lstrip("\n") + "<|im_end|>\n"
//!              "/no_think": "<|im_start|>assistant\n<think>\n\n</think>\n" + content.lstrip("\n") + "<|im_end|>\n"
//!   tool:      "<|im_start|>user\n" + content + "<|im_end|>\n" (refused here: no tool support in this layer)
//! add_generation_prompt: "<|im_start|>assistant\n" ("/think") or "<|im_start|>assistant\n<think>\n\n</think>\n"
//! ```
//!
//! The date: transformers defines `strftime_now` with the LOCAL date (`datetime.now()`). `chat_prompt` uses today's
//! local date too (`crate::localdate`, libc `localtime_r`); pass the date with `chat_prompt_on` to
//! render a fixed date (tests).
//! The cutoff and both default instruction texts are read from the folder's template by `template_kind` and checked
//! against the ported ones; the family default (`chat_prompt`) uses the SmolLM3-3B texts.
//!
//! Prompt encoding: no BOS; `Tokenizer::encode`.
use std::path::Path;

const CUTOFF: &str = "June 2025";
const THINK_DEFAULT: &str = "You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.";
const NO_THINK_DEFAULT: &str = "You are a helpful AI assistant named SmolLM, trained by Hugging Face.";
/// Template statements that must all be present for `template_kind` (escaped as in the .jinja file).
const MARKERS: [&str; 9] = [
    "{%- set enable_thinking = true -%}",
    "{%- if \"/no_think\" in system_message -%}",
    "{%- set custom_instructions = system_message.replace(\"/no_think\", \"\").replace(\"/think\", \"\").rstrip() -%}",
    "{{- \"Knowledge Cutoff Date: June 2025\\n\" -}}",
    "{%- set today = strftime_now(\"%d %B %Y\") -%}",
    "{{ \"<|im_start|>\" + message.role + \"\\n\"  + content + \"<|im_end|>\\n\" }}",
    "{{ \"<|im_start|>assistant\\n\" + \"<think>\\n\\n</think>\\n\" + content.lstrip(\"\\n\") + \"<|im_end|>\\n\" }}",
    "{{ \"<|im_start|>assistant\\n\" + content.lstrip(\"\\n\") + \"<|im_end|>\\n\" }}",
    "{{ \"<|im_start|>assistant\\n\" + \"<think>\\n\\n</think>\\n\"  }}",
];

/// Python `str.isspace` (Rust whitespace plus U+001C..U+001F).
fn py_space(c: char) -> bool { c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c) }

/// Render with an explicit "Today Date" text (`%d %B %Y`, e.g. "15 November 2026").
pub fn chat_prompt_on(messages: &[(String, String)], add_generation_prompt: bool, date: &str) -> Result<String, String> {
    let first = messages.first().ok_or("chat template smollm3: no messages")?;
    let system = (first.0 == "system").then_some(first.1.as_str());
    let mut think = true;
    if let Some(s) = system {
        if s.contains("/no_think") { think = false; } else if s.contains("/think") { think = true; }
    }
    let custom = system.map(|s| s.replace("/no_think", "").replace("/think", "").trim_end_matches(py_space).to_string());
    let mut out = String::from("<|im_start|>system\n");
    if system.is_some_and(|s| s.contains("/system_override")) {
        let c = custom.as_deref().unwrap_or("");
        out += c.replace("/system_override", "").trim_end_matches(py_space);
        out += "<|im_end|>\n";
    } else {
        out += &format!("## Metadata\n\nKnowledge Cutoff Date: {}\nToday Date: {}\nReasoning Mode: {}\n\n", CUTOFF, date,
                        if think { "/think" } else { "/no_think" });
        out += "## Custom Instructions\n\n";
        match custom.as_deref() {
            Some(c) if !c.is_empty() => out += c,
            _ if think => out += THINK_DEFAULT,
            _ => out += NO_THINK_DEFAULT,
        }
        out += "\n\n";
    }
    for (i, (r, c)) in messages.iter().enumerate() {
        match r.as_str() {
            "system" if i == 0 => {}
            "system" => return Err("chat template smollm3: a system message is read only as the first message (the template \
                                    drops later ones)".into()),
            "user" => out += &format!("<|im_start|>user\n{}<|im_end|>\n", c),
            "assistant" => {
                out += "<|im_start|>assistant\n";
                if !think { out += "<think>\n\n</think>\n"; }
                out += c.trim_start_matches('\n');
                out += "<|im_end|>\n";
            }
            o => return Err(format!("chat template smollm3: role {:?} not supported (system, user, assistant only)", o)),
        }
    }
    if add_generation_prompt {
        out += "<|im_start|>assistant\n";
        if !think { out += "<think>\n\n</think>\n"; }
    }
    Ok(out)
}

pub use crate::localdate::{date_from_days, local_date_at, local_offset_at, today_local, today_utc};

/// Chat prompt with today's local date (as HF).
pub fn chat_prompt(messages: &[(String, String)], add_generation_prompt: bool) -> Result<String, String> {
    chat_prompt_on(messages, add_generation_prompt, &today_local())
}

/// Check that a template text is the ported SmolLM3 template (statements + both default instruction texts).
pub fn check_template(tpl: Option<&str>) -> Result<(), String> {
    let t = tpl.ok_or("chat template: none in this model folder (base model?): use plain-text prompts")?;
    if let Some(m) = MARKERS.iter().find(|m| !t.contains(**m)) {
        return Err(format!("chat template smollm3: not the ported SmolLM3 template (missing {:?})", m));
    }
    for d in [THINK_DEFAULT, NO_THINK_DEFAULT] {
        if !t.contains(&format!("{{{{- \"{}\\n\\n\" -}}}}", d)) {
            return Err("chat template smollm3: default custom instructions differ from the ported template".into());
        }
    }
    Ok(())
}

/// Name of the template `model_dir` declares, or why it is refused.
pub fn template_kind(model_dir: &Path) -> Result<&'static str, String> {
    let tpl = crate::llama::template::template_text(model_dir)?;
    check_template(tpl.as_deref())?;
    Ok("smollm3 chatml (/think default)")
}
