//! Chat KV bookkeeping for `qwen serve` (op "chat"): which tokens the engine state holds, and how much of a new
//! prompt can reuse it. Pure logic (no engine), so it is unit-tested without a model.
//!
//! `seq` is the logical token sequence of the engine state: every fed token plus the last generated token,
//! which the engine keeps "pending" (emitted, not yet fed). Modes, tried in order when `continue` is set:
//! * Prefix:   `seq` is a token prefix of the newly rendered + tokenized prompt -> feed the rest;
//! * Followup: the earlier Python UI rule (`followup`): the new messages are the previous
//!   request's messages + one assistant message + new turns, with the same thinking options -> feed
//!   `<|im_end|>` (if the reply did not end with it) + "\n" + the template text of the new turns + generation
//!   prompt; the model's own reply tokens stay in the state;
//!   With tools ([`ChatState::plan_chat`]) the new turns may also be `tool` results, and the tool definitions must
//!   equal the previous request's;
//! * Full:     reset + prefill of the whole rendered prompt.
use crate::chat::{render_chat, ChatMessage, ChatOptions, Message, ToolDef};
use crate::tokenizer::Tokenizer;

pub const IM_END: u32 = 248046;
pub const ENDOFTEXT: u32 = 248044;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode { Prefix, Followup, Full }

impl Mode {
    pub fn as_str(&self) -> &'static str { match self { Mode::Prefix => "prefix", Mode::Followup => "followup", Mode::Full => "full" } }
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub mode: Mode,
    /// Tokens to append after `seq` (Prefix / Followup), or the whole prompt (Full).
    pub tokens: Vec<u32>,
    /// Length of the full rendered prompt (for Full this is `tokens.len()`).
    pub rendered_len: usize,
    /// The full rendered prompt, kept so the caller can fall back to Full (e.g. context limit) without re-encoding.
    pub full: Vec<u32>,
}

#[derive(Default)]
pub struct ChatState {
    /// Logical sequence of the engine state (fed tokens + pending last token). Empty = nothing reusable.
    pub seq: Vec<u32>,
    prev: Option<(Vec<ChatMessage>, Vec<ToolDef>, ChatOptions)>,
}

impl ChatState {
    pub fn new() -> Self { Self::default() }
    /// Forget everything (after reset, a token-level `gen`, an error, ctx_full).
    pub fn invalidate(&mut self) { self.seq.clear(); self.prev = None; }
    pub fn is_valid(&self) -> bool { !self.seq.is_empty() }

    /// Decide how to feed `messages` (no tools). `allow_reuse` = the request's `continue`.
    pub fn plan(&self, tk: &Tokenizer, messages: &[Message], opt: &ChatOptions, allow_reuse: bool) -> Result<Plan, String> {
        let m: Vec<ChatMessage> = messages.iter().map(ChatMessage::from).collect();
        self.plan_chat(tk, &m, &[], opt, allow_reuse)
    }

    /// Decide how to feed `messages` rendered with `tools`. `allow_reuse` = the request's `continue`.
    pub fn plan_chat(&self, tk: &Tokenizer, messages: &[ChatMessage], tools: &[ToolDef], opt: &ChatOptions, allow_reuse: bool) -> Result<Plan, String> {
        let text = render_chat(messages, tools, opt)?;
        let full = tk.encode(&text);
        let n = full.len();
        if allow_reuse && !self.seq.is_empty() {
            if full.len() > self.seq.len() && full[..self.seq.len()] == self.seq[..] {
                return Ok(Plan { mode: Mode::Prefix, tokens: full[self.seq.len()..].to_vec(), rendered_len: n, full });
            }
            if let Some(t) = self.followup(tk, messages, tools, opt)? {
                return Ok(Plan { mode: Mode::Followup, tokens: t, rendered_len: n, full });
            }
        }
        Ok(Plan { mode: Mode::Full, tokens: full.clone(), rendered_len: n, full })
    }

    fn followup(&self, tk: &Tokenizer, messages: &[ChatMessage], tools: &[ToolDef], opt: &ChatOptions) -> Result<Option<Vec<u32>>, String> {
        let Some((pm, pt, po)) = &self.prev else { return Ok(None) };
        let k = pm.len();
        if po.enable_thinking != opt.enable_thinking || po.reasoning_effort != opt.reasoning_effort || pt[..] != tools[..] { return Ok(None); }
        if messages.len() < k + 2 || messages[..k] != pm[..] || messages[k].role != "assistant" { return Ok(None); }
        if messages[k + 1..].iter().any(|m| m.role != "user" && m.role != "assistant" && m.role != "tool") { return Ok(None); }
        let base = ChatOptions { add_generation_prompt: false, ..opt.clone() };
        let a = render_chat(&messages[..=k], tools, &base)?;
        let b = render_chat(messages, tools, opt)?;
        let Some(tail) = b.strip_prefix(a.as_str()) else { return Ok(None) };
        let pre = if self.seq.last() == Some(&IM_END) { "" } else { "<|im_end|>" };
        Ok(Some(tk.encode(&format!("{}\n{}", pre, tail))))
    }

    /// Record a finished request: the engine state now holds `seq_before ++ plan.tokens ++ generated`
    /// (for Full: `plan.tokens ++ generated`). `ok` = false (error, ctx_full, nothing generated) drops it all.
    pub fn commit(&mut self, plan_mode: Mode, fed: &[u32], generated: &[u32], messages: &[Message], opt: &ChatOptions, ok: bool) {
        let m: Vec<ChatMessage> = messages.iter().map(ChatMessage::from).collect();
        self.commit_chat(plan_mode, fed, generated, &m, &[], opt, ok)
    }

    /// [`ChatState::commit`] for a request planned with [`ChatState::plan_chat`].
    #[allow(clippy::too_many_arguments)]
    pub fn commit_chat(&mut self, plan_mode: Mode, fed: &[u32], generated: &[u32], messages: &[ChatMessage], tools: &[ToolDef],
                       opt: &ChatOptions, ok: bool) {
        if !ok || generated.is_empty() { self.invalidate(); return; }
        if plan_mode == Mode::Full { self.seq.clear(); }
        self.seq.extend_from_slice(fed);
        self.seq.extend_from_slice(generated);
        self.prev = Some((messages.to_vec(), tools.to_vec(), ChatOptions { add_generation_prompt: true, ..opt.clone() }));
    }
}
