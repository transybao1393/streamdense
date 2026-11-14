//! KV reuse planner: which tokens the engine state holds, and how much of a new prompt can reuse it.
//! Pure logic over token lists + messages (render / encode are passed in), unit-tested without a model.
//!
//! `KvState.seq` is the logical token sequence of the engine state: every fed token plus the last generated
//! token, which the engine keeps pending (emitted, not yet fed), so `seq.len() == engine.pos() + 1` after a
//! reply. Modes, in order:
//! * prefix:   the new prompt's ids start with `seq` -> feed the rest (the rendered history re-tokenizes to exactly
//!             what the engine saw and generated: the usual case with Claude Code, measured 0.7-1.6 s vs 16 s);
//! * followup: the new messages are the previous request's messages + one assistant message (our reply, maybe
//!             re-rendered differently) + new turns, same tools and thinking -> feed `<|im_end|>` (unless the reply
//!             already ended with it) + "\n" + the template text of the new turns + generation prompt; the model's
//!             own reply tokens stay in the state (as `plan` in gw27b.py, the project's earlier Python gateway, not kept);
//! * full:     reset + prefill of the whole rendered prompt.
use crate::chatmap::{ChatMsg, ToolDef};
use streamdense_tokenizer::IM_END;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode { Prefix, Followup, Full }

impl Mode {
    pub fn as_str(&self) -> &'static str { match self { Mode::Prefix => "prefix", Mode::Followup => "followup", Mode::Full => "full" } }
}

#[derive(Clone, Debug, Default)]
pub struct KvState {
    /// Fed tokens + the pending last generated token. Empty = nothing reusable.
    pub seq: Vec<u32>,
    /// The request that produced the state (messages before our reply, tools, thinking flag).
    pub msgs: Vec<ChatMsg>,
    pub tools: Vec<ToolDef>,
    pub think: bool,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub mode: Mode,
    /// Tokens to append after `seq` (prefix / followup) or the whole prompt (full).
    pub tokens: Vec<u32>,
    /// The whole rendered prompt (kept for a fallback to full, e.g. the context limit).
    pub full: Vec<u32>,
}

impl KvState {
    pub fn invalidate(&mut self) { self.seq.clear(); self.msgs.clear(); self.tools.clear(); }
    pub fn is_valid(&self) -> bool { !self.seq.is_empty() }

    /// `full` = encode(render(msgs, tools, think, gen prompt)); `render(msgs, add_generation_prompt)` and `encode`
    /// are the request's renderer / tokenizer.
    pub fn plan(&self, msgs: &[ChatMsg], tools: &[ToolDef], think: bool, full: Vec<u32>,
                render: &dyn Fn(&[ChatMsg], bool) -> Result<String, String>, encode: &dyn Fn(&str) -> Vec<u32>) -> Plan {
        if self.is_valid() {
            let n = self.seq.len();
            if full.len() > n && full[..n] == self.seq[..] {
                return Plan { mode: Mode::Prefix, tokens: full[n..].to_vec(), full };
            }
            if let Some(t) = self.followup(msgs, tools, think, render, encode) {
                return Plan { mode: Mode::Followup, tokens: t, full };
            }
        }
        Plan { mode: Mode::Full, tokens: full.clone(), full }
    }

    fn followup(&self, msgs: &[ChatMsg], tools: &[ToolDef], think: bool,
                render: &dyn Fn(&[ChatMsg], bool) -> Result<String, String>, encode: &dyn Fn(&str) -> Vec<u32>) -> Option<Vec<u32>> {
        let k = self.msgs.len();
        if k == 0 || self.tools != tools || self.think != think { return None; }
        if msgs.len() < k + 2 || msgs[..k] != self.msgs[..] || msgs[k].role != "assistant" { return None; }
        if msgs[k + 1..].iter().any(|m| m.role == "system" || m.role == "developer") { return None; }
        let a = render(&msgs[..=k], false).ok()?;
        let b = render(msgs, true).ok()?;
        if !a.ends_with("<|im_end|>\n") { return None; }
        let tail = b.strip_prefix(a.as_str())?;
        let pre = if self.seq.last() == Some(&IM_END) { "" } else { "<|im_end|>" };
        Some(encode(&format!("{}\n{}", pre, tail)))
    }

    /// Record a finished request: the state now holds `seq_before ++ fed ++ generated` (full: `fed ++ generated`).
    /// `ok` false (error, ctx_full before any token) drops it all.
    pub fn commit(&mut self, mode: Mode, fed: &[u32], generated: &[u32], msgs: &[ChatMsg], tools: &[ToolDef], think: bool, ok: bool) {
        if !ok || generated.is_empty() { self.invalidate(); return; }
        if mode == Mode::Full { self.seq.clear(); }
        self.seq.extend_from_slice(fed);
        self.seq.extend_from_slice(generated);
        self.msgs = msgs.to_vec();
        self.tools = tools.to_vec();
        self.think = think;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chatmap::{render, ChatMsg};

    // char-level fake tokenizer: one id per char, "<|im_end|>" -> IM_END
    fn enc(s: &str) -> Vec<u32> {
        let mut v = vec![];
        let mut r = s;
        while !r.is_empty() {
            if let Some(x) = r.strip_prefix("<|im_end|>") { v.push(IM_END); r = x; continue; }
            let c = r.chars().next().unwrap();
            v.push(c as u32);
            r = &r[c.len_utf8()..];
        }
        v
    }
    fn rend(m: &[ChatMsg], g: bool) -> Result<String, String> { render(m, &[], false, g) }
    fn full(m: &[ChatMsg]) -> Vec<u32> { enc(&rend(m, true).unwrap()) }

    #[test]
    fn modes() {
        let mut kv = KvState::default();
        let m1 = vec![ChatMsg::new("system", "s"), ChatMsg::new("user", "hi")];
        let p = kv.plan(&m1, &[], false, full(&m1), &rend, &enc);
        assert_eq!(p.mode, Mode::Full);
        // the model answers "Hello" + <|im_end|>; the state holds prompt + reply (last token pending)
        let gen: Vec<u32> = enc("Hello<|im_end|>");
        kv.commit(p.mode, &p.tokens, &gen, &m1, &[], false, true);

        // turn 2, re-rendered history matches token for token -> prefix, feed = "\n" + new turn + gen prompt
        let mut m2 = m1.clone();
        m2.push(ChatMsg::new("assistant", "Hello"));
        m2.push(ChatMsg::new("user", "more"));
        let p = kv.plan(&m2, &[], false, full(&m2), &rend, &enc);
        assert_eq!(p.mode, Mode::Prefix);
        assert_eq!(p.tokens, enc("\n<|im_start|>user\nmore<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));

        // the client re-sends our reply edited (history differs after it) -> followup keeps the model's tokens
        let mut m3 = m1.clone();
        m3.push(ChatMsg::new("assistant", "Hello (edited)"));
        m3.push(ChatMsg::new("user", "more"));
        let p = kv.plan(&m3, &[], false, full(&m3), &rend, &enc);
        assert_eq!(p.mode, Mode::Followup);
        assert_eq!(p.tokens, enc("\n<|im_start|>user\nmore<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));

        // a reply cut by max_tokens (no <|im_end|>) -> followup closes the turn first
        let mut kv2 = kv.clone();
        kv2.seq.pop();
        let mut m5 = m1.clone();
        m5.push(ChatMsg::new("assistant", "Bye"));
        m5.push(ChatMsg::new("user", "more"));
        let p = kv2.plan(&m5, &[], false, full(&m5), &rend, &enc);
        assert_eq!(p.mode, Mode::Followup);
        assert_eq!(p.tokens[0], IM_END);

        // other thinking flag / other tools / changed history -> full
        let p = kv.plan(&m3, &[], true, enc(&render(&m3, &[], true, true).unwrap()), &|m, g| render(m, &[], true, g), &enc);
        assert_eq!(p.mode, Mode::Full);
        let mut m4 = m3.clone();
        m4[1] = ChatMsg::new("user", "changed");
        assert_eq!(kv.plan(&m4, &[], false, full(&m4), &rend, &enc).mode, Mode::Full);

        // commit of a failed request drops the state
        kv.commit(Mode::Prefix, &[1], &[], &m2, &[], false, true);
        assert!(!kv.is_valid());
        assert_eq!(kv.plan(&m2, &[], false, full(&m2), &rend, &enc).mode, Mode::Full);
    }
}
