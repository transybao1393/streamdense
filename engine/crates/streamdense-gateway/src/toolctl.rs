//! Tool-call control of one reply (worker.rs `Prepared::tc`): forced tool calls (`tool_choice` any / tool) and the
//! tool-name constraint (`streamdense.allowed_tools`, or the tools of a forced call).
//!
//! Forced call: the template's tool-call opener (`ToolForce::prefix`, Qwen3.8 format `<tool_call>\n<function=NAME>\n`)
//! is appended to the prompt as ordinary prompt tokens (prefilled; spec / MTP exactness and prefix reuse unchanged:
//! the KV bookkeeping holds prompt ++ opener ++ reply like any reply). The reply text the parser sees is opener + the
//! generated text (the opener is streamed as the first text piece, which the visible-text filter hides as a call).
//!
//! Name constraint: while the reply is inside an open `<function` name (after `<tool_call>`, before its `>`; with
//! thinking only after `</think>`), the decode stops (the token that opened the name is held pending) and the name is
//! decoded one token per pass with the engine's allow-list (`Backend::set_allow`: every other logit -inf before the
//! argmax), the list = every vocab token that keeps `=NAME>` a possible continuation for some allowed NAME. So the name
//! is the greedy decode among the allowed names (identical to the plain greedy reply whenever the model picks an allowed
//! name anyway); the normal (spec) decode resumes after the name. Stats-free, deterministic. A backend without
//! `set_allow`, more candidates than the list holds, or a name already outside the set (a token spanning `<function=X`)
//! leaves that call unconstrained; server.rs drops any call whose name is not allowed.
use crate::anthropic::ChatReq;
use streamdense_tokenizer::Tokenizer;

/// Opening of a tool call's name in the template (`<function=NAME>`).
const FN_OPEN: &str = "<function";

#[derive(Clone, Debug, Default)]
pub struct ToolCtl {
    /// forced opener text (prompt tokens appended by the worker), "" = none
    pub force_text: String,
    /// allowed tool names (None = no constraint)
    pub names: Option<Vec<String>>,
    /// the solo decode loop (worker `generate`) runs the constrained name steps; false = batched (unconstrained)
    pub solo: bool,
    /// opener + the reply text so far (kept only with `names`)
    pub raw: String,
    /// thinking on: calls only count after `</think>`
    pub think: bool,
    /// set by `push` when the decode must stop for a constrained name; cleared by the worker
    pub hold: bool,
    /// a constrained name step is running (no hold inside it)
    pub stepping: bool,
    /// byte offset of a `<function` left unconstrained (`skip`)
    skip_at: Option<usize>,
    /// constrained name steps run (stats)
    pub steps: usize,
}

impl ToolCtl {
    pub fn new(req: &ChatReq, think: bool, solo: bool) -> ToolCtl {
        let force_text = req.tool_force.as_ref().map(|f| f.prefix()).unwrap_or_default();
        let names = req.name_constraint();
        ToolCtl { raw: if names.is_some() { force_text.clone() } else { String::new() }, force_text, names, solo, think, ..Default::default() }
    }

    /// A decoded piece of the reply. true = stop the decode now (an open call name needs the constrained steps).
    pub fn push(&mut self, piece: &str) -> bool {
        if self.names.is_none() { return false; }
        self.raw.push_str(piece);
        if self.solo && !self.stepping && self.open_name().is_some() { self.hold = true; }
        self.hold
    }

    /// (byte offset of the `<function`, name text after it so far) when the reply is inside an open call name that is
    /// not skipped.
    pub fn open_name(&self) -> Option<(usize, &str)> {
        let base = if self.think { self.raw.rfind("</think>")? } else { 0 };
        let s = &self.raw[base..];
        let tc = s.rfind("<tool_call>")?;
        let f = tc + s[tc..].rfind(FN_OPEN)?;
        let rest = &s[f + FN_OPEN.len()..];
        if rest.contains('>') || self.skip_at == Some(base + f) { return None; }
        Some((base + f, rest))
    }

    /// Leave the current open name unconstrained.
    pub fn skip(&mut self) { self.skip_at = self.open_name().map(|x| x.0); }

    /// The next constrained step's allow-list (`fold`: test ids modulo the tiny vocab); None = not inside an open name
    /// (or no constraint / batched).
    pub fn name_allow(&self, tk: &Tokenizer, fold: Option<u32>) -> Option<Vec<u32>> {
        if !self.solo { return None; }
        let names = self.names.as_ref()?;
        let (_, partial) = self.open_name()?;
        let mut ids = name_candidates(tk, names, partial);
        if let Some(m) = fold { for t in ids.iter_mut() { *t %= m; } ids.sort_unstable(); ids.dedup(); }
        Some(ids)
    }
}

/// Every non-special token whose bytes continue `partial` (the text after `<function` so far) along `=NAME>` of some
/// allowed NAME: either still inside that string, or completing it (bytes past the `>` are free, e.g. `>\n`).
pub fn name_candidates(tk: &Tokenizer, names: &[String], partial: &str) -> Vec<u32> {
    let p = partial.as_bytes();
    let rest: Vec<Vec<u8>> = names.iter().map(|n| format!("={}>", n).into_bytes())
        .filter(|s| s.starts_with(p)).map(|s| s[p.len()..].to_vec()).collect();
    if rest.is_empty() { return vec![]; }
    (0..tk.vocab_size() as u32).filter(|&id| {
        if tk.is_special(id) { return false; }
        let b = tk.id_to_bytes(id);
        !b.is_empty() && rest.iter().any(|r| if b.len() <= r.len() { r.starts_with(b) } else { b.starts_with(r) })
    }).collect()
}

// ---------------------------------------------------------------- streamed tool calls (server.rs SSE)
//
// `ToolStream` turns the growing reply text into tool_use block events while the call is written: `Start(name)` once
// the function name is complete, `Delta(json)` pieces of the JSON-encoded input as the arguments are decoded, `Stop`
// at the end of the call (`</function>`; a truncated call: at the reply end). The concatenated deltas of a call are
// exactly `input.to_string()` of the batch path (server.rs `finish_reply`: `parse_tool_calls` + `coerce_args`):
// the region is found with the parser's own `stream_visible`, the region / block / parameter structure mirrors the
// tokenizer crate's `parse_tool_region` / `parse_block` / `parse_params`, a string value streams with a hold-back of
// what may still change (trailing whitespace, a partial closing tag, a partial U+FFFD byte), a typed parameter (its
// schema has no "string" type) is sent once complete, coerced like `coerce_args`, and `finish` reconciles with the
// batch result (rest of the open call, calls never streamed, `mismatch` when an emitted piece is not a prefix).
use serde_json::{Map, Value};

const S_TOOL_OPEN: &str = "<tool_call>";
const S_TOOL_CLOSE: &str = "</tool_call>";
const S_FUNC_OPEN: &str = "<function=";
const S_FUNC_CLOSE: &str = "</function>";
const S_PARAM_OPEN: &str = "<parameter=";
const S_PARAM_CLOSE: &str = "</parameter>";
/// Tags that end a value (or the call) when they complete: a proper prefix at the end of a value is held back.
const S_ENDERS: [&str; 6] = [S_PARAM_CLOSE, S_PARAM_OPEN, S_FUNC_CLOSE, S_FUNC_OPEN, S_TOOL_CLOSE, S_TOOL_OPEN];

/// One tool_use block event (the server adds the block index and id).
#[derive(Clone, Debug, PartialEq)]
pub enum ToolEv { Start(String), Delta(String), Stop }

struct SCall { name: String, emitted: String, stopped: bool, args: Args }

/// Where `args_step` is in a function body: everything before `q` is emitted.
#[derive(Default)]
struct Args {
    opened: bool,
    closed: bool,
    /// stopped emitting (a repeated parameter name, or a mismatch)
    broken: bool,
    /// body offset after the last finished parameter
    q: usize,
    /// parameters emitted
    n: usize,
    seen: Vec<String>,
    /// the parameter being written has its key out (a string one: and its opening quote)
    key_open: bool,
    /// bytes of that string value emitted
    cur_raw: usize,
}

pub struct ToolStream {
    /// `reply_text` of the reply so far (thinking: "<think>\n" first)
    text: String,
    /// `tool_choice: none`: never a call
    none: bool,
    /// allowed names (`ChatReq::name_constraint`); None = any
    allowed: Option<Vec<String>>,
    /// (tool name, its JSON schema) of the request
    schemas: Vec<(String, Value)>,
    /// byte offset of the tool-call region in `text` (the first `<tool_call>` / bare `<function=` of the parser)
    region: Option<usize>,
    /// region views done (finished or skipped)
    vdone: usize,
    calls: Vec<SCall>,
    /// an emitted piece turned out not to be a prefix of the final input (never seen in tests; logged)
    pub mismatch: bool,
}

/// A call in the region: an XML function (`body` after `<function=NAME>`, `fin` = it cannot grow any more) or a
/// complete JSON-form `<tool_call>` block.
enum View<'a> { Xml { name: &'a str, body: &'a str, fin: bool }, Json(&'a str) }

fn find_at(s: &str, from: usize, pat: &str) -> Option<usize> { s.get(from..)?.find(pat).map(|i| from + i) }

/// The tokenizer's `parse_tool_region` over a region that may still grow.
fn views(r: &str) -> Vec<View<'_>> {
    let mut out = vec![];
    let mut p = 0;
    loop {
        let a = find_at(r, p, S_TOOL_OPEN);
        let b = find_at(r, p, S_FUNC_OPEN);
        match (a, b) {
            (Some(a), b) if b.map_or(true, |b| a < b) => {
                let st = a + S_TOOL_OPEN.len();
                let (end, next, closed) = match find_at(r, st, S_TOOL_CLOSE) { Some(j) => (j, j + S_TOOL_CLOSE.len(), true), None => (r.len(), r.len(), false) };
                block_views(&r[st..end], closed, &mut out);
                if !closed { break; }
                p = next;
            }
            (_, Some(b)) => {
                let close = find_at(r, b, S_FUNC_CLOSE).map(|j| j + S_FUNC_CLOSE.len());
                let next_tc = find_at(r, b, S_TOOL_OPEN);
                let (end, closed) = match (close, next_tc) {
                    (Some(c), Some(n)) => (c.min(n), true),
                    (Some(c), None) => (c, true),
                    (None, Some(n)) => (n, true),
                    (None, None) => (r.len(), false),
                };
                block_views(&r[b..end], closed, &mut out);
                if !closed { break; }
                p = end;
            }
            _ => break,
        }
        if p >= r.len() { break; }
    }
    out
}

/// The tokenizer's `parse_block` (`closed`: the block's end is known).
fn block_views<'a>(block: &'a str, closed: bool, out: &mut Vec<View<'a>>) {
    let b = block.trim();
    if !b.contains(S_FUNC_OPEN) {
        if closed { out.push(View::Json(b)); }
        return;
    }
    let mut p = 0;
    while let Some(i) = find_at(b, p, S_FUNC_OPEN) {
        let ns = i + S_FUNC_OPEN.len();
        let Some(gt) = b[ns..].find('>') else { break };
        let name = b[ns..ns + gt].trim();
        let bs = ns + gt + 1;
        let (body, next, fin) = match find_at(b, bs, S_FUNC_CLOSE) {
            Some(j) => (&b[bs..j], j + S_FUNC_CLOSE.len(), true),
            None => match find_at(b, bs, S_FUNC_OPEN) { Some(j) => (&b[bs..j], j, true), None => (&b[bs..], b.len(), closed) },
        };
        if !name.is_empty() { out.push(View::Xml { name, body, fin }); }
        p = next;
    }
}

/// The JSON types a property schema allows (tokenizer `schema_types`).
fn schema_types(p: &Value) -> Vec<String> {
    let mut t = vec![];
    match &p["type"] {
        Value::String(s) => t.push(s.clone()),
        Value::Array(a) => t.extend(a.iter().filter_map(|x| x.as_str().map(|s| s.to_string()))),
        _ => {}
    }
    for k in ["anyOf", "oneOf"] {
        if let Some(a) = p[k].as_array() { for x in a { t.extend(schema_types(x)); } }
    }
    t
}

/// The JSON string body (no quotes) of `s`, as serde_json writes it.
fn esc(s: &str) -> String { let q = Value::String(s.to_string()).to_string(); q[1..q.len() - 1].to_string() }

/// The part of a growing string value that can no longer change: without a trailing proper prefix of a closing tag,
/// a trailing U+FFFD, then trailing whitespace (the parser trims / strips it at the end of a value).
fn settled(v: &str) -> &str {
    let mut v = v;
    'm: for m in S_ENDERS {
        for l in (1..m.len()).rev() {
            if v.ends_with(&m[..l]) { v = &v[..v.len() - l]; break 'm; }
        }
    }
    let v = v.strip_suffix('\u{FFFD}').unwrap_or(v);
    v.trim_end()
}

impl ToolStream {
    pub fn new(req: &ChatReq, think: bool) -> ToolStream {
        ToolStream {
            text: crate::chatmap::reply_text("", think), none: req.tool_choice_none, allowed: req.name_constraint(),
            schemas: req.tools.iter().map(|t| (t.name.clone(), t.parameters.clone())).collect(),
            region: None, vdone: 0, calls: vec![], mismatch: false,
        }
    }

    /// Calls streamed so far (a tool_use block was started).
    pub fn started(&self) -> usize { self.calls.len() }
    /// A tool_use block is open.
    pub fn open(&self) -> bool { self.calls.last().is_some_and(|c| !c.stopped) }

    fn schema(&self, name: &str) -> Option<&Value> { self.schemas.iter().find(|s| s.0 == name).map(|s| &s.1) }
    fn allowed(&self, name: &str) -> bool { !self.none && self.allowed.as_ref().map_or(true, |a| a.iter().any(|n| n == name)) }

    /// A decoded piece of the reply -> block events.
    pub fn push(&mut self, piece: &str) -> Vec<ToolEv> {
        self.text.push_str(piece);
        if self.none { return vec![]; }
        if self.region.is_none() { self.region = region_start(&self.text); }
        let Some(r0) = self.region else { return vec![] };
        let text = std::mem::take(&mut self.text);
        let ev = self.sync(&text[r0..]);
        self.text = text;
        ev
    }

    fn sync(&mut self, r: &str) -> Vec<ToolEv> {
        let mut ev = vec![];
        let vs = views(r);
        for v in vs.iter().skip(self.vdone) {
            match v {
                View::Json(b) => {
                    // complete JSON-form block: the parser's own result for it, whole
                    let (_, calls) = streamdense_tokenizer::parse_tool_calls(&format!("{}{}{}", S_TOOL_OPEN, b, S_TOOL_CLOSE));
                    for c in calls {
                        if !self.allowed(&c.name) { continue; }
                        let js = crate::chatmap::coerce_args(&c.args, self.schema(&c.name)).to_string();
                        ev.push(ToolEv::Start(c.name.clone()));
                        ev.push(ToolEv::Delta(js.clone()));
                        ev.push(ToolEv::Stop);
                        self.calls.push(SCall { name: c.name, emitted: js, stopped: true, args: Args::default() });
                    }
                    self.vdone += 1;
                }
                View::Xml { name, body, fin } => {
                    if !self.allowed(name) {
                        if *fin { self.vdone += 1; continue; }
                        break;
                    }
                    if !self.open() {
                        ev.push(ToolEv::Start(name.to_string()));
                        self.calls.push(SCall { name: name.to_string(), emitted: String::new(), stopped: false, args: Args::default() });
                    }
                    let schema = self.schemas.iter().find(|s| s.0 == *name).map(|s| &s.1);
                    let c = self.calls.last_mut().unwrap();
                    let mut d = String::new();
                    args_step(&mut c.args, body, *fin, schema, &mut d, &mut self.mismatch);
                    if !d.is_empty() { c.emitted.push_str(&d); ev.push(ToolEv::Delta(d)); }
                    if !*fin { break; }
                    ev.push(ToolEv::Stop);
                    c.stopped = true;
                    self.vdone += 1;
                }
            }
        }
        ev
    }

    /// The reply was cut by the token limit inside the open call (`cut_call`, `[gateway] cut_tool_call = "length"`):
    /// close its tool_use block (its input stays partial; the reply ends with stop_reason "max_tokens") and leave it
    /// out of the reconciliation, so `finish` with the batch calls minus the cut one finds no mismatch.
    pub fn close_cut(&mut self) -> Vec<ToolEv> {
        if !self.open() { return vec![]; }
        self.calls.pop();
        vec![ToolEv::Stop]
    }

    /// End of the reply: reconcile with the batch calls (`finish_reply`, coerced): the rest of the open call, the
    /// calls never streamed (whole), stops for anything still open.
    pub fn finish(&mut self, calls: &[(String, Value)]) -> Vec<ToolEv> {
        let mut ev = vec![];
        for (i, (n, a)) in calls.iter().enumerate() {
            let js = a.to_string();
            if let Some(c) = self.calls.get_mut(i) {
                if c.name != *n || !js.starts_with(&c.emitted) || (c.stopped && c.emitted != js) {
                    self.mismatch = true;
                } else if !c.stopped && js.len() > c.emitted.len() {
                    ev.push(ToolEv::Delta(js[c.emitted.len()..].to_string()));
                    c.emitted = js;
                }
                if !c.stopped { ev.push(ToolEv::Stop); c.stopped = true; }
            } else {
                ev.push(ToolEv::Start(n.clone()));
                ev.push(ToolEv::Delta(js.clone()));
                ev.push(ToolEv::Stop);
                self.calls.push(SCall { name: n.clone(), emitted: js, stopped: true, args: Args::default() });
            }
        }
        if self.calls.len() > calls.len() {
            self.mismatch = true;
            if let Some(c) = self.calls.last_mut().filter(|c| !c.stopped) { c.stopped = true; ev.push(ToolEv::Stop); }
        }
        ev
    }
}

/// The reply (`reply_text`) ends inside an unfinished XML tool call (no `</function>` / `</tool_call>` / next call
/// after it): (its function name, chars of its body written so far). None when the reply has no call or its last call
/// is complete. Used when the generation stopped on the token limit (server.rs `finish_reply`).
pub fn cut_call(text: &str) -> Option<(String, usize)> {
    let r0 = region_start(text)?;
    match views(&text[r0..]).pop()? {
        View::Xml { name, body, fin: false } => Some((name.to_string(), body.chars().count())),
        _ => None,
    }
}

/// Byte offset of the parser's tool-call region (its first `<tool_call>` / bare `<function=`, the scan respecting think
/// blocks): the shortest prefix `stream_visible` already sees a call in ends with that marker (monotone: binary search).
fn region_start(text: &str) -> Option<usize> {
    if !streamdense_tokenizer::stream_visible(text).tool_call { return None; }
    let pred = |i: usize| streamdense_tokenizer::stream_visible(&text[..i]).tool_call;
    let (mut lo, mut hi) = (0usize, text.len());
    while lo + 1 < hi {
        let mut mid = lo + (hi - lo) / 2;
        while mid > lo && !text.is_char_boundary(mid) { mid -= 1; }
        if mid == lo {
            mid = lo + (hi - lo) / 2;
            while mid < hi && !text.is_char_boundary(mid) { mid += 1; }
            if mid == hi { break; }
        }
        if pred(mid) { hi = mid } else { lo = mid }
    }
    [S_TOOL_OPEN, S_FUNC_OPEN].iter().find(|m| text[..hi].ends_with(**m)).map(|m| hi - m.len())
}

/// The next piece of a function's arguments JSON (the tokenizer's `parse_params` + `coerce_args` over a body that may
/// still grow; `fin`: complete, the object is closed), appended to `out`. Resumes at the parameter being written, so a
/// long value costs only its new bytes; every piece extends what went out before (else `mismatch`, and it stops).
fn args_step(a: &mut Args, body: &str, fin: bool, schema: Option<&Value>, out: &mut String, mismatch: &mut bool) {
    if !a.opened { out.push('{'); a.opened = true; }
    if a.broken || a.closed { return; }
    let key = |out: &mut String, n: usize, name: &str| { if n > 0 { out.push(','); } out.push_str(&Value::String(name.to_string()).to_string()); out.push(':'); };
    while let Some(i) = find_at(body, a.q, S_PARAM_OPEN) {
        let ns = i + S_PARAM_OPEN.len();
        let Some(gt) = body[ns..].find('>') else { break };
        let name = body[ns..ns + gt].trim();
        let vs = ns + gt + 1;
        let rest = &body[vs..];
        let close = rest.find(S_PARAM_CLOSE);
        let next = rest.find(S_PARAM_OPEN);
        let (vend, adv, done) = match (close, next) {
            (Some(c), Some(n)) if n < c => (n, n, true),
            (Some(c), _) => (c, c + S_PARAM_CLOSE.len(), true),
            (None, Some(n)) => (n, n, true),
            (None, None) => (rest.len(), rest.len(), fin),
        };
        // a repeated name replaces the earlier value in place (map insert): what went out is no longer a prefix
        if !name.is_empty() && !a.key_open && a.seen.iter().any(|x| x == name) { a.broken = true; return; }
        let typed = schema.is_some_and(|s| { let t = schema_types(&s["properties"][name]); !t.is_empty() && !t.iter().any(|x| x == "string") });
        if done {
            let mut v = rest[..vend].strip_prefix('\n').unwrap_or(&rest[..vend]);
            v = v.strip_suffix('\n').unwrap_or(v);
            if close.is_none() && next.is_none() { v = v.trim_end_matches('\n'); }
            if !name.is_empty() {
                if !a.key_open { key(out, a.n, name); }
                if typed {
                    let mut m = Map::new();
                    m.insert(name.to_string(), Value::String(v.to_string()));
                    out.push_str(&streamdense_tokenizer::coerce_args(&m, schema.unwrap()).remove(name).unwrap_or(Value::Null).to_string());
                } else if a.key_open {
                    // the streamed bytes are the start of the final value (same text offset): only its length can disagree
                    if v.len() < a.cur_raw || !v.is_char_boundary(a.cur_raw) { *mismatch = true; a.broken = true; return; }
                    out.push_str(&esc(&v[a.cur_raw..]));
                    out.push('"');
                } else {
                    out.push_str(&Value::String(v.to_string()).to_string());
                }
                a.seen.push(name.to_string());
                a.n += 1;
            }
            a.key_open = false;
            a.cur_raw = 0;
            a.q = vs + adv;
        } else {
            // the value being written (the last parameter): a string streams, a typed value waits for its end
            if name.is_empty() { return; }
            if !a.key_open {
                key(out, a.n, name);
                if !typed { out.push('"'); }
                a.key_open = true;
                a.cur_raw = 0;
            }
            if !typed {
                let v = settled(rest.strip_prefix('\n').unwrap_or(rest));
                if v.len() > a.cur_raw && v.is_char_boundary(a.cur_raw) { out.push_str(&esc(&v[a.cur_raw..])); a.cur_raw = v.len(); }
                else if v.len() < a.cur_raw { *mismatch = true; a.broken = true; }
            }
            return;
        }
    }
    if fin { out.push('}'); a.closed = true; }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anthropic::{convert, ToolForce};
    use serde_json::json;

    const TOK_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models/qwen38_27b_mlx4");

    fn req(extra: serde_json::Value) -> ChatReq {
        let mut b = json!({"messages": [{"role": "user", "content": "x"}], "tools": [
            {"name": "Bash", "input_schema": {"type": "object"}}, {"name": "Read", "input_schema": {"type": "object"}},
            {"name": "TodoWrite", "input_schema": {"type": "object"}}, {"name": "WebFetch", "input_schema": {"type": "object"}}]});
        for (k, v) in extra.as_object().unwrap() { b[k] = v.clone(); }
        convert(&b).unwrap()
    }

    #[test]
    fn open_name_and_hold() {
        let mut c = ToolCtl::new(&req(json!({"streamdense": {"allowed_tools": ["Bash", "Read"]}})), false, true);
        assert_eq!(c.names, Some(vec!["Bash".to_string(), "Read".to_string()]));
        assert!(!c.push("I will run <function it.\n\n"), "no <tool_call> yet");
        assert!(!c.push("<tool_call>\n<"));
        assert!(c.push("function"), "inside the name");
        assert_eq!(c.open_name().map(|x| x.1), Some(""));
        c.hold = false;
        c.stepping = true;
        assert!(!c.push("=B"));
        assert_eq!(c.open_name().map(|x| x.1), Some("=B"));
        assert!(!c.push("ash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"));
        assert_eq!(c.open_name(), None);
        c.stepping = false;
        assert!(!c.push("\n<tool_call>\n<func"));
        assert!(c.push("tion"), "second call");
        c.skip();
        assert_eq!(c.open_name(), None, "skipped");
        // thinking: a call inside the reasoning is not a call
        let mut t = ToolCtl::new(&req(json!({"tool_choice": {"type": "any"}})), true, true);
        assert_eq!(t.raw, "<tool_call>\n<function");
        assert!(t.open_name().is_none(), "think on and no </think>: not a call");
        t.raw.clear();
        assert!(!t.push("<tool_call>\n<function"));
        assert!(t.push("</think>\n\n<tool_call>\n<function"));
        // forced any without thinking: inside the name from the start
        let a = ToolCtl::new(&req(json!({"tool_choice": {"type": "any"}})), false, true);
        assert_eq!(a.open_name().map(|x| x.1), Some(""));
        assert_eq!(a.names.as_ref().unwrap().len(), 4);
        // no constraint: nothing tracked; batched: never holds
        let mut n = ToolCtl::new(&req(json!({})), false, true);
        assert!(n.names.is_none() && !n.push("<tool_call>\n<function") && n.raw.is_empty());
        let mut b = ToolCtl::new(&req(json!({"streamdense": {"allowed_tools": ["Bash"]}})), false, false);
        assert!(!b.push("<tool_call>\n<function"));
    }

    #[test]
    fn candidates_follow_the_allowed_names() {
        if !std::path::Path::new(&format!("{}/tokenizer.json", TOK_DIR)).exists() { eprintln!("SKIP: 27B tokenizer missing"); return; }
        let tk = Tokenizer::from_path(TOK_DIR).unwrap();
        let names = vec!["Bash".to_string(), "TodoWrite".to_string()];
        let txt = |ids: &[u32]| ids.iter().map(|&i| String::from_utf8_lossy(tk.id_to_bytes(i)).to_string()).collect::<Vec<_>>();
        let c0 = name_candidates(&tk, &names, "");
        let t0 = txt(&c0);
        for want in ["=", "=B"] { assert!(t0.iter().any(|x| x == want), "{:?} missing from {:?}", want, t0); }
        assert!(t0.iter().all(|x| "=Bash>".starts_with(x.as_str()) || "=TodoWrite>".starts_with(x.as_str())), "{:?}", t0);
        assert!(!t0.iter().any(|x| x == "=R" || x == "=read"), "{:?}", t0);
        assert!(c0.len() <= streamdense_model::qwen_engine::BAN_MAX, "{} candidates", c0.len());
        // greedy walk along the template's own tokenization of each name: every token is a candidate at its step
        for n in &names {
            let ids = tk.encode(&format!("<tool_call>\n<function={}>\n", n));
            let s = ids.iter().position(|&i| tk.id_to_bytes(i) == b"function").unwrap() + 1;
            let mut partial = String::new();
            for &id in &ids[s..] {
                if partial.contains('>') { break; }
                assert!(name_candidates(&tk, &names, &partial).contains(&id), "{}: {:?} not a candidate after {:?}", n, txt(&[id]), partial);
                partial.push_str(&String::from_utf8_lossy(tk.id_to_bytes(id)));
            }
        }
        assert!(txt(&name_candidates(&tk, &names, "=Bash")).iter().all(|x| x.starts_with('>')));
        assert!(name_candidates(&tk, &names, "=R").is_empty(), "a name outside the set has no continuation");
    }

    // ------------------------------------------------ ToolStream: incremental tool_use input == the batch result

    fn sreq(extra: serde_json::Value) -> ChatReq {
        let mut b = json!({"messages": [{"role": "user", "content": "x"}], "tools": [
            {"name": "Write", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}}},
            {"name": "Read", "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "limit": {"type": "integer"},
                "offset": {"type": "number"}, "flag": {"type": "boolean"}, "lines": {"type": "array"}, "opt": {"type": ["integer", "string"]},
                "o": {"anyOf": [{"type": "object"}, {"type": "null"}]}}}},
            {"name": "Bash", "input_schema": {"type": "object"}}]});
        for (k, v) in extra.as_object().unwrap() { b[k] = v.clone(); }
        convert(&b).unwrap()
    }

    /// server.rs `finish_reply`'s calls: parse, tool_choice none, allowed names, coerce.
    fn batch(r: &ChatReq, text: &str, think: bool) -> Vec<(String, Value)> {
        let (_, mut calls) = crate::chatmap::parse_tool_calls(&crate::chatmap::reply_text(text, think));
        if r.tool_choice_none { calls.clear(); }
        if let Some(ok) = r.name_constraint() { calls.retain(|c| ok.contains(&c.name)); }
        calls.into_iter().map(|c| { let a = crate::chatmap::coerce_args(&c.args, r.tool_schema(&c.name)); (c.name, a) }).collect()
    }

    /// Feed `text` in random pieces (1..=max chars); (blocks (name, concatenated input JSON), deltas before the end,
    /// mismatch). Checks the event order (start, deltas, stop; one block open at a time).
    fn run(r: &ChatReq, think: bool, text: &str, seed: u64, max: usize) -> (Vec<(String, String)>, usize, bool) {
        let mut ts = ToolStream::new(r, think);
        let chars: Vec<char> = text.chars().collect();
        let (mut evs, mut i, mut x) = (vec![], 0, seed);
        while i < chars.len() {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let n = 1 + (x >> 33) as usize % max;
            evs.extend(ts.push(&chars[i..(i + n).min(chars.len())].iter().collect::<String>()));
            i += n;
        }
        let early = evs.iter().filter(|e| matches!(e, ToolEv::Delta(_))).count();
        evs.extend(ts.finish(&batch(r, text, think)));
        let (mut out, mut open): (Vec<(String, String)>, bool) = (vec![], false);
        for e in evs {
            match e {
                ToolEv::Start(n) => { assert!(!open, "start inside a block"); open = true; out.push((n, String::new())); }
                ToolEv::Delta(d) => { assert!(open && !d.is_empty(), "delta outside a block / empty"); out.last_mut().unwrap().1.push_str(&d); }
                ToolEv::Stop => { assert!(open, "stop outside a block"); open = false; }
            }
        }
        assert!(!open, "a block left open");
        (out, early, ts.mismatch)
    }

    fn check(r: &ChatReq, think: bool, text: &str) -> usize {
        let want: Vec<(String, String)> = batch(r, text, think).into_iter().map(|(n, a)| (n, a.to_string())).collect();
        let mut early = usize::MAX;
        for seed in 0..40u64 {
            for max in [1usize, 3, 9, 40] {
                let (got, e, mm) = run(r, think, text, seed, max);
                assert!(!mm, "mismatch seed {} max {}: {:?}", seed, max, text);
                assert_eq!(got, want, "seed {} max {} text {:?}", seed, max, text);
                early = early.min(e);
            }
        }
        early
    }

    const LONG_WRITE: &str = "I'll write it.\n\n<tool_call>\n<function=Write>\n<parameter=file_path>\n/tmp/a \"b\".py\n</parameter>\n<parameter=content>\n\
import os\n\ndef f(x):\n\treturn \"q\\\\n\" + x  # </par not a close </function x\n\n    s = '\u{e9}\u{2713} \u{1} <tool_c'\n   \n</parameter>\n</function>\n</tool_call>";

    #[test]
    fn stream_equals_batch() {
        let r = sreq(json!({}));
        // a long string argument streams while it is written (many deltas before the reply ends)
        let early = check(&r, false, LONG_WRITE);
        assert!(early >= 5, "only {} deltas before the end", early);
        for t in [
            // typed parameters (sent once complete) between strings; no newlines around a value; spaces kept
            "<tool_call>\n<function=Read>\n<parameter=file_path>\n/x  \n</parameter>\n<parameter=limit>\n20\n</parameter>\n<parameter=offset>1.5</parameter>\n<parameter=flag>\nTrue\n</parameter>\n<parameter=lines>\n[1, 2]\n</parameter>\n<parameter=opt>\n7\n</parameter>\n<parameter=o>\nnull\n</parameter>\n<parameter=extra>\nx\n</parameter>\n</function>\n</tool_call>",
            // a typed value that does not parse stays a string; no </parameter> before the next parameter / the end
            "<tool_call>\n<function=Read>\n<parameter=limit>\nmany\n<parameter=file_path>\n\n\nb\n\n</function>\n</tool_call>",
            // two calls, two functions in one block, a call to a tool without a schema, text between blocks
            "a\n<tool_call>\n<function=Bash>\n<parameter=command>\nls -la\n</parameter>\n<parameter=timeout>\n5000\n</parameter>\n</function>\n<function=Write>\n<parameter=content>\nx\n</parameter>\n</function>\n</tool_call>\nnoise\n<tool_call>\n<function=Read>\n<parameter=file_path>\nq\n</parameter>\n</function>\n</tool_call>",
            // truncated at the end (max_tokens): no closing tags
            "<tool_call>\n<function=Write>\n<parameter=file_path>\nf.txt\n</parameter>\n<parameter=content>\nhalf a line   \n\n",
            // bare function without <tool_call>; then a <tool_call> cutting it
            "ok <function=Bash>\n<parameter=command>\necho </tool_call> hi\n</parameter>\n<tool_call>\n<function=Read>\n<parameter=limit>\n3\n</parameter>\n</function>\n</tool_call>",
            // JSON-form call; an empty name; a name with spaces
            "<tool_call>\n{\"name\": \"Read\", \"arguments\": {\"file_path\": \"j\", \"limit\": \"5\"}}\n</tool_call>\n<tool_call>\n<function=>\n<parameter=a>\nb\n</parameter>\n</function>\n<function= Write >\n<parameter=content>\nz\n</parameter>\n</function>\n</tool_call>",
            // no arguments; no call at all
            "<tool_call>\n<function=Bash>\n</function>\n</tool_call>",
            "just text with a <b>tag</b> and </tool_call> stray",
        ] { check(&r, false, t); }
        // thinking: a call inside the reasoning is not a call
        check(&r, true, "let me <tool_call>\n<function=Bash>\n<parameter=command>\nrm\n</parameter>\n</function>\n</tool_call> think\n</think>\n\nNow.\n\n<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>");
        // allowed names: a call to another tool is never streamed; tool_choice none: nothing
        let two = "<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=Write>\n<parameter=content>\nok\n</parameter>\n</function>\n</tool_call>";
        let a = sreq(json!({"streamdense": {"allowed_tools": ["Write"]}}));
        check(&a, false, two);
        assert_eq!(run(&a, false, two, 1, 4).0.iter().map(|x| x.0.as_str()).collect::<Vec<_>>(), vec!["Write"]);
        let n = sreq(json!({"tool_choice": {"type": "none"}}));
        assert!(run(&n, false, two, 1, 4).0.is_empty());
        // forced call: the opener is the first piece
        let f = sreq(json!({"tool_choice": {"type": "tool", "name": "Write"}}));
        check(&f, false, &format!("{}<parameter=content>\nforced\n</parameter>\n</function>\n</tool_call>", ToolForce::Tool("Write".into()).prefix()));
    }

    #[test]
    fn stream_starts_at_the_name_and_holds_what_may_change() {
        let r = sreq(json!({}));
        let mut ts = ToolStream::new(&r, false);
        assert!(ts.push("Sure.\n\n<tool_call>\n<function=Wri").is_empty());
        assert_eq!(ts.push("te>\n"), vec![ToolEv::Start("Write".into()), ToolEv::Delta("{".into())]);
        assert_eq!(ts.push("<parameter=content>\nab"), vec![ToolEv::Delta("\"content\":\"ab".into())]);
        assert!(ts.push("  \n</para").is_empty(), "trailing spaces / a partial close tag are held");
        assert_eq!(ts.push("graph x"), vec![ToolEv::Delta("  \\n</paragraph x".into())]);
        assert_eq!(ts.push("\n</parameter>\n</function>"), vec![ToolEv::Delta("\"}".into()), ToolEv::Stop]);
        // typed: the key goes out, the value once complete
        let mut ts = ToolStream::new(&r, false);
        ts.push("<tool_call>\n<function=Read>\n");
        assert_eq!(ts.push("<parameter=limit>\n12"), vec![ToolEv::Delta("\"limit\":".into())]);
        assert!(ts.push("3").is_empty());
        assert_eq!(ts.push("\n</parameter>\n"), vec![ToolEv::Delta("123".into())]);
        assert_eq!(ts.finish(&batch(&r, "<tool_call>\n<function=Read>\n<parameter=limit>\n123\n</parameter>\n", false)), vec![ToolEv::Delta("}".into()), ToolEv::Stop]);
    }

    #[test]
    fn cut_call_detects_an_open_call() {
        let w = "Writing.\n<tool_call>\n<function=Write>\n<parameter=file_path>\nchess.js\n</parameter>\n<parameter=content>\nif (castling[color].k &&";
        assert_eq!(cut_call(w), Some(("Write".into(), "\n<parameter=file_path>\nchess.js\n</parameter>\n<parameter=content>\nif (castling[color].k &&".chars().count())));
        // complete calls, no call, a call in the thinking only, a name not complete yet: not cut
        assert_eq!(cut_call(&format!("{}\n</parameter>\n</function>\n</tool_call>", w)), None);
        assert_eq!(cut_call(&format!("{}\n</parameter>\n</function>\n", w)), None, "</function> written: complete");
        assert_eq!(cut_call("plain text, cut mid-sentence"), None);
        assert_eq!(cut_call("<think>\n<tool_call>\n<function=Bash>\n<parameter=command>\nls"), None);
        assert_eq!(cut_call("<tool_call>\n<function=Wri"), None);
        // a second call cut after a complete one
        let two = "<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=Read>\n<parameter=file_path>\na";
        assert_eq!(cut_call(two).map(|c| c.0), Some("Read".into()));
    }

    #[test]
    fn close_cut_closes_the_streamed_block_without_mismatch() {
        let r = sreq(json!({}));
        let text = "Sure.\n<tool_call>\n<function=Bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=Write>\n<parameter=file_path>\nchess.js\n</parameter>\n<parameter=content>\nif (castling[color].k &&";
        let mut ts = ToolStream::new(&r, false);
        let mut evs = vec![];
        for ch in text.chars().collect::<Vec<_>>().chunks(5) { evs.extend(ts.push(&ch.iter().collect::<String>())); }
        assert!(ts.open() && ts.started() == 2);
        let mut calls = batch(&r, text, false);
        assert_eq!(calls.len(), 2, "the parser closes the cut call");
        calls.pop();
        let cut = ts.close_cut();
        assert_eq!(cut, vec![ToolEv::Stop]);
        evs.extend(cut);
        evs.extend(ts.finish(&calls));
        assert!(!ts.mismatch);
        // the first call complete, the second streamed partial and closed
        let starts: Vec<&ToolEv> = evs.iter().filter(|e| matches!(e, ToolEv::Start(_))).collect();
        assert_eq!(starts, vec![&ToolEv::Start("Bash".into()), &ToolEv::Start("Write".into())]);
        assert_eq!(evs.iter().filter(|e| **e == ToolEv::Stop).count(), 2);
        let w: String = evs.iter().skip_while(|e| **e != ToolEv::Start("Write".into())).filter_map(|e| if let ToolEv::Delta(d) = e { Some(d.as_str()) } else { None }).collect();
        assert!(w.starts_with("{\"file_path\":\"chess.js\",\"content\":\"if (castling") && !w.ends_with('}'), "{}", w);
        assert!(ts.close_cut().is_empty(), "nothing open any more");
    }

    #[test]
    fn stream_cost_is_small_for_a_long_write() {
        let r = sreq(json!({}));
        let body: String = (0..1500).map(|i| format!("    line {} = \"x\" + str({})\n", i, i)).collect();
        let text = format!("<tool_call>\n<function=Write>\n<parameter=file_path>\nbig.py\n</parameter>\n<parameter=content>\n{}</parameter>\n</function>\n</tool_call>", body);
        let t0 = std::time::Instant::now();
        let (got, early, mm) = run(&r, false, &text, 7, 4);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        assert!(!mm && early > 1000);
        assert_eq!(got[0].1, batch(&r, &text, false)[0].1.to_string());
        eprintln!("{} bytes, {} deltas: {:.1} ms", text.len(), early, ms);
    }
}
