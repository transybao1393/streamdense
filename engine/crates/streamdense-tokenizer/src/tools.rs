//! Tool calling helpers for the Qwen3.8 chat format: parse tool calls out of generated text (XML + JSON forms),
//! a streaming "how much visible text is safe to emit" helper, argument coercion by JSON schema, and a
//! Python-compatible JSON dumper (the template's `tojson`). See the crate docs for the contract.
use serde_json::{Map, Value};

/// A tool call: function name + arguments (insertion-ordered; `serde_json` is built with `preserve_order`).
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub args: Map<String, Value>,
}

impl ToolCall {
    pub fn new(name: &str, args: Map<String, Value>) -> ToolCall { ToolCall { name: name.into(), args } }
}

/// A fully parsed assistant reply.
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedReply {
    /// Text of the think block(s), trimmed; None when there was none.
    pub reasoning: Option<String>,
    /// Visible text: outside think blocks, before the first tool call, trimmed.
    pub visible: String,
    pub calls: Vec<ToolCall>,
}

/// Result of [`stream_visible`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamVisible {
    /// Visible text safe to emit so far (a prefix of the final visible text).
    pub text: String,
    /// A tool call has started: nothing after it is visible text.
    pub tool_call: bool,
    /// Currently inside an unclosed think block.
    pub in_think: bool,
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";
const FUNC_OPEN: &str = "<function=";
const FUNC_CLOSE: &str = "</function>";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";
// stray closing tool tags (no opening call: the model copying the format from context) are dropped like a stray </think>
const MARKERS: [&str; 7] = [THINK_OPEN, THINK_CLOSE, TOOL_OPEN, FUNC_OPEN, TOOL_CLOSE, FUNC_CLOSE, PARAM_CLOSE];

struct Scan<'a> {
    visible: String,
    reasoning: Vec<&'a str>,
    /// byte offset of the first tool-call marker (`<tool_call>` or bare `<function=`)
    tool_at: Option<usize>,
    in_think: bool,
}

fn earliest(text: &str, from: usize) -> Option<(usize, &'static str)> {
    let mut best: Option<(usize, &'static str)> = None;
    for m in MARKERS {
        if let Some(i) = text[from..].find(m) {
            let i = from + i;
            if best.map_or(true, |(b, _)| i < b) { best = Some((i, m)); }
        }
    }
    best
}

fn scan(text: &str) -> Scan<'_> {
    let mut s = Scan { visible: String::new(), reasoning: vec![], tool_at: None, in_think: false };
    let mut pos = 0;
    // leading reasoning without an opening tag (thinking-on output passed without the "<think>\n" prefix)
    if let Some(k) = text.find(THINK_CLOSE) {
        let first_other = [THINK_OPEN, TOOL_OPEN, FUNC_OPEN].iter().filter_map(|m| text.find(m)).min();
        if first_other.map_or(true, |o| k < o) {
            s.reasoning.push(&text[..k]);
            pos = k + THINK_CLOSE.len();
        }
    }
    while let Some((i, m)) = earliest(text, pos) {
        s.visible.push_str(&text[pos..i]);
        match m {
            THINK_OPEN => {
                let a = i + THINK_OPEN.len();
                match text[a..].find(THINK_CLOSE) {
                    Some(j) => { s.reasoning.push(&text[a..a + j]); pos = a + j + THINK_CLOSE.len(); }
                    None => { s.reasoning.push(&text[a..]); s.in_think = true; return s; }
                }
            }
            THINK_CLOSE | TOOL_CLOSE | FUNC_CLOSE | PARAM_CLOSE => pos = i + m.len(), // stray closing tag: dropped
            _ => { s.tool_at = Some(i); return s; }
        }
    }
    s.visible.push_str(&text[pos..]);
    s
}

/// Parse a generated assistant turn into reasoning, visible text and tool calls (see crate docs).
pub fn parse_reply(text: &str) -> ParsedReply {
    let s = scan(text);
    let reasoning = if s.reasoning.is_empty() { None } else {
        Some(s.reasoning.iter().map(|r| r.trim()).filter(|r| !r.is_empty()).collect::<Vec<_>>().join("\n\n"))
    };
    let calls = match s.tool_at { Some(i) => parse_tool_region(&text[i..]), None => vec![] };
    ParsedReply { reasoning, visible: s.visible.trim().to_string(), calls }
}

/// `(visible_text, tool_calls)` of a generated assistant turn (see crate docs).
pub fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let p = parse_reply(text);
    (p.visible, p.calls)
}

/// Visible text that is safe to emit while `text_so_far` is still growing (see crate docs).
pub fn stream_visible(text_so_far: &str) -> StreamVisible {
    let s = scan(text_so_far);
    let mut v = s.visible;
    if s.tool_at.is_none() && !s.in_think {
        // the tail of `v` is the tail of the text here: hold back what may still turn into something else
        if v.ends_with('\u{FFFD}') { v.truncate(v.len() - '\u{FFFD}'.len_utf8()); }
        'outer: for m in MARKERS.iter() {
            for l in (1..m.len()).rev() {
                if v.ends_with(&m[..l]) { v.truncate(v.len() - l); break 'outer; }
            }
        }
    }
    let t = v.trim();
    StreamVisible { text: t.to_string(), tool_call: s.tool_at.is_some(), in_think: s.in_think }
}

/// Everything from the first tool-call marker on: a sequence of `<tool_call>..</tool_call>` blocks and/or bare
/// `<function=..>..</function>` blocks; missing closing tags run to the end.
fn parse_tool_region(r: &str) -> Vec<ToolCall> {
    let mut calls = vec![];
    let mut p = 0;
    loop {
        let a = r[p..].find(TOOL_OPEN).map(|i| p + i);
        let b = r[p..].find(FUNC_OPEN).map(|i| p + i);
        match (a, b) {
            (Some(a), b) if b.map_or(true, |b| a < b) => {
                let st = a + TOOL_OPEN.len();
                let (end, next) = match r[st..].find(TOOL_CLOSE) {
                    Some(j) => (st + j, st + j + TOOL_CLOSE.len()),
                    None => (r.len(), r.len()),
                };
                calls.extend(parse_block(&r[st..end]));
                p = next;
            }
            (_, Some(b)) => {
                let st = b;
                let close = r[st..].find(FUNC_CLOSE).map(|j| st + j + FUNC_CLOSE.len());
                let next_tc = r[st..].find(TOOL_OPEN).map(|j| st + j);
                let end = match (close, next_tc) {
                    (Some(c), Some(n)) => c.min(n),
                    (Some(c), None) => c,
                    (None, Some(n)) => n,
                    (None, None) => r.len(),
                };
                calls.extend(parse_block(&r[st..end]));
                p = end;
            }
            _ => break,
        }
        if p >= r.len() { break; }
    }
    calls
}

/// Content of one `<tool_call>` block: one or more `<function=..>` blocks, or a JSON object.
fn parse_block(block: &str) -> Vec<ToolCall> {
    let b = block.trim();
    if b.contains(FUNC_OPEN) {
        let mut out = vec![];
        let mut p = 0;
        while let Some(i) = b[p..].find(FUNC_OPEN) {
            let ns = p + i + FUNC_OPEN.len();
            let Some(gt) = b[ns..].find('>') else { break };
            let name = b[ns..ns + gt].trim();
            let bs = ns + gt + 1;
            let (body, next) = match b[bs..].find(FUNC_CLOSE) {
                Some(j) => (&b[bs..bs + j], bs + j + FUNC_CLOSE.len()),
                None => {
                    // truncated: runs to the next function or the end
                    let j = b[bs..].find(FUNC_OPEN).map(|j| bs + j).unwrap_or(b.len());
                    (&b[bs..j], j)
                }
            };
            if !name.is_empty() { out.push(ToolCall { name: name.to_string(), args: parse_params(body) }); }
            p = next;
        }
        return out;
    }
    parse_json_call(b).into_iter().collect()
}

/// Strip exactly one leading and one trailing "\n" (the template writes `<parameter=k>\n{v}\n</parameter>`).
fn strip_one_nl(v: &str) -> &str {
    let v = v.strip_prefix('\n').unwrap_or(v);
    v.strip_suffix('\n').unwrap_or(v)
}

fn parse_params(body: &str) -> Map<String, Value> {
    let mut args = Map::new();
    let mut q = 0;
    while let Some(i) = body[q..].find(PARAM_OPEN) {
        let ns = q + i + PARAM_OPEN.len();
        let Some(gt) = body[ns..].find('>') else { break };
        let name = body[ns..ns + gt].trim().to_string();
        let vs = ns + gt + 1;
        let rest = &body[vs..];
        let close = rest.find(PARAM_CLOSE);
        let next = rest.find(PARAM_OPEN);
        let (vend, adv) = match (close, next) {
            (Some(c), Some(n)) if n < c => (n, n),
            (Some(c), _) => (c, c + PARAM_CLOSE.len()),
            (None, Some(n)) => (n, n),
            (None, None) => (rest.len(), rest.len()),
        };
        let mut v = strip_one_nl(&rest[..vend]);
        if close.is_none() && next.is_none() { v = v.trim_end_matches('\n'); }
        if !name.is_empty() { args.insert(name, Value::String(v.to_string())); }
        q = vs + adv;
    }
    args
}

/// `{"name": .., "arguments": {..} | "<json>"}` (also "parameters" for the arguments).
fn parse_json_call(b: &str) -> Option<ToolCall> {
    let v: Value = serde_json::from_str(b).ok()?;
    let o = v.as_object()?;
    let o = match o.get("function").and_then(|f| f.as_object()) { Some(f) if !o.contains_key("name") => f, _ => o };
    let name = o.get("name")?.as_str()?.to_string();
    let args = match o.get("arguments").or_else(|| o.get("parameters")) {
        Some(Value::Object(m)) => m.clone(),
        Some(Value::String(s)) if !s.trim().is_empty() => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(m)) => m,
            _ => return None,
        },
        _ => Map::new(),
    };
    Some(ToolCall { name, args })
}

/// The JSON types a property schema allows (`"type": "x"` or `"type": [..]`, also looked up in anyOf/oneOf).
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

fn coerce_one(s: &str, ty: &str) -> Option<Value> {
    let t = s.trim();
    match ty {
        "boolean" => match t.to_ascii_lowercase().as_str() { "true" => Some(Value::Bool(true)), "false" => Some(Value::Bool(false)), _ => None },
        "integer" => {
            if let Ok(i) = t.parse::<i64>() { return Some(i.into()); }
            if let Ok(u) = t.parse::<u64>() { return Some(u.into()); }
            let f: f64 = t.parse().ok()?;
            if f.is_finite() && f.fract() == 0.0 && f.abs() < 9.007_199_254_740_992e15 { Some((f as i64).into()) } else { None }
        }
        "number" => {
            if let Ok(i) = t.parse::<i64>() { return Some(i.into()); }
            let f: f64 = t.parse().ok()?;
            serde_json::Number::from_f64(f).map(Value::Number)
        }
        "array" => match serde_json::from_str::<Value>(t) { Ok(v @ Value::Array(_)) => Some(v), _ => None },
        "object" => match serde_json::from_str::<Value>(t) { Ok(v @ Value::Object(_)) => Some(v), _ => None },
        "null" => if t == "null" { Some(Value::Null) } else { None },
        _ => None,
    }
}

/// Convert string argument values to the type `schema["properties"][k]` asks for (see crate docs).
pub fn coerce_args(args: &Map<String, Value>, schema: &Value) -> Map<String, Value> {
    let props = &schema["properties"];
    let mut out = Map::new();
    for (k, v) in args {
        let mut nv = v.clone();
        if let Value::String(s) = v {
            let types = schema_types(&props[k.as_str()]);
            if !types.iter().any(|t| t == "string") {
                for t in &types {
                    if let Some(c) = coerce_one(s, t) { nv = c; break; }
                }
            }
        }
        out.insert(k.clone(), nv);
    }
    out
}

/// Python `repr(float)` (what `json.dumps` writes for a float).
fn py_float(f: f64) -> String {
    if f.is_nan() { return "NaN".into(); }
    if f.is_infinite() { return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() }; }
    if f == 0.0 { return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() }; }
    let e = format!("{:e}", f); // shortest round-trip digits, e.g. "-1.5e20"
    let (mant, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let mut out = String::new();
    if neg { out.push('-'); }
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let ip = (exp + 1) as usize;
            if digits.len() <= ip {
                out.push_str(&digits);
                out.extend(std::iter::repeat('0').take(ip - digits.len()));
                out.push_str(".0");
            } else {
                out.push_str(&digits[..ip]);
                out.push('.');
                out.push_str(&digits[ip..]);
            }
        } else {
            out.push_str("0.");
            out.extend(std::iter::repeat('0').take((-exp - 1) as usize));
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 { out.push('.'); out.push_str(&digits[1..]); }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", exp.abs()));
    }
    out
}

fn py_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn py_json_into(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() { out.push_str(&n.to_string()); } else { out.push_str(&py_float(n.as_f64().unwrap_or(0.0))); }
        }
        Value::String(s) => py_str(s, out),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() { if i > 0 { out.push_str(", "); } py_json_into(x, out); }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 { out.push_str(", "); }
                py_str(k, out);
                out.push_str(": ");
                py_json_into(x, out);
            }
            out.push('}');
        }
    }
}

/// Python `json.dumps(v, ensure_ascii=False)` = the chat template's `tojson` filter (keys in insertion order).
pub fn py_json(v: &Value) -> String {
    let mut s = String::new();
    py_json_into(v, &mut s);
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn obj(v: Value) -> Map<String, Value> { v.as_object().unwrap().clone() }

    // raw outputs of the 27B captured by the throwaway gateway (gw.jsonl)
    const RAW_READ: &str = "<tool_call>\n<function=Read>\n<parameter=file_path>\n/tmp/x/task/hello.py\n</parameter>\n</function>\n</tool_call>";
    const RAW_EDIT: &str = "<tool_call>\n<function=Edit>\n<parameter=replace_all>\nfalse\n</parameter>\n<parameter=file_path>\n/tmp/x/task/hello.py\n</parameter>\n<parameter=old_string>\n    return \"Hello, \" + nme\n</parameter>\n<parameter=new_string>\n    return \"Hello, \" + name\n</parameter>\n</function>\n</tool_call>";
    const RAW_BASH: &str = "<tool_call>\n<function=Bash>\n<parameter=command>\npython3 hello.py\n</parameter>\n<parameter=description>\nRun hello.py to confirm the fix\n</parameter>\n</function>\n</tool_call>";
    const RAW_TEXT: &str = "Fixed. The bug was a typo: `nme` instead of `name` on line 2, which would have raised a `NameError`. The script now runs and prints `Hello, world`.";

    fn edit_schema() -> Value {
        json!({"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "properties": {
            "file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"},
            "replace_all": {"default": false, "type": "boolean"}}, "required": ["file_path", "old_string", "new_string"]})
    }

    #[test]
    fn parse_real_outputs() {
        let (v, c) = parse_tool_calls(RAW_READ);
        assert_eq!(v, "");
        assert_eq!(c, vec![ToolCall::new("Read", obj(json!({"file_path": "/tmp/x/task/hello.py"})))]);
        let (v, c) = parse_tool_calls(RAW_EDIT);
        assert_eq!(v, "");
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].name, "Edit");
        let keys: Vec<&str> = c[0].args.keys().map(|k| k.as_str()).collect();
        assert_eq!(keys, ["replace_all", "file_path", "old_string", "new_string"]);
        assert_eq!(c[0].args["old_string"], json!("    return \"Hello, \" + nme"));
        assert_eq!(c[0].args["replace_all"], json!("false"));
        let (_, c) = parse_tool_calls(RAW_BASH);
        assert_eq!(c[0].args, obj(json!({"command": "python3 hello.py", "description": "Run hello.py to confirm the fix"})));
        let (v, c) = parse_tool_calls(RAW_TEXT);
        assert_eq!((v.as_str(), c.len()), (RAW_TEXT, 0));
    }

    #[test]
    fn coerce_edit_replace_all() {
        let (_, c) = parse_tool_calls(RAW_EDIT);
        let a = coerce_args(&c[0].args, &edit_schema());
        assert_eq!(a["replace_all"], json!(false));
        assert_eq!(a["file_path"], json!("/tmp/x/task/hello.py"));
        let keys: Vec<&str> = a.keys().map(|k| k.as_str()).collect();
        assert_eq!(keys, ["replace_all", "file_path", "old_string", "new_string"]);
        let schema = json!({"properties": {"n": {"type": "integer"}, "x": {"type": "number"}, "t": {"type": "number"},
            "l": {"type": "array"}, "o": {"type": "object"}, "b": {"type": "boolean"}, "s": {"type": ["string", "null"]},
            "u": {"anyOf": [{"type": "integer"}, {"type": "null"}]}, "bad": {"type": "integer"}}});
        let a = coerce_args(&obj(json!({"n": " 42\n", "x": "2.5", "t": "600000", "l": "[1, \"a\"]", "o": "{\"k\": true}",
            "b": "True", "s": "123", "u": "null", "bad": "abc", "extra": "7"})), &schema);
        assert_eq!(Value::Object(a), json!({"n": 42, "x": 2.5, "t": 600000, "l": [1, "a"], "o": {"k": true}, "b": true,
            "s": "123", "u": null, "bad": "abc", "extra": "7"}));
    }

    #[test]
    fn parse_variants() {
        // text before the call, think block, JSON form, two calls, truncated end
        let t = "<think>\nI should read it.\n</think>\n\nLet me look.\n\n<tool_call>\n{\"name\": \"Read\", \"arguments\": {\"file_path\": \"/a\", \"limit\": 5}}\n</tool_call>\n<tool_call>\n<function=Bash>\n<parameter=command>\nls -la\n";
        let p = parse_reply(t);
        assert_eq!(p.reasoning.as_deref(), Some("I should read it."));
        assert_eq!(p.visible, "Let me look.");
        assert_eq!(p.calls, vec![ToolCall::new("Read", obj(json!({"file_path": "/a", "limit": 5}))),
                                 ToolCall::new("Bash", obj(json!({"command": "ls -la"})))]);
        // arguments as a JSON string; reasoning without the opening tag
        let p = parse_reply("thinking...\n</think>\n\n<tool_call>{\"name\": \"X\", \"arguments\": \"{\\\"a\\\": 1}\"}");
        assert_eq!(p.reasoning.as_deref(), Some("thinking..."));
        assert_eq!(p.calls, vec![ToolCall::new("X", obj(json!({"a": 1})))]);
        // bare function block, multi-line value with a trailing newline (template writes value + "\n")
        let p = parse_reply("<function=Write>\n<parameter=content>\nline1\nline2\n\n</parameter>\n</function>");
        assert_eq!(p.calls[0].args["content"], json!("line1\nline2\n"));
        // unclosed think: everything is reasoning, nothing visible
        let p = parse_reply("Card: goal done</parameter>\n</function> next</tool_call>");
        assert_eq!(p.visible, "Card: goal done\n next");
        assert!(p.calls.is_empty());
        let p = parse_reply("<think>\nstill going");
        assert_eq!((p.visible.as_str(), p.calls.len()), ("", 0));
        // missing </parameter> before the next parameter
        let p = parse_reply("<tool_call>\n<function=F>\n<parameter=a>\n1\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>");
        assert_eq!(p.calls[0].args, obj(json!({"a": "1", "b": "2"})));
    }

    #[test]
    fn streaming_is_prefix_of_final() {
        for raw in [RAW_READ, RAW_EDIT, RAW_TEXT, "<think>\nhmm <tool_call> not\n</think>\n\nOK, reading.\n\n<tool_call>\n<function=Read>\n</function>\n</tool_call>",
                    "Use a < b and <thinking> words, then </tool_call stuff \u{FFFD}x  end"] {
            let fin = parse_reply(raw).visible;
            let mut prev = String::new();
            let chars: Vec<(usize, char)> = raw.char_indices().collect();
            for (i, _) in chars.iter().skip(1) {
                let s = stream_visible(&raw[..*i]);
                assert!(s.text.starts_with(&prev), "{:?} !startswith {:?}", s.text, prev);
                assert!(fin.starts_with(&s.text), "{:?} not prefix of final {:?}", s.text, fin);
                prev = s.text;
            }
        }
        assert_eq!(stream_visible("Hi <tool_ca").text, "Hi");
        assert_eq!(stream_visible("Hi </thi").text, "Hi");
        assert_eq!(stream_visible("Hi \u{FFFD}").text, "Hi");
        let s = stream_visible("Reading now.\n\n<tool_call>\n<func");
        assert_eq!((s.text.as_str(), s.tool_call), ("Reading now.", true));
        let s = stream_visible("<think>\nplanning");
        assert_eq!((s.text.as_str(), s.in_think), ("", true));
    }

    #[test]
    fn py_json_matches_python() {
        let v = json!({"b": 1, "a": [true, null, 1.5, 1e20, 1e-5, 100.0, -0.25, 12345678901234567.0], "s": "q\"\\\n\t\u{1}é😀"});
        assert_eq!(py_json(&v), "{\"b\": 1, \"a\": [true, null, 1.5, 1e+20, 1e-05, 100.0, -0.25, 1.2345678901234568e+16], \"s\": \"q\\\"\\\\\\n\\t\\u0001é😀\"}");
        assert_eq!(py_float(0.0001), "0.0001");
        assert_eq!(py_float(1e15), "1000000000000000.0");
        assert_eq!(py_float(123.456), "123.456");
    }
}
