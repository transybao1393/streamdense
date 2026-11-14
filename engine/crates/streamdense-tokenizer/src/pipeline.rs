//! tokenizer.json components other than the BPE model: normalizers, pre-tokenizers, decoders and the
//! post-processor (BOS/EOS template). Each one is parsed from its JSON object into a small enum; anything not
//! listed here is refused at load time with an error that names the component.
//!
//! Supported (HF `tokenizers` semantics):
//! * normalizer: `null`, `NFC`/`NFD`/`NFKC`/`NFKD`, `Lowercase`, `Replace` (String or Regex pattern), `Prepend`,
//!   `Strip`, `Sequence` of these.
//! * pre-tokenizer: `null`, `Split` (Regex or String pattern; Isolated/Removed/MergedWithPrevious/MergedWithNext/
//!   Contiguous; `invert`), `ByteLevel` (`add_prefix_space`, `use_regex` with the GPT-2 pattern; must be the
//!   last stage), `Metaspace` (`replacement`, `prepend_scheme` always/first/never or legacy `add_prefix_space`,
//!   `split`), `Digits` (`individual_digits`), `Sequence` of these.
//! * decoder: `ByteLevel` (alone), or `Replace` (String pattern), `ByteFallback`, `Fuse`, `Strip`, `Metaspace`,
//!   alone or in a `Sequence`.
//! * post-processor: `null`, `ByteLevel`, `TemplateProcessing` (single), `RobertaProcessing`, `BertProcessing`,
//!   `Sequence` of these.
//!
//! Regexes: the `regex` crate has no look-around. The common HF tail `\s+(?!\S)|\s+` is emulated (see
//! [`SplitRe`]); any other look-around is refused.
use regex::Regex;
use serde_json::Value;
use std::borrow::Cow;
use unicode_normalization::UnicodeNormalization;

/// GPT-2 pattern that the `ByteLevel` pre-tokenizer uses when `use_regex` is true.
pub const GPT2_PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const LOOKAHEAD_TAIL: &str = r"|\s+(?!\S)|\s+";

fn ty(v: &Value) -> &str { v["type"].as_str().unwrap_or("?") }

/// A pattern field `{"String": s}` or `{"Regex": r}` -> regex source.
fn pattern_src(v: &Value, what: &str) -> Result<String, String> {
    if let Some(s) = v["String"].as_str() { return Ok(regex::escape(s)); }
    if let Some(r) = v["Regex"].as_str() { return Ok(r.to_string()); }
    Err(format!("{}: pattern {:?}: want String or Regex", what, v))
}

// ---------------------------------------------------------------- regex split

/// Split regex. When the HF pattern ends in `\s+(?!\S)|\s+` it is compiled as `head|\s+`, and a whitespace-only
/// match that did not come from `head`, is at least two chars long and is followed by a non-space char gives
/// back its last char (what `\s+(?!\S)` would have done).
pub struct SplitRe {
    full: Regex,
    head: Option<Regex>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Behavior { Isolated, Removed, MergedWithPrevious, MergedWithNext, Contiguous }

impl Behavior {
    fn parse(v: &Value) -> Result<Behavior, String> {
        Ok(match v.as_str() {
            Some("Isolated") => Behavior::Isolated,
            Some("Removed") => Behavior::Removed,
            Some("MergedWithPrevious") => Behavior::MergedWithPrevious,
            Some("MergedWithNext") => Behavior::MergedWithNext,
            Some("Contiguous") => Behavior::Contiguous,
            _ => return Err(format!("Split behavior {:?} not supported", v)),
        })
    }
}

fn has_lookaround(p: &str) -> bool { p.contains("(?=") || p.contains("(?!") || p.contains("(?<=") || p.contains("(?<!") }

impl SplitRe {
    pub fn new(p: &str) -> Result<SplitRe, String> {
        if let Some(head) = p.strip_suffix(LOOKAHEAD_TAIL) {
            if has_lookaround(head) { return Err(format!("unsupported look-around in pre-tokenizer regex {:?}", p)); }
            let full = Regex::new(&format!("{}|\\s+", head)).map_err(|e| format!("regex: {}", e))?;
            let head = Regex::new(head).map_err(|e| format!("regex: {}", e))?;
            return Ok(SplitRe { full, head: Some(head) });
        }
        if has_lookaround(p) {
            return Err(format!("unsupported look-around in pre-tokenizer regex {:?} (only a trailing {} is emulated)", p, LOOKAHEAD_TAIL));
        }
        Ok(SplitRe { full: Regex::new(p).map_err(|e| format!("regex: {}", e))?, head: None })
    }

    /// Non-empty match spans, left to right.
    fn for_each_match(&self, s: &str, mut f: impl FnMut(usize, usize)) {
        let mut pos = 0;
        while pos < s.len() {
            let Some(m) = self.full.find_at(s, pos) else { break };
            let mut end = m.end();
            if m.start() == end {
                match s[end..].chars().next() { Some(c) => { pos = end + c.len_utf8(); continue; } None => break }
            }
            if let Some(head) = &self.head {
                let t = m.as_str();
                if end < s.len() && t.chars().all(char::is_whitespace) && !head.find(t).is_some_and(|h| h.start() == 0) {
                    let next = s[end..].chars().next().unwrap();
                    let mut cs = t.chars();
                    if !next.is_whitespace() && cs.next().is_some() && cs.next().is_some() {
                        end -= t.chars().next_back().unwrap().len_utf8();
                    }
                }
            }
            f(m.start(), end);
            pos = end;
        }
    }

    /// Split `s` like HF `NormalizedString::split(pattern, behavior)` (+ `invert`); calls `f(piece, start)`.
    pub fn split(&self, s: &str, behavior: Behavior, invert: bool, f: &mut dyn FnMut(&str, usize)) {
        let mut segs: Vec<(usize, usize, bool)> = Vec::new();
        let mut last = 0;
        self.for_each_match(s, |a, b| {
            if a > last { segs.push((last, a, invert)); }
            segs.push((a, b, !invert));
            last = b;
        });
        if last < s.len() { segs.push((last, s.len(), invert)); }
        let mut out: Vec<(usize, usize)> = Vec::with_capacity(segs.len());
        let mut prev_match = false;
        for (a, b, m) in segs {
            match behavior {
                Behavior::Isolated => out.push((a, b)),
                Behavior::Removed => if !m { out.push((a, b)) },
                Behavior::MergedWithPrevious => match out.last_mut() {
                    Some(l) if m && !prev_match => l.1 = b,
                    _ => out.push((a, b)),
                },
                // a delimiter joins the next piece only if that one is not a delimiter too ("--x" -> "-", "-x")
                Behavior::MergedWithNext => match out.last_mut() {
                    Some(l) if prev_match && !m => l.1 = b,
                    _ => out.push((a, b)),
                },
                Behavior::Contiguous => match out.last_mut() {
                    Some(l) if m == prev_match => l.1 = b,
                    _ => out.push((a, b)),
                },
            }
            prev_match = m;
        }
        for (a, b) in out { f(&s[a..b], a); }
    }
}

// ---------------------------------------------------------------- normalizers

pub enum Norm {
    Nfc, Nfd, Nfkc, Nfkd, Lowercase,
    Replace(String, String),
    ReplaceRe(Regex, String),
    Prepend(String),
    Strip { left: bool, right: bool },
}

pub fn parse_normalizer(v: &Value, out: &mut Vec<Norm>) -> Result<(), String> {
    if v.is_null() { return Ok(()); }
    out.push(match ty(v) {
        "Sequence" => {
            for n in v["normalizers"].as_array().ok_or("normalizer Sequence: want normalizers")? { parse_normalizer(n, out)?; }
            return Ok(());
        }
        "NFC" => Norm::Nfc,
        "NFD" => Norm::Nfd,
        "NFKC" => Norm::Nfkc,
        "NFKD" => Norm::Nfkd,
        "Lowercase" => Norm::Lowercase,
        "Replace" => {
            let content = v["content"].as_str().ok_or("normalizer Replace: want content")?.to_string();
            match v["pattern"]["String"].as_str() {
                Some(p) => Norm::Replace(p.to_string(), content),
                None => {
                    let src = pattern_src(&v["pattern"], "normalizer Replace")?;
                    if has_lookaround(&src) { return Err(format!("normalizer Replace: look-around in {:?} not supported", src)); }
                    Norm::ReplaceRe(Regex::new(&src).map_err(|e| format!("normalizer Replace regex: {}", e))?, content)
                }
            }
        }
        "Prepend" => Norm::Prepend(v["prepend"].as_str().ok_or("normalizer Prepend: want prepend")?.to_string()),
        "Strip" => Norm::Strip { left: v["strip_left"].as_bool().unwrap_or(false), right: v["strip_right"].as_bool().unwrap_or(false) },
        t => return Err(format!("normalizer {:?} not supported", t)),
    });
    Ok(())
}

/// Apply normalizers in order. Borrows `s` when nothing changes (NFC quick check for the common case).
pub fn normalize<'a>(norms: &[Norm], s: &'a str) -> Cow<'a, str> {
    use unicode_normalization::{is_nfc_quick, is_nfd_quick, is_nfkc_quick, is_nfkd_quick, IsNormalized::Yes};
    let mut cur: Cow<'a, str> = Cow::Borrowed(s);
    for n in norms {
        if cur.is_empty() { break; }
        let next: Option<String> = match n {
            Norm::Nfc => if is_nfc_quick(cur.chars()) == Yes { None } else { Some(cur.nfc().collect()) },
            Norm::Nfd => if is_nfd_quick(cur.chars()) == Yes { None } else { Some(cur.nfd().collect()) },
            Norm::Nfkc => if is_nfkc_quick(cur.chars()) == Yes { None } else { Some(cur.nfkc().collect()) },
            Norm::Nfkd => if is_nfkd_quick(cur.chars()) == Yes { None } else { Some(cur.nfkd().collect()) },
            Norm::Lowercase => if cur.chars().any(|c| c.is_uppercase()) { Some(cur.to_lowercase()) } else { None },
            Norm::Replace(p, c) => if !p.is_empty() && cur.contains(p.as_str()) { Some(cur.replace(p.as_str(), c)) } else { None },
            Norm::ReplaceRe(re, c) => match re.replace_all(&cur, regex::NoExpand(c)) { Cow::Owned(o) => Some(o), Cow::Borrowed(_) => None },
            Norm::Prepend(p) => Some(format!("{}{}", p, cur)),
            Norm::Strip { left, right } => {
                let mut t: &str = &cur;
                if *left { t = t.trim_start(); }
                if *right { t = t.trim_end(); }
                if t.len() == cur.len() { None } else { Some(t.to_string()) }
            }
        };
        if let Some(o) = next { cur = Cow::Owned(o); }
    }
    cur
}

// ---------------------------------------------------------------- pre-tokenizers

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrependScheme { Always, First, Never }

pub enum Pre {
    Split { re: SplitRe, behavior: Behavior, invert: bool },
    /// GPT-2 byte-level stage: optional prefix space, optional GPT-2 regex split. The byte -> char mapping itself
    /// is implicit (the BPE model then works on raw bytes).
    ByteLevel { add_prefix_space: bool, re: Option<SplitRe> },
    /// `split`: the MergedWithNext splitter on `repl` (None = no split).
    Metaspace { repl: char, prepend: PrependScheme, split: Option<SplitRe> },
}

fn prepend_scheme(v: &Value) -> Result<PrependScheme, String> {
    match v["prepend_scheme"].as_str() {
        Some("always") => Ok(PrependScheme::Always),
        Some("first") => Ok(PrependScheme::First),
        Some("never") => Ok(PrependScheme::Never),
        None => Ok(if v["add_prefix_space"].as_bool().unwrap_or(true) { PrependScheme::Always } else { PrependScheme::Never }),
        Some(x) => Err(format!("Metaspace prepend_scheme {:?} not supported", x)),
    }
}

fn single_char(v: &Value, what: &str) -> Result<char, String> {
    let s = v.as_str().ok_or_else(|| format!("{}: want a string", what))?;
    let mut cs = s.chars();
    match (cs.next(), cs.next()) { (Some(c), None) => Ok(c), _ => Err(format!("{}: want one char, got {:?}", what, s)) }
}

/// `pretokenize`: replaces the regex of the first Split stage (transformers' `pretokenize_regex` override).
pub fn parse_pre(v: &Value, pretokenize: &mut Option<&str>, out: &mut Vec<Pre>) -> Result<(), String> {
    if v.is_null() { return Ok(()); }
    out.push(match ty(v) {
        "Sequence" => {
            for p in v["pretokenizers"].as_array().ok_or("pre_tokenizer Sequence: want pretokenizers")? { parse_pre(p, pretokenize, out)?; }
            return Ok(());
        }
        "Split" => {
            let src = match pretokenize.take() { Some(p) => p.to_string(), None => pattern_src(&v["pattern"], "pre_tokenizer Split")? };
            Pre::Split { re: SplitRe::new(&src)?, behavior: Behavior::parse(&v["behavior"])?, invert: v["invert"].as_bool().unwrap_or(false) }
        }
        // HF `Digits`: split on `char::is_numeric` (= \p{N}); individual_digits isolates each one, else runs
        "Digits" => {
            let individual = v["individual_digits"].as_bool().unwrap_or(false);
            Pre::Split { re: SplitRe::new(if individual { r"\p{N}" } else { r"\p{N}+" })?, behavior: Behavior::Isolated, invert: false }
        }
        "ByteLevel" => Pre::ByteLevel {
            add_prefix_space: v["add_prefix_space"].as_bool().unwrap_or(true),
            re: if v["use_regex"].as_bool().unwrap_or(true) { Some(SplitRe::new(GPT2_PATTERN)?) } else { None },
        },
        "Metaspace" => {
            let repl = single_char(&v["replacement"], "Metaspace replacement")?;
            let split = if v["split"].as_bool().unwrap_or(true) { Some(SplitRe::new(&regex::escape(repl.encode_utf8(&mut [0; 4])))?) } else { None };
            Pre::Metaspace { repl, prepend: prepend_scheme(v)?, split }
        }
        t => return Err(format!("pre_tokenizer {:?} not supported", t)),
    });
    Ok(())
}

// ---------------------------------------------------------------- decoders

/// Per-token decode step (applied to every token's text at load time).
pub enum DStep {
    Replace(String, String),
    ByteFallback,
    Strip { ch: char, start: usize, stop: usize },
    Metaspace(char),
}

/// Sequence-level part of the decoder.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lead {
    None,
    /// `Strip` after `Fuse`: up to `n` leading `ch` of the whole decoded text are removed.
    Fused { ch: char, n: usize },
    /// `Metaspace` decoder (prepend_scheme != never): the first token loses its spaces (= its replacement chars).
    FirstToken,
}

pub struct DecoderSpec {
    pub byte_level: bool,
    pub steps: Vec<DStep>,
    pub lead: Lead,
}

impl DecoderSpec {
    pub fn byte_fallback(&self) -> bool { self.steps.iter().any(|s| matches!(s, DStep::ByteFallback)) }
}

pub fn parse_decoder(v: &Value) -> Result<DecoderSpec, String> {
    let mut d = DecoderSpec { byte_level: false, steps: Vec::new(), lead: Lead::None };
    if v.is_null() { return Err("decoder null not supported (HF would join tokens with spaces)".into()); }
    let list: Vec<&Value> = match ty(v) {
        "Sequence" => v["decoders"].as_array().ok_or("decoder Sequence: want decoders")?.iter().collect(),
        _ => vec![v],
    };
    if list.iter().any(|x| ty(x) == "ByteLevel") {
        if list.iter().any(|x| !matches!(ty(x), "ByteLevel" | "Fuse")) {
            return Err(format!("decoder ByteLevel combined with {:?} not supported", list.iter().map(|x| ty(x)).collect::<Vec<_>>()));
        }
        d.byte_level = true;
        return Ok(d);
    }
    let mut fused = false;
    for x in list {
        match ty(x) {
            "Fuse" => fused = true,
            "Replace" => {
                let p = x["pattern"]["String"].as_str().ok_or_else(|| format!("decoder Replace: only String patterns supported, got {:?}", x["pattern"]))?;
                if fused && p.chars().count() > 1 { return Err("decoder Replace of a multi-char pattern after Fuse not supported".into()); }
                d.steps.push(DStep::Replace(p.to_string(), x["content"].as_str().ok_or("decoder Replace: want content")?.to_string()));
            }
            "ByteFallback" => {
                if fused { return Err("decoder ByteFallback after Fuse not supported".into()); }
                d.steps.push(DStep::ByteFallback);
            }
            "Strip" => {
                let ch = single_char(&x["content"], "decoder Strip content")?;
                let (start, stop) = (x["start"].as_u64().unwrap_or(0) as usize, x["stop"].as_u64().unwrap_or(0) as usize);
                if fused {
                    if stop != 0 { return Err("decoder Strip with stop > 0 after Fuse not supported (not streamable)".into()); }
                    if d.lead != Lead::None { return Err("decoder: two leading strips not supported".into()); }
                    d.lead = Lead::Fused { ch, n: start };
                } else {
                    d.steps.push(DStep::Strip { ch, start, stop });
                }
            }
            "Metaspace" => {
                if fused { return Err("decoder Metaspace after Fuse not supported".into()); }
                let repl = single_char(&x["replacement"], "decoder Metaspace replacement")?;
                d.steps.push(DStep::Metaspace(repl));
                if prepend_scheme(x)? != PrependScheme::Never {
                    if d.lead != Lead::None { return Err("decoder: two leading strips not supported".into()); }
                    d.lead = Lead::FirstToken;
                }
            }
            t => return Err(format!("decoder {:?} not supported", t)),
        }
    }
    Ok(d)
}

/// `<0xNN>` -> NN.
pub fn byte_fallback_token(t: &str) -> Option<u8> {
    let h = t.strip_prefix("<0x")?.strip_suffix('>')?;
    if h.len() != 2 { return None; }
    u8::from_str_radix(h, 16).ok()
}

/// Apply the per-token steps to one token text. Returns (bytes, is a byte-fallback byte).
pub fn decode_token(steps: &[DStep], text: &str) -> (Vec<u8>, bool) {
    let mut s = Cow::Borrowed(text);
    for st in steps {
        match st {
            DStep::Replace(p, c) => if s.contains(p.as_str()) { s = Cow::Owned(s.replace(p.as_str(), c)); },
            DStep::ByteFallback => if let Some(b) = byte_fallback_token(&s) { return (vec![b], true); },
            DStep::Metaspace(r) => if s.contains(*r) { s = Cow::Owned(s.replace(*r, " ")); },
            DStep::Strip { ch, start, stop } => {
                let mut t: &str = &s;
                for _ in 0..*start { match t.strip_prefix(*ch) { Some(r) => t = r, None => break } }
                for _ in 0..*stop { match t.strip_suffix(*ch) { Some(r) => t = r, None => break } }
                if t.len() != s.len() { s = Cow::Owned(t.to_string()); }
            }
        }
    }
    (s.into_owned().into_bytes(), false)
}

// ---------------------------------------------------------------- post-processor

/// Special ids around a single sequence (HF `add_special_tokens=True`).
pub fn parse_post(v: &Value, prefix: &mut Vec<u32>, suffix: &mut Vec<u32>) -> Result<(), String> {
    if v.is_null() { return Ok(()); }
    match ty(v) {
        "ByteLevel" => {}
        "Sequence" => for p in v["processors"].as_array().ok_or("post_processor Sequence: want processors")? { parse_post(p, prefix, suffix)?; },
        "TemplateProcessing" => {
            let mut seen_a = false;
            for piece in v["single"].as_array().ok_or("TemplateProcessing: want single")? {
                if piece.get("Sequence").is_some() { seen_a = true; continue; }
                let name = piece["SpecialToken"]["id"].as_str().ok_or_else(|| format!("TemplateProcessing piece {:?} not supported", piece))?;
                let ids = v["special_tokens"][name]["ids"].as_array().ok_or_else(|| format!("TemplateProcessing: special token {:?} has no ids", name))?;
                let dst = if seen_a { &mut *suffix } else { &mut *prefix };
                for id in ids { dst.push(id.as_u64().ok_or("TemplateProcessing: bad id")? as u32); }
            }
        }
        "RobertaProcessing" | "BertProcessing" => {
            let id = |k: &str| v[k][1].as_u64().map(|x| x as u32).ok_or_else(|| format!("{}: want {} = [token, id]", ty(v), k));
            prefix.push(id("cls")?);
            suffix.push(id("sep")?);
        }
        t => return Err(format!("post_processor {:?} not supported", t)),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn pieces(re: &SplitRe, s: &str, b: Behavior, inv: bool) -> Vec<String> {
        let mut v = vec![];
        re.split(s, b, inv, &mut |p, _| v.push(p.to_string()));
        v
    }

    #[test]
    fn lookahead_emulation_gpt2() {
        let re = SplitRe::new(GPT2_PATTERN).unwrap();
        let p = |s| pieces(&re, s, Behavior::Isolated, false);
        assert_eq!(p("a  b"), ["a", " ", " b"]);
        assert_eq!(p("a \nb"), ["a", " ", "\n", "b"]); // no newline branch in GPT-2: the run gives back its last char
        assert_eq!(p("a   "), ["a", "   "]); // run at the end keeps everything
        assert_eq!(p("it's 12"), ["it", "'s", " 12"]);
        assert!(SplitRe::new(r"(?=x)y").is_err());
    }

    #[test]
    fn lookahead_emulation_newline_branch() {
        // Qwen / Llama 3 style: `\s*[\r\n]+` comes first and must keep its whole match
        let re = SplitRe::new(r"\p{L}+|\s*[\r\n]+|\s+(?!\S)|\s+").unwrap();
        let p = |s| pieces(&re, s, Behavior::Isolated, false);
        assert_eq!(p("a  \nb"), ["a", "  \n", "b"]);
        assert_eq!(p("a   b"), ["a", "  ", " ", "b"]);
    }

    #[test]
    fn split_behaviors() {
        let re = SplitRe::new(&regex::escape("-")).unwrap();
        let p = |b, inv| pieces(&re, "--ab-c", b, inv);
        assert_eq!(p(Behavior::Isolated, false), ["-", "-", "ab", "-", "c"]);
        assert_eq!(p(Behavior::Removed, false), ["ab", "c"]);
        assert_eq!(p(Behavior::MergedWithPrevious, false), ["-", "-", "ab-", "c"]);
        assert_eq!(p(Behavior::MergedWithNext, false), ["-", "-ab", "-c"]);
        assert_eq!(p(Behavior::Contiguous, false), ["--", "ab", "-", "c"]);
        assert_eq!(p(Behavior::Removed, true), ["-", "-", "-"]);
    }

    #[test]
    fn normalizers() {
        let mut n = vec![];
        parse_normalizer(&json!({"type": "Sequence", "normalizers": [{"type": "NFKC"}, {"type": "Lowercase"},
            {"type": "Strip", "strip_left": true, "strip_right": false}, {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
            {"type": "Prepend", "prepend": "▁"}]}), &mut n).unwrap();
        assert_eq!(normalize(&n, "  Ab ﬁ "), "▁ab▁fi▁");
        assert_eq!(normalize(&n, ""), "");
        let mut n = vec![];
        parse_normalizer(&json!({"type": "NFC"}), &mut n).unwrap();
        assert!(matches!(normalize(&n, "plain"), Cow::Borrowed(_)));
        assert_eq!(normalize(&n, "e\u{301}"), "\u{e9}");
        let mut n = vec![];
        parse_normalizer(&json!({"type": "Replace", "pattern": {"Regex": " +"}, "content": "$1"}), &mut n).unwrap();
        assert_eq!(normalize(&n, "a  b"), "a$1b"); // content is literal, no `$` expansion
        assert!(parse_normalizer(&json!({"type": "Precompiled"}), &mut vec![]).unwrap_err().contains("Precompiled"));
    }

    #[test]
    fn decoders() {
        let d = parse_decoder(&json!({"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"}, {"type": "Fuse"}, {"type": "Strip", "content": " ", "start": 1, "stop": 0}]})).unwrap();
        assert!(!d.byte_level && d.byte_fallback());
        assert_eq!(d.lead, Lead::Fused { ch: ' ', n: 1 });
        assert_eq!(decode_token(&d.steps, "▁hi▁"), (b" hi ".to_vec(), false));
        assert_eq!(decode_token(&d.steps, "<0x0A>"), (vec![10], true));
        assert_eq!(decode_token(&d.steps, "<0x0G>"), (b"<0x0G>".to_vec(), false));
        let d = parse_decoder(&json!({"type": "Metaspace", "replacement": "▁", "prepend_scheme": "first", "split": true})).unwrap();
        assert_eq!(d.lead, Lead::FirstToken);
        let d = parse_decoder(&json!({"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never", "split": true})).unwrap();
        assert_eq!(d.lead, Lead::None);
        let d = parse_decoder(&json!({"type": "Sequence", "decoders": [{"type": "Strip", "content": "x", "start": 2, "stop": 1}]})).unwrap();
        assert_eq!(decode_token(&d.steps, "xxxaxx").0, b"xax");
        assert!(parse_decoder(&json!({"type": "ByteLevel"})).unwrap().byte_level);
        assert!(parse_decoder(&json!({"type": "Sequence", "decoders": [{"type": "ByteLevel"}, {"type": "Strip", "content": " ", "start": 1, "stop": 0}]})).is_err());
    }

    #[test]
    fn post_processors() {
        let (mut p, mut s) = (vec![], vec![]);
        parse_post(&json!({"type": "Sequence", "processors": [{"type": "ByteLevel"}, {"type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "<eos>", "type_id": 0}}],
            "special_tokens": {"<bos>": {"id": "<bos>", "ids": [7], "tokens": ["<bos>"]}, "<eos>": {"id": "<eos>", "ids": [8], "tokens": ["<eos>"]}}}]}), &mut p, &mut s).unwrap();
        assert_eq!((p, s), (vec![7], vec![8]));
        let (mut p, mut s) = (vec![], vec![]);
        parse_post(&json!({"type": "RobertaProcessing", "sep": ["</s>", 2], "cls": ["<s>", 0]}), &mut p, &mut s).unwrap();
        assert_eq!((p, s), (vec![0], vec![2]));
    }
}
