//! Byte-level BPE tokenizer for Qwen3.x `tokenizer.json` (HF `tokenizers` semantics, encode + decode).
//!
//! Pipeline (matches tokenizer.json of models/qwen38_27b_mlx4):
//!   1. added tokens (`<|im_start|>`, `<think>`, ...; all `normalized: false`) are split out of the raw text first
//!      (leftmost, longest match), and map straight to their ids;
//!   2. every other segment: NFC normalizer -> regex Split (Isolated) -> ByteLevel (GPT-2 byte -> char map)
//!      -> BPE with merge ranks (lowest rank first, leftmost on ties; no `ignore_merges` shortcut).
//!
//! The pre-tokenizer regex has a look-ahead (`\s+(?!\S)`), which the `regex` crate does not support. It is
//! emulated: the pattern uses `\s+` in its place, and a whitespace-only match without a line break that is
//! followed by a non-space character gives back its last character (what `\s+(?!\S)` would have done).
//!
//! Decode: ids -> bytes (byte-level chars back to bytes, added tokens as their text) -> UTF-8 with U+FFFD for
//! invalid sequences (`String::from_utf8_lossy`, like HF). [`StreamDecoder`] emits only complete characters.
use regex::Regex;
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

/// Pre-tokenizer regex that transformers' `Qwen3_5Tokenizer` installs (it overrides tokenizer.json, whose
/// pattern lacks `\p{M}`; tokenizer_config.json carries the same text as `pretokenize_regex`). The Python UI
/// tokenized through transformers, so this is the reference.
pub const QWEN35_PRETOKENIZE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const LOOKAHEAD_TAIL: &str = r"|\s+(?!\S)|\s+";

/// HF pattern -> `regex`-crate pattern: the `\s+(?!\S)|\s+` tail becomes `\s+` (look-ahead emulated in code).
fn convert_pattern(p: &str) -> Result<String, String> {
    let head = p.strip_suffix(LOOKAHEAD_TAIL).ok_or_else(|| format!("unsupported pre-tokenizer regex (want a ...{} tail): {:?}", LOOKAHEAD_TAIL, p))?;
    if head.contains("(?=") || head.contains("(?!") || head.contains("(?<") { return Err(format!("unsupported look-around in {:?}", p)); }
    Ok(format!("{}|\\s+", head))
}

pub struct Tokenizer {
    vocab: HashMap<Vec<u8>, u32>,
    /// id -> bytes (model vocab decoded from byte-level chars; added tokens as their UTF-8 text)
    id_bytes: Vec<Vec<u8>>,
    /// (left id, right id) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    byte_id: [u32; 256],
    /// added tokens, longest first
    added: Vec<(String, u32)>,
    special: Vec<bool>,
    re: Regex,
}

/// GPT-2 `bytes_to_unicode`: byte -> printable char.
fn byte_to_char() -> [char; 256] {
    let mut m = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        m[b as usize] = if printable { char::from_u32(b).unwrap() } else { n += 1; char::from_u32(255 + n).unwrap() };
    }
    m
}

impl Tokenizer {
    /// Load `DIR/tokenizer.json` (or a direct path to a tokenizer.json file). The pre-tokenizer regex is the one
    /// transformers uses: `pretokenize_regex` of tokenizer_config.json next to it when present, else
    /// [`QWEN35_PRETOKENIZE`] for a `Qwen3_5Tokenizer` class, else the tokenizer.json pattern.
    pub fn from_path(path: &str) -> Result<Tokenizer, String> {
        let p = std::path::Path::new(path);
        let file = if p.is_dir() { p.join("tokenizer.json") } else { p.to_path_buf() };
        let s = std::fs::read_to_string(&file).map_err(|e| format!("read {}: {}", file.display(), e))?;
        let cfg_path = file.with_file_name("tokenizer_config.json");
        let mut pat: Option<String> = None;
        if let Ok(c) = std::fs::read_to_string(&cfg_path) {
            let c: serde_json::Value = serde_json::from_str(&c).map_err(|e| format!("{}: {}", cfg_path.display(), e))?;
            pat = c["pretokenize_regex"].as_str().map(|x| x.to_string());
            if pat.is_none() && c["tokenizer_class"].as_str() == Some("Qwen3_5Tokenizer") { pat = Some(QWEN35_PRETOKENIZE.into()); }
        }
        Self::from_json_str(&s, pat.as_deref()).map_err(|e| format!("{}: {}", file.display(), e))
    }

    /// `pretokenize`: the HF Split regex to use instead of the one in tokenizer.json (None = tokenizer.json's).
    pub fn from_json_str(s: &str, pretokenize: Option<&str>) -> Result<Tokenizer, String> {
        let j: serde_json::Value = serde_json::from_str(s).map_err(|e| format!("parse: {}", e))?;
        let model = &j["model"];
        if model["type"].as_str() != Some("BPE") { return Err(format!("model type {:?}, want BPE", model["type"])); }
        if j["normalizer"]["type"].as_str() != Some("NFC") { return Err(format!("normalizer {:?}, want NFC", j["normalizer"])); }
        let pre = j["pre_tokenizer"]["pretokenizers"].as_array().ok_or("pre_tokenizer: want a Sequence")?;
        let pat = pre.first().and_then(|p| p["pattern"]["Regex"].as_str()).ok_or("pre_tokenizer[0]: want a Split regex")?;
        let pat = convert_pattern(pretokenize.unwrap_or(pat))?;
        if pre.get(1).and_then(|p| p["type"].as_str()) != Some("ByteLevel") { return Err("pre_tokenizer[1]: want ByteLevel".into()); }

        let b2c = byte_to_char();
        let mut c2b: HashMap<char, u8> = HashMap::with_capacity(256);
        for (b, c) in b2c.iter().enumerate() { c2b.insert(*c, b as u8); }
        let to_bytes = |t: &str| -> Result<Vec<u8>, String> {
            t.chars().map(|c| c2b.get(&c).copied().ok_or_else(|| format!("token {:?}: char {:?} is not byte-level", t, c))).collect()
        };

        let vj = model["vocab"].as_object().ok_or("model.vocab: want an object")?;
        let mut vocab: HashMap<Vec<u8>, u32> = HashMap::with_capacity(vj.len());
        let mut max_id = 0u32;
        for (t, id) in vj {
            let id = id.as_u64().ok_or("vocab id")? as u32;
            max_id = max_id.max(id);
            vocab.insert(to_bytes(t)?, id);
        }
        let added_j = j["added_tokens"].as_array().cloned().unwrap_or_default();
        for a in &added_j { max_id = max_id.max(a["id"].as_u64().unwrap_or(0) as u32); }
        let n = max_id as usize + 1;
        let mut id_bytes = vec![Vec::new(); n];
        for (b, &id) in &vocab { id_bytes[id as usize] = b.clone(); }
        let mut special = vec![false; n];
        let mut added = Vec::with_capacity(added_j.len());
        for a in &added_j {
            let (c, id) = (a["content"].as_str().ok_or("added token content")?.to_string(), a["id"].as_u64().ok_or("added token id")? as u32);
            if a["normalized"].as_bool() == Some(true) || a["lstrip"].as_bool() == Some(true) || a["rstrip"].as_bool() == Some(true)
                || a["single_word"].as_bool() == Some(true) {
                return Err(format!("added token {:?}: normalized/lstrip/rstrip/single_word not supported", c));
            }
            special[id as usize] = a["special"].as_bool().unwrap_or(false);
            id_bytes[id as usize] = c.as_bytes().to_vec();
            added.push((c, id));
        }
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let mut byte_id = [u32::MAX; 256];
        for b in 0..256usize {
            byte_id[b] = *vocab.get(&vec![b as u8]).ok_or_else(|| format!("byte {} has no token", b))?;
        }
        let mj = model["merges"].as_array().ok_or("model.merges: want an array")?;
        let mut merges = HashMap::with_capacity(mj.len());
        for (rank, m) in mj.iter().enumerate() {
            let (a, b) = match m {
                serde_json::Value::Array(p) if p.len() == 2 => (p[0].as_str().unwrap_or("").to_string(), p[1].as_str().unwrap_or("").to_string()),
                serde_json::Value::String(s) => { let mut it = s.splitn(2, ' '); (it.next().unwrap_or("").to_string(), it.next().unwrap_or("").to_string()) }
                _ => return Err(format!("merge {}: bad entry", rank)),
            };
            let (ab, bb) = (to_bytes(&a)?, to_bytes(&b)?);
            let ia = *vocab.get(&ab).ok_or_else(|| format!("merge {}: {:?} not in vocab", rank, a))?;
            let ib = *vocab.get(&bb).ok_or_else(|| format!("merge {}: {:?} not in vocab", rank, b))?;
            let mut cat = ab; cat.extend_from_slice(&bb);
            let ic = *vocab.get(&cat).ok_or_else(|| format!("merge {}: {:?}+{:?} not in vocab", rank, a, b))?;
            merges.entry((ia, ib)).or_insert((rank as u32, ic));
        }
        let re = Regex::new(&pat).map_err(|e| format!("regex: {}", e))?;
        Ok(Tokenizer { vocab, id_bytes, merges, byte_id, added, special, re })
    }

    /// Number of ids (model vocab + added tokens) = highest id + 1.
    pub fn vocab_size(&self) -> usize { self.id_bytes.len() }
    pub fn model_vocab_len(&self) -> usize { self.vocab.len() }
    pub fn merges_len(&self) -> usize { self.merges.len() }
    /// Id of an added token (`<|im_end|>`) or of a model token given as plain text.
    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.added.iter().find(|(c, _)| c == s).map(|x| x.1).or_else(|| self.vocab.get(s.as_bytes()).copied())
    }
    pub fn is_special(&self, id: u32) -> bool { self.special.get(id as usize).copied().unwrap_or(false) }
    /// Raw bytes of one id (empty for unknown ids).
    pub fn id_to_bytes(&self, id: u32) -> &[u8] { self.id_bytes.get(id as usize).map(|v| v.as_slice()).unwrap_or(&[]) }

    /// Encode text (no BOS/EOS added; `add_special_tokens=False`). Added tokens in the text become their ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 4);
        let mut seg = 0; // start of the current plain segment
        let mut i = 0;
        let b = text.as_bytes();
        while i < b.len() {
            if b[i] == b'<' {
                if let Some((c, id)) = self.added.iter().find(|(c, _)| text[i..].starts_with(c.as_str())) {
                    self.encode_plain(&text[seg..i], &mut out);
                    out.push(*id);
                    i += c.len();
                    seg = i;
                    continue;
                }
            }
            i += 1;
        }
        self.encode_plain(&text[seg..], &mut out);
        out
    }

    fn encode_plain(&self, s: &str, out: &mut Vec<u32>) {
        if s.is_empty() { return; }
        let norm: String;
        let s = if unicode_normalization::is_nfc_quick(s.chars()) == unicode_normalization::IsNormalized::Yes { s }
                else { norm = s.nfc().collect(); &norm };
        let mut pos = 0;
        while pos < s.len() {
            let Some(m) = self.re.find_at(s, pos) else { self.bpe(s[pos..].as_bytes(), out); break };
            if m.start() > pos { self.bpe(s[pos..m.start()].as_bytes(), out); }
            let mut end = m.end();
            let t = m.as_str();
            // `\s+(?!\S)`: a whitespace run (no line break) followed by a non-space char gives back its last char
            if end < s.len() && !t.ends_with(['\r', '\n']) && t.chars().all(char::is_whitespace) {
                let next = s[end..].chars().next().unwrap();
                let mut cs = t.chars();
                if !next.is_whitespace() && cs.next().is_some() && cs.next().is_some() {
                    end -= t.chars().next_back().unwrap().len_utf8();
                }
            }
            self.bpe(s[m.start()..end].as_bytes(), out);
            pos = end;
        }
    }

    fn bpe(&self, bytes: &[u8], out: &mut Vec<u32>) {
        if bytes.is_empty() { return; }
        let mut sym: Vec<u32> = bytes.iter().map(|&b| self.byte_id[b as usize]).collect();
        loop {
            let mut best: Option<(u32, usize, u32)> = None; // (rank, position, merged id)
            for i in 0..sym.len().saturating_sub(1) {
                if let Some(&(r, id)) = self.merges.get(&(sym[i], sym[i + 1])) {
                    if best.map_or(true, |(br, _, _)| r < br) { best = Some((r, i, id)); }
                }
            }
            let Some((_, i, id)) = best else { break };
            sym[i] = id;
            sym.remove(i + 1);
        }
        out.extend_from_slice(&sym);
    }

    /// Decode ids to text (special tokens kept as their text; invalid UTF-8 -> U+FFFD).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut b = Vec::with_capacity(ids.len() * 4);
        for &id in ids { b.extend_from_slice(self.id_to_bytes(id)); }
        String::from_utf8_lossy(&b).into_owned()
    }
}

/// Incremental decoder: push ids, get back only complete UTF-8 text (a char split across tokens is held).
#[derive(Default)]
pub struct StreamDecoder {
    buf: Vec<u8>,
}

impl StreamDecoder {
    pub fn new() -> Self { Self::default() }
    /// Append one token; returns the text that became complete (may be empty).
    pub fn push(&mut self, tk: &Tokenizer, id: u32) -> String {
        self.buf.extend_from_slice(tk.id_to_bytes(id));
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.buf) {
                Ok(s) => { out.push_str(s); self.buf.clear(); break; }
                Err(e) => {
                    let ok = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.buf[..ok]).unwrap());
                    match e.error_len() {
                        None => { self.buf.drain(..ok); break; } // incomplete tail: wait for more bytes
                        Some(n) => { out.push('\u{FFFD}'); self.buf.drain(..ok + n); }
                    }
                }
            }
        }
        out
    }
    /// Bytes still held (an unfinished character).
    pub fn pending(&self) -> usize { self.buf.len() }
    /// End of stream: whatever is held, with U+FFFD for an unfinished character.
    pub fn finish(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.buf).into_owned();
        self.buf.clear();
        s
    }
}
