//! BPE tokenizer for HF `tokenizer.json` files (HF `tokenizers` semantics, encode + decode), for dense model
//! families with a BPE model: Qwen, GPT-2 style byte-level (Granite, SmolLM2, ...), Llama 3 style
//! (Split + ByteLevel, `ignore_merges`), and SentencePiece-style BPE (Llama 2, Gemma, Mistral: Prepend/Replace
//! normalizers or Metaspace, `byte_fallback`). Unigram / WordPiece / WordLevel models are refused.
//!
//! Pipeline (same order as HF):
//!   1. added tokens with `normalized: false` are split out of the raw text first (leftmost, longest match;
//!      `lstrip`/`rstrip` take the whitespace next to them, `single_word` needs non-word chars around);
//!   2. every other segment: normalizers (see [`crate::pipeline`]) -> added tokens with `normalized: true`
//!      (matched against their normalized content) -> pre-tokenizer stages -> BPE.
//!   3. BPE: byte-level models (a ByteLevel pre-tokenizer) start from bytes; others start from chars, with
//!      `<0xNN>` byte tokens for chars missing from the vocab (`byte_fallback`) or `unk_token` (`fuse_unk`).
//!      Merges: lowest rank first, leftmost on ties; `ignore_merges` emits a whole word found in the vocab as is.
//!
//! Regex look-ahead (`\s+(?!\S)`) is emulated, see [`crate::pipeline::SplitRe`].
//!
//! Decode: ids -> bytes (byte-level chars back to bytes / decoder steps such as `Replace("▁", " ")` and
//! `ByteFallback`; added tokens as their text) -> UTF-8 with U+FFFD for invalid sequences (`from_utf8_lossy`;
//! an invalid run of `<0xNN>` tokens gives one U+FFFD per token, like HF) -> leading strip (`Strip` after `Fuse`,
//! `Metaspace` first token). [`StreamDecoder`] emits only complete characters.
//!
//! BOS/EOS: [`Tokenizer::encode`] adds none (`add_special_tokens=False`); [`Tokenizer::encode_with_special`] adds
//! the post-processor's template ids ([`Tokenizer::special_prefix`] / [`Tokenizer::special_suffix`]).
use crate::pipeline::{self, Behavior, Lead, Norm, Pre, PrependScheme};
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Reverse;

/// Pre-tokenizer regex that transformers' `Qwen3_5Tokenizer` installs (it overrides tokenizer.json, whose
/// pattern lacks `\p{M}`; tokenizer_config.json carries the same text as `pretokenize_regex`). The Python UI
/// tokenized through transformers, so this is the reference.
pub const QWEN35_PRETOKENIZE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Words longer than this (in initial symbols) use the heap merge loop instead of the quadratic scan.
const HEAP_BPE_MIN: usize = 64;

struct Added {
    content: String,
    id: u32,
    lstrip: bool,
    rstrip: bool,
    single_word: bool,
}

/// A set of added tokens matched in one pass (longest first, so the first hit at a position is the longest).
#[derive(Default)]
struct AddedSet {
    toks: Vec<Added>,
    first: Vec<bool>, // 256 entries once non-empty: first byte of some token
}

impl AddedSet {
    fn finish(&mut self) {
        self.toks.sort_by(|a, b| b.content.len().cmp(&a.content.len()));
        self.first = vec![false; 256];
        for t in &self.toks { if let Some(&b) = t.content.as_bytes().first() { self.first[b as usize] = true; } }
    }

    /// Split `text` into plain spans and added-token ids (HF `AddedVocabulary::find_matches`).
    fn split(&self, text: &str, f: &mut dyn FnMut(Piece)) {
        let b = text.as_bytes();
        let is_word = |c: Option<char>| c.is_some_and(|c| c.is_alphanumeric() || c == '_');
        let (mut seg, mut i) = (0, 0);
        while !self.toks.is_empty() && i < b.len() {
            if self.first[b[i] as usize] {
                if let Some(t) = self.toks.iter().find(|t| !t.content.is_empty() && text[i..].starts_with(t.content.as_str())) {
                    let stop = i + t.content.len();
                    if t.single_word && (is_word(text[..i].chars().next_back()) || is_word(text[stop..].chars().next())) {
                        i = stop; // HF skips this match and searches on after it
                        continue;
                    }
                    let mut start = i;
                    if t.lstrip { start = seg + text[seg..i].trim_end().len(); }
                    let mut end = stop;
                    if t.rstrip { end = text.len() - text[stop..].trim_start().len(); }
                    if start > seg { f(Piece::Text(seg, start)); }
                    f(Piece::Token(t.id));
                    seg = end;
                    i = end;
                    continue;
                }
            }
            i += 1;
        }
        if seg < text.len() { f(Piece::Text(seg, text.len())); }
    }
}

enum Piece { Text(usize, usize), Token(u32) }

pub struct Tokenizer {
    /// model token -> id. Keys: raw bytes for byte-level models, the token's UTF-8 text otherwise.
    vocab: HashMap<Vec<u8>, u32>,
    /// id -> decoded bytes (decoder steps applied; added tokens as their UTF-8 text)
    id_bytes: Vec<Vec<u8>>,
    /// (left id, right id) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    byte_level: bool,
    /// byte-level: byte -> id. Otherwise `<0xNN>` -> id when `byte_fallback` (u32::MAX = missing).
    byte_id: [u32; 256],
    byte_fallback: bool,
    unk: Option<u32>,
    fuse_unk: bool,
    ignore_merges: bool,
    added_raw: AddedSet,
    added_norm: AddedSet,
    special: Vec<bool>,
    norms: Vec<Norm>,
    pre: Vec<Pre>,
    /// ids decoded from `<0xNN>` byte-fallback tokens (empty when the decoder has no ByteFallback)
    bf_ids: Vec<bool>,
    lead: Lead,
    prefix: Vec<u32>,
    suffix: Vec<u32>,
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
    /// Load `DIR/tokenizer.json` (or a direct path to a tokenizer.json file). For a Split pre-tokenizer the regex
    /// is the one transformers uses: `pretokenize_regex` of tokenizer_config.json next to it when present, else
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

    /// `pretokenize`: the HF Split regex to use instead of the first Split's regex in tokenizer.json
    /// (None = tokenizer.json's). Errors name the unsupported component.
    pub fn from_json_str(s: &str, pretokenize: Option<&str>) -> Result<Tokenizer, String> {
        let j: serde_json::Value = serde_json::from_str(s).map_err(|e| format!("parse: {}", e))?;
        let model = &j["model"];
        match model["type"].as_str() {
            Some("BPE") => {}
            Some(t @ ("Unigram" | "WordPiece" | "WordLevel")) => return Err(format!("model type {} not supported (BPE only)", t)),
            t => return Err(format!("model type {:?} not supported (BPE only)", t)),
        }
        if model["dropout"].as_f64().is_some_and(|d| d > 0.0) { return Err("BPE dropout not supported".into()); }
        for k in ["continuing_subword_prefix", "end_of_word_suffix"] {
            if model[k].as_str().is_some_and(|x| !x.is_empty()) { return Err(format!("BPE {} not supported", k)); }
        }

        let mut norms = Vec::new();
        pipeline::parse_normalizer(&j["normalizer"], &mut norms)?;
        let mut pre = Vec::new();
        let mut pt = pretokenize;
        pipeline::parse_pre(&j["pre_tokenizer"], &mut pt, &mut pre)?;
        if pt.is_some() { return Err("pretokenize regex override given but the pre_tokenizer has no Split".into()); }
        let byte_level = match pre.iter().position(|p| matches!(p, Pre::ByteLevel { .. })) {
            Some(i) if i + 1 == pre.len() => true,
            Some(_) => return Err("pre_tokenizer: ByteLevel must be the last stage".into()),
            None => false,
        };
        let dec = pipeline::parse_decoder(&j["decoder"])?;
        if dec.byte_level != byte_level {
            return Err(format!("decoder ByteLevel = {} but pre_tokenizer ByteLevel = {}: mixed byte-level setup not supported", dec.byte_level, byte_level));
        }
        let mut prefix = Vec::new();
        let mut suffix = Vec::new();
        pipeline::parse_post(&j["post_processor"], &mut prefix, &mut suffix)?;

        let b2c = byte_to_char();
        let mut c2b: HashMap<char, u8> = HashMap::with_capacity(256);
        for (b, c) in b2c.iter().enumerate() { c2b.insert(*c, b as u8); }
        let to_bytes = |t: &str| -> Result<Vec<u8>, String> {
            if !byte_level { return Ok(t.as_bytes().to_vec()); }
            t.chars().map(|c| c2b.get(&c).copied().ok_or_else(|| format!("token {:?}: char {:?} is not byte-level", t, c))).collect()
        };

        let vj = model["vocab"].as_object().ok_or("model.vocab: want an object")?;
        let mut vocab: HashMap<Vec<u8>, u32> = HashMap::with_capacity(vj.len());
        let mut max_id = 0u32;
        // byte-level vocab entries with chars outside the GPT-2 byte alphabet (Cohere2: 1403 raw Unicode tokens such as
        // U+200D): HF's ByteLevel decoder gives them their raw UTF-8 bytes, and the byte-level pre-tokenizer can never
        // produce them, so they are decode-only (kept out of `vocab`: their raw bytes may equal another token's)
        let mut raw_ids: Vec<(u32, Vec<u8>)> = Vec::new();
        for (t, id) in vj {
            let id = id.as_u64().ok_or("vocab id")? as u32;
            max_id = max_id.max(id);
            if byte_level && t.chars().any(|c| !c2b.contains_key(&c)) { raw_ids.push((id, t.as_bytes().to_vec())); continue; }
            vocab.insert(to_bytes(t)?, id);
        }
        let added_j = j["added_tokens"].as_array().cloned().unwrap_or_default();
        for a in &added_j { max_id = max_id.max(a["id"].as_u64().unwrap_or(0) as u32); }
        let n = max_id as usize + 1;
        let mut id_bytes = vec![Vec::new(); n];
        let use_bf = dec.byte_fallback();
        let mut bf_ids = if use_bf { vec![false; n] } else { Vec::new() };
        if byte_level {
            for (b, &id) in &vocab { id_bytes[id as usize] = b.clone(); }
            for (id, b) in raw_ids { id_bytes[id as usize] = b; }
        } else {
            for (t, &id) in &vocab {
                let (b, is_bf) = pipeline::decode_token(&dec.steps, std::str::from_utf8(t).unwrap());
                id_bytes[id as usize] = b;
                if is_bf { bf_ids[id as usize] = true; }
            }
        }
        let mut special = vec![false; n];
        let (mut added_raw, mut added_norm) = (AddedSet::default(), AddedSet::default());
        for a in &added_j {
            let (c, id) = (a["content"].as_str().ok_or("added token content")?.to_string(), a["id"].as_u64().ok_or("added token id")? as u32);
            special[id as usize] = a["special"].as_bool().unwrap_or(false);
            let normalized = a["normalized"].as_bool().unwrap_or(!special[id as usize]);
            // HF decodes a normalized added token as its normalized text (e.g. "a b" -> "▁a▁b" -> " a b")
            let content = if normalized { pipeline::normalize(&norms, &c).into_owned() } else { c };
            if byte_level {
                id_bytes[id as usize] = content.as_bytes().to_vec();
            } else {
                let (b, is_bf) = pipeline::decode_token(&dec.steps, &content);
                id_bytes[id as usize] = b;
                if is_bf { bf_ids[id as usize] = true; }
            }
            let t = Added { content, id, lstrip: a["lstrip"].as_bool().unwrap_or(false), rstrip: a["rstrip"].as_bool().unwrap_or(false),
                            single_word: a["single_word"].as_bool().unwrap_or(false) };
            if normalized { added_norm.toks.push(t) } else { added_raw.toks.push(t) }
        }
        added_raw.finish();
        added_norm.finish();

        let byte_fallback = model["byte_fallback"].as_bool().unwrap_or(false);
        if byte_fallback && byte_level { return Err("BPE byte_fallback with a ByteLevel pre-tokenizer not supported".into()); }
        let mut byte_id = [u32::MAX; 256];
        for b in 0..256usize {
            if byte_level {
                // a byte without a token (SmolLM's vocab lacks 21) is dropped as HF BPE drops a char it has no id
                // for when there is no unk_token (`model`); with an unk_token it becomes that
                byte_id[b] = vocab.get(&vec![b as u8]).copied().unwrap_or(u32::MAX);
            } else if byte_fallback {
                byte_id[b] = vocab.get(format!("<0x{:02X}>", b).as_bytes()).copied().unwrap_or(u32::MAX);
            }
        }
        let unk = match model["unk_token"].as_str() {
            Some(u) => Some(*vocab.get(&to_bytes(u)?).ok_or_else(|| format!("unk_token {:?} not in vocab", u))?),
            None => None,
        };
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
        Ok(Tokenizer {
            vocab, id_bytes, merges, byte_level, byte_id, byte_fallback, unk,
            fuse_unk: model["fuse_unk"].as_bool().unwrap_or(false),
            ignore_merges: model["ignore_merges"].as_bool().unwrap_or(false),
            added_raw, added_norm, special, norms, pre, bf_ids, lead: dec.lead, prefix, suffix,
        })
    }

    /// Number of ids (model vocab + added tokens) = highest id + 1.
    pub fn vocab_size(&self) -> usize { self.id_bytes.len() }
    pub fn model_vocab_len(&self) -> usize { self.vocab.len() }
    pub fn merges_len(&self) -> usize { self.merges.len() }
    /// True for byte-level BPE (GPT-2 / Qwen / Llama 3 style), false for SentencePiece-style char BPE.
    pub fn is_byte_level(&self) -> bool { self.byte_level }
    /// Id of an added token (`<|im_end|>`, `<s>`) or of a model token given as its text (byte-level models: the
    /// decoded text, e.g. " the"; SentencePiece models: the vocab text, e.g. "▁the").
    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.added_raw.toks.iter().chain(&self.added_norm.toks).find(|t| t.content == s).map(|t| t.id)
            .or_else(|| self.vocab.get(s.as_bytes()).copied())
    }
    pub fn is_special(&self, id: u32) -> bool { self.special.get(id as usize).copied().unwrap_or(false) }
    /// Decoded bytes of one id (empty for unknown ids). Does not apply the sequence-level leading strip.
    pub fn id_to_bytes(&self, id: u32) -> &[u8] { self.id_bytes.get(id as usize).map(|v| v.as_slice()).unwrap_or(&[]) }

    /// Ids the post-processor puts before the text (e.g. `[<s>]` / `[<bos>]` / `[<|begin_of_text|>]`; empty for Qwen).
    pub fn special_prefix(&self) -> &[u32] { &self.prefix }
    /// Ids the post-processor puts after the text (e.g. `[</s>]` when the template adds EOS).
    pub fn special_suffix(&self) -> &[u32] { &self.suffix }
    /// The template adds a BOS (some id before the text).
    pub fn add_bos(&self) -> bool { !self.prefix.is_empty() }
    /// The BOS id the template adds, if any.
    pub fn bos_id(&self) -> Option<u32> { self.prefix.first().copied() }

    /// Encode with the post-processor's special tokens (HF `add_special_tokens=True`): prefix + text + suffix.
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let mut out = self.prefix.clone();
        out.extend(self.encode(text));
        out.extend_from_slice(&self.suffix);
        out
    }

    /// Encode text (no BOS/EOS added; `add_special_tokens=False`). Added tokens in the text become their ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 4);
        self.added_raw.split(text, &mut |p| match p {
            Piece::Token(id) => out.push(id),
            Piece::Text(a, b) => self.encode_segment(&text[a..b], a == 0, &mut out),
        });
        out
    }

    /// One segment between raw added tokens: normalize, split normalized added tokens, pre-tokenize, BPE.
    fn encode_segment(&self, s: &str, first: bool, out: &mut Vec<u32>) {
        if s.is_empty() { return; }
        let n = pipeline::normalize(&self.norms, s);
        if self.added_norm.toks.is_empty() { return self.pre_tokenize(0, &n, first, out); }
        self.added_norm.split(&n, &mut |p| match p {
            Piece::Token(id) => out.push(id),
            Piece::Text(a, b) => self.pre_tokenize(0, &n[a..b], first && a == 0, out),
        });
    }

    /// Run pre-tokenizer stage `stage` on `s` (`first`: the piece starts at offset 0 of the input text).
    fn pre_tokenize(&self, stage: usize, s: &str, first: bool, out: &mut Vec<u32>) {
        if s.is_empty() { return; }
        let Some(p) = self.pre.get(stage) else { return self.model(s, out) };
        match p {
            Pre::Split { re, behavior, invert } => re.split(s, *behavior, *invert, &mut |piece, at| self.pre_tokenize(stage + 1, piece, first && at == 0, out)),
            Pre::ByteLevel { add_prefix_space, re } => {
                let owned;
                let s = if *add_prefix_space && !s.starts_with(' ') { owned = format!(" {}", s); owned.as_str() } else { s };
                match re {
                    Some(re) => re.split(s, Behavior::Isolated, false, &mut |piece, at| self.pre_tokenize(stage + 1, piece, first && at == 0, out)),
                    None => self.pre_tokenize(stage + 1, s, first, out),
                }
            }
            Pre::Metaspace { repl, prepend, split } => {
                let mut t = s.replace(' ', repl.encode_utf8(&mut [0; 4]));
                if (*prepend == PrependScheme::Always || (*prepend == PrependScheme::First && first)) && !t.starts_with(*repl) { t.insert(0, *repl); }
                if let Some(re) = split {
                    re.split(&t, Behavior::MergedWithNext, false, &mut |piece, at| self.pre_tokenize(stage + 1, piece, first && at == 0, out));
                } else {
                    self.pre_tokenize(stage + 1, &t, first, out);
                }
            }
        }
    }

    /// BPE model on one pre-tokenized word.
    fn model(&self, s: &str, out: &mut Vec<u32>) {
        if s.is_empty() { return; }
        if self.ignore_merges {
            if let Some(&id) = self.vocab.get(s.as_bytes()) { out.push(id); return; }
        }
        let mut sym: Vec<u32>;
        if self.byte_level {
            sym = s.bytes().filter_map(|b| match self.byte_id[b as usize] { u32::MAX => self.unk, id => Some(id) }).collect();
            if sym.is_empty() { return; }
        } else {
            sym = Vec::with_capacity(s.len());
            let mut last_unk = false;
            for c in s.chars() {
                let mut buf = [0u8; 4];
                let cb = c.encode_utf8(&mut buf).as_bytes();
                if let Some(&id) = self.vocab.get(cb) { sym.push(id); last_unk = false; continue; }
                if self.byte_fallback && cb.iter().all(|&b| self.byte_id[b as usize] != u32::MAX) {
                    sym.extend(cb.iter().map(|&b| self.byte_id[b as usize]));
                    last_unk = false;
                    continue;
                }
                if let Some(u) = self.unk {
                    if !(self.fuse_unk && last_unk) { sym.push(u); }
                    last_unk = true;
                }
            }
        }
        if sym.len() > HEAP_BPE_MIN { self.merge_heap(&mut sym) } else { self.merge_scan(&mut sym) }
        out.extend_from_slice(&sym);
    }

    /// Repeatedly apply the lowest-rank merge (leftmost on ties). O(n^2); for short words.
    fn merge_scan(&self, sym: &mut Vec<u32>) {
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
    }

    /// Same result as [`Self::merge_scan`], O(n log n): a min-heap of (rank, left position) over a linked list
    /// (HF `Word::merge_all`). Stale heap entries are skipped by re-checking the pair's rank.
    fn merge_heap(&self, sym: &mut Vec<u32>) {
        let n = sym.len();
        let none = usize::MAX;
        let mut prev: Vec<usize> = (0..n).map(|i| if i == 0 { none } else { i - 1 }).collect();
        let mut next: Vec<usize> = (0..n).map(|i| if i + 1 == n { none } else { i + 1 }).collect();
        let mut alive = vec![true; n];
        let mut heap = BinaryHeap::new();
        for i in 0..n.saturating_sub(1) {
            if let Some(&(r, _)) = self.merges.get(&(sym[i], sym[i + 1])) { heap.push(Reverse((r, i))); }
        }
        while let Some(Reverse((r, i))) = heap.pop() {
            if !alive[i] { continue; }
            let j = next[i];
            if j == none { continue; }
            let Some(&(r2, id)) = self.merges.get(&(sym[i], sym[j])) else { continue };
            if r2 != r { continue; }
            sym[i] = id;
            alive[j] = false;
            next[i] = next[j];
            if next[j] != none { prev[next[j]] = i; }
            if prev[i] != none {
                if let Some(&(r, _)) = self.merges.get(&(sym[prev[i]], sym[i])) { heap.push(Reverse((r, prev[i]))); }
            }
            if next[i] != none {
                if let Some(&(r, _)) = self.merges.get(&(sym[i], sym[next[i]])) { heap.push(Reverse((r, i))); }
            }
        }
        let mut k = 0;
        for i in 0..n { if alive[i] { sym[k] = sym[i]; k += 1; } }
        sym.truncate(k);
    }

    /// Decode ids to text (special tokens kept as their text; invalid UTF-8 -> U+FFFD).
    pub fn decode(&self, ids: &[u32]) -> String { self.decode_impl(ids, false) }

    /// Decode, dropping special tokens (HF `skip_special_tokens=True`).
    pub fn decode_skip_special(&self, ids: &[u32]) -> String { self.decode_impl(ids, true) }

    fn decode_impl(&self, ids: &[u32], skip_special: bool) -> String {
        let mut b: Vec<u8> = Vec::with_capacity(ids.len() * 4);
        let mut run: Vec<u8> = Vec::new(); // pending byte-fallback bytes
        let mut run_n = 0usize;
        let flush = |b: &mut Vec<u8>, run: &mut Vec<u8>, run_n: &mut usize| {
            if *run_n == 0 { return; }
            if std::str::from_utf8(run).is_ok() { b.extend_from_slice(run); } else { for _ in 0..*run_n { b.extend_from_slice("\u{FFFD}".as_bytes()); } }
            run.clear();
            *run_n = 0;
        };
        let mut first = true;
        for &id in ids {
            if (id as usize) >= self.id_bytes.len() || (skip_special && self.is_special(id)) { continue; }
            let t = self.id_to_bytes(id);
            if self.bf_ids.get(id as usize).copied().unwrap_or(false) {
                run.extend_from_slice(t);
                run_n += 1;
            } else {
                flush(&mut b, &mut run, &mut run_n);
                if first && self.lead == Lead::FirstToken { b.extend(t.iter().filter(|&&c| c != b' ')); } else { b.extend_from_slice(t); }
            }
            first = false;
        }
        flush(&mut b, &mut run, &mut run_n);
        let mut s = String::from_utf8_lossy(&b).into_owned();
        if let Lead::Fused { ch, n } = self.lead {
            let mut cut = 0;
            for c in s.chars().take(n) { if c == ch { cut += c.len_utf8() } else { break } }
            s.drain(..cut);
        }
        s
    }
}

/// Incremental decoder: push ids, get back only complete UTF-8 text (a char split across tokens is held).
/// Applies the same leading strip as [`Tokenizer::decode`]. An invalid `<0xNN>` byte run gives one U+FFFD per
/// invalid sequence here (HF `decode` gives one per token); valid text is identical.
#[derive(Default)]
pub struct StreamDecoder {
    buf: Vec<u8>,
    pushed: usize,
    lead_cut: usize,
    lead_done: bool,
}

impl StreamDecoder {
    pub fn new() -> Self { Self::default() }
    /// Append one token; returns the text that became complete (may be empty).
    pub fn push(&mut self, tk: &Tokenizer, id: u32) -> String {
        let t = tk.id_to_bytes(id);
        match tk.lead {
            Lead::FirstToken if self.pushed == 0 => self.buf.extend(t.iter().filter(|&&c| c != b' ')),
            _ => self.buf.extend_from_slice(t),
        }
        self.pushed += 1;
        if let Lead::Fused { ch, n } = tk.lead {
            let mut cb = [0u8; 4];
            let cb = ch.encode_utf8(&mut cb).as_bytes();
            while !self.lead_done && !self.buf.is_empty() {
                if self.lead_cut < n && self.buf.starts_with(cb) { self.buf.drain(..cb.len()); self.lead_cut += 1; }
                else if self.lead_cut < n && cb.starts_with(&self.buf) { return String::new(); } // maybe a split char
                else { self.lead_done = true; }
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Tiny xorshift so the test needs no extra crate.
    struct Rng(u64);
    impl Rng { fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 } }

    /// Byte-level tokenizer over "abc" with pseudo-random merges (ranks in creation order).
    fn random_merges_tokenizer(rng: &mut Rng, ignore_merges: bool) -> Tokenizer {
        let b2c = byte_to_char();
        let mut vocab = serde_json::Map::new();
        for (i, c) in b2c.iter().enumerate() { vocab.insert(c.to_string(), json!(i)); }
        let mut toks: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let mut merges = vec![];
        while merges.len() < 80 {
            let (x, y) = (toks[(rng.next() % toks.len() as u64) as usize].clone(), toks[(rng.next() % toks.len() as u64) as usize].clone());
            let xy = format!("{}{}", x, y);
            if xy.len() > 6 || vocab.contains_key(&xy) { continue; }
            vocab.insert(xy.clone(), json!(vocab.len()));
            merges.push(json!([x, y]));
            toks.push(xy);
        }
        let j = json!({"added_tokens": [], "normalizer": null, "post_processor": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false},
            "decoder": {"type": "ByteLevel"},
            "model": {"type": "BPE", "vocab": vocab, "merges": merges, "ignore_merges": ignore_merges}});
        Tokenizer::from_json_str(&j.to_string(), None).unwrap()
    }

    /// Byte-level vocab entry with a char outside the byte alphabet (Cohere2's raw U+200D, U+2122): loads, decodes to
    /// its raw UTF-8 bytes (HF ByteLevel decoder), and is never produced by encode even when its raw bytes equal the
    /// byte-level spelling of another token.
    #[test]
    fn non_byte_level_vocab_entry_is_decode_only() {
        let b2c = byte_to_char();
        let mut vocab = serde_json::Map::new();
        for (i, c) in b2c.iter().enumerate() { vocab.insert(c.to_string(), json!(i)); }
        let zwj: String = "\u{200d}".bytes().map(|b| b2c[b as usize]).collect();
        vocab.insert(zwj.clone(), json!(257));
        vocab.insert("\u{200d}".into(), json!(258));
        vocab.insert("\u{2122}".into(), json!(259));
        let first2: String = zwj.chars().take(2).collect();
        vocab.insert(first2.clone(), json!(256));
        let last: String = zwj.chars().skip(2).collect();
        let (c0, c1): (String, String) = (zwj.chars().take(1).collect(), zwj.chars().skip(1).take(1).collect());
        let j = json!({"added_tokens": [], "normalizer": null, "post_processor": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": false},
            "decoder": {"type": "ByteLevel"},
            "model": {"type": "BPE", "vocab": vocab, "merges": [[c0, c1], [first2, last]]}});
        let tk = Tokenizer::from_json_str(&j.to_string(), None).unwrap();
        assert_eq!(tk.id_to_bytes(258), "\u{200d}".as_bytes());
        assert_eq!(tk.id_to_bytes(259), "\u{2122}".as_bytes());
        // ids 0..=255 are the bytes; U+200D encodes through the byte-level merge (257), never as the raw entry (258)
        assert_eq!(tk.encode("a\u{200d}b"), vec![b'a' as u32, 257, b'b' as u32]);
        assert!(!tk.encode("\u{200d}\u{2122}").iter().any(|&i| i == 258 || i == 259));
        assert_eq!(tk.decode(&[258, 259]), "\u{200d}\u{2122}");
    }

    #[test]
    fn heap_merge_equals_scan_merge() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for round in 0..4 {
            let t = random_merges_tokenizer(&mut rng, false);
            for _ in 0..200 {
                let n = 1 + (rng.next() % 200) as usize;
                let s: String = (0..n).map(|_| ['a', 'b', 'c'][(rng.next() % 3) as usize]).collect();
                let base: Vec<u32> = s.bytes().map(|b| t.byte_id[b as usize]).collect();
                let (mut x, mut y) = (base.clone(), base);
                t.merge_scan(&mut x);
                t.merge_heap(&mut y);
                assert_eq!(x, y, "round {} word {:?}", round, s);
                assert_eq!(t.decode(&x), s);
            }
        }
    }

    #[test]
    fn ignore_merges_takes_whole_word() {
        let b2c = byte_to_char();
        let mut vocab = serde_json::Map::new();
        for (i, c) in b2c.iter().enumerate() { vocab.insert(c.to_string(), json!(i)); }
        vocab.insert("ab".into(), json!(256));
        vocab.insert("abc".into(), json!(257)); // reachable only through ignore_merges (no "ab"+"c" merge)
        let mk = |im: bool| {
            let j = json!({"added_tokens": [], "normalizer": null, "post_processor": null,
                "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}, "decoder": {"type": "ByteLevel"},
                "model": {"type": "BPE", "vocab": vocab.clone(), "merges": [["a", "b"]], "ignore_merges": im}});
            Tokenizer::from_json_str(&j.to_string(), None).unwrap()
        };
        assert_eq!(mk(false).encode("abc"), [256, b'c' as u32]);
        assert_eq!(mk(true).encode("abc"), [257]);
        assert_eq!(mk(true).encode("abcc"), [256, b'c' as u32, b'c' as u32]);
    }

    #[test]
    fn byte_fallback_and_unk() {
        let mk = |bf: bool| {
            let mut vocab = serde_json::Map::new();
            vocab.insert("<unk>".into(), json!(0));
            for b in 0..256 { vocab.insert(format!("<0x{:02X}>", b), json!(1 + b)); }
            for (i, t) in ["▁", "a", "▁a"].iter().enumerate() { vocab.insert(t.to_string(), json!(257 + i)); }
            let j = json!({"added_tokens": [{"id": 0, "content": "<unk>", "special": true, "normalized": false, "lstrip": false, "rstrip": false, "single_word": false}],
                "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}, "pre_tokenizer": null, "post_processor": null,
                "decoder": {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "ByteFallback"}, {"type": "Fuse"}]},
                "model": {"type": "BPE", "vocab": vocab, "merges": [["▁", "a"]], "unk_token": "<unk>", "fuse_unk": true, "byte_fallback": bf}});
            Tokenizer::from_json_str(&j.to_string(), None).unwrap()
        };
        let t = mk(true);
        let ids = t.encode(" aé");
        assert_eq!(ids, [259, 1 + 0xC3, 1 + 0xA9]);
        assert_eq!(t.decode(&ids), " aé");
        assert_eq!(t.decode(&[1 + 0xC3]), "\u{FFFD}");
        assert_eq!(t.decode(&[1 + 0xE2, 1 + 0x82]), "\u{FFFD}\u{FFFD}"); // one per byte token, like HF
        let mut d = StreamDecoder::new();
        assert_eq!(d.push(&t, 1 + 0xC3), "");
        assert_eq!(d.push(&t, 1 + 0xA9), "é");
        let t = mk(false);
        assert_eq!(t.encode("aéé a"), [258, 0, 259]); // fuse_unk: one <unk> for the run
    }

    /// SmolLM-style: Digits(individual) before ByteLevel, and a byte-level vocab without some bytes (HF drops them
    /// when there is no unk_token). Expected ids follow HF `tokenizers` semantics (checked on SmolLM's tokenizer.json).
    #[test]
    fn digits_and_missing_bytes() {
        let mut vocab = serde_json::Map::new();
        for c in ['1', '2', 'a', 'b'] { vocab.insert(c.to_string(), json!(vocab.len())); }
        vocab.insert("12".into(), json!(4));
        vocab.insert("ab".into(), json!(5));
        let j = json!({"added_tokens": [], "normalizer": null, "post_processor": null,
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Digits", "individual_digits": true},
                              {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}]},
            "decoder": {"type": "ByteLevel"},
            "model": {"type": "BPE", "vocab": vocab, "merges": [["1", "2"], ["a", "b"]], "unk_token": null}});
        let t = Tokenizer::from_json_str(&j.to_string(), None).unwrap();
        assert_eq!(t.encode("12"), [0, 1]); // digits isolated: no "12" merge
        assert_eq!(t.encode("a\u{4}b"), [2, 3]); // \x04 (its own pre-token) has no token: dropped
        assert_eq!(t.encode("ab"), [5]);
    }

    #[test]
    fn stream_decoder_leading_strip() {
        let mut vocab = serde_json::Map::new();
        for (i, t) in ["▁", "a", "▁a"].iter().enumerate() { vocab.insert(t.to_string(), json!(i)); }
        let j = json!({"added_tokens": [], "normalizer": {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"}, {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}]},
            "pre_tokenizer": null, "post_processor": null,
            "decoder": {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "Fuse"}, {"type": "Strip", "content": " ", "start": 1, "stop": 0}]},
            "model": {"type": "BPE", "vocab": vocab, "merges": [["▁", "a"]]}});
        let t = Tokenizer::from_json_str(&j.to_string(), None).unwrap();
        let ids = t.encode("a  a");
        assert_eq!(ids, [2, 0, 2]);
        assert_eq!(t.decode(&ids), "a  a");
        let mut d = StreamDecoder::new();
        let s: String = ids.iter().map(|&i| d.push(&t, i)).collect();
        assert_eq!(s, "a  a");
        let mut d = StreamDecoder::new();
        let s: String = [0u32, 0, 1].iter().map(|&i| d.push(&t, i)).collect();
        assert_eq!(s, t.decode(&[0, 0, 1]));
        assert_eq!(s, " a");
    }
}
