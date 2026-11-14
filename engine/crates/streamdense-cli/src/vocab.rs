//! streamdense vocab keep --model DIR [--out FILE] [--drop cjk,ja,ko,tl,id,ms] [--drop-list FILE]
//! Keep-list of the pruned MTP draft head (`--draft-vocab FILE` on qwen / gateway; streamdense_model::draft_vocab).
//! Each token id is decoded to its bytes with the Rust tokenizer and dropped when it holds a character of a dropped
//! script, by code point (no corpus):
//!   cjk: Han (CJK radicals / Kangxi / IDC 2E80-2FFF, Ext A 3400-4DBF, 4E00-9FFF, compat F900-FAFF, planes 2-3
//!        20000-3FFFF, strokes 31C0-31EF), Yijing 4DC0-4DFF, Bopomofo 3100-312F / 31A0-31BF, CJK symbols and punctuation
//!        3000-303F, kanbun 3190-319F, enclosed CJK / compat 3200-33FF, vertical / compat / small forms FE10-FE1F,
//!        FE30-FE6F, fullwidth forms FF00-FF64, FFE0-FFEF;
//!   ja:  Hiragana 3040-309F, Katakana 30A0-30FF, 31F0-31FF, halfwidth FF65-FF9F, kana supplements 1AFF0-1B16F;
//!   ko:  Hangul jamo 1100-11FF, compat jamo 3130-318F, jamo ext A960-A97F / D7B0-D7FF, syllables AC00-D7AF, halfwidth
//!        FFA0-FFDF.
//! Byte-level BPE pieces that end inside a character are dropped only when EVERY code point the known prefix bytes
//! can start lies in the dropped set (e.g. lone E3 / E5-E9 / EB-ED leads, F0 A0-BF; not E2, EA, EF, F0 alone, which
//! also start general punctuation, Yi / Latin ext, emoji variation selectors / U+FFFD, emoji). Pieces that START
//! mid-character have no lead byte and stay unless they also hold a complete dropped character. Latin (incl.
//! Vietnamese 1E00-1EFF and combining marks), Thai, Devanagari, Cyrillic, Arabic, emoji etc. are never dropped.
//! tl / id / ms (Filipino, Indonesian, Malay) are Latin script, their subwords are shared with English and other
//! languages: without a corpus nothing is dropped for them (a note is printed); --drop-list FILE (ids, whitespace /
//! comma separated, # comments) removes extra ids later. Always kept: special tokens, single-byte tokens,
//! all-ASCII tokens (digits, code symbols). Dropped too: model vocab rows with no token (lm_head padding rows).
use crate::Args;
use std::path::Path;
use streamdense_model::draft_vocab::KeepList;
use streamdense_tokenizer::Tokenizer;

fn die(e: String) -> ! { eprintln!("ERROR: {}", e); std::process::exit(2) }

const CJK: &[(u32, u32)] = &[
    (0x2E80, 0x2FFF), (0x3000, 0x303F), (0x3100, 0x312F), (0x3190, 0x319F), (0x31A0, 0x31BF), (0x31C0, 0x31EF),
    (0x3200, 0x33FF), (0x3400, 0x4DBF), (0x4DC0, 0x4DFF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0xFE10, 0xFE1F),
    (0xFE30, 0xFE6F), (0xFF00, 0xFF64), (0xFFE0, 0xFFEF), (0x20000, 0x3FFFF),
];
const JA: &[(u32, u32)] = &[(0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF65, 0xFF9F), (0x1AFF0, 0x1B16F)];
const KO: &[(u32, u32)] = &[(0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xAC00, 0xD7AF), (0xD7B0, 0xD7FF), (0xFFA0, 0xFFDF)];
const SCRIPTS: [(&str, &[(u32, u32)]); 3] = [("cjk", CJK), ("ja", JA), ("ko", KO)];

/// Script tag of code point c among the enabled sets.
fn script_of(c: u32, on: &[bool; 3]) -> Option<usize> {
    (0..3).find(|&s| on[s] && SCRIPTS[s].1.iter().any(|&(a, b)| (a..=b).contains(&c)))
}
/// Script tag when every code point of [lo, hi] (surrogates excepted) is in the enabled sets: the script of lo.
fn range_script(lo: u32, hi: u32, on: &[bool; 3]) -> Option<usize> {
    let mut iv: Vec<(u32, u32)> = (0..3).filter(|&s| on[s]).flat_map(|s| SCRIPTS[s].1.iter().copied()).collect();
    iv.push((0xD800, 0xDFFF));
    iv.sort();
    let mut c = lo;
    for (a, b) in iv {
        if a > c { break; }
        if b >= c { if b >= hi { return script_of(lo, on).or_else(|| (0..3).find(|&s| on[s])); } c = b + 1; }
    }
    None
}

#[derive(Debug, PartialEq, Eq)]
enum Class { Keep, Mid, Drop { script: usize, partial: bool } }

/// Classify a token's bytes (see the module doc).
fn classify(b: &[u8], on: &[bool; 3]) -> Class {
    let cont = |x: u8| (0x80..=0xBF).contains(&x);
    let mut i = 0;
    let mut mid = false;
    while i < b.len() && cont(b[i]) { i += 1; mid = true; }
    while i < b.len() {
        let l = b[i];
        let n = match l { 0x00..=0x7F => 1, 0xC2..=0xDF => 2, 0xE0..=0xEF => 3, 0xF0..=0xF4 => 4, _ => { i += 1; continue; } };
        let have = (1..n).take_while(|&k| i + k < b.len() && cont(b[i + k])).count() + 1;
        let lead_bits = match n { 1 => l as u32, 2 => (l & 0x1F) as u32, 3 => (l & 0x0F) as u32, _ => (l & 0x07) as u32 };
        let mut lo = lead_bits;
        let mut hi = lead_bits;
        for k in 1..n {
            let (bl, bh) = if k < have { ((b[i + k] & 0x3F) as u32, (b[i + k] & 0x3F) as u32) } else { (0, 0x3F) };
            lo = (lo << 6) | bl;
            hi = (hi << 6) | bh;
        }
        if have == n {
            if let Some(s) = script_of(lo, on) { return Class::Drop { script: s, partial: false }; }
        } else if i + have == b.len() {
            // truncated last character: range of code points the known bytes can start (valid UTF-8 only)
            let lo = lo.max([0, 0, 0x80, 0x800, 0x10000][n]);
            let hi = hi.min(0x10FFFF);
            if lo <= hi { if let Some(s) = range_script(lo, hi, on) { return Class::Drop { script: s, partial: true }; } }
        }
        i += have;
    }
    if mid { Class::Mid } else { Class::Keep }
}

pub fn main(a: &Args) {
    let sub = a.0.get(2).cloned().unwrap_or_default();
    if sub != "keep" { die("usage: streamdense vocab keep --model DIR [--out FILE] [--drop cjk,ja,ko,tl,id,ms] [--drop-list FILE]".into()); }
    let model = a.sd("--model", &format!("{}/{}", crate::ROOT, crate::qwen::DEFAULT_MODEL));
    let out = a.s("--out").unwrap_or_else(|| format!("{}/draft_vocab.keep", model.trim_end_matches('/')));
    let drop = a.sd("--drop", "cjk,ja,ko,tl,id,ms");
    let mut on = [false; 3];
    let mut latin: Vec<&str> = Vec::new();
    for t in drop.split(',').map(|x| x.trim()).filter(|x| !x.is_empty()) {
        match t {
            "cjk" | "zh" => on[0] = true, "ja" => on[1] = true, "ko" => on[2] = true,
            "tl" | "id" | "ms" => latin.push(t),
            _ => die(format!("--drop {}: want cjk,ja,ko,tl,id,ms", t)),
        }
    }
    let vocab = streamdense_model::QwenConfig::from_model_dir(&model).unwrap_or_else(|e| die(e)).vocab;
    let tk = Tokenizer::from_path(&model).unwrap_or_else(|e| die(format!("tokenizer {}: {}", model, e)));
    let (mut special, mut byte1, mut ascii, mut other, mut mid, mut pad) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    let mut dropped = [[0usize; 2]; 3]; // [script][complete, partial]
    let mut keep: Vec<u32> = Vec::with_capacity(vocab);
    let mut protect = vec![false; vocab];
    for id in 0..vocab as u32 {
        let b = tk.id_to_bytes(id);
        if (id as usize) >= tk.vocab_size() || b.is_empty() { pad += 1; continue; }
        if tk.is_special(id) {
            special += 1; protect[id as usize] = true; keep.push(id); continue;
        }
        if b.len() == 1 { byte1 += 1; protect[id as usize] = true; keep.push(id); continue; }
        if b.is_ascii() { ascii += 1; protect[id as usize] = true; keep.push(id); continue; }
        match classify(b, &on) {
            Class::Drop { script, partial } => dropped[script][partial as usize] += 1,
            Class::Mid => { mid += 1; keep.push(id); }
            Class::Keep => { other += 1; keep.push(id); }
        }
    }
    let mut extra = (0usize, 0usize);
    if let Some(f) = a.s("--drop-list") {
        let s = std::fs::read_to_string(&f).unwrap_or_else(|e| die(format!("{}: {}", f, e)));
        let mut del = std::collections::HashSet::new();
        for line in s.lines() {
            for t in line.split('#').next().unwrap_or("").split(|c: char| c.is_whitespace() || c == ',').filter(|x| !x.is_empty()) {
                let id: u32 = t.parse().unwrap_or_else(|_| die(format!("{}: bad id {:?}", f, t)));
                if (id as usize) < vocab && protect[id as usize] { extra.1 += 1; } else { del.insert(id); }
            }
        }
        let n0 = keep.len();
        keep.retain(|i| !del.contains(i));
        extra.0 = n0 - keep.len();
    }
    let kl = KeepList::new(vocab as u32, keep).unwrap_or_else(|e| die(e));
    println!("vocab keep: model {} (vocab {}, tokenizer {} ids), drop {}", model, vocab, tk.vocab_size(), drop);
    println!("| class | kept | dropped |");
    println!("|---|---|---|");
    println!("| special tokens | {} | 0 |", special);
    println!("| single-byte tokens | {} | 0 |", byte1);
    println!("| all-ASCII tokens | {} | 0 |", ascii);
    println!("| other scripts (Latin incl. Vietnamese, Cyrillic, Thai, Arabic, emoji, ...) | {} | 0 |", other);
    println!("| pieces starting mid-character (no lead byte, ambiguous) | {} | 0 |", mid);
    for (s, (name, _)) in SCRIPTS.iter().enumerate() {
        let label = match *name { "cjk" => "cjk: Han + Bopomofo + CJK punctuation / fullwidth", "ja" => "ja: Hiragana / Katakana", _ => "ko: Hangul" };
        println!("| {} (complete chars + partial UTF-8 pieces) | 0 | {} + {} = {} |", label, dropped[s][0], dropped[s][1], dropped[s][0] + dropped[s][1]);
    }
    println!("| lm_head rows with no token (padding) | 0 | {} |", pad);
    if a.s("--drop-list").is_some() { println!("| --drop-list (protected ids ignored: {}) | 0 | {} |", extra.1, extra.0); }
    let nd = vocab - kl.ids.len();
    println!("total: kept {} / {} ({:.1}%), dropped {} -> draft lm_head reads {:.1}% of its bytes", kl.ids.len(), vocab,
             100.0 * kl.ids.len() as f64 / vocab as f64, nd, 100.0 * kl.ids.len() as f64 / vocab as f64);
    if !latin.is_empty() {
        println!("note: {} are Latin-script; their subword tokens are shared with English and other languages and cannot be \
                  separated by script without a corpus, so nothing was dropped for them (use --drop-list FILE later)", latin.join(", "));
    }
    kl.write(Path::new(&out)).unwrap_or_else(|e| die(e));
    println!("wrote {} ({} ids, {} bytes)", out, kl.ids.len(), 16 + 4 * kl.ids.len());
}

#[cfg(test)]
mod tests {
    use super::*;
    const ALL: [bool; 3] = [true, true, true];
    fn c(s: &str) -> Class { classify(s.as_bytes(), &ALL) }
    #[test]
    fn scripts() {
        assert_eq!(c("的"), Class::Drop { script: 0, partial: false });
        assert_eq!(c(" 中国"), Class::Drop { script: 0, partial: false });
        assert_eq!(c("。"), Class::Drop { script: 0, partial: false });
        assert_eq!(c("，"), Class::Drop { script: 0, partial: false });
        assert_eq!(c("の"), Class::Drop { script: 1, partial: false });
        assert_eq!(c("カ"), Class::Drop { script: 1, partial: false });
        assert_eq!(c("한국"), Class::Drop { script: 2, partial: false });
        assert_eq!(c("ㄅ"), Class::Drop { script: 0, partial: false });
        for k in ["Việt", " những", "ạ", "ữ", "đ", "Привет", "สวัสดี", "नमस्ते", "مرحبا", "😀", "→", "…", "é", "\u{0301}", "’", "·"] {
            assert_eq!(c(k), Class::Keep, "{:?}", k);
        }
    }
    #[test]
    fn partial() {
        let p = |b: &[u8]| classify(b, &ALL);
        assert_eq!(p(&[0xE4, 0xBD]), Class::Drop { script: 0, partial: true }); // 你 prefix
        assert_eq!(p(&[0xE3]), Class::Drop { script: 0, partial: true }); // U+3000-3FFF: all CJK / kana / hangul
        assert!(matches!(p(&[0xEB]), Class::Drop { script: 2, partial: true }));
        assert!(matches!(p(&[0xED, 0x95]), Class::Drop { script: 2, partial: true }));
        assert!(matches!(p(&[0xF0, 0xA0]), Class::Drop { script: 0, partial: true }));
        assert!(matches!(p(&[0xE3, 0x81]), Class::Drop { script: 1, partial: true })); // U+3040-307F hiragana
        // ambiguous leads stay: E2 (punctuation), E1 (Vietnamese 1E00), EA (Yi / Latin ext), EF (FFFD / VS), F0 (emoji)
        for b in [&[0xE2u8][..], &[0xE1], &[0xE1, 0xBA], &[0xEA], &[0xEF], &[0xEF, 0xB8], &[0xEF, 0xBF], &[0xF0], &[0xF0, 0x9F], &[0x20, 0xE2, 0x80]] {
            assert_eq!(p(b), Class::Keep, "{:x?}", b);
        }
        assert_eq!(p(&[0xBD, 0xA0]), Class::Mid);
        assert_eq!(p(&[0xA0, 0xE4, 0xBD, 0xA0]), Class::Drop { script: 0, partial: false });
        // only ko enabled: E3 alone is not all Hangul
        assert_eq!(classify(&[0xE3], &[false, false, true]), Class::Keep);
    }
}
