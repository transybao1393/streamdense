//! Non-Qwen tokenizer.json formats, checked against HF `tokenizers` golden output.
//! * tests/data/synthetic_fixtures.json: small hand-made tokenizer.json files (SentencePiece-style BPE with byte
//!   fallback, Metaspace, GPT-2 / Llama 3 byte-level, ignore_merges, Split behaviors, normalizers, added-token
//!   flags) + HF encode/decode output for each.
//! * tests/data/granite_golden.json: HF ids for models/granite-4.2-3b-MLX-4bit/tokenizer.json (GPT-2 style
//!   ByteLevel pre-tokenizer, no normalizer). Skipped when that file is missing.
use serde_json::Value;
use streamdense_tokenizer::*;

fn ids(v: &Value) -> Vec<u32> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect() }

fn load(p: &str) -> Value { serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap() }

fn stream(t: &Tokenizer, ids: &[u32]) -> String {
    let mut d = StreamDecoder::new();
    let mut s: String = ids.iter().map(|&i| d.push(t, i)).collect();
    s.push_str(&d.finish());
    s
}

/// Compare one tokenizer against its HF cases; returns mismatch descriptions.
fn check(name: &str, t: &Tokenizer, f: &Value) -> Vec<String> {
    let mut bad = vec![];
    for c in f["cases"].as_array().unwrap() {
        let text = c["text"].as_str().unwrap();
        let (want, want_sp) = (ids(&c["ids"]), ids(&c["ids_special"]));
        let got = t.encode(text);
        if got != want { bad.push(format!("{}: encode {:?}\n  want {:?}\n  got  {:?}", name, text, want, got)); continue; }
        let got = t.encode_with_special(text);
        if got != want_sp { bad.push(format!("{}: encode_with_special {:?}\n  want {:?}\n  got  {:?}", name, text, want_sp, got)); }
        let dec = c["decoded"].as_str().unwrap();
        if t.decode(&want) != dec { bad.push(format!("{}: decode {:?}: want {:?} got {:?}", name, want, dec, t.decode(&want))); }
        if !dec.contains('\u{FFFD}') && stream(t, &want) != dec { bad.push(format!("{}: stream decode {:?}: got {:?}", name, dec, stream(t, &want))); }
        let skip = c["decoded_skip"].as_str().unwrap();
        let skip_ids = if f.get("source").is_some() { &want } else { &want_sp };
        if t.decode_skip_special(skip_ids) != skip { bad.push(format!("{}: decode_skip_special {:?}: want {:?} got {:?}", name, skip_ids, skip, t.decode_skip_special(skip_ids))); }
    }
    for d in f["decode"].as_array().map(|v| v.as_slice()).unwrap_or(&[]) {
        let i = ids(&d["ids"]);
        if t.decode(&i) != d["decoded"].as_str().unwrap() { bad.push(format!("{}: decode {:?}: want {:?} got {:?}", name, i, d["decoded"], t.decode(&i))); }
        if t.decode_skip_special(&i) != d["decoded_skip"].as_str().unwrap() {
            bad.push(format!("{}: decode_skip_special {:?}: want {:?} got {:?}", name, i, d["decoded_skip"], t.decode_skip_special(&i)));
        }
    }
    bad
}

#[test]
fn synthetic_fixtures_match_hf() {
    let all = load(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/synthetic_fixtures.json"));
    let mut bad = vec![];
    let mut n_cases = 0;
    for f in all.as_array().unwrap() {
        let name = f["name"].as_str().unwrap();
        let t = Tokenizer::from_json_str(&f["tokenizer"].to_string(), None).unwrap_or_else(|e| panic!("{}: {}", name, e));
        n_cases += f["cases"].as_array().unwrap().len();
        bad.extend(check(name, &t, f));
    }
    eprintln!("{} fixtures, {} cases, {} mismatches", all.as_array().unwrap().len(), n_cases, bad.len());
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn fixture_bos_api() {
    let all = load(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/synthetic_fixtures.json"));
    let get = |n: &str| {
        let f = all.as_array().unwrap().iter().find(|f| f["name"] == n).unwrap();
        Tokenizer::from_json_str(&f["tokenizer"].to_string(), None).unwrap()
    };
    let llama = get("sp_llama");
    assert!(llama.add_bos());
    assert_eq!(llama.bos_id(), llama.token_id("<s>"));
    assert_eq!(llama.special_suffix(), &[] as &[u32]);
    assert!(!llama.is_byte_level());
    let ms = get("sp_metaspace_first_split");
    assert_eq!(ms.special_prefix(), &[ms.token_id("<s>").unwrap()]);
    assert_eq!(ms.special_suffix(), &[ms.token_id("</s>").unwrap()]);
    let g2 = get("bl_gpt2");
    assert!(!g2.add_bos() && g2.bos_id().is_none() && g2.is_byte_level());
    assert_eq!(g2.encode_with_special("hello"), g2.encode("hello"));
}

#[test]
fn granite_golden() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models/granite-4.2-3b-MLX-4bit");
    if !std::path::Path::new(&format!("{}/tokenizer.json", dir)).exists() {
        eprintln!("SKIP: needs models/granite-4.2-3b-MLX-4bit/tokenizer.json (missing)");
        return;
    }
    let t0 = std::time::Instant::now();
    let t = Tokenizer::from_path(dir).expect("load granite tokenizer");
    eprintln!("granite tokenizer loaded in {:.1} ms: {} ids, {} merges", t0.elapsed().as_secs_f64() * 1e3, t.vocab_size(), t.merges_len());
    let f = load(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/granite_golden.json"));
    assert_eq!(t.vocab_size() as u64, f["vocab_size"].as_u64().unwrap());
    let bad = check("granite", &t, &f);
    eprintln!("granite: {}/{} cases match", f["cases"].as_array().unwrap().len() - bad.len(), f["cases"].as_array().unwrap().len());
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// Unsupported components fail at load with an error naming them.
#[test]
fn refuses_unsupported_components() {
    let base = || -> Value {
        serde_json::json!({"added_tokens": [], "normalizer": null, "post_processor": null,
            "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
            "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
            "model": {"type": "BPE", "vocab": {"▁": 0, "a": 1, "▁a": 2}, "merges": [["▁", "a"]]}})
    };
    let err = |j: Value| Tokenizer::from_json_str(&j.to_string(), None).err().unwrap_or_else(|| panic!("should fail: {}", j));
    assert!(Tokenizer::from_json_str(&base().to_string(), None).is_ok());
    let cases: Vec<(&str, Value, &str)> = vec![
        ("/model", serde_json::json!({"type": "Unigram", "vocab": [["a", 0.0]], "unk_id": 0}), "Unigram"),
        ("/model", serde_json::json!({"type": "WordPiece", "vocab": {"a": 0}, "unk_token": "a"}), "WordPiece"),
        ("/normalizer", serde_json::json!({"type": "Precompiled", "precompiled_charsmap": ""}), "Precompiled"),
        ("/normalizer", serde_json::json!({"type": "BertNormalizer"}), "BertNormalizer"),
        ("/pre_tokenizer", serde_json::json!({"type": "BertPreTokenizer"}), "BertPreTokenizer"),
        ("/pre_tokenizer", serde_json::json!({"type": "Split", "pattern": {"Regex": "(?=a)b"}, "behavior": "Isolated", "invert": false}), "look-around"),
        ("/decoder", Value::Null, "decoder null"),
        ("/decoder", serde_json::json!({"type": "WordPiece", "prefix": "##"}), "WordPiece"),
        ("/decoder", serde_json::json!({"type": "Sequence", "decoders": [{"type": "Fuse"}, {"type": "Strip", "content": " ", "start": 0, "stop": 1}]}), "stop"),
        ("/post_processor", serde_json::json!({"type": "Weird"}), "Weird"),
    ];
    for (ptr, v, want) in cases {
        let mut j = base();
        *j.pointer_mut(ptr).unwrap() = v;
        let e = err(j);
        assert!(e.contains(want), "{} error {:?} should name {:?}", ptr, e, want);
    }
    let mut j = base();
    j["model"]["dropout"] = serde_json::json!(0.1);
    assert!(err(j).contains("dropout"));
    let mut j = base();
    j["pre_tokenizer"] = serde_json::json!({"type": "Sequence", "pretokenizers": [{"type": "ByteLevel"}, {"type": "Metaspace", "replacement": "▁"}]});
    assert!(err(j).contains("ByteLevel must be the last"));
}
