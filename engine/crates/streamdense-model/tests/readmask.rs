//! read mask + unsure-token full read (readmask.rs) on the tiny model (models/tiny_qwen35, has mtp/), Metal.
//! - all-ones mask == no mask == an engine without the region, BIT-IDENTICAL (tokens and rec margins): plain sync +
//!   pipe, MTP depth 1, deep spec fixed depth (pipelined + sync), adaptive; masked kernels forced on every pass
//!   (gap rule and out-of-profile off) and forced off (every pass "read everything").
//! - random masks: never panic; the prefill (first token, its top-1 / top-2) is unaffected; output changes are
//!   allowed only after the first token; the masked FFN block / head outputs are exactly 0 (recorder sees none).
//! - union: decode reads profile ∪ prompt-active; re-running with that union as the profile gives the same tokens.
//! - unsure-pass counting: full passes == emitted tokens whose gap < gap_thr (plain sync), learned ∩ read = ∅.
//! - the activity recorder is unchanged by an all-ones mask (bit for bit).
//! Run: cargo test --release --offline -p streamdense-model --test readmask -- --nocapture --test-threads 1
use serde_json::Value;
use streamdense_model::readmask::{MaskPolicy, ReadMask};
use streamdense_model::spec::{generate_mtp, DraftOverride};
use streamdense_model::spec_deep::{generate_spec_mode, generate_spec_sync, SpecMode};
use streamdense_model::{QwenEngine, QwenOptions};

const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");

fn model_dir() -> Option<String> {
    let d = format!("{}/models/tiny_qwen35", ROOT);
    if std::path::Path::new(&format!("{}/mtp/weights.safetensors", d)).exists() { Some(d) } else {
        eprintln!("SKIP: {} (with mtp/) missing", d);
        None
    }
}
fn open(read_mask: bool) -> Option<QwenEngine> {
    let dir = model_dir()?;
    let opt = QwenOptions { max_ctx: 2048, mtp: true, spec_depth: 7, spec_auto: true, read_mask, ..Default::default() };
    Some(QwenEngine::open(&dir, opt).expect("open tiny"))
}
fn prompts() -> Vec<(String, Vec<u32>)> {
    let p = format!("{}/engine/refs/tiny_qwen35_mlxref_prompts.json", ROOT);
    let mut v: Vec<(String, Vec<u32>)> = match std::fs::read_to_string(&p) {
        Ok(s) => {
            let j: Value = serde_json::from_str(&s).unwrap();
            j["prompts"].as_array().unwrap().iter().map(|r| (r["id"].as_str().unwrap().to_string(),
                r["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect())).collect()
        }
        Err(_) => vec![],
    };
    let mut s = 0x1234_5678u64;
    let mut rnd = |n: usize| -> Vec<u32> { (0..n).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; (s % 2000) as u32 + 10 }).collect() };
    v.push(("syn5".into(), rnd(5)));
    v.push(("syn300".into(), rnd(300)));
    v.into_iter().take(5).collect()
}

const N: usize = 48;
fn always_masked() -> MaskPolicy { MaskPolicy { gap_thr: f32::NEG_INFINITY, oop_min: 0, ..Default::default() } }
fn always_full() -> MaskPolicy { MaskPolicy { gap_thr: f32::INFINITY, oop_min: 0, ..Default::default() } }
fn bits(v: &[f32]) -> Vec<u32> { v.iter().map(|x| x.to_bits()).collect() }
/// Recorder thresholds that leave prompt-active empty (decode reads exactly the profile).
fn no_prompt(p: MaskPolicy) -> MaskPolicy { MaskPolicy { ffn_thr: 1e30, head_thr: 1e30, ..p } }
/// A threshold at which the prompt `p` activates some but not all FFN blocks (tiny model: the default 0.01 marks
/// every block active).
fn mid_thr(eng: &QwenEngine, p: &[u32]) -> f32 {
    let total = ReadMask::full(&eng.cfg).count_ffn();
    let base = eng.mask_policy();
    for t in [0.03f32, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0] {
        eng.set_mask_policy(MaskPolicy { ffn_thr: t, head_thr: t, ..base });
        eng.reset();
        eng.prefill(p).unwrap();
        let n = eng.prompt_active().unwrap().count_ffn();
        if n > 0 && n * 10 < total * 7 { eng.set_mask_policy(base); return t; }
    }
    eng.set_mask_policy(base);
    1e30
}

/// (plain sync, plain pipe, mtp d1, deep fixed 3 pipe, deep fixed 3 sync, auto 7): tokens + margins bits
fn all_modes(eng: &QwenEngine, p: &[u32]) -> Vec<(String, Vec<u32>, Vec<u32>)> {
    let mut out = Vec::new();
    let g = eng.generate(p, N, false).unwrap();
    out.push(("plain sync".into(), g.tokens.clone(), bits(&g.margins)));
    let g = eng.generate(p, N, true).unwrap();
    out.push(("plain pipe".into(), g.tokens.clone(), bits(&g.margins)));
    let m = generate_mtp(eng, p, N, DraftOverride::None, false).unwrap();
    out.push(("mtp d1".into(), m.tokens.clone(), bits(&m.margins)));
    let m = generate_spec_mode(eng, p, N, SpecMode::Fixed(3), DraftOverride::None).unwrap();
    out.push(("deep 3 pipe".into(), m.tokens.clone(), bits(&m.margins)));
    let m = generate_spec_sync(eng, p, N, 3, DraftOverride::None).unwrap();
    out.push(("deep 3 sync".into(), m.tokens.clone(), bits(&m.margins)));
    let m = generate_spec_mode(eng, p, N, SpecMode::Auto { max: 7 }, DraftOverride::None).unwrap();
    out.push(("auto 7".into(), m.tokens.clone(), bits(&m.margins)));
    out
}

#[test]
fn all_ones_mask_bit_identical() {
    let (Some(base), Some(eng)) = (open(false), open(true)) else { return };
    assert!(base.set_read_mask(Some(&ReadMask::full(&base.cfg))).is_err(), "no region: set_read_mask must refuse");
    let full = ReadMask::full(&eng.cfg);
    let mut ok = true;
    for (id, p) in prompts() {
        let r0 = all_modes(&base, &p);
        eng.set_read_mask(None).unwrap();
        let r1 = all_modes(&eng, &p);
        for (pol, name) in [(always_masked(), "masked kernels"), (always_full(), "read everything")] {
            eng.set_mask_policy(pol);
            eng.set_read_mask(Some(&full)).unwrap();
            eng.reset_mask_stats();
            let r2 = all_modes(&eng, &p);
            let s = eng.mask_stats();
            if pol.gap_thr < 0.0 { assert!(s.masked_passes > 0 && s.full_passes == 0, "{}: {:?}", id, s); }
            else { assert!(s.full_passes > 0 && s.masked_passes == 0, "{}: {:?}", id, s); }
            assert_eq!(s.learned_blocks + s.learned_heads, 0, "all-ones: nothing to learn");
            for ((a, b), c) in r0.iter().zip(&r1).zip(&r2) {
                let eq = a.1 == b.1 && a.2 == b.2 && a.1 == c.1 && a.2 == c.2;
                println!("  {:7} {:12} {:16}: none == no-region {} | all-ones == no-region {} ({} tokens)", id, a.0, name,
                         a.1 == b.1 && a.2 == b.2, a.1 == c.1 && a.2 == c.2, a.1.len());
                ok &= eq;
            }
        }
        eng.set_read_mask(None).unwrap();
    }
    assert!(ok, "all-ones mask / no mask not bit-identical");
}

#[test]
fn random_mask_prefill_untouched_and_union() {
    let (Some(base), Some(eng)) = (open(false), open(true)) else { return };
    let cfg = eng.cfg.clone();
    let mut changed = 0;
    let mut masked_any = 0;
    for (id, p) in prompts() {
        let t = mid_thr(&eng, &p);
        let r = base.generate(&p, N, false).unwrap();
        for (i, (keep, pol)) in [(0.0, false), (0.3, false), (0.7, false), (0.5, true), (0.0, true)].into_iter().enumerate() {
            let mp = MaskPolicy { ffn_thr: t, head_thr: t, ..always_masked() };
            eng.set_mask_policy(if pol { no_prompt(mp) } else { mp });
            let prof = ReadMask::random(&cfg, keep, 11 + i as u64);
            eng.set_read_mask(Some(&prof)).unwrap();
            let g = eng.generate(&p, N, false).unwrap();
            // prefill is unmasked: first token and its top-1 / top-2 bit-identical
            assert_eq!(g.tokens[0], r.tokens[0], "{} keep {}: first token", id, keep);
            assert_eq!(g.margins[0].to_bits(), r.margins[0].to_bits(), "{} keep {}: first margin", id, keep);
            let pl = p.len();
            assert_eq!(eng.rec_token(pl - 1), base.rec_token(pl - 1), "{} keep {}: prefill rec", id, keep);
            // union: decode reads profile ∪ prompt-active (nothing learned with the gap / oop rules off)
            let pa = eng.prompt_active().unwrap();
            assert!(eng.learned().unwrap().is_empty());
            let dm = eng.decode_mask().expect("masked decode active");
            assert_eq!(dm, prof.union(&pa), "{} keep {}: decode mask != profile ∪ prompt", id, keep);
            if !pol { assert!(pa.count_ffn() > 0, "{}: prompt activated no block", id); } else { assert!(pa.is_empty()); }
            masked_any += (dm.count_ffn() < ReadMask::full(&cfg).count_ffn()) as usize;
            // the same union given as the profile: same decode mask, same tokens
            eng.set_read_mask(Some(&dm)).unwrap();
            let g2 = eng.generate(&p, N, false).unwrap();
            assert_eq!(g2.tokens, g.tokens, "{} keep {}: union as profile", id, keep);
            assert_eq!(eng.decode_mask().unwrap(), dm);
            if g.tokens != r.tokens { changed += 1; }
            // every decode mode runs with the mask (no panic); the first token is the prefill's
            let m = all_modes(&eng, &p);
            for (name, t, _) in &m { assert_eq!(t[0], r.tokens[0], "{} keep {} {}: first token", id, keep, name); }
            println!("  {:7} keep {:.1} thr {}: profile {} blocks, prompt-active {} / {} blocks, {} / {} heads; decode reads {} blocks; plain == unmasked {}",
                     id, keep, if pol { "inf".to_string() } else { t.to_string() }, prof.count_ffn(), pa.count_ffn(), ReadMask::full(&cfg).count_ffn(), pa.count_heads(),
                     ReadMask::full(&cfg).count_heads(), dm.count_ffn(), g.tokens == r.tokens);
        }
    }
    eng.set_read_mask(None).unwrap();
    println!("  masked outputs that differ from unmasked: {}, runs with masked blocks: {}", changed, masked_any);
    assert!(masked_any > 0 && changed > 0, "no run actually masked / changed anything");
}

#[test]
fn masked_units_are_zero_in_the_recorder() {
    // prompt-active off (huge thresholds): decode reads exactly a random profile. Over masked decode passes only
    // (recorder reset after the prefill), every masked block's activations and masked head's output are exactly 0.
    let Some(eng) = open(true) else { return };
    let cfg = eng.cfg.clone();
    eng.set_mask_policy(no_prompt(always_masked()));
    eng.set_read_mask(Some(&ReadMask::random(&cfg, 0.5, 9))).unwrap();
    let p = prompts().remove(0).1;
    eng.reset();
    let mut tok = eng.prefill(&p).unwrap();
    let dm = eng.decode_mask().unwrap();
    let tap = eng.activity_tap().unwrap();
    tap.reset();
    for _ in 0..12 {
        eng.reader.read_rows(&[tok], eng.arena.ptr(eng.sl().stage)).unwrap();
        let cb = eng.gpu.command_buffer();
        eng.encode_pass(&cb, 1, 0, 1, false).unwrap();
        streamdense_metal::commit(&cb);
        streamdense_metal::wait(&cb).unwrap();
        tok = eng.rec_token(eng.pos() - 1).0;
    }
    let s = tap.snapshot();
    assert_eq!(s.passes, 12);
    let (mut zb, mut nb, mut zh, mut nh, mut readnz) = (0, 0, 0, 0, 0);
    for l in 0..cfg.layers {
        for b in 0..cfg.inter / 64 {
            let z = s.neuron[l * cfg.inter + b * 64..l * cfg.inter + (b + 1) * 64].iter().all(|x| *x == 0.0);
            if dm.ffn(l, b) { readnz += !z as usize; continue; }
            nb += 1;
            zb += z as usize;
        }
        for (h, v) in s.head_mass(l).iter().enumerate() { if !dm.head(l, h) { nh += 1; zh += (*v == 0.0) as usize; } }
    }
    println!("  decode reads {} blocks ({} with activity), masks {} blocks / {} heads: exactly 0 in the recorder {} / {}",
             dm.count_ffn(), readnz, nb, nh, zb, zh);
    assert!(nb > 0 && nh > 0, "nothing masked");
    assert_eq!((zb, zh), (nb, nh), "masked units not exactly 0");
    assert!(readnz > 0);
    eng.set_read_mask(None).unwrap();
}


#[test]
fn unsure_pass_counting_and_learning() {
    let Some(eng) = open(true) else { return };
    let cfg = eng.cfg.clone();
    for (id, p) in prompts() {
        let prof = ReadMask::random(&cfg, 0.5, 5);
        let t = mid_thr(&eng, &p);
        let pm = |q: MaskPolicy| MaskPolicy { ffn_thr: t, head_thr: t, ..q };
        eng.set_read_mask(Some(&prof)).unwrap();
        // pick the threshold from an unmasked run's margins (median)
        eng.set_mask_policy(pm(always_full()));
        let g0 = eng.generate(&p, N, false).unwrap();
        let mut m: Vec<f32> = g0.margins.clone();
        m.sort_by(|a, b| a.total_cmp(b));
        let thr = m[m.len() / 2];
        eng.set_mask_policy(pm(MaskPolicy { gap_thr: thr, oop_min: 0, ..Default::default() }));
        eng.take_learned();
        eng.reset_mask_stats();
        let g = eng.generate(&p, N, false).unwrap();
        let s = eng.mask_stats();
        let passes = g.tokens.len() - 1; // the first token is the prefill's
        let want_full = g.margins[..passes].iter().filter(|x| **x < thr).count() as u64;
        println!("  {:7} gap_thr {:.3}: passes {} full {} (want {}) masked {} learned {} blocks {} heads",
                 id, thr, passes, s.full_passes, want_full, s.masked_passes, s.learned_blocks, s.learned_heads);
        assert_eq!(s.full_passes, want_full, "{}: full passes", id);
        assert_eq!(s.full_passes + s.masked_passes, passes as u64, "{}: passes", id);
        // learned blocks were outside the read set (profile ∪ prompt), and are read now
        let l = eng.learned().unwrap();
        let pa = eng.prompt_active().unwrap();
        assert!(l.and(&prof.union(&pa)).is_empty(), "{}: learned overlaps the read set", id);
        assert_eq!(eng.decode_mask().unwrap(), prof.union(&pa).union(&l));
        assert_eq!(l.count_ffn() as u64, s.learned_blocks);
        let t = eng.take_learned().unwrap();
        assert_eq!(t, l);
        assert!(eng.learned().unwrap().is_empty());
        // every pass full: identical to the unmasked run
        eng.set_mask_policy(pm(always_full()));
        let gf = eng.generate(&p, N, false).unwrap();
        assert_eq!(gf.tokens, g0.tokens, "{}: all-full != unmasked", id);
        // out-of-profile rule: profile = nothing, so any activity in the read rows (prompt-active) is out of profile
        eng.set_read_mask(Some(&ReadMask::empty(&cfg))).unwrap();
        eng.set_mask_policy(pm(MaskPolicy { gap_thr: f32::NEG_INFINITY, oop_min: 1, ..Default::default() }));
        eng.reset_mask_stats();
        let _ = eng.generate(&p, N, false).unwrap();
        let s = eng.mask_stats();
        println!("  {:7} oop (empty profile): full {} masked {} oop hits {}", id, s.full_passes, s.masked_passes, s.full_oop);
        assert!(s.full_oop > 0 && s.full_passes > 0, "{}: out-of-profile never fired", id);
        eng.take_learned();
    }
    eng.set_read_mask(None).unwrap();
}

#[test]
fn recorder_unchanged_by_all_ones() {
    let Some(eng) = open(true) else { return };
    let tap = eng.activity_tap().unwrap();
    for (_, p) in prompts().into_iter().take(3) {
        let mut snaps = Vec::new();
        for mask in [None, Some(ReadMask::full(&eng.cfg))] {
            eng.set_mask_policy(always_masked());
            eng.set_read_mask(mask.as_ref()).unwrap();
            tap.reset();
            let _ = eng.generate(&p, 24, false).unwrap();
            let s = tap.snapshot();
            snaps.push((bits(&s.neuron), bits(&s.dn_head), bits(&s.attn_head), bits(&s.resid), s.rows, s.passes));
        }
        assert!(snaps[0] == snaps[1], "activity recorder differs with an all-ones mask");
    }
    eng.set_read_mask(None).unwrap();
}
