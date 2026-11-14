//! Family detection from a model folder's config.json. `FAMILIES` is the only family list: another family is one more
//! entry there, plus its `Family` arm in engine.rs / report.rs (the routing: each arm calls that family's folder).
//!
//! model_type comes from `text_config` when present, else the root, as `DenseConfig::from_json` reads it. The
//! model_type strings of each entry are the family folder's (`qwen::config` for the Qwen ones).
use dense_engine::config::is_hybrid;
use serde_json::Value;

/// The existing engine a family goes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineKind {
    /// engine/dense-engine: `DenseConfig`, `Engine` (standalone crate; greedy; all-attention, head dim 128)
    DenseEngine,
    /// engine/crates/streamdense-model: `QwenConfig`, `QwenEngine` (hybrid DeltaNet + attention, head dim 256)
    Main,
}

impl EngineKind {
    pub fn name(self) -> &'static str { match self { EngineKind::DenseEngine => "dense-engine", EngineKind::Main => "main engine (streamdense-model)" } }
}

/// Whose code a folder goes to, in `Model::load` and `report::inspect_text`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family { Qwen, Llama, Granite, Gemma }

/// What the layer types of a config must look like for this family (`dense_engine::config::is_hybrid`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape { Dense, Hybrid }

#[derive(Clone, Copy, Debug)]
pub struct FamilyEntry {
    pub name: &'static str,
    pub family: Family,
    pub model_types: &'static [&'static str],
    pub shape: Shape,
    pub engine: EngineKind,
}

/// Every supported family.
pub const FAMILIES: &[FamilyEntry] = &[
    FamilyEntry { name: "qwen2", family: Family::Qwen, model_types: crate::qwen::QWEN2_TYPES, shape: Shape::Dense, engine: EngineKind::DenseEngine },
    FamilyEntry { name: "qwen3", family: Family::Qwen, model_types: crate::qwen::QWEN3_TYPES, shape: Shape::Dense, engine: EngineKind::DenseEngine },
    FamilyEntry { name: "qwen3.8-hybrid", family: Family::Qwen, model_types: crate::qwen::HYBRID_TYPES, shape: Shape::Hybrid, engine: EngineKind::Main },
    FamilyEntry { name: "llama", family: Family::Llama, model_types: crate::llama::MODEL_TYPES, shape: Shape::Dense, engine: EngineKind::DenseEngine },
    FamilyEntry { name: "granite", family: Family::Granite, model_types: crate::granite::MODEL_TYPES, shape: Shape::Dense, engine: EngineKind::DenseEngine },
    FamilyEntry { name: "gemma", family: Family::Gemma, model_types: crate::gemma::MODEL_TYPES, shape: Shape::Dense, engine: EngineKind::DenseEngine },
];

/// Keys that mark a mixture-of-experts config (same list as `DenseConfig::from_json`).
const MOE_KEYS: [&str; 3] = ["num_experts", "n_routed_experts", "num_experts_per_tok"];

#[derive(Clone, Debug)]
pub struct Detected {
    pub entry: &'static FamilyEntry,
    pub model_type: String,
}

/// Family of `dir/config.json`.
pub fn detect(dir: &str) -> Result<Detected, String> {
    let path = format!("{}/config.json", dir);
    let txt = std::fs::read_to_string(&path).map_err(|e| format!("{}: {}", path, e))?;
    let root: Value = serde_json::from_str(&txt).map_err(|e| format!("{}: {}", path, e))?;
    detect_json(&root)
}

/// Family of an already-parsed config.json.
pub fn detect_json(root: &Value) -> Result<Detected, String> {
    let c = if root["text_config"].is_object() { &root["text_config"] } else { root };
    let model_type = c["model_type"].as_str().or(root["model_type"].as_str()).ok_or("config.json has no model_type")?;
    let root_type = root["model_type"].as_str().unwrap_or("");
    // the other Granite members (MoE, hybrid, shared-MLP) get the granite folder's own reason, not the generic ones
    if model_type.starts_with("granite") && !crate::granite::MODEL_TYPES.contains(&model_type) {
        let b = crate::granite::blockers(root);
        return Err(if b.is_empty() { format!("model_type {:?} is not supported yet", model_type) } else { b.join("; ") });
    }
    let moe = MOE_KEYS.iter().any(|k| !c[*k].is_null() || !root[*k].is_null());
    if moe || model_type.contains("moe") || root_type.contains("moe") {
        return Err(format!("model_type {:?}: mixture-of-experts model rejected: dense models only", model_type));
    }
    let entry = FAMILIES.iter().find(|f| f.model_types.contains(&model_type)).ok_or_else(|| {
        let known: Vec<&str> = FAMILIES.iter().flat_map(|f| f.model_types.iter().copied()).collect();
        format!("model_type {:?} is not supported yet (supported: {})", model_type, known.join(", "))
    })?;
    let hybrid = is_hybrid(root);
    if hybrid != (entry.shape == Shape::Hybrid) {
        return Err(format!("model_type {:?}: config {} linear-attention fields, but family {} goes to {}, which {} them",
                           model_type, if hybrid { "has" } else { "has no" }, entry.name, entry.engine.name(),
                           if hybrid { "does not take" } else { "needs" }));
    }
    Ok(Detected { entry, model_type: model_type.to_string() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dense_families_go_to_dense_engine() {
        for t in ["qwen2", "qwen3"] {
            let d = detect_json(&json!({"model_type": t, "hidden_size": 512})).unwrap();
            assert_eq!(d.entry.name, t);
            assert_eq!(d.entry.engine, EngineKind::DenseEngine);
        }
    }

    #[test]
    fn hybrid_goes_to_main_engine() {
        // the shape of the MLX Qwen3.8 config: both model_type strings, hybrid fields in text_config
        let v = json!({"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text", "linear_num_key_heads": 4,
                       "layer_types": ["linear_attention", "full_attention"]}});
        let d = detect_json(&v).unwrap();
        assert_eq!(d.entry.name, "qwen3.8-hybrid");
        assert_eq!(d.model_type, "qwen3_5_text");
        assert_eq!(d.entry.engine, EngineKind::Main);
    }

    #[test]
    fn moe_rejected() {
        for v in [json!({"model_type": "qwen2_moe"}), json!({"model_type": "qwen3_moe"}),
                  json!({"model_type": "qwen3", "num_experts": 8}),
                  json!({"model_type": "qwen3", "text_config": {"model_type": "qwen3", "n_routed_experts": 8}}),
                  json!({"model_type": "qwen3_5_moe", "text_config": {"model_type": "qwen3_5_text"}})] {
            let e = detect_json(&v).unwrap_err();
            assert!(e.contains("dense models only"), "{}", e);
        }
    }

    #[test]
    fn unknown_type_named() {
        let e = detect_json(&json!({"model_type": "falcon"})).unwrap_err();
        assert!(e.contains("\"falcon\"") && e.contains("not supported yet"), "{}", e);
        assert!(detect_json(&json!({"hidden_size": 1})).unwrap_err().contains("no model_type"));
    }

    #[test]
    fn shape_must_fit_the_family() {
        assert!(detect_json(&json!({"model_type": "qwen3", "linear_num_key_heads": 4})).unwrap_err().contains("does not take"));
        assert!(detect_json(&json!({"model_type": "qwen3_5"})).unwrap_err().contains("needs"));
    }

    #[test]
    fn other_families_go_to_dense_engine() {
        for (t, name) in [("llama", "llama"), ("granite", "granite"), ("gemma", "gemma"), ("gemma2", "gemma"), ("gemma3_text", "gemma")] {
            let d = detect_json(&json!({"model_type": t, "hidden_size": 512})).unwrap();
            assert_eq!((d.entry.name, d.entry.engine), (name, EngineKind::DenseEngine));
        }
        // multimodal gemma3 root: model_type from text_config
        let d = detect_json(&json!({"model_type": "gemma3", "text_config": {"model_type": "gemma3_text"}, "vision_config": {}})).unwrap();
        assert_eq!(d.entry.name, "gemma");
    }

    #[test]
    fn granite_members_get_the_granite_reason() {
        let e = detect_json(&json!({"model_type": "granitemoehybrid", "num_local_experts": 0,
                                    "layer_types": ["mamba", "attention"]})).unwrap_err();
        assert!(e.contains("mamba") && e.contains("not supported yet"), "{}", e);
        let e = detect_json(&json!({"model_type": "granitemoehybrid", "num_local_experts": 62})).unwrap_err();
        assert!(e.contains("dense models only") && e.contains("62 experts"), "{}", e);
        let e = detect_json(&json!({"model_type": "granitemoe"})).unwrap_err();
        assert!(e.contains("dense models only"), "{}", e);
    }

    #[test]
    fn each_model_type_listed_once() {
        let mut all: Vec<&str> = FAMILIES.iter().flat_map(|f| f.model_types.iter().copied()).collect();
        let n = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), n);
    }
}
