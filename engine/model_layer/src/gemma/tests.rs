//! Config fixtures + chat strings. No model weights, no Metal. Fixture values are from memory of the public HF
//! config.json files; fields marked "unverified" are ones not certain. Every asserted size field is one I am sure of.
use super::config::map_config;
use super::{blockers, chat_prompt, dense_config, MODEL_TYPES};
use dense_engine::config::{Activation, Family, LayerKind, RopeScaling};
use serde_json::{json, Value};
use streamdense_core::{write_safetensors, DType};

// google/gemma-3-270m. unverified: max_position_embeddings 32768, eos_token_id (1 here; the -it generation_config adds
// 106), presence of `_sliding_window_pattern`, transformers_version.
fn gemma3_270m() -> Value {
    let lt: Vec<&str> = (0..18).map(|i| if (i + 1) % 6 == 0 { "full_attention" } else { "sliding_attention" }).collect();
    json!({
        "_sliding_window_pattern": 6, "architectures": ["Gemma3ForCausalLM"], "attention_bias": false,
        "attention_dropout": 0.0, "attn_logit_softcapping": null, "bos_token_id": 2, "eos_token_id": 1,
        "final_logit_softcapping": null, "head_dim": 256, "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 640,
        "initializer_range": 0.02, "intermediate_size": 2048, "layer_types": lt, "max_position_embeddings": 32768,
        "model_type": "gemma3_text", "num_attention_heads": 4, "num_hidden_layers": 18, "num_key_value_heads": 1,
        "pad_token_id": 0, "query_pre_attn_scalar": 256, "rms_norm_eps": 1e-06, "rope_local_base_freq": 10000.0,
        "rope_scaling": null, "rope_theta": 1000000.0, "sliding_window": 512, "torch_dtype": "bfloat16",
        "transformers_version": "4.55.0.dev0", "use_bidirectional_attention": false, "use_cache": true, "vocab_size": 262144
    })
}

// google/gemma-2-2b. unverified: eos_token_id (1 vs [1, 107]), torch_dtype, transformers_version, cache_implementation.
fn gemma2_2b() -> Value {
    json!({
        "architectures": ["Gemma2ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0,
        "attn_logit_softcapping": 50.0, "bos_token_id": 2, "cache_implementation": "hybrid", "eos_token_id": 1,
        "final_logit_softcapping": 30.0, "head_dim": 256, "hidden_act": "gelu_pytorch_tanh",
        "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 2304, "initializer_range": 0.02,
        "intermediate_size": 9216, "max_position_embeddings": 8192, "model_type": "gemma2", "num_attention_heads": 8,
        "num_hidden_layers": 26, "num_key_value_heads": 4, "pad_token_id": 0, "query_pre_attn_scalar": 256,
        "rms_norm_eps": 1e-06, "rope_theta": 10000.0, "sliding_window": 4096, "torch_dtype": "float32",
        "transformers_version": "4.42.4", "use_cache": true, "vocab_size": 256000
    })
}

// google/gemma-2b (Gemma 1). unverified: whether the current file also carries `hidden_activation`, transformers_version.
fn gemma1_2b() -> Value {
    json!({
        "architectures": ["GemmaForCausalLM"], "attention_bias": false, "attention_dropout": 0.0, "bos_token_id": 2,
        "eos_token_id": 1, "head_dim": 256, "hidden_act": "gelu", "hidden_size": 2048, "initializer_range": 0.02,
        "intermediate_size": 16384, "max_position_embeddings": 8192, "model_type": "gemma", "num_attention_heads": 8,
        "num_hidden_layers": 18, "num_key_value_heads": 1, "pad_token_id": 0, "rms_norm_eps": 1e-06,
        "rope_scaling": null, "rope_theta": 10000.0, "torch_dtype": "bfloat16", "transformers_version": "4.38.0.dev0",
        "use_cache": true, "vocab_size": 256000
    })
}

fn map(v: &Value) -> dense_engine::config::DenseConfig { map_config(v, None, &[]).unwrap() }
fn err(v: &Value) -> String { map_config(v, None, &[]).unwrap_err() }
fn edit(mut v: Value, f: impl FnOnce(&mut Value)) -> Value { f(&mut v); v }
fn remove(v: Value, k: &str) -> Value { edit(v, |v| { v.as_object_mut().unwrap().remove(k); }) }
fn dims(c: &dense_engine::config::DenseConfig) -> (usize, usize, usize, usize, usize, usize, usize) {
    (c.hidden, c.inter, c.layers, c.heads, c.kv_heads, c.head_dim, c.vocab)
}

#[test]
fn gemma3_270m_mapping() {
    let c = map(&gemma3_270m());
    assert_eq!((c.family, c.model_type.as_str()), (Family::Gemma3, "gemma3_text"));
    assert_eq!(dims(&c), (640, 2048, 18, 4, 1, 256, 262144));
    assert_eq!((c.rope_dims, c.head_dim_explicit), (256, true));
    assert_eq!(c.attn_scale, Some(0.0625)); // 256^-0.5
    assert_eq!((c.attn_softcap, c.final_softcap), (None, None));
    assert_eq!(c.activation, Activation::GeluTanh);
    assert!(c.norm_plus_one && c.four_norms && c.qk_norm && c.tie_word_embeddings);
    assert!(!c.attention_bias && !c.o_bias && !c.mlp_bias);
    assert_eq!(c.embed_scale, 640f64.sqrt());
    let full: Vec<usize> = (0..18).filter(|&i| c.kinds[i] == LayerKind::Full).collect();
    assert_eq!(full, vec![5, 11, 17]);
    assert_eq!(c.sliding_window, Some(512));
    assert_eq!((c.rope_theta, c.rope_local_theta, c.rope_scale, c.rope_scaling.clone()), (1e6, Some(1e4), RopeScaling::None, None));
    assert_eq!((c.bos, c.eos.clone(), c.quant, c.max_pos), (Some(2), vec![1], None, 32768));
    c.validate().unwrap();
    // without layer_types: the pattern gives the same kinds (both spellings)
    let p = remove(gemma3_270m(), "layer_types");
    assert_eq!(map(&p).kinds, c.kinds);
    let p2 = edit(remove(p, "_sliding_window_pattern"), |v| v["sliding_window_pattern"] = 6.into());
    assert_eq!(map(&p2).kinds, c.kinds);
    assert!(err(&remove(p2, "sliding_window_pattern")).contains("sliding_window_pattern"));
    // MLX 4-bit block
    let q = edit(gemma3_270m(), |v| v["quantization"] = json!({"group_size": 64, "bits": 4}));
    let cq = map(&q);
    assert_eq!(cq.quant, Some((64, 4)));
    cq.validate().unwrap();
}

#[test]
fn gemma3_rope_forms() {
    // Gemma 3 4B+ style: linear factor 8 on the global layers only
    let c = map(&edit(gemma3_270m(), |v| v["rope_scaling"] = json!({"rope_type": "linear", "factor": 8.0})));
    assert_eq!((c.rope_scale, c.rope_local_theta, c.rope_scaling.clone()), (RopeScaling::Linear { factor: 8.0 }, Some(1e4), None));
    c.validate().unwrap();
    // transformers v5 rope_parameters form
    let v = edit(remove(remove(remove(gemma3_270m(), "rope_theta"), "rope_local_base_freq"), "rope_scaling"), |v| {
        v["rope_parameters"] = json!({"full_attention": {"rope_type": "linear", "factor": 8.0, "rope_theta": 1e6},
                                      "sliding_attention": {"rope_type": "default", "rope_theta": 1e4}});
    });
    let c = map(&v);
    assert_eq!((c.rope_theta, c.rope_local_theta, c.rope_scale), (1e6, Some(1e4), RopeScaling::Linear { factor: 8.0 }));
    // scaling the engine does not have is named, never dropped
    let y = edit(gemma3_270m(), |v| v["rope_scaling"] = json!({"rope_type": "yarn", "factor": 4.0}));
    assert!(blockers(&y).iter().any(|b| b.contains("rope_scaling yarn factor 4")), "{:?}", blockers(&y));
    let s = edit(v, |v| v["rope_parameters"]["sliding_attention"] = json!({"rope_type": "linear", "factor": 2.0, "rope_theta": 1e4}));
    assert!(blockers(&s).iter().any(|b| b.contains("linear factor 2 on sliding layers")), "{:?}", blockers(&s));
    // no local theta: not guessed
    assert!(err(&remove(gemma3_270m(), "rope_local_base_freq")).contains("rope_local_base_freq"));
}

#[test]
fn gemma2_2b_mapping() {
    let c = map(&gemma2_2b());
    assert_eq!((c.family, c.model_type.as_str()), (Family::Gemma2, "gemma2"));
    assert_eq!(dims(&c), (2304, 9216, 26, 8, 4, 256, 256000));
    assert_eq!(c.attn_scale, Some(0.0625));
    assert_eq!((c.attn_softcap, c.final_softcap), (Some(50.0), Some(30.0)));
    assert_eq!(c.activation, Activation::GeluTanh);
    assert!(c.norm_plus_one && c.four_norms && !c.qk_norm && c.tie_word_embeddings);
    assert_eq!(c.embed_scale, 48.0);
    // layer 0 sliding, then alternating
    assert_eq!((c.kinds[0], c.kinds[1], c.kinds[25]), (LayerKind::Sliding, LayerKind::Full, LayerKind::Full));
    assert_eq!(c.n_sliding(), 13);
    assert_eq!((c.sliding_window, c.rope_theta, c.rope_local_theta, c.rope_scale), (Some(4096), 1e4, None, RopeScaling::None));
    c.validate().unwrap();
    // absent caps = class defaults 50 / 30, null = off
    let a = map(&remove(remove(gemma2_2b(), "attn_logit_softcapping"), "final_logit_softcapping"));
    assert_eq!((a.attn_softcap, a.final_softcap), (Some(50.0), Some(30.0)));
    let n = map(&edit(gemma2_2b(), |v| { v["attn_logit_softcapping"] = Value::Null; v["final_logit_softcapping"] = Value::Null; }));
    assert_eq!((n.attn_softcap, n.final_softcap), (None, None));
    // Gemma 2 27B style scalar (144 = 4608 / 32): scale from the field, not head_dim
    assert_eq!(map(&edit(gemma2_2b(), |v| v["query_pre_attn_scalar"] = 144.into())).attn_scale, Some(144f64.powf(-0.5)));
    assert!(err(&remove(gemma2_2b(), "query_pre_attn_scalar")).contains("query_pre_attn_scalar"));
    assert!(err(&remove(gemma2_2b(), "sliding_window")).contains("sliding_window"));
    // explicit layer_types wins
    let lt: Vec<&str> = (0..26).map(|_| "full_attention").collect();
    assert_eq!(map(&edit(gemma2_2b(), |v| v["layer_types"] = json!(lt))).n_sliding(), 0);
}

#[test]
fn gemma1_2b_mapping() {
    let c = map(&gemma1_2b());
    assert_eq!((c.family, c.model_type.as_str()), (Family::Gemma, "gemma"));
    assert_eq!(dims(&c), (2048, 16384, 18, 8, 1, 256, 256000));
    // hidden_act "gelu" in the file; transformers (PR 29402) runs gelu_pytorch_tanh
    assert_eq!(c.activation, Activation::GeluTanh);
    assert!(c.hidden_act.contains("\"gelu\""), "{}", c.hidden_act);
    assert_eq!((c.attn_scale, c.attn_softcap, c.final_softcap), (None, None, None));
    assert_eq!(c.attn_scale_f32(), 0.0625);
    assert!(c.norm_plus_one && !c.four_norms && !c.qk_norm && c.tie_word_embeddings);
    assert_eq!(c.embed_scale, 2048f64.sqrt());
    assert!(c.kinds.iter().all(|k| *k == LayerKind::Full) && c.sliding_window.is_none());
    assert_eq!((c.rope_theta, c.rope_local_theta), (1e4, None));
    c.validate().unwrap();
    // an explicit hidden_activation wins; exact GELU is then named by validate()
    let e = map(&edit(gemma1_2b(), |v| v["hidden_activation"] = "gelu".into()));
    assert!(e.validate().unwrap_err().contains("hidden_act must be silu or gelu_pytorch_tanh (got gelu)"));
    // attention_bias covers q/k/v and o_proj
    let b = map(&edit(gemma1_2b(), |v| v["attention_bias"] = true.into()));
    assert!(b.attention_bias && b.o_bias && !b.mlp_bias);
}

#[test]
fn model_types_and_blockers_clean() {
    assert_eq!(MODEL_TYPES, &["gemma", "gemma2", "gemma3_text"]);
    for v in [gemma1_2b(), gemma2_2b(), gemma3_270m()] {
        assert!(blockers(&v).is_empty(), "{:?}", blockers(&v));
    }
    let q = edit(gemma2_2b(), |v| v["quantization"] = json!({"group_size": 32, "bits": 3}));
    assert!(blockers(&q).iter().any(|b| b.contains("group_size 32 bits 3")));
    let bad = edit(gemma2_2b(), |v| v["quantization_config"] = json!({"load_in_4bit": true}));
    assert!(blockers(&bad)[0].contains("not MLX"));
}

#[test]
fn refusals_named() {
    for t in ["gemma3n", "gemma3n_text", "gemma4", "gemma4_text", "paligemma", "recurrent_gemma", "t5gemma"] {
        let e = err(&edit(gemma2_2b(), |v| v["model_type"] = t.into()));
        assert!(e.contains("refused") && e.contains(t), "{}", e);
    }
    assert!(err(&edit(gemma2_2b(), |v| v["model_type"] = "llama".into())).contains("is not gemma"));
    assert!(err(&edit(gemma1_2b(), |v| v["num_experts"] = 8.into())).contains("dense models only"));
    assert!(err(&edit(gemma3_270m(), |v| v["use_bidirectional_attention"] = true.into())).contains("not a causal model"));
    assert!(err(&edit(gemma3_270m(), |v| v["vision_config"] = json!({}))).contains("multimodal"));
    assert!(err(&edit(gemma3_270m(), |v| v["architectures"] = json!(["Gemma3TextForSequenceClassification"]))).contains("causal LM"));
    for k in ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size"] {
        for v in [gemma1_2b(), gemma2_2b(), gemma3_270m()] {
            assert!(err(&remove(v, k)).contains(k), "{}", k);
        }
    }
    assert!(err(&remove(gemma1_2b(), "rope_theta")).contains("rope_theta"));
    let lt = edit(gemma3_270m(), |v| v["layer_types"] = json!(["full_attention"]));
    assert!(err(&lt).contains("layer_types has 1 entries"));
    // multimodal gemma3 without a text_config, or with another text model
    assert!(err(&json!({"model_type": "gemma3"})).contains("text_config"));
    assert!(err(&json!({"model_type": "gemma3", "text_config": {"model_type": "siglip"}})).contains("refused"));
}

#[test]
fn chat_strings() {
    let m = |v: &[(&str, &str)]| v.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect::<Vec<_>>();
    assert_eq!(chat_prompt(&m(&[("user", "Hi")]), true).unwrap(),
               "<bos><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n");
    // system merged into the first user turn: system text + "\n\n" (untrimmed), user text trimmed
    assert_eq!(chat_prompt(&m(&[("system", "Be brief."), ("user", "  Hi \n")]), true).unwrap(),
               "<bos><start_of_turn>user\nBe brief.\n\nHi<end_of_turn>\n<start_of_turn>model\n");
    assert_eq!(chat_prompt(&m(&[("system", "S"), ("user", "a"), ("assistant", " b "), ("user", "c")]), false).unwrap(),
               "<bos><start_of_turn>user\nS\n\na<end_of_turn>\n<start_of_turn>model\nb<end_of_turn>\n<start_of_turn>user\nc<end_of_turn>\n");
    assert_eq!(chat_prompt(&m(&[("user", "a"), ("model", "b")]), false).unwrap(),
               "<bos><start_of_turn>user\na<end_of_turn>\n<start_of_turn>model\nb<end_of_turn>\n");
    assert_eq!(chat_prompt(&[], true).unwrap(), "<bos><start_of_turn>model\n");
    // errors
    assert!(chat_prompt(&m(&[("assistant", "x")]), true).unwrap_err().contains("alternate"));
    assert!(chat_prompt(&m(&[("user", "a"), ("user", "b")]), true).unwrap_err().contains("alternate"));
    assert!(chat_prompt(&m(&[("user", "a"), ("system", "b")]), true).unwrap_err().contains("first message"));
    assert!(chat_prompt(&m(&[("tool", "x")]), true).unwrap_err().contains("\"tool\""));
    assert!(chat_prompt(&m(&[("system", "x")]), true).unwrap_err().contains("user turn"));
}

/// A temp folder with the given files; removed on drop.
struct Dir(std::path::PathBuf);
impl Dir {
    fn new(tag: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("sd_gemma_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Dir(d)
    }
    fn json(&self, name: &str, v: &Value) { std::fs::write(self.0.join(name), v.to_string()).unwrap(); }
    fn weights(&self, names: &[&str]) {
        let z = [0u8; 4];
        let t: Vec<(&str, DType, &[usize], &[u8])> = names.iter().map(|n| (*n, DType::BF16, &[2usize][..], &z[..])).collect();
        write_safetensors(&self.0.join("model.safetensors").to_string_lossy(), &t).unwrap();
    }
}
impl Drop for Dir { fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); } }

#[test]
fn dense_config_folder_eos() {
    let d = Dir::new("eos");
    let v = gemma3_270m();
    assert_eq!(dense_config(&v, &d.0).unwrap().eos, vec![1]);
    d.json("generation_config.json", &json!({"eos_token_id": [1, 106]}));
    assert_eq!(dense_config(&v, &d.0).unwrap().eos, vec![1, 106]);
    // <end_of_turn> from tokenizer_config.json when generation_config lacks it (Gemma 2 vocab: 107)
    let d2 = Dir::new("eot");
    d2.json("tokenizer_config.json", &json!({"added_tokens_decoder": {"1": {"content": "<eos>"}, "107": {"content": "<end_of_turn>"}}}));
    assert_eq!(dense_config(&gemma2_2b(), &d2.0).unwrap().eos, vec![1, 107]);
    // no eos anywhere
    assert!(dense_config(&remove(gemma1_2b(), "eos_token_id"), &d.0.join("none")).unwrap_err().contains("eos"));
    // validate() runs
    let q = edit(gemma2_2b(), |v| v["quantization"] = json!({"group_size": 64, "bits": 3}));
    assert!(dense_config(&q, &d.0).unwrap_err().contains("bits 3"));
}

#[test]
fn multimodal_gemma3_only_with_text_weights() {
    let tc = remove(remove(gemma3_270m(), "architectures"), "eos_token_id");
    let root = json!({"model_type": "gemma3", "architectures": ["Gemma3ForConditionalGeneration"], "eos_token_id": [1, 106],
                      "text_config": tc, "vision_config": {"model_type": "siglip_vision_model"},
                      "quantization": {"group_size": 64, "bits": 4}});
    // config alone maps (text_config, root eos and quantization)
    let c = map(&root);
    assert_eq!((c.model_type.as_str(), c.eos.clone(), c.quant), ("gemma3_text", vec![1, 106], Some((64, 4))));
    // no weights: cannot confirm, refused
    let d = Dir::new("mm_none");
    assert!(dense_config(&root, &d.0).unwrap_err().contains("weights not readable"));
    // multimodal checkpoint names
    let d = Dir::new("mm_lm");
    d.weights(&["language_model.model.embed_tokens.weight", "vision_tower.x.weight"]);
    assert!(dense_config(&root, &d.0).unwrap_err().contains("language_model."));
    let d = Dir::new("mm_vis");
    d.weights(&["model.embed_tokens.weight", "model.vision_tower.x.weight"]);
    assert!(dense_config(&root, &d.0).unwrap_err().contains("vision"));
    // text-only weights under model.*: taken
    let d = Dir::new("mm_text");
    d.weights(&["model.embed_tokens.weight", "model.norm.weight"]);
    let c = dense_config(&root, &d.0).unwrap();
    assert_eq!((c.family, dims(&c)), (Family::Gemma3, (640, 2048, 18, 4, 1, 256, 262144)));
    // the same through an index.json weight_map
    let d = Dir::new("mm_idx");
    d.json("model.safetensors.index.json", &json!({"weight_map": {"language_model.model.norm.weight": "a.safetensors"}}));
    assert!(dense_config(&root, &d.0).unwrap_err().contains("language_model."));
}
