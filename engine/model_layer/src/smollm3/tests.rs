//! Config fixture and chat strings only: no model, no Metal, no weights.
//!
//! Fixture: config.json of mlx-community/SmolLM3-3B-4bit, verbatim (fetched from huggingface.co 2026-11-15, metadata
//! only). Expected chat strings: transformers 5.21 `apply_chat_template(tokenize=False)` on that repo's own tokenizer
//! files and chat_template.jinja (rendered on 2026-11-15: "Today Date: 15 November 2026").
use dense_engine::config::{Family, LayerKind, RopeScaling};
use serde_json::{json, Value};

use super::chat::{check_template, date_from_days, local_date_at, local_offset_at, today_local, today_utc};
use super::*;

const SMOLLM3_3B: &str = r#"{"architectures": ["SmolLM3ForCausalLM"], "attention_bias": false, "attention_dropout": 0.0, "bos_token_id": 128000, "eos_token_id": 128012, "hidden_act": "silu", "hidden_size": 2048, "initializer_range": 0.02, "intermediate_size": 11008, "layer_types": ["full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention"], "max_position_embeddings": 65536, "max_window_layers": 28, "mlp_bias": false, "model_type": "smollm3", "no_rope_layer_interval": 4, "no_rope_layers": [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0], "num_attention_heads": 16, "num_hidden_layers": 36, "num_key_value_heads": 4, "pad_token_id": 128004, "pretraining_tp": 2, "quantization": {"group_size": 64, "bits": 4}, "quantization_config": {"group_size": 64, "bits": 4}, "rms_norm_eps": 1e-06, "rope_scaling": null, "rope_theta": 5000000.0, "sliding_window": null, "torch_dtype": "bfloat16", "transformers_version": "4.54.0.dev0", "use_cache": false, "use_sliding_window": false, "vocab_size": 128256}"#;
const DATE: &str = "15 November 2026";

fn j(s: &str) -> Value { serde_json::from_str(s).unwrap() }
fn edit(mut v: Value, f: impl FnOnce(&mut Value)) -> Value { f(&mut v); v }
fn m(r: &str, c: &str) -> (String, String) { (r.to_string(), c.to_string()) }

#[test]
fn smollm3_3b_mapping() {
    let d = config::parse(&j(SMOLLM3_3B), None, Some(128012)).unwrap();
    assert_eq!(d.family, Family::SmolLM3);
    assert_eq!((d.hidden, d.inter, d.layers, d.heads, d.kv_heads, d.head_dim, d.vocab), (2048, 11008, 36, 16, 4, 128, 128256));
    assert_eq!((d.rope_dims, d.rope_theta, d.rope_scale, d.rope_traditional), (128, 5000000.0, RopeScaling::None, false));
    // no tie_word_embeddings key: mlx llama default true (the repo's weights have no lm_head)
    assert!(d.tie_word_embeddings && !d.attention_bias && !d.o_bias && !d.mlp_bias && !d.qk_norm);
    assert_eq!((d.eps, d.max_pos, d.bos, d.eos.clone(), d.quant), (1e-6, 65536, Some(128000), vec![128012], Some((64, 4))));
    assert!(d.kinds.iter().all(|k| *k == LayerKind::Full) && d.sliding_window.is_none());
    // config no_rope_layers [1, 1, 1, 0, ...]: layers 3, 7, ..., 35 without RoPE
    let off: Vec<usize> = (0..36).filter(|i| d.no_rope_layers[*i]).collect();
    assert_eq!(off, (0..9).map(|k| 4 * k + 3).collect::<Vec<_>>());
    assert!(blockers(&j(SMOLLM3_3B)).is_empty(), "{:?}", blockers(&j(SMOLLM3_3B)));
}

#[test]
fn smollm3_no_rope_defaults_and_refusals() {
    let c = j(SMOLLM3_3B);
    // absent no_rope_layers: interval rule (default 4) gives the same list
    let d = config::parse(&edit(c.clone(), |v| { v.as_object_mut().unwrap().remove("no_rope_layers"); v.as_object_mut().unwrap().remove("no_rope_layer_interval"); }), None, None).unwrap();
    assert_eq!(d.no_rope_layers, config::parse(&c, None, None).unwrap().no_rope_layers);
    let d = config::parse(&edit(c.clone(), |v| { v.as_object_mut().unwrap().remove("no_rope_layers"); v["no_rope_layer_interval"] = json!(3); }), None, None).unwrap();
    assert_eq!((0..36).filter(|i| d.no_rope_layers[*i]).count(), 12);
    let e = |v: Value| config::parse(&v, None, None).unwrap_err();
    assert!(e(edit(c.clone(), |v| v["no_rope_layers"] = json!([1, 0]))).contains("length mismatch"));
    assert!(e(edit(c.clone(), |v| v["model_type"] = json!("llama"))).contains("not \"smollm3\""));
    assert!(e(edit(c.clone(), |v| v["num_experts"] = json!(8))).contains("mixture-of-experts"));
    // generation_config eos replaces config eos (mlx_lm)
    let d = config::parse(&c, Some(&json!({"eos_token_id": 128012})), Some(128012)).unwrap();
    assert_eq!(d.eos, vec![128012]);
    let b = blockers(&edit(c, |v| { v.as_object_mut().unwrap().remove("quantization"); v.as_object_mut().unwrap().remove("quantization_config"); }));
    assert!(b.iter().any(|x| x.contains("no MLX quantization")), "{:?}", b);
}

#[test]
fn dates() {
    assert_eq!(date_from_days(0), "01 January 1970");
    assert_eq!(date_from_days(20407), "15 November 2025");
    assert_eq!(date_from_days(20772), DATE);
}

/// the chat date is the LOCAL date (HF `strftime_now` = `datetime.now()`), not the UTC one.
#[test]
fn local_date() {
    // 2026-11-15 23:30 UTC and 2026-11-16 00:30 UTC: the date follows the local offset across midnight.
    for t in [1_794_785_400i64, 1_794_789_000, 0, 1_700_000_000] {
        let off = local_offset_at(t).expect("localtime_r");
        assert!(off.abs() <= 14 * 3600, "offset {}", off);
        assert_eq!(local_date_at(t), date_from_days((t + off).div_euclid(86400)), "t={}", t);
    }
    let (l, u) = (today_local(), today_utc());
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
    let around: Vec<String> = [-1i64, 0, 1].iter().map(|d| date_from_days(now.div_euclid(86400) + d)).collect();
    assert!(around.contains(&l) && around.contains(&u), "{} {}", l, u);
    assert!(chat_prompt(&[m("user", "Hello")], true).unwrap().contains(&format!("Today Date: {}\n", l)));
}

#[test]
fn smollm3_template_check() {
    let t = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/smollm3/smollm3_chat_template.jinja")).unwrap();
    assert!(check_template(Some(&t)).is_ok());
    assert!(check_template(Some(&t.replace("June 2025", "May 2025"))).is_err());
    assert!(check_template(Some(&t.replace("trained by Hugging Face.\\n\\n", "x\\n\\n"))).is_err());
    assert!(check_template(None).is_err());
}

#[test]
fn smollm3_strings() {
    assert_eq!(chat_prompt_on(&[m("user", "Hello")], true, DATE).unwrap(),
               r###"<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 November 2026
Reasoning Mode: /think

## Custom Instructions

You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.

<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
"###);
    assert_eq!(chat_prompt_on(&[m("user", "Hello")], false, DATE).unwrap(),
               r###"<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 November 2026
Reasoning Mode: /think

## Custom Instructions

You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.

<|im_start|>user
Hello<|im_end|>
"###);
    assert_eq!(chat_prompt_on(&[m("system", "You are terse."), m("user", "Hi there")], true, DATE).unwrap(),
               r###"<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 November 2026
Reasoning Mode: /think

## Custom Instructions

You are terse.

<|im_start|>user
Hi there<|im_end|>
<|im_start|>assistant
"###);
    assert_eq!(chat_prompt_on(&[m("system", "You are terse."), m("user", "Hi there")], false, DATE).unwrap(),
               r###"<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 November 2026
Reasoning Mode: /think

## Custom Instructions

You are terse.

<|im_start|>user
Hi there<|im_end|>
"###);
    assert_eq!(chat_prompt_on(&[m("user", "What is 2+2?"), m("assistant", "4"), m("user", "And 3+3?")], true, DATE).unwrap(),
               r###"<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 November 2026
Reasoning Mode: /think

## Custom Instructions

You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.

<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
4<|im_end|>
<|im_start|>user
And 3+3?<|im_end|>
<|im_start|>assistant
"###);
    assert_eq!(chat_prompt_on(&[m("user", "What is 2+2?"), m("assistant", "4"), m("user", "And 3+3?")], false, DATE).unwrap(),
               r###"<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 November 2026
Reasoning Mode: /think

## Custom Instructions

You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.

<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
4<|im_end|>
<|im_start|>user
And 3+3?<|im_end|>
"###);
    assert_eq!(chat_prompt_on(&[m("system", "Be brief."), m("user", "A"), m("assistant", "B"), m("user", "C")], true, DATE).unwrap(),
               r###"<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 November 2026
Reasoning Mode: /think

## Custom Instructions

Be brief.

<|im_start|>user
A<|im_end|>
<|im_start|>assistant
B<|im_end|>
<|im_start|>user
C<|im_end|>
<|im_start|>assistant
"###);
    assert_eq!(chat_prompt_on(&[m("system", "Be brief."), m("user", "A"), m("assistant", "B"), m("user", "C")], false, DATE).unwrap(),
               r###"<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 November 2026
Reasoning Mode: /think

## Custom Instructions

Be brief.

<|im_start|>user
A<|im_end|>
<|im_start|>assistant
B<|im_end|>
<|im_start|>user
C<|im_end|>
"###);
    assert_eq!(chat_prompt_on(&[m("user", "  padded \n")], true, DATE).unwrap(),
               r###"<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 November 2026
Reasoning Mode: /think

## Custom Instructions

You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.

<|im_start|>user
  padded 
<|im_end|>
<|im_start|>assistant
"###);
    assert_eq!(chat_prompt_on(&[m("user", "  padded \n")], false, DATE).unwrap(),
               r###"<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 November 2026
Reasoning Mode: /think

## Custom Instructions

You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.

<|im_start|>user
  padded 
<|im_end|>
"###);
    assert_eq!(chat_prompt_on(&[m("system", "Be brief. /no_think"), m("user", "Hi")], true, DATE).unwrap(),
               r###"<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 November 2026
Reasoning Mode: /no_think

## Custom Instructions

Be brief.

<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
<think>

</think>
"###);
    assert_eq!(chat_prompt_on(&[m("system", "Be brief. /no_think"), m("user", "Hi")], false, DATE).unwrap(),
               r###"<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 November 2026
Reasoning Mode: /no_think

## Custom Instructions

Be brief.

<|im_start|>user
Hi<|im_end|>
"###);
    assert_eq!(chat_prompt_on(&[m("system", "/think"), m("user", "Hi")], true, DATE).unwrap(),
               r###"<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 November 2026
Reasoning Mode: /think

## Custom Instructions

You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.

<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
"###);
    assert_eq!(chat_prompt_on(&[m("system", "/think"), m("user", "Hi")], false, DATE).unwrap(),
               r###"<|im_start|>system
## Metadata

Knowledge Cutoff Date: June 2025
Today Date: 15 November 2026
Reasoning Mode: /think

## Custom Instructions

You are a helpful AI assistant named SmolLM, trained by Hugging Face. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracking, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> Thought section </think> Solution section. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion.

<|im_start|>user
Hi<|im_end|>
"###);
    // assistant history: leading newlines stripped; /no_think adds the empty think block (HF renders)
    assert_eq!(chat_prompt_on(&[m("system", "Be brief /no_think"), m("user", "a"), m("assistant", "\n\nb"), m("user", "c")], true, DATE).unwrap(),
               "<|im_start|>system\n## Metadata\n\nKnowledge Cutoff Date: June 2025\nToday Date: 15 November 2026\nReasoning Mode: /no_think\n\n## Custom Instructions\n\nBe brief\n\n<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\nb<|im_end|>\n<|im_start|>user\nc<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n");
    assert!(chat_prompt_on(&[m("user", "a"), m("assistant", "\n\nb\n"), m("user", "c")], true, DATE).unwrap()
        .ends_with("<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\nb\n<|im_end|>\n<|im_start|>user\nc<|im_end|>\n<|im_start|>assistant\n"));
    // /system_override
    assert_eq!(chat_prompt_on(&[m("system", "Keep it /system_override  "), m("user", "a")], true, DATE).unwrap(),
               "<|im_start|>system\nKeep it<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n");
    assert_eq!(chat_prompt_on(&[m("system", "x /system_override /no_think"), m("user", "a")], true, DATE).unwrap(),
               "<|im_start|>system\nx<|im_end|>\n<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n");
    // empty system: think default text
    assert_eq!(chat_prompt_on(&[m("system", ""), m("user", "a")], true, DATE).unwrap(),
               chat_prompt_on(&[m("user", "a")], true, DATE).unwrap());
    assert!(chat_prompt_on(&[m("user", "a"), m("tool", "b")], true, DATE).unwrap_err().contains("role"));
    assert!(chat_prompt_on(&[m("user", "a"), m("system", "b")], true, DATE).is_err());
    assert!(chat_prompt(&[m("user", "Hello")], true).unwrap().starts_with("<|im_start|>system\n## Metadata\n"));
}

#[test]
fn model_type_list() {
    assert_eq!(MODEL_TYPES, &["smollm3"]);
}
