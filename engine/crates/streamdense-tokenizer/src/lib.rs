//! streamdense-tokenizer: BPE tokenizer for HF `tokenizer.json` files (Qwen, GPT-2 style byte-level such as
//! Granite / SmolLM2, Llama 3 style, SentencePiece-style BPE such as Llama 2 / Gemma / Mistral), the Qwen chat
//! template (incl. tools) and chat KV bookkeeping, in Rust (replaces HF transformers' AutoTokenizer /
//! apply_chat_template that the Python UI and gateway used). Chat templates of other families live with those
//! families, not here.
//!
//! # Public API (stable; the gateway codes against this)
//!
//! ## Tokenizer
//! * [`Tokenizer::from_path`]`(dir) -> Result<Tokenizer, String>`; `encode(&str) -> Vec<u32>`, `decode(&[u32]) -> String`,
//!   `token_id(&str) -> Option<u32>`; [`StreamDecoder`] (`push(&tk, id) -> String` emits only complete UTF-8 chars,
//!   `finish() -> String`). Stop ids: [`IM_END`], [`ENDOFTEXT`] (Qwen).
//! * Any family: `encode` adds no BOS/EOS (HF `add_special_tokens=False`); `encode_with_special(&str)` adds the
//!   post-processor template ids (`special_prefix()` / `special_suffix()`, `add_bos()`, `bos_id()`), e.g. Llama /
//!   Gemma BOS. `decode_skip_special(&[u32])` = HF `skip_special_tokens=True`. An unsupported tokenizer.json
//!   component (Unigram/WordPiece model, Precompiled normalizer, other look-around regexes, ...) fails at load
//!   with an error naming it; supported components are listed in [`pipeline`].
//!
//! ## Chat template (exact to transformers `apply_chat_template` with models/qwen38_27b_mlx4/chat_template.jinja)
//! ```ignore
//! pub struct ChatOptions { pub add_generation_prompt: bool, pub enable_thinking: bool, pub reasoning_effort: Option<String> }
//!
//! /// One message. role: "system" | "developer" | "user" | "assistant" | "tool".
//! /// content: plain text (callers flatten content-part lists by concatenating the text parts with NO separator,
//! /// which is what the template's render_content does; the template `|trim`s it).
//! pub struct ChatMessage {
//!     pub role: String,
//!     pub content: String,
//!     pub reasoning_content: Option<String>,   // assistant only; rendered inside <think>..</think>
//!     pub tool_calls: Vec<ToolCall>,           // assistant only; rendered as <tool_call><function=..> XML
//! }
//! impl ChatMessage {
//!     pub fn new(role: &str, content: &str) -> ChatMessage;
//!     pub fn tool(content: &str) -> ChatMessage;                                      // role "tool" (a tool result)
//!     pub fn assistant(content: &str, reasoning: Option<&str>, calls: Vec<ToolCall>) -> ChatMessage;
//!     /// Generated assistant text -> message (reasoning, visible content, parsed tool calls). `raw` is what the
//!     /// model produced after `<|im_start|>assistant\n` (prefix "<think>\n" when thinking was on; see parse_reply).
//!     pub fn assistant_from_reply(raw: &str) -> ChatMessage;
//! }
//! impl From<Message> for ChatMessage   // legacy 3-field Message (no tools), still used by `qwen serve` op "chat"
//!
//! /// A tool definition; rendered in the system turn as
//! /// {"type": "function", "function": {"name": .., "description": .., "parameters": ..}} via Python-json `tojson`.
//! pub struct ToolDef { pub name: String, pub description: String, pub parameters: serde_json::Value }
//! impl ToolDef {
//!     pub fn from_anthropic(tool: &Value) -> Result<ToolDef, String>;   // {"name","description","input_schema"}
//!     pub fn from_openai(tool: &Value) -> Result<ToolDef, String>;      // {"type":"function","function":{..}} or the inner object
//!     pub fn to_template_value(&self) -> Value;
//! }
//!
//! /// A parsed / to-be-rendered tool call. `args` keeps insertion order (serde_json `preserve_order`), which is the
//! /// order the template renders the <parameter=..> blocks in.
//! pub struct ToolCall { pub name: String, pub args: serde_json::Map<String, Value> }
//!
//! pub fn render_chat(messages: &[ChatMessage], tools: &[ToolDef], opt: &ChatOptions) -> Result<String, String>;
//! pub fn render(messages: &[Message], opt: &ChatOptions) -> Result<String, String>;   // = render_chat(.., &[], ..)
//! ```
//!
//! ## Tool-call parsing (model output -> calls)
//! ```ignore
//! /// `text` = the generated assistant turn. When thinking was ON, prefix it with "<think>\n" (the prompt's
//! /// generation prompt opened the think block); with thinking OFF pass it as generated. A `</think>` with no
//! /// opening tag is also treated as the end of leading reasoning.
//! pub fn parse_tool_calls(text: &str) -> (String /*visible, trimmed*/, Vec<ToolCall>);
//! pub struct ParsedReply { pub reasoning: Option<String>, pub visible: String, pub calls: Vec<ToolCall> }
//! pub fn parse_reply(text: &str) -> ParsedReply;
//! ```
//! Handles the Qwen XML form `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>`
//! (values: exactly one leading and one trailing "\n" removed = inverse of the template; values stay JSON strings,
//! use [`coerce_args`]), the JSON form `<tool_call>{"name":..,"arguments":{..} or "<json string>"}</tool_call>`,
//! a bare `<function=..>` without `<tool_call>`, and missing closing tags at the end (truncated output).
//! `<think>..</think>` is stripped; visible text = everything outside think blocks before the first tool call, trimmed.
//!
//! ## Streaming
//! ```ignore
//! pub struct StreamVisible { pub text: String, pub tool_call: bool, pub in_think: bool }
//! pub fn stream_visible(text_so_far: &str) -> StreamVisible;
//! ```
//! `text_so_far` = all text decoded so far (same "<think>\n" prefix rule as parse_tool_calls). `.text` is the
//! visible text that is safe to emit now: it stops at `<tool_call>` / `<function=` (`tool_call` = true then; emit
//! nothing more as text), skips think blocks (`in_think` = inside an unclosed one), and holds back a trailing partial
//! tag (`<`, `<tool_ca`, `</thi`, ...), trailing whitespace, and a trailing U+FFFD (incomplete UTF-8 from a lossy
//! decode). Guarantee: as text grows, each `.text` is a prefix of the next one and of `parse_reply(final).visible`,
//! so a gateway emits `vis.text[sent..]`, and at the end emits `parse_reply(final).visible[sent..]`.
//!
//! ## Argument coercion
//! ```ignore
//! /// String values -> boolean / integer / number / array / object / null per `schema["properties"][k]["type"]`
//! /// (type may be a list; when it allows "string" the value stays a string). Unparseable values stay strings.
//! pub fn coerce_args(args: &Map<String, Value>, schema: &Value) -> Map<String, Value>;
//! ```
//!
//! ## Misc
//! * [`py_json`]`(&Value) -> String`: Python `json.dumps(v, ensure_ascii=False)` (", " / ": " separators,
//!   Python float repr) = the template's `tojson`.
//! * [`ChatState`]: KV reuse planning. `plan(tk, &[Message], opt, allow_reuse)` / `commit(..)` (legacy) and
//!   `plan_chat(tk, &[ChatMessage], &[ToolDef], opt, allow_reuse) -> Result<Plan, String>` /
//!   `commit_chat(mode, fed, generated, &[ChatMessage], &[ToolDef], opt, ok)`. Followup mode also covers
//!   tool-result turns after an assistant tool call (tools must be identical to the previous request's).
pub mod chat;
pub mod pipeline;
pub mod session;
pub mod tokenizer;
pub mod tools;
pub use chat::{render, render_chat, ChatMessage, ChatOptions, Message, ToolDef};
pub use session::{ChatState, Mode, Plan, ENDOFTEXT, IM_END};
pub use tokenizer::{StreamDecoder, Tokenizer, QWEN35_PRETOKENIZE};
pub use tools::{coerce_args, parse_reply, parse_tool_calls, py_json, stream_visible, ParsedReply, StreamVisible, ToolCall};

/// Default tokenizer directory (repo-relative): the Qwen3.8-27B MLX 4-bit model.
pub const DEFAULT_TOKENIZER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../models/qwen38_27b_mlx4");
