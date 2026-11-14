//! Chat KV planner (what `qwen serve` op "chat" feeds), simulated without an engine: the "engine state" is the
//! token sequence ChatState says it holds; every reuse must equal what a full re-render would give, or follow
//! the Python UI followup rule.
use std::sync::OnceLock;
use streamdense_tokenizer::*;

fn tk() -> Option<&'static Tokenizer> {
    static T: OnceLock<Option<Tokenizer>> = OnceLock::new();
    let t = T.get_or_init(|| {
        if !std::path::Path::new(&format!("{}/tokenizer.json", DEFAULT_TOKENIZER_DIR)).exists() { return None; }
        Some(Tokenizer::from_path(DEFAULT_TOKENIZER_DIR).unwrap())
    }).as_ref();
    // printed per test (not once in the OnceLock) so every skipped test says why
    if t.is_none() { eprintln!("SKIP: needs models/qwen38_27b_mlx4/tokenizer.json (missing)"); }
    t
}
fn opt(think: bool) -> ChatOptions { ChatOptions { enable_thinking: think, ..Default::default() } }

/// One turn: plan, check the plan against a full render, "generate" `reply` (+ stop token), commit.
fn turn(t: &Tokenizer, st: &mut ChatState, msgs: &[Message], o: &ChatOptions, cont: bool, reply: &str, stop: Option<u32>) -> (Mode, usize) {
    let p = st.plan(t, msgs, o, cont).unwrap();
    let full = t.encode(&render(msgs, o).unwrap());
    assert_eq!(p.full, full);
    let reused = if p.mode == Mode::Full { 0 } else { st.seq.len() };
    if p.mode == Mode::Prefix {
        let mut s = st.seq.clone(); s.extend_from_slice(&p.tokens);
        assert_eq!(s, full, "prefix mode must feed exactly the rest of the rendered prompt");
    }
    if p.mode == Mode::Full { assert_eq!(p.tokens, full); }
    let mut gen = t.encode(reply);
    if let Some(s) = stop { gen.push(s); }
    st.commit(p.mode, &p.tokens, &gen, msgs, o, true);
    (p.mode, reused)
}

#[test]
fn multi_turn_reuse_think_off() {
    let Some(t) = tk() else { return };
    let mut st = ChatState::new();
    let o = opt(false);
    let mut m = vec![Message::new("system", "Be brief."), Message::new("user", "Hi")];
    assert_eq!(turn(t, &mut st, &m, &o, true, "Hello! How can I help?", Some(IM_END)).0, Mode::Full);
    m.push(Message::assistant_from_raw("Hello! How can I help?"));
    m.push(Message::new("user", "Write a haiku about 东京 😀"));
    let (mode, reused) = turn(t, &mut st, &m, &o, true, "Silent GPU hums", Some(IM_END));
    assert_eq!(mode, Mode::Prefix);
    assert!(reused > 10);
    // third turn still a prefix
    m.push(Message::assistant_from_raw("Silent GPU hums"));
    m.push(Message::new("user", "Thanks"));
    assert_eq!(turn(t, &mut st, &m, &o, true, "You're welcome.", Some(IM_END)).0, Mode::Prefix);
    // continue=false -> full
    assert_eq!(turn(t, &mut st, &m, &o, false, "ok", Some(IM_END)).0, Mode::Full);
}

#[test]
fn multi_turn_reuse_think_on() {
    let Some(t) = tk() else { return };
    let mut st = ChatState::new();
    let o = opt(true);
    let mut m = vec![Message::new("user", "2+2?")];
    // with thinking on the prompt ends in "<think>\n"; the model writes reasoning, </think>, the answer
    let raw = "The user asks 2+2. That is 4.\n</think>\n\n4";
    assert_eq!(turn(t, &mut st, &m, &o, true, raw, Some(IM_END)).0, Mode::Full);
    let a = Message::assistant_from_raw(raw);
    assert_eq!(a.reasoning_content.as_deref(), Some("The user asks 2+2. That is 4.\n"));
    assert_eq!(a.content, "\n\n4");
    m.push(a);
    m.push(Message::new("user", "And 3+3?"));
    assert_eq!(turn(t, &mut st, &m, &o, true, "6\n</think>\n\n6", Some(IM_END)).0, Mode::Prefix);
    // toggling thinking changes the system turn -> full
    m.push(Message::assistant_from_raw("6\n</think>\n\n6"));
    m.push(Message::new("user", "bye"));
    assert_eq!(turn(t, &mut st, &m, &opt(false), true, "bye", Some(IM_END)).0, Mode::Full);
}

#[test]
fn followup_after_length_stop_and_edits() {
    let Some(t) = tk() else { return };
    let mut st = ChatState::new();
    let o = opt(false);
    let mut m = vec![Message::new("user", "Count to 100")];
    turn(t, &mut st, &m, &o, true, "1, 2, 3, 4", None); // stopped at nnew: no <|im_end|>
    let before = st.seq.clone();
    // the same text sent back: the KV (no <|im_end|> yet) is a token prefix of the re-render -> prefix
    let mut same = m.clone();
    same.push(Message::new("assistant", "1, 2, 3, 4"));
    same.push(Message::new("user", "Stop."));
    assert_eq!(st.plan(t, &same, &o, true).unwrap().mode, Mode::Prefix);
    // the app sends a different assistant text (e.g. trimmed): Python-UI followup, the KV keeps the model's tokens
    m.push(Message::new("assistant", "1, 2, 3"));
    m.push(Message::new("user", "Stop."));
    let p = st.plan(t, &m, &o, true).unwrap();
    assert_eq!(p.mode, Mode::Followup);
    // Python UI: "<|im_end|>" + "\n" + the new turn + generation prompt
    assert_eq!(t.decode(&p.tokens), "<|im_end|>\n<|im_start|>user\nStop.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    st.commit(p.mode, &p.tokens, &t.encode("OK"), &m, &o, true);
    assert_eq!(&st.seq[..before.len()], &before[..]);
    // an edited earlier user message -> full
    let mut e = m.clone();
    e[0].content = "Count to 10".into();
    e.push(Message::new("assistant", "OK"));
    e.push(Message::new("user", "again"));
    let pe = st.plan(t, &e, &o, true).unwrap();
    assert_eq!(pe.mode, Mode::Full, "{:?}", t.decode(&pe.tokens));
    // an error / ctx_full invalidates
    st.commit(Mode::Prefix, &[], &[1], &m, &o, false);
    assert!(!st.is_valid());
    assert_eq!(st.plan(t, &m, &o, true).unwrap().mode, Mode::Full);
}

#[test]
fn template_errors() {
    assert!(render(&[], &ChatOptions::default()).is_err());
    let bad = [Message::new("user", "a"), Message::new("system", "late")];
    assert!(render(&bad, &ChatOptions::default()).is_err());
    assert!(render(&[Message::new("bogus", "x")], &ChatOptions::default()).is_err());
    let o = ChatOptions { enable_thinking: true, reasoning_effort: Some("max".into()), ..Default::default() };
    assert!(render(&[Message::new("user", "x")], &o).is_err());
}
