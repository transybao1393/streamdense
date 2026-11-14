//! `Engine`: one Metal arena holding the weights, a bf16 KV cache per layer and the pass scratch; prefill in passes
//! of up to `attn.max_pass()` tokens (8 split-K / sdpa, 128 flash / auto), decode one token per pass, greedy argmax.
//! With `AttnImpl::Auto` the last prompt token runs as its own 1-row (sdpa) pass, as mlx_lm's generate_step does
//! (prompt[:-1] in prefill chunks, then the last token through the decode step). `forward` / `prefill` / `step`: one
//! command buffer per pass, waited on; the position lives in `SdCtrl.pos`, written by the host before every pass.
//! `generate` decodes with `DecodeMode::Pipelined` by default (pass t+1 encoded while pass t runs, its token id and
//! ctrl.pos set on the GPU) or `DecodeMode::Sync` (the `step` loop). No MTP / speculation.
use crate::config::DenseConfig;
use crate::gpu::{commit, run, wait, Arena, CmdBuf, Gpu, SdCtrl};
use crate::config::{rope_inv_freq, RopeScaling};
use crate::model::{self, AttnImpl, Dims, Kv, PassIo, BF, CHUNK, F32, K_MAX};
use crate::prefill::{Prefill, PrefillModel};
use crate::weights::{self, SafeSource, TensorSource, Weights};
use std::collections::VecDeque;
use std::time::Instant;

/// How `prefill` runs prompts of >= `GEMM_MIN_TOKENS` tokens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrefillMode {
    /// prompt[..n-1] through the GEMM prefill (src/prefill.rs: q4b_gemm + attn128_flash, chunks of
    /// `GEMM_T_MAX` rows), then the last token as a 1-row pass of `attn` (MLX generate_step order).
    #[default]
    Gemm,
    /// Engine passes (`forward`) only: up to `attn.max_pass()` rows each, decode matmul kernels.
    Passes,
}

impl PrefillMode {
    pub fn parse(s: &str) -> Result<PrefillMode, String> {
        match s { "gemm" => Ok(PrefillMode::Gemm), "passes" => Ok(PrefillMode::Passes), _ => Err(format!("prefill {:?}: want gemm | passes", s)) }
    }
    pub fn name(self) -> &'static str { match self { PrefillMode::Gemm => "gemm", PrefillMode::Passes => "passes" } }
}

/// How `generate` runs the decode passes (tokens 2..). Both dispatch the same kernels in the same order on the same
/// buffers, so tokens and logits are bit-identical; only the CPU / GPU overlap differs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DecodeMode {
    /// Up to `PIPELINE_DEPTH` command buffers in flight: pass t+1 is encoded and committed while pass t runs. Its
    /// token row is filled on the GPU (`qd_next_tok`: ctrl.tok[0] of pass t -> tok[0], ctrl.pos = pos), so no CPU
    /// round trip sits between passes; the host reads each token afterwards from the argmax record rec[pos]. On a
    /// stop token the already-queued look-ahead pass is waited for and discarded (it wrote KV at a position past
    /// `pos()`, which later passes overwrite).
    #[default]
    Pipelined,
    /// One command buffer per pass, committed and waited on (host writes the token and ctrl.pos, reads ctrl.tok).
    Sync,
}

impl DecodeMode {
    pub fn parse(s: &str) -> Result<DecodeMode, String> {
        match s { "pipelined" => Ok(DecodeMode::Pipelined), "sync" => Ok(DecodeMode::Sync), _ => Err(format!("decode {:?}: want pipelined | sync", s)) }
    }
    pub fn name(self) -> &'static str { match self { DecodeMode::Pipelined => "pipelined", DecodeMode::Sync => "sync" } }
}

/// Command buffers in flight in `DecodeMode::Pipelined` (the running pass + the next one, already encoded).
pub const PIPELINE_DEPTH: usize = 2;

/// Shortest prompt the GEMM prefill takes (shorter ones run as passes).
pub const GEMM_MIN_TOKENS: usize = 16;
/// Rows per GEMM prefill chunk (scratch ~19 MB at Qwen3-1.7B shapes).
pub const GEMM_T_MAX: usize = 512;

#[derive(Clone, Copy, Debug)]
pub struct EngineOptions {
    /// KV cache length (tokens); prompt + generated must fit.
    pub max_ctx: usize,
    pub attn: AttnImpl,
    pub prefill: PrefillMode,
    pub decode: DecodeMode,
}

impl Default for EngineOptions {
    fn default() -> Self { EngineOptions { max_ctx: 4096, attn: AttnImpl::Auto, prefill: PrefillMode::Gemm, decode: DecodeMode::Pipelined } }
}

pub struct Engine {
    pub cfg: DenseConfig,
    pub gpu: Gpu,
    pub arena: Arena,
    pub w: Weights,
    pub kv: Vec<Kv>,
    pub dims: Dims,
    /// per layer: sliding window (None = full attention)
    pub windows: Vec<Option<u32>>,
    pub attn: AttnImpl,
    pub max_ctx: usize,
    /// max rows of one pass (attn.max_pass())
    pub pass_max: usize,
    pub prefill_mode: PrefillMode,
    pub decode_mode: DecodeMode,
    /// GEMM prefill (own scratch buffer), built on first use
    pf: Option<Prefill>,
    /// scratch layout (k / pos filled per pass)
    io: PassIo,
    pos: usize,
    /// f32 [vocab] logits slots: io.logits and a second one; pipelined decode alternates them so the look-ahead
    /// pass never overwrites the logits of the last accepted pass
    logits_alt: usize,
    /// slot holding the logits of the last head pass
    cur_logits: usize,
    /// seconds spent compiling shaders + reading weights
    pub load_s: f64,
    /// bytes of the arena (weights + KV + scratch)
    pub arena_bytes: usize,
}

/// Result of `generate`.
#[derive(Clone, Debug, Default)]
pub struct GenStats {
    pub prompt_tokens: usize,
    /// generated tokens (first one comes from the prefill pass), stop token included if hit
    pub tokens: Vec<u32>,
    pub prefill_s: f64,
    /// time of the decode passes (tokens 2..)
    pub decode_s: f64,
    pub decode_passes: usize,
}

impl GenStats {
    pub fn decode_tok_s(&self) -> f64 { if self.decode_s > 0.0 { self.decode_passes as f64 / self.decode_s } else { 0.0 } }
    pub fn prefill_tok_s(&self) -> f64 { if self.prefill_s > 0.0 { self.prompt_tokens as f64 / self.prefill_s } else { 0.0 } }
}

/// Lifetime peak phys_footprint of this process (bytes; what Activity Monitor / jetsam count, Metal buffers included).
pub fn peak_footprint() -> u64 {
    let mut ri: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
    // SAFETY: proc_pid_rusage fills a rusage_info_v4 for our own pid.
    let r = unsafe { libc::proc_pid_rusage(std::process::id() as i32, libc::RUSAGE_INFO_V4, &mut ri as *mut _ as *mut libc::rusage_info_t) };
    if r == 0 { ri.ri_lifetime_max_phys_footprint } else { 0 }
}

/// Current phys_footprint (bytes).
pub fn footprint() -> u64 { streamdense_core::footprint::self_footprint() }

impl Engine {
    /// Load an MLX 4-bit Qwen dense model dir (config.json + model.safetensors[.index.json]).
    pub fn load(dir: &str, opts: EngineOptions) -> Result<Engine, String> {
        let cfg = DenseConfig::from_model_dir(dir)?;
        let src = SafeSource::open(dir)?;
        Engine::from_source(cfg, &src, opts)
    }

    /// Build from a parsed config and any tensor source (tests: `weights::MemSource`).
    pub fn from_source(cfg: DenseConfig, src: &dyn TensorSource, opts: EngineOptions) -> Result<Engine, String> {
        let t0 = Instant::now();
        cfg.validate()?;
        let (mut w, jobs) = weights::layout(&cfg, src)?;
        let max_ctx = opts.max_ctx.max(1);
        let nch_max = max_ctx.div_ceil(CHUNK);
        let (h, nh, nkv, hd) = (cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim);
        let dims = Dims::new(&cfg);
        let windows = model::layer_windows(&cfg);
        let prow = cfg.attn_in_rows();
        let kv_layer = nkv * max_ctx * hd * BF;
        let al = |n: usize| (n + 255) / 256 * 256;
        let r = opts.attn.max_pass();
        // split-K partials only for passes of <= K_MAX rows (split-K / the sdpa long-row fallback). The last three
        // entries came after the Qwen-only layout (earlier offsets unchanged): f2 (up rows, unfused FFN only), r
        // (residual branch), the sliding layers' RoPE table.
        let scratch = [
            r * 4, r * h * BF, r * h * BF, r * prow * BF, r * nh * hd * BF, r * nh * hd * BF,
            r * cfg.inter * BF, K_MAX * nh * nch_max * hd * F32, K_MAX * nh * nch_max * 2 * F32, h * BF,
            cfg.vocab * F32, (max_ctx + r) * 16, (cfg.rope_dims / 2) * F32, std::mem::size_of::<SdCtrl>(), 256, cfg.vocab * F32,
            if dims.ffn_unfused { r * cfg.inter * BF } else { 0 }, r * h * BF, (cfg.rope_dims / 2) * F32,
        ];
        let total = w.bytes + cfg.layers * 2 * al(kv_layer) + scratch.iter().map(|&n| al(n)).sum::<usize>() + 256;

        let mut gpu = Gpu::from_sources(&crate::ENGINE_SHADER_SOURCE)?;
        if total > gpu.max_buffer() {
            return Err(format!("engine: arena {} B > max Metal buffer {} B (lower --max-ctx or split the arena)", total, gpu.max_buffer()));
        }
        for p in model::pipelines(opts.attn, &dims, &windows) { gpu.pipeline(&p)?; }
        let mut arena = Arena::new(&gpu, total)?;
        let base = arena.alloc(w.bytes);
        weights::load_into(src, &jobs, arena.bytes_mut(base, w.bytes))?;
        w.rebase(base);
        let kv: Vec<Kv> = (0..cfg.layers).map(|_| Kv { k: arena.alloc(kv_layer), v: arena.alloc(kv_layer) }).collect();
        let mut s = scratch.iter().map(|&n| arena.alloc(n));
        let mut nx = || s.next().unwrap();
        let mut io = PassIo {
            k: 0, pos: 0, tok: nx(), h: nx(), x: nx(), p: nx(), qr: nx(), ao: nx(), f: nx(), po: nx(), pml: nx(),
            nch_max, hf: nx(), logits: nx(), rec: nx(), invf: nx(), ctrl: nx(), dummy: nx(), max_ctx, ..Default::default()
        };
        let logits_alt = nx();
        io.f2 = nx();
        io.r = nx();
        let invf_local = nx();
        arena.put(io.ctrl, &[SdCtrl::default()]);
        // RoPE tables: plain ones from the kernel (Qwen: as before), scaled ones computed on the host
        if cfg.rope_scale == RopeScaling::None {
            run(&gpu, &arena, |e| model::encode_rope_invf(e, cfg.rope_dims, cfg.rope_theta, io.invf))?;
        } else {
            arena.put(io.invf, &rope_inv_freq(cfg.rope_dims, cfg.rope_theta, cfg.rope_scale));
        }
        io.invf_local = io.invf;
        if let Some(t) = cfg.rope_local_theta {
            io.invf_local = invf_local;
            run(&gpu, &arena, |e| model::encode_rope_invf(e, cfg.rope_dims, t, invf_local))?;
        }
        Ok(Engine {
            cfg, gpu, arena, w, kv, dims, windows, attn: opts.attn, max_ctx, pass_max: r, prefill_mode: opts.prefill, decode_mode: opts.decode, pf: None, io, pos: 0,
            logits_alt, cur_logits: io.logits,
            load_s: t0.elapsed().as_secs_f64(), arena_bytes: total,
        })
    }

    /// Tokens already in the KV cache.
    pub fn pos(&self) -> usize { self.pos }
    /// Start a new sequence (the cache is overwritten as positions are reused).
    pub fn reset(&mut self) { self.pos = 0; }

    /// One pass over `tokens` (1..=pass_max) at the current position; with `head`, returns the greedy next token
    /// after the last of them (logits of that row stay readable via `logits()`).
    pub fn forward(&mut self, tokens: &[u32], head: bool) -> Result<Option<u32>, String> {
        let k = tokens.len();
        if k == 0 || k > self.pass_max { return Err(format!("engine: pass of {} tokens (1..={})", k, self.pass_max)); }
        if self.pos + k > self.max_ctx { return Err(format!("engine: context full ({} + {} > max_ctx {})", self.pos, k, self.max_ctx)); }
        if let Some(&t) = tokens.iter().find(|&&t| t as usize >= self.cfg.vocab) {
            return Err(format!("engine: token {} >= vocab {}", t, self.cfg.vocab));
        }
        let io = PassIo { k, pos: self.pos, ..self.io };
        self.arena.put(io.tok, tokens);
        self.arena.put(io.ctrl, &[SdCtrl { pos: self.pos as u32, ..Default::default() }]);
        let (d, w, kv, attn, wins) = (&self.dims, &self.w, &self.kv, self.attn, &self.windows);
        run(&self.gpu, &self.arena, |e| {
            model::encode_embed(e, d, w, &io);
            for ((lw, kvl), win) in w.layers.iter().zip(kv).zip(wins) { model::encode_layer(e, d, lw, kvl, &io, attn, *win); }
            if head { model::encode_head(e, d, w, &io); }
        })?;
        self.pos += k;
        if !head { return Ok(None); }
        self.cur_logits = io.logits;
        let c: Vec<SdCtrl> = self.arena.get(io.ctrl, 1);
        Ok(Some(c[0].tok[0]))
    }

    /// Prefill `tokens` (any length >= 1) at the current position; returns the greedy next token.
    /// Gemm mode and >= GEMM_MIN_TOKENS tokens: tokens[..n-1] through the GEMM prefill, then the last token as a
    /// 1-row pass. Otherwise passes of up to `pass_max`; Auto: the last token is a pass of its own (MLX
    /// generate_step order: prompt[:-1], then the decode path).
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32, String> {
        if tokens.is_empty() { return Err("engine: empty prompt".into()); }
        let n = tokens.len();
        if self.prefill_mode == PrefillMode::Gemm && n >= GEMM_MIN_TOKENS {
            if self.pos + n > self.max_ctx {
                return Err(format!("engine: context full ({} + {} > max_ctx {})", self.pos, n, self.max_ctx));
            }
            let pf = match self.pf.take() { Some(p) => p, None => Prefill::new(&mut self.gpu, &self.cfg, GEMM_T_MAX)? };
            let m = PrefillModel {
                cfg: &self.cfg, w: &self.w, kv: self.kv.iter().map(|k| (k.k, k.v)).collect(), max_ctx: self.max_ctx,
                arena: &self.arena.buf,
            };
            let r = pf.run(&self.gpu, &m, &tokens[..n - 1], self.pos, false);
            self.pf = Some(pf);
            r?;
            self.pos += n - 1;
            return self.forward(&tokens[n - 1..], true)?.ok_or_else(|| "engine: no head pass".into());
        }
        let (body, last) = if self.attn == AttnImpl::Auto { tokens.split_at(tokens.len() - 1) } else { (tokens, &[][..]) };
        let mut passes: Vec<&[u32]> = body.chunks(self.pass_max).collect();
        if !last.is_empty() { passes.push(last); }
        let n = passes.len();
        let mut next = None;
        for (i, c) in passes.into_iter().enumerate() { next = self.forward(c, i + 1 == n)?; }
        next.ok_or_else(|| "engine: no head pass".into())
    }

    /// Decode one token: feed `tok`, return the greedy next.
    pub fn step(&mut self, tok: u32) -> Result<u32, String> {
        self.forward(&[tok], true)?.ok_or_else(|| "engine: no head pass".into())
    }

    /// f32 logits of the last head pass.
    pub fn logits(&self) -> Vec<f32> { self.arena.get(self.cur_logits, self.cfg.vocab) }

    /// Greedy generation from `prompt` (cache reset first): up to `max_new` tokens, stopping after any token in
    /// `stop`. `on_token` sees every generated token as it is produced.
    pub fn generate(&mut self, prompt: &[u32], max_new: usize, stop: &[u32], mut on_token: impl FnMut(u32))
                    -> Result<GenStats, String> {
        self.reset();
        let mut st = GenStats { prompt_tokens: prompt.len(), ..Default::default() };
        if max_new == 0 { return Ok(st); }
        let t0 = Instant::now();
        let mut tok = self.prefill(prompt)?;
        st.prefill_s = t0.elapsed().as_secs_f64();
        st.tokens.push(tok);
        on_token(tok);
        let t1 = Instant::now();
        match self.decode_mode {
            DecodeMode::Sync => {
                while st.tokens.len() < max_new && !stop.contains(&tok) {
                    tok = self.step(tok)?;
                    st.decode_passes += 1;
                    st.tokens.push(tok);
                    on_token(tok);
                }
            }
            DecodeMode::Pipelined => self.decode_pipelined(tok, max_new, stop, &mut st, &mut on_token)?,
        }
        st.decode_s = t1.elapsed().as_secs_f64();
        Ok(st)
    }

    /// Encode + commit (no wait) the 1-row decode pass at `pos` whose input token is the greedy output of the
    /// previous pass (ctrl.tok[0]); logits into slot `slot` (0 = io.logits, 1 = logits_alt).
    fn commit_step(&self, pos: usize, slot: usize) -> CmdBuf {
        let logits = if slot == 0 { self.io.logits } else { self.logits_alt };
        let io = PassIo { k: 1, pos, logits, ..self.io };
        let (d, w, kv, attn, wins) = (&self.dims, &self.w, &self.kv, self.attn, &self.windows);
        commit(&self.gpu, &self.arena, |e| {
            model::encode_next_tok(e, &io);
            model::encode_embed(e, d, w, &io);
            for ((lw, kvl), win) in w.layers.iter().zip(kv).zip(wins) { model::encode_layer(e, d, lw, kvl, &io, attn, *win); }
            model::encode_head(e, d, w, &io);
        })
    }

    /// `DecodeMode::Pipelined` part of `generate`: `first` (already in `st.tokens`) is the greedy output of the last
    /// waited pass. Same passes, kernels and buffers as the `step` loop (bit-identical tokens / logits); up to
    /// `PIPELINE_DEPTH` command buffers in flight, never more passes than `max_new` needs or `max_ctx` holds.
    fn decode_pipelined(&mut self, first: u32, max_new: usize, stop: &[u32], st: &mut GenStats,
                        on_token: &mut impl FnMut(u32)) -> Result<(), String> {
        if st.tokens.len() >= max_new || stop.contains(&first) { return Ok(()); }
        let want = max_new - st.tokens.len();
        let base = self.pos;
        let n = want.min(self.max_ctx.saturating_sub(base)); // passes that fit in the cache
        // nothing in flight: seed ctrl.tok[0] with `first` (the last head pass already left it there)
        self.arena.put(self.io.ctrl, &[SdCtrl { pos: base as u32, tok: [first, 0], ..Default::default() }]);
        let mut q: VecDeque<(CmdBuf, usize)> = VecDeque::new();
        let mut enc = 0;
        while enc < n && q.len() < PIPELINE_DEPTH {
            q.push_back((self.commit_step(base + enc, enc % 2), enc));
            enc += 1;
        }
        while let Some((cb, i)) = q.pop_front() {
            if let Err(e) = wait(&cb) {
                for (c, _) in q.drain(..) { let _ = wait(&c); }
                return Err(e);
            }
            let p = base + i;
            // argmax_b record of this pass's row: rec[p] = (token bits, top1, top2, 0)
            let tok = self.arena.get::<u32>(self.io.rec + p * 16, 1)[0];
            self.pos = p + 1;
            self.cur_logits = if i % 2 == 0 { self.io.logits } else { self.logits_alt };
            let done = stop.contains(&tok);
            // keep the GPU fed before handing the token out (on_token may print / detokenize)
            if !done && enc < n {
                q.push_back((self.commit_step(base + enc, enc % 2), enc));
                enc += 1;
            }
            st.decode_passes += 1;
            st.tokens.push(tok);
            on_token(tok);
            if done {
                // the look-ahead pass past the stop token: let it finish, discard it
                for (c, _) in q.drain(..) { wait(&c)?; }
                return Ok(());
            }
        }
        if n < want {
            return Err(format!("engine: context full ({} + {} > max_ctx {})", self.pos, 1, self.max_ctx));
        }
        Ok(())
    }
}
