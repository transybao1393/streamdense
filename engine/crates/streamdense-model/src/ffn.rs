//! FFN variants behind one trait. SparseFatRelu = the exact-gate sparse FFN (decode7b encFFN k = 1);
//! DenseGated = gate GEMV -> up GEMV * act(gate) -> down GEMV (row-major W_down): act FATReLU (decode7b FFN=dense)
//! or SiLU (SwiGLU models, gemv4 MODE 4).
use streamdense_core::arena::Region;
use streamdense_metal::Enc;

#[derive(Clone, Copy, Debug)]
pub struct Q4 { pub r: usize, pub c: usize, pub w: Region, pub s: Region, pub b: Region }

/// decode7b encGemv4: MODE 0 out = v; 1 out = fatrelu(g) * v; 2 outf = res + half(v); 3 outf = v; 4 out = silu(g) * v
pub fn gemv4(e: &Enc, m: &Q4, x: usize, mode: u32, out: usize, g: usize, res: usize, outf: usize, thr: f32) {
    let name = ["gemv4_m0", "gemv4_m1", "gemv4_m2", "gemv4_m3", "gemv4_m4"][mode as usize];
    e.pso(name).b(0, m.w.off).b(1, m.s.off).b(2, m.b.off).b(3, x).b(4, out).u(5, m.r as u32).u(6, m.c as u32)
        .b(7, g).b(8, res).b(9, outf).f(10, thr);
    e.groups(((m.r + 3) / 4, 1), 64);
}

pub trait Ffn {
    fn kernels(&self) -> Vec<&'static str>;
    fn sparse(&self) -> bool;
    /// x = normed input (half), res = residual in (fp32), out = residual out (fp32)
    fn encode(&self, e: &Enc, l: usize, x: usize, res: usize, out: usize);
}

pub struct SparseFatRelu {
    pub w: Vec<[Q4; 3]>, // gate, up, downT (neuron-major)
    pub g: usize, pub hbuf: usize, pub idx: usize, pub partial: usize, pub counts: usize, pub args: usize,
    pub thr: f32, pub h: usize, pub d: usize, pub ordered: bool, pub dummy: usize,
}
impl Ffn for SparseFatRelu {
    fn kernels(&self) -> Vec<&'static str> {
        vec!["gemv4_m0", "compact_thr", "compact_thr_ordered", "make_args", "gemv_gather_act", "axpy_rows", "reduce_res"]
    }
    fn sparse(&self) -> bool { true }
    fn encode(&self, e: &Enc, l: usize, x: usize, res: usize, out: usize) {
        let [gate, up, down_t] = &self.w[l];
        let cnt = self.counts + l * 4;
        gemv4(e, gate, x, 0, self.g, self.dummy, self.dummy, self.dummy, self.thr);
        if self.ordered {
            e.pso("compact_thr_ordered").b(0, self.g).b(1, self.idx).b(2, cnt).u(3, self.d as u32).f(4, self.thr);
            e.groups((1, 1), 1024);
        } else {
            e.pso("compact_thr").b(0, self.g).b(1, self.idx).b(2, cnt).u(3, self.d as u32).f(4, self.thr);
            e.threads((self.d, 1), 256);
        }
        e.pso("make_args").b(0, cnt).b(1, self.args + l * 12);
        e.threads((1, 1), 1);
        e.pso("gemv_gather_act").b(0, up.w.off).b(1, up.s.off).b(2, up.b.off).b(3, x).b(4, self.idx).b(5, cnt).b(6, self.g)
            .b(7, self.hbuf).u(8, self.h as u32);
        e.indirect(self.args + l * 12, 256);
        e.pso("axpy_rows").b(0, down_t.w.off).b(1, down_t.s.off).b(2, down_t.b.off).b(3, self.idx).b(4, cnt).b(5, self.hbuf)
            .b(6, self.partial).u(7, self.h as u32).u(8, crate::engine::SPLITS as u32);
        e.threads((self.h / 8, crate::engine::SPLITS), 256);
        e.pso("reduce_res").b(0, self.partial).b(1, res).b(2, out).u(3, self.h as u32).u(4, crate::engine::SPLITS as u32);
        e.threads((self.h, 1), 256);
    }
}

pub struct DenseGated { pub w: Vec<[Q4; 3]>, pub g: usize, pub hbuf: usize, pub thr: f32, pub mode: u32, pub dummy: usize }
impl Ffn for DenseGated {
    fn kernels(&self) -> Vec<&'static str> { vec!["gemv4_m0", "gemv4_m1", "gemv4_m2", "gemv4_m4"] }
    fn sparse(&self) -> bool { false }
    fn encode(&self, e: &Enc, l: usize, x: usize, res: usize, out: usize) {
        let [gate, up, down] = &self.w[l];
        gemv4(e, gate, x, 0, self.g, self.dummy, self.dummy, self.dummy, self.thr);
        gemv4(e, up, x, self.mode, self.hbuf, self.g, self.dummy, self.dummy, self.thr);
        gemv4(e, down, self.hbuf, 2, self.dummy, self.dummy, res, out, self.thr);
    }
}
