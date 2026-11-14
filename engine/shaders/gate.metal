// gate.metal - streamdense shared MSL (Rust engine + Swift KernelLab).
// Gate = gemv4_m0 (gemv_q4.metal, tuned r2_d1_s0 tpt64); FATReLU threshold is applied in compact.metal (active iff g > thr) and in sparse_up (relu(g) * up).
#include <metal_stdlib>
using namespace metal;

// (no kernels of its own: kept so the layout in ARCHITECTURE.md section 2 maps 1:1)

