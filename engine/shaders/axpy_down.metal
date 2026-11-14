// axpy_down.metal - streamdense shared MSL (Rust engine + Swift KernelLab).
// Sparse down: decode7b axpy_rows / axpy_rows_k over neuron-major W_down^T + reduce_res.
#include <metal_stdlib>
using namespace metal;

kernel void axpy_rows(device const uint* W [[buffer(0)]], device const half* S [[buffer(1)]],
                      device const half* B [[buffer(2)]], device const uint* idx [[buffer(3)]],
                      device const uint* count [[buffer(4)]], device const half* coef [[buffer(5)]],
                      device float* partial [[buffer(6)]], constant uint& C [[buffer(7)]],
                      constant uint& splits [[buffer(8)]], uint2 gid [[thread_position_in_grid]]) {
    uint w = gid.x, s = gid.y, C8 = C / 8, G = C / 64;
    if (w >= C8) return;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint n = count[0];
    for (uint t = s; t < n; t += splits) {
        uint row = idx[t];
        float c = float(coef[t]);
        uint q = W[row * C8 + w];
        float sc = float(S[row * G + w / 8]) * c, bi = float(B[row * G + w / 8]) * c;
        for (uint k = 0; k < 8; k++) acc[k] += sc * float((q >> (4 * k)) & 15u) + bi;
    }
    for (uint k = 0; k < 8; k++) partial[s * C + w * 8 + k] = acc[k];
}
template <uint K>
kernel void axpy_rows_k(device const uint* W [[buffer(0)]], device const half* S [[buffer(1)]],
                        device const half* B [[buffer(2)]], device const uint* idx [[buffer(3)]],
                        device const uint* count [[buffer(4)]], device const half* coef [[buffer(5)]],
                        device float* partial [[buffer(6)]], constant uint& C [[buffer(7)]],
                        constant uint& splits [[buffer(8)]], constant uint& R [[buffer(9)]], uint2 gid [[thread_position_in_grid]]) {
    uint w = gid.x, s = gid.y, C8 = C / 8, G = C / 64;
    if (w >= C8) return;
    float acc[K][8];
    for (uint k = 0; k < K; k++) for (uint e = 0; e < 8; e++) acc[k][e] = 0;
    uint n = count[0];
    for (uint t = s; t < n; t += splits) {
        uint row = idx[t];
        uint q = W[row * C8 + w];
        float sc = float(S[row * G + w / 8]), bi = float(B[row * G + w / 8]);
        float wv[8];
        for (uint e = 0; e < 8; e++) wv[e] = sc * float((q >> (4 * e)) & 15u) + bi;
        for (uint k = 0; k < K; k++) {
            float c = float(coef[k * R + t]);
            for (uint e = 0; e < 8; e++) acc[k][e] += c * wv[e];
        }
    }
    for (uint k = 0; k < K; k++) for (uint e = 0; e < 8; e++) partial[s * (K * C) + k * C + w * 8 + e] = acc[k][e];
}
#define KTOKD(K_) template [[host_name("axpy_rows_k" #K_)]] kernel void axpy_rows_k<K_>(device const uint*, device const half*, device const half*, device const uint*, device const uint*, device const half*, device float*, constant uint&, constant uint&, constant uint&, uint2);
KTOKD(2) KTOKD(3) KTOKD(4)
// partial [splits][C] -> out = res + half(sum)
kernel void reduce_res(device const float* partial [[buffer(0)]], device const float* res [[buffer(1)]],
                       device float* out [[buffer(2)]], constant uint& C [[buffer(3)]], constant uint& splits [[buffer(4)]],
                       uint j [[thread_position_in_grid]]) {
    if (j >= C) return;
    float s = 0;
    for (uint k = 0; k < splits; k++) s += partial[k * C + j];
    out[j] = res[j] + float(half(s));
}
