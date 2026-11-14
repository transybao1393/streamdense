// bench_bw.metal - streamdense bench-only kernel (decode_bench): pure streaming read of a buffer, the practical DRAM
// read ceiling of this GPU. Not part of the engine library (QWEN_SHADERS); loaded by the bench only.
// bw_read: n uint4 words at src [0], each thread reads words i, i + grid, ... and xors them; out [1] gets the xor only
// when it equals `never` [3] (practically never), so the loads are not dead code.
kernel void bw_read(device const uint4* src [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& n [[buffer(2)]],
                    constant uint& never [[buffer(3)]], uint i [[thread_position_in_grid]], uint grid [[threads_per_grid]]) {
    uint4 a = uint4(0);
    for (uint j = i; j < n; j += grid) a ^= src[j];
    uint v = a.x ^ a.y ^ a.z ^ a.w;
    if (v == never) out[0] = v;
}
