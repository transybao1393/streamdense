//! IEEE fp16 <-> fp32 (round to nearest, ties to even), as Swift Float16(Float) / Float(Float16).
pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 { s } else {
            // subnormal: value = m * 2^-24
            let f = m as f32 * (1.0 / 16777216.0);
            return if s != 0 { -f } else { f };
        }
    } else if e == 31 { s | 0x7f80_0000 | (m << 13) } else { s | ((e + 112) << 23) | (m << 13) };
    f32::from_bits(bits)
}
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 255 { return s | 0x7c00 | if m != 0 { 0x200 } else { 0 }; }
    let ue = e - 127 + 15;
    if ue >= 31 { return s | 0x7c00; }
    if ue <= 0 {
        if ue < -10 { return s; }
        let mm = m | 0x80_0000;
        let shift = (14 - ue) as u32;
        let mut r = mm >> shift;
        let rem = mm & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if rem > half || (rem == half && (r & 1) == 1) { r += 1; }
        return s | r as u16;
    }
    let mut r = ((ue as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) { r += 1; }
    s | r as u16
}
/// bf16 -> fp32 (exact: bf16 is the top 16 bits of an fp32).
pub fn bf16_to_f32(h: u16) -> f32 { f32::from_bits((h as u32) << 16) }
/// fp32 -> bf16, round to nearest, ties to even (as MLX / Metal `bfloat`). NaN stays NaN (sign kept, quiet bit set);
/// values past the bf16 max round to +-inf.
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() { return ((b >> 16) as u16) | 0x0040; }
    let round = 0x7fff + ((b >> 16) & 1);
    (b.wrapping_add(round) >> 16) as u16
}
#[cfg(test)]
mod t {
    use super::*;
    #[test]
    fn roundtrip_all() {
        for h in 0..=0xffffu16 {
            let f = f16_to_f32(h);
            if f.is_nan() { continue; }
            assert_eq!(f32_to_f16(f), h, "h {:#x}", h);
        }
        // ties to even and rounding: 1 + 2^-11 (tie) -> 1.0 ; 1 + 3*2^-11 -> 1 + 2^-9
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 / 2048.0), 0x3c02);
        assert_eq!(f16_to_f32(f32_to_f16(0.01)), 0.01000213623046875);
        assert_eq!(f32_to_f16(65520.0), 0x7c00);
        assert_eq!(f32_to_f16(5.9604645e-8), 0x0001);
    }
}
