//! Deterministic fixed-point 2D IDCT over a caller-supplied 64×`i16` scale table.
//!
//! Uses only integer arithmetic (zero `f64`, zero heap allocations) and supports
//! arbitrary custom scale tables loaded via `MDEC(3)`.

/// Performs a separable 2D IDCT on an 8×8 block of dequantized DCT coefficients
/// (`src`, in natural row-major order) using `scale_table` (`scale_table[u * 8 + x]`),
/// writing clamped `-128..=127` spatial samples into `dst`.
///
/// Zero heap allocations.
pub fn idct_2d(src: &[i16; 64], scale_table: &[i16; 64], dst: &mut [i16; 64]) {
    let mut tmp = [0i32; 64];

    // Pass 1: 1D IDCT along columns (for each column x, transform v -> y).
    for x in 0..8 {
        for y in 0..8 {
            let mut sum: i32 = 0;
            for v in 0..8 {
                sum += i32::from(src[v * 8 + x]) * i32::from(scale_table[v * 8 + y]);
            }
            // Keep 3 fractional bits (shift right by 13 with rounding? Let's use
            // standard PS1 16-bit intermediate: (sum + 0x1000) >> 13 or >> 16).
            // Wait: 2^13 * 2^19 = 2^32; let's check max magnitude:
            // |src| <= 1024, |scale| <= 32768 -> 8 * 1024 * 32768 = 2^28, fits in i32!
            // Shifting by 13 gives max ~32768, then Pass 2 max = 8 * 32768 * 32768 = 2^33 (use i64).
            tmp[y * 8 + x] = (sum + (1 << 12)) >> 13;
        }
    }

    // Pass 2: 1D IDCT along rows (for each row y, transform u -> x).
    for y in 0..8 {
        for x in 0..8 {
            let mut sum: i64 = 0;
            for u in 0..8 {
                sum += i64::from(tmp[y * 8 + u]) * i64::from(scale_table[u * 8 + x]);
            }
            let val = ((sum + (1 << 18)) >> 19) as i32;
            dst[y * 8 + x] = val.clamp(-128, 127) as i16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::DEFAULT_SCALE_TABLE;

    #[test]
    fn dc_only_block_produces_flat_output() {
        let mut src = [0i16; 64];
        src[0] = 512; // 512 / 8 = 64
        let mut dst = [0i16; 64];
        idct_2d(&src, &DEFAULT_SCALE_TABLE, &mut dst);
        for &v in &dst {
            assert_eq!(v, 64);
        }
    }

    #[test]
    fn clamps_extremes_to_i8_range() {
        let mut src = [0i16; 64];
        src[0] = 1023;
        let mut dst = [0i16; 64];
        idct_2d(&src, &DEFAULT_SCALE_TABLE, &mut dst);
        assert!(dst.iter().all(|&v| v == 127));

        src[0] = -1024;
        idct_2d(&src, &DEFAULT_SCALE_TABLE, &mut dst);
        assert!(dst.iter().all(|&v| v == -128));
    }
}
