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
    // Fast path for DC-only (and all-zero) blocks: bit-identical to the full
    // two-pass transform for any `scale_table`.
    let mut ac_or: i16 = 0;
    for &c in &src[1..64] {
        ac_or |= c;
    }
    if ac_or == 0 {
        let dc = i32::from(src[0]);
        if dc == 0 {
            dst.fill(0);
            return;
        }
        for y in 0..8 {
            let t = i64::from((dc * i32::from(scale_table[y]) + (1 << 12)) >> 13);
            let dst_row = &mut dst[y * 8..y * 8 + 8];
            for x in 0..8 {
                let val = ((t * i64::from(scale_table[x]) + (1 << 18)) >> 19) as i32;
                dst_row[x] = val.clamp(-128, 127) as i16;
            }
        }
        return;
    }

    let mut tmp = [0i32; 64];

    // Pass 1: 1D IDCT along columns (v -> y).
    // Loops are ordered so `scale_table[v * 8 .. v * 8 + 8]` is accessed
    // contiguously and zero input coefficients skip all 8 MACs.
    // Rounding shift by 13 preserves 3 fractional bits (max magnitude ~2^15).
    for x in 0..8 {
        let mut col_acc = [0i32; 8];
        for v in 0..8 {
            let c = i32::from(src[v * 8 + x]);
            if c == 0 {
                continue;
            }
            let scale_row = &scale_table[v * 8..v * 8 + 8];
            for y in 0..8 {
                col_acc[y] += c * i32::from(scale_row[y]);
            }
        }
        for y in 0..8 {
            tmp[y * 8 + x] = (col_acc[y] + (1 << 12)) >> 13;
        }
    }

    // Pass 2: 1D IDCT along rows (u -> x).
    // Accumulates in `i64` (products can reach ~2^33) and rounds with shift 19.
    for y in 0..8 {
        let mut row_acc = [0i64; 8];
        let tmp_row = &tmp[y * 8..y * 8 + 8];
        for u in 0..8 {
            let t = i64::from(tmp_row[u]);
            if t == 0 {
                continue;
            }
            let scale_row = &scale_table[u * 8..u * 8 + 8];
            for x in 0..8 {
                row_acc[x] += t * i64::from(scale_row[x]);
            }
        }
        let dst_row = &mut dst[y * 8..y * 8 + 8];
        for x in 0..8 {
            let val = ((row_acc[x] + (1 << 18)) >> 19) as i32;
            dst_row[x] = val.clamp(-128, 127) as i16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::DEFAULT_SCALE_TABLE;

    fn idct_2d_reference(src: &[i16; 64], scale_table: &[i16; 64], dst: &mut [i16; 64]) {
        let mut tmp = [0i32; 64];
        for x in 0..8 {
            for y in 0..8 {
                let mut sum: i32 = 0;
                for v in 0..8 {
                    sum += i32::from(src[v * 8 + x]) * i32::from(scale_table[v * 8 + y]);
                }
                tmp[y * 8 + x] = (sum + (1 << 12)) >> 13;
            }
        }
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

    #[test]
    fn matches_reference_on_sparse_dense_and_custom_scale_tables() {
        let mut custom_scale = [0i16; 64];
        let mut state: u32 = 0x1234_5678;
        let mut next_i16 = || -> i16 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as i16
        };
        for slot in &mut custom_scale {
            *slot = next_i16();
        }

        // Test all-zero, DC-only, sparse, and dense blocks against both scale tables.
        for case in 0..40 {
            let mut src = [0i16; 64];
            match case {
                0 => {}
                1..=10 => {
                    src[0] = (next_i16() % 1024).clamp(-1024, 1023);
                }
                11..=25 => {
                    for _ in 0..6 {
                        let idx = (next_i16().unsigned_abs() as usize) & 63;
                        src[idx] = (next_i16() % 1024).clamp(-1024, 1023);
                    }
                }
                _ => {
                    for slot in &mut src {
                        *slot = (next_i16() % 1024).clamp(-1024, 1023);
                    }
                }
            }

            for scale in [&DEFAULT_SCALE_TABLE, &custom_scale] {
                let mut got = [0i16; 64];
                let mut expected = [0i16; 64];
                idct_2d(&src, scale, &mut got);
                idct_2d_reference(&src, scale, &mut expected);
                assert_eq!(got, expected, "mismatch on case {case}");
            }
        }
    }
}
