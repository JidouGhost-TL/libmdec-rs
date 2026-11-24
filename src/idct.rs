//! Deterministic fixed-point 2D IDCT over a caller-supplied 64×`i16` scale table.
//!
//! Uses only integer arithmetic (zero `f64`, zero heap allocations) and supports
//! arbitrary custom scale tables loaded via `MDEC(3)`. When using the standard
//! PS1 [`DEFAULT_SCALE_TABLE`], dispatches to a 100% bit-exact 22-multiply
//! butterfly 1D IDCT with an overflow-checked `i32` Pass 2 fast path.

use crate::tables::DEFAULT_SCALE_TABLE;

/// Returns `true` if `table` is element-wise identical to [`DEFAULT_SCALE_TABLE`].
#[inline]
pub(crate) const fn is_default_scale_table(table: &[i16; 64]) -> bool {
    let mut i = 0;
    while i < 64 {
        if table[i] != DEFAULT_SCALE_TABLE[i] {
            return false;
        }
        i += 1;
    }
    true
}

#[inline(always)]
fn clamp_u8_branchless(v: i32) -> u8 {
    let non_neg = v & !(v >> 31);
    let overflow = (255 - non_neg) >> 31;
    (non_neg | overflow) as u8
}

#[inline(always)]
fn clamp_i8_branchless(v: i32) -> i16 {
    i16::from(clamp_u8_branchless(v + 128)) - 128
}

/// 100% bit-exact 22-multiply butterfly 1D IDCT for [`DEFAULT_SCALE_TABLE`],
/// exploiting even symmetry and one's-complement odd symmetry
/// (`S[v, 7-k] == -S[v, k] - 1`) across 8 columns simultaneously.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn idct_1d_butterfly_i32_8col(
    c0: [i32; 8],
    c1: [i32; 8],
    c2: [i32; 8],
    c3: [i32; 8],
    c4: [i32; 8],
    c5: [i32; 8],
    c6: [i32; 8],
    c7: [i32; 8],
    out: &mut [[i32; 8]; 8],
) {
    for x in 0..8 {
        let v0 = c0[x];
        let v1 = c1[x];
        let v2 = c2[x];
        let v3 = c3[x];
        let v4 = c4[x];
        let v5 = c5[x];
        let v6 = c6[x];
        let v7 = c7[x];

        // Even part (6 muls)
        let ee0 = (v0 + v4) * 23170;
        let ee1 = (v0 - v4) * 23170 - v4;
        let eo0 = v2 * 30273 + v6 * 12539;
        let eo1 = v2 * 12539 - v6 * 30274;
        let c26 = v2 + v6;
        let e0 = ee0 + eo0;
        let e3 = ee0 - eo0 - c26;
        let e1 = ee1 + eo1;
        let e2 = ee1 - eo1 - c26;

        // Odd part (16 muls)
        let o0 = v1 * 32138 + v3 * 27245 + v5 * 18204 + v7 * 6392;
        let o1 = v1 * 27245 - v3 * 6393 - v5 * 32139 - v7 * 18205;
        let o2 = v1 * 18204 - v3 * 32139 + v5 * 6392 + v7 * 27245;
        let o3 = v1 * 6392 - v3 * 18205 + v5 * 27245 - v7 * 32139;
        let o_sum = v1 + v3 + v5 + v7;

        out[0][x] = (e0 + o0 + (1 << 12)) >> 13;
        out[1][x] = (e1 + o1 + (1 << 12)) >> 13;
        out[2][x] = (e2 + o2 + (1 << 12)) >> 13;
        out[3][x] = (e3 + o3 + (1 << 12)) >> 13;
        out[4][x] = (e3 - o3 - o_sum + (1 << 12)) >> 13;
        out[5][x] = (e2 - o2 - o_sum + (1 << 12)) >> 13;
        out[6][x] = (e1 - o1 - o_sum + (1 << 12)) >> 13;
        out[7][x] = (e0 - o0 - o_sum + (1 << 12)) >> 13;
    }
}

/// Fast DC-only 2D IDCT for [`DEFAULT_SCALE_TABLE`].
#[inline]
fn idct_2d_default_dc_only(dc: i16, dst: &mut [i16; 64]) {
    let dc = i32::from(dc);
    if dc == 0 {
        dst.fill(0);
        return;
    }
    let t = i64::from((dc * 23170 + (1 << 12)) >> 13);
    let val = (((t * 23170 + (1 << 18)) >> 19) as i32).clamp(-128, 127) as i16;
    dst.fill(val);
}

/// 100% bit-exact fast 2D IDCT specialized for [`DEFAULT_SCALE_TABLE`],
/// when the caller already knows whether any AC coefficient is non-zero.
#[inline]
pub(crate) fn idct_2d_default_with_ac(src: &[i16; 64], has_ac: bool, dst: &mut [i16; 64]) {
    if !has_ac {
        idct_2d_default_dc_only(src[0], dst);
        return;
    }

    let load_row = |r: usize| -> [i32; 8] {
        let s = &src[r * 8..r * 8 + 8];
        [
            i32::from(s[0]),
            i32::from(s[1]),
            i32::from(s[2]),
            i32::from(s[3]),
            i32::from(s[4]),
            i32::from(s[5]),
            i32::from(s[6]),
            i32::from(s[7]),
        ]
    };

    let mut tmp = [[0i32; 8]; 8];
    idct_1d_butterfly_i32_8col(
        load_row(0),
        load_row(1),
        load_row(2),
        load_row(3),
        load_row(4),
        load_row(5),
        load_row(6),
        load_row(7),
        &mut tmp,
    );

    // Check if all Pass 1 outputs lie in `-8192..=8191`, which guarantees that
    // Pass 2 (`sum(|S|) = 173135`, `8192 * 173135 + 2^18 < 2^31`) fits in `i32`.
    let mut range_check: u32 = 0;
    for row in &tmp {
        for &v in row {
            range_check |= ((v.wrapping_add(8192)) as u32) & !0x3FFF;
        }
    }

    if range_check == 0 {
        // Pass 2 in i32 (transpose to column-major so 8 rows vectorize cleanly).
        let mut t_col = [[0i32; 8]; 8];
        for (u, col) in t_col.iter_mut().enumerate() {
            for y in 0..8 {
                col[y] = tmp[y][u];
            }
        }
        let mut out_col = [[0i16; 8]; 8];
        for y in 0..8 {
            let v0 = t_col[0][y];
            let v1 = t_col[1][y];
            let v2 = t_col[2][y];
            let v3 = t_col[3][y];
            let v4 = t_col[4][y];
            let v5 = t_col[5][y];
            let v6 = t_col[6][y];
            let v7 = t_col[7][y];

            let ee0 = (v0 + v4) * 23170;
            let ee1 = (v0 - v4) * 23170 - v4;
            let eo0 = v2 * 30273 + v6 * 12539;
            let eo1 = v2 * 12539 - v6 * 30274;
            let c26 = v2 + v6;
            let e0 = ee0 + eo0;
            let e3 = ee0 - eo0 - c26;
            let e1 = ee1 + eo1;
            let e2 = ee1 - eo1 - c26;

            let o0 = v1 * 32138 + v3 * 27245 + v5 * 18204 + v7 * 6392;
            let o1 = v1 * 27245 - v3 * 6393 - v5 * 32139 - v7 * 18205;
            let o2 = v1 * 18204 - v3 * 32139 + v5 * 6392 + v7 * 27245;
            let o3 = v1 * 6392 - v3 * 18205 + v5 * 27245 - v7 * 32139;
            let o_sum = v1 + v3 + v5 + v7;

            out_col[0][y] = clamp_i8_branchless((e0 + o0 + (1 << 18)) >> 19);
            out_col[1][y] = clamp_i8_branchless((e1 + o1 + (1 << 18)) >> 19);
            out_col[2][y] = clamp_i8_branchless((e2 + o2 + (1 << 18)) >> 19);
            out_col[3][y] = clamp_i8_branchless((e3 + o3 + (1 << 18)) >> 19);
            out_col[4][y] = clamp_i8_branchless((e3 - o3 - o_sum + (1 << 18)) >> 19);
            out_col[5][y] = clamp_i8_branchless((e2 - o2 - o_sum + (1 << 18)) >> 19);
            out_col[6][y] = clamp_i8_branchless((e1 - o1 - o_sum + (1 << 18)) >> 19);
            out_col[7][y] = clamp_i8_branchless((e0 - o0 - o_sum + (1 << 18)) >> 19);
        }
        for y in 0..8 {
            let d = &mut dst[y * 8..y * 8 + 8];
            for x in 0..8 {
                d[x] = out_col[x][y];
            }
        }
    } else {
        idct_2d_default_pass2_i64(&tmp, dst);
    }
}

#[cold]
fn idct_2d_default_pass2_i64(tmp: &[[i32; 8]; 8], dst: &mut [i16; 64]) {
    for y in 0..8 {
        let v0 = i64::from(tmp[y][0]);
        let v1 = i64::from(tmp[y][1]);
        let v2 = i64::from(tmp[y][2]);
        let v3 = i64::from(tmp[y][3]);
        let v4 = i64::from(tmp[y][4]);
        let v5 = i64::from(tmp[y][5]);
        let v6 = i64::from(tmp[y][6]);
        let v7 = i64::from(tmp[y][7]);

        let ee0 = (v0 + v4) * 23170;
        let ee1 = (v0 - v4) * 23170 - v4;
        let eo0 = v2 * 30273 + v6 * 12539;
        let eo1 = v2 * 12539 - v6 * 30274;
        let c26 = v2 + v6;
        let e0 = ee0 + eo0;
        let e3 = ee0 - eo0 - c26;
        let e1 = ee1 + eo1;
        let e2 = ee1 - eo1 - c26;

        let o0 = v1 * 32138 + v3 * 27245 + v5 * 18204 + v7 * 6392;
        let o1 = v1 * 27245 - v3 * 6393 - v5 * 32139 - v7 * 18205;
        let o2 = v1 * 18204 - v3 * 32139 + v5 * 6392 + v7 * 27245;
        let o3 = v1 * 6392 - v3 * 18205 + v5 * 27245 - v7 * 32139;
        let o_sum = v1 + v3 + v5 + v7;

        let d = &mut dst[y * 8..y * 8 + 8];
        d[0] = (((e0 + o0 + (1 << 18)) >> 19) as i32).clamp(-128, 127) as i16;
        d[1] = (((e1 + o1 + (1 << 18)) >> 19) as i32).clamp(-128, 127) as i16;
        d[2] = (((e2 + o2 + (1 << 18)) >> 19) as i32).clamp(-128, 127) as i16;
        d[3] = (((e3 + o3 + (1 << 18)) >> 19) as i32).clamp(-128, 127) as i16;
        d[4] = (((e3 - o3 - o_sum + (1 << 18)) >> 19) as i32).clamp(-128, 127) as i16;
        d[5] = (((e2 - o2 - o_sum + (1 << 18)) >> 19) as i32).clamp(-128, 127) as i16;
        d[6] = (((e1 - o1 - o_sum + (1 << 18)) >> 19) as i32).clamp(-128, 127) as i16;
        d[7] = (((e0 - o0 - o_sum + (1 << 18)) >> 19) as i32).clamp(-128, 127) as i16;
    }
}

/// 100% bit-exact fast 2D IDCT specialized for [`DEFAULT_SCALE_TABLE`].
#[inline]
pub(crate) fn idct_2d_default(src: &[i16; 64], dst: &mut [i16; 64]) {
    let mut ac_or: i16 = 0;
    for &c in &src[1..64] {
        ac_or |= c;
    }
    idct_2d_default_with_ac(src, ac_or != 0, dst);
}

/// General separable 2D IDCT for arbitrary custom `MDEC(3)` scale tables.
pub(crate) fn idct_2d_general_with_ac(
    src: &[i16; 64],
    scale_table: &[i16; 64],
    has_ac: bool,
    dst: &mut [i16; 64],
) {
    if !has_ac {
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

/// Performs a separable 2D IDCT on an 8×8 block of dequantized DCT coefficients
/// (`src`, in natural row-major order) using `scale_table` (`scale_table[u * 8 + x]`),
/// writing clamped `-128..=127` spatial samples into `dst`.
///
/// Zero heap allocations.
pub fn idct_2d(src: &[i16; 64], scale_table: &[i16; 64], dst: &mut [i16; 64]) {
    if scale_table == &DEFAULT_SCALE_TABLE {
        idct_2d_default(src, dst);
        return;
    }

    let mut ac_or: i16 = 0;
    for &c in &src[1..64] {
        ac_or |= c;
    }
    idct_2d_general_with_ac(src, scale_table, ac_or != 0, dst);
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn matches_reference_on_sparse_dense_random_and_extreme_blocks() {
        let mut custom_scale = [0i16; 64];
        let mut state: u32 = 0x1234_5678;
        let mut next_i16 = || -> i16 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as i16
        };
        for slot in &mut custom_scale {
            *slot = next_i16();
        }

        // Test all-zero, DC-only, sparse, dense, random, and extreme -1024..=1023 blocks
        // (including blocks that trigger the i64 Pass 2 fallback!).
        for case in 0..200 {
            let mut src = [0i16; 64];
            match case {
                0 => {}
                1..=10 => {
                    src[0] = (next_i16() % 1024).clamp(-1024, 1023);
                }
                11..=40 => {
                    for _ in 0..6 {
                        let idx = (next_i16().unsigned_abs() as usize) & 63;
                        src[idx] = (next_i16() % 1024).clamp(-1024, 1023);
                    }
                }
                41..=120 => {
                    for slot in &mut src {
                        *slot = (next_i16() % 1024).clamp(-1024, 1023);
                    }
                }
                121..=160 => {
                    // Extreme full-amplitude -1024/+1023 blocks that push tmp > 8191
                    // to exercise the i64 Pass 2 fallback path.
                    for slot in &mut src {
                        *slot = if (next_i16() & 1) == 0 { 1023 } else { -1024 };
                    }
                }
                _ => {
                    // Worst-case sign-aligned with DEFAULT_SCALE_TABLE column 0
                    let sign = if case % 2 == 0 { 1 } else { -1 };
                    for v in 0..8 {
                        for x in 0..8 {
                            let s = DEFAULT_SCALE_TABLE[v * 8];
                            src[v * 8 + x] = if (s >= 0) == (sign > 0) { 1023 } else { -1024 };
                        }
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
