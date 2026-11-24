//! Fixed-point YCbCr-to-RGB and monochrome color conversion.
//!
//! Implements the PS1 MDEC hardware color space conversion in pure integer
//! arithmetic (zero `f64`, zero heap allocations), supporting:
//! - 4-bit monochrome (2 pixels per byte, rounded to nearest nibble)
//! - 8-bit monochrome
//! - 15-bit RGB555 (with configurable STP bit 15)
//! - 24-bit RGB888
//! - 32-bit RGBA8888 (for high-level ergonomic decoding)
//! - Both unsigned (`0..=255`) and signed (`-128..=127`) output modes

/// Clamps an `i32` to the signed 8-bit range `-128..=127`, then converts to `u8`
/// according to `signed` (`false` adds `128` for `0..=255`; `true` keeps two's complement).
#[must_use]
#[inline]
pub const fn clamp_channel(v: i32, signed: bool) -> u8 {
    let c = if v < -128 {
        -128
    } else if v > 127 {
        127
    } else {
        v
    };
    if signed {
        (c as i8) as u8
    } else {
        (c + 128) as u8
    }
}

/// Computes the fixed-point `(R, G, B)` chroma offsets for a `(Cr, Cb)` sample.
#[inline]
const fn chroma_offsets(cr: i16, cb: i16) -> (i32, i32, i32) {
    let cr = cr as i32;
    let cb = cb as i32;
    ((359 * cr) >> 8, (-88 * cb - 183 * cr) >> 8, (454 * cb) >> 8)
}

/// Converts a single `(Y, Cr, Cb)` pixel (each in `-128..=127`) to `[R, G, B]`.
///
/// Uses the PS1 MDEC fixed-point coefficients:
/// - `R = Y + ((359 * Cr) >> 8)` (`1.40234375 * Cr`)
/// - `G = Y + ((-88 * Cb - 183 * Cr) >> 8)` (`-0.34375 * Cb - 0.71484375 * Cr`)
/// - `B = Y + ((454 * Cb) >> 8)` (`1.7734375 * Cb`)
#[must_use]
#[inline]
pub const fn ycbcr_to_rgb(y: i16, cr: i16, cb: i16, signed: bool) -> [u8; 3] {
    let y = y as i32;
    let (r_off, g_off, b_off) = chroma_offsets(cr, cb);

    [
        clamp_channel(y + r_off, signed),
        clamp_channel(y + g_off, signed),
        clamp_channel(y + b_off, signed),
    ]
}

/// Packs 8-bit `[R, G, B]` channels into PS1 15-bit `RGB555` (`0bbbbbgggggrrrrr`),
/// rounding to the nearest 5-bit level (matching retail PS1 hardware) and
/// setting bit 15 when `stp` is `true`.
#[must_use]
#[inline]
pub const fn rgb8_to_rgb15(r: u8, g: u8, b: u8, stp: bool) -> u16 {
    let r5 = (r as u16 + 4) >> 3;
    let g5 = (g as u16 + 4) >> 3;
    let b5 = (b as u16 + 4) >> 3;
    let r5 = if r5 > 31 { 31 } else { r5 };
    let g5 = if g5 > 31 { 31 } else { g5 };
    let b5 = if b5 > 31 { 31 } else { b5 };
    let stp_bit = if stp { 0x8000 } else { 0 };
    stp_bit | (b5 << 10) | (g5 << 5) | r5
}

/// Converts a single 8-bit monochrome sample (`-128..=127`) to `u8`.
#[must_use]
#[inline]
pub const fn y_to_gray8(y: i16, signed: bool) -> u8 {
    clamp_channel(y as i32, signed)
}

/// Converts a single 4-bit monochrome sample (`-128..=127`) to a 4-bit nibble (`0..=15`).
///
/// Retail PS1 hardware rounds to the nearest 4-bit level (`(u + 8) >> 4`, clamped to 15)
/// in unsigned mode, or arithmetic-shifts in signed mode.
#[must_use]
#[inline]
pub const fn y_to_gray4_nibble(y: i16, signed: bool) -> u8 {
    let c = if y < -128 {
        -128
    } else if y > 127 {
        127
    } else {
        y as i32
    };
    if signed {
        // Signed 4-bit two's complement (-8..=7) in low nibble.
        let s = (c + 8) >> 4;
        let clamped = if s < -8 {
            -8
        } else if s > 7 {
            7
        } else {
            s
        };
        (clamped as u8) & 0x0F
    } else {
        let u = c + 128; // 0..=255
        let q = (u + 8) >> 4;
        if q > 15 {
            15
        } else {
            q as u8
        }
    }
}

/// Converts an 8×8 monochrome block into 32 packed 4-bit bytes (2 pixels per byte:
/// low nibble = even column, high nibble = odd column).
pub fn convert_mono_4bit(y: &[i16; 64], signed: bool, out: &mut [u8; 32]) {
    for i in 0..32 {
        let lo = y_to_gray4_nibble(y[i * 2], signed);
        let hi = y_to_gray4_nibble(y[i * 2 + 1], signed);
        out[i] = (hi << 4) | lo;
    }
}

/// Converts an 8×8 monochrome block directly into a strided 4-bit raster buffer
/// (4 bytes per row, `stride` bytes between rows).
#[inline]
pub(crate) fn convert_mono_4bit_strided(
    y: &[i16; 64],
    signed: bool,
    out: &mut [u8],
    stride: usize,
) {
    for row in 0..8 {
        let y_row = &y[row * 8..row * 8 + 8];
        let dst = &mut out[row * stride..row * stride + 4];
        for col in 0..4 {
            let lo = y_to_gray4_nibble(y_row[col * 2], signed);
            let hi = y_to_gray4_nibble(y_row[col * 2 + 1], signed);
            dst[col] = (hi << 4) | lo;
        }
    }
}

/// Converts an 8×8 monochrome block into 64 8-bit grayscale bytes.
pub fn convert_mono_8bit(y: &[i16; 64], signed: bool, out: &mut [u8; 64]) {
    for i in 0..64 {
        out[i] = y_to_gray8(y[i], signed);
    }
}

/// Converts an 8×8 monochrome block directly into a strided 8-bit raster buffer
/// (8 bytes per row, `stride` bytes between rows).
#[inline]
pub(crate) fn convert_mono_8bit_strided(
    y: &[i16; 64],
    signed: bool,
    out: &mut [u8],
    stride: usize,
) {
    for row in 0..8 {
        let y_row = &y[row * 8..row * 8 + 8];
        let dst = &mut out[row * stride..row * stride + 8];
        for col in 0..8 {
            dst[col] = y_to_gray8(y_row[col], signed);
        }
    }
}

/// Converts one 8×8 luminance sub-block (`quad_x` in `{0, 4}`, `quad_y` in `{0, 4}`
/// selecting the 4×4 chroma quadrant of `cr`/`cb`) into 64 15-bit RGB555 pixels.
#[allow(clippy::too_many_arguments)]
pub fn convert_subblock_rgb15(
    y_blk: &[i16; 64],
    cr: &[i16; 64],
    cb: &[i16; 64],
    quad_x: usize,
    quad_y: usize,
    signed: bool,
    stp: bool,
    out: &mut [u16; 64],
) {
    for qy in 0..4 {
        let cy = quad_y + qy;
        let r0 = qy * 2;
        let r1 = r0 + 1;
        for qx in 0..4 {
            let cx = quad_x + qx;
            let (r_off, g_off, b_off) = chroma_offsets(cr[cy * 8 + cx], cb[cy * 8 + cx]);
            let c0 = qx * 2;
            let c1 = c0 + 1;

            for (row, col) in [(r0, c0), (r0, c1), (r1, c0), (r1, c1)] {
                let idx = row * 8 + col;
                let yv = i32::from(y_blk[idx]);
                let r = clamp_channel(yv + r_off, signed);
                let g = clamp_channel(yv + g_off, signed);
                let b = clamp_channel(yv + b_off, signed);
                out[idx] = rgb8_to_rgb15(r, g, b, stp);
            }
        }
    }
}

/// Converts one 8×8 luminance sub-block into 192 packed 24-bit RGB bytes.
pub fn convert_subblock_rgb24(
    y_blk: &[i16; 64],
    cr: &[i16; 64],
    cb: &[i16; 64],
    quad_x: usize,
    quad_y: usize,
    signed: bool,
    out: &mut [u8; 192],
) {
    for qy in 0..4 {
        let cy = quad_y + qy;
        let r0 = qy * 2;
        let r1 = r0 + 1;
        for qx in 0..4 {
            let cx = quad_x + qx;
            let (r_off, g_off, b_off) = chroma_offsets(cr[cy * 8 + cx], cb[cy * 8 + cx]);
            let c0 = qx * 2;
            let c1 = c0 + 1;

            for (row, col) in [(r0, c0), (r0, c1), (r1, c0), (r1, c1)] {
                let idx = row * 8 + col;
                let yv = i32::from(y_blk[idx]);
                let off = idx * 3;
                out[off] = clamp_channel(yv + r_off, signed);
                out[off + 1] = clamp_channel(yv + g_off, signed);
                out[off + 2] = clamp_channel(yv + b_off, signed);
            }
        }
    }
}

/// Converts a full 16×16 macroblock (`y = [Y0, Y1, Y2, Y3]`, `cr`, `cb`) directly
/// into a strided RGB24 raster buffer (`stride` bytes between rows).
#[inline]
pub(crate) fn convert_macroblock_rgb24_strided(
    y: &[[i16; 64]; 4],
    cr: &[i16; 64],
    cb: &[i16; 64],
    signed: bool,
    out: &mut [u8],
    stride: usize,
) {
    for by in 0..2 {
        for bx in 0..2 {
            let y_blk = &y[by * 2 + bx];
            for qy in 0..4 {
                let cy = by * 4 + qy;
                let row0 = by * 8 + qy * 2;
                let row1 = row0 + 1;
                let ly0 = qy * 2;
                let ly1 = ly0 + 1;
                for qx in 0..4 {
                    let cx = bx * 4 + qx;
                    let (r_off, g_off, b_off) = chroma_offsets(cr[cy * 8 + cx], cb[cy * 8 + cx]);
                    let col0 = bx * 8 + qx * 2;
                    let lx0 = qx * 2;

                    let off0 = row0 * stride + col0 * 3;
                    let y00 = i32::from(y_blk[ly0 * 8 + lx0]);
                    let y01 = i32::from(y_blk[ly0 * 8 + lx0 + 1]);
                    out[off0] = clamp_channel(y00 + r_off, signed);
                    out[off0 + 1] = clamp_channel(y00 + g_off, signed);
                    out[off0 + 2] = clamp_channel(y00 + b_off, signed);
                    out[off0 + 3] = clamp_channel(y01 + r_off, signed);
                    out[off0 + 4] = clamp_channel(y01 + g_off, signed);
                    out[off0 + 5] = clamp_channel(y01 + b_off, signed);

                    let off1 = row1 * stride + col0 * 3;
                    let y10 = i32::from(y_blk[ly1 * 8 + lx0]);
                    let y11 = i32::from(y_blk[ly1 * 8 + lx0 + 1]);
                    out[off1] = clamp_channel(y10 + r_off, signed);
                    out[off1 + 1] = clamp_channel(y10 + g_off, signed);
                    out[off1 + 2] = clamp_channel(y10 + b_off, signed);
                    out[off1 + 3] = clamp_channel(y11 + r_off, signed);
                    out[off1 + 4] = clamp_channel(y11 + g_off, signed);
                    out[off1 + 5] = clamp_channel(y11 + b_off, signed);
                }
            }
        }
    }
}

/// Converts a full 16×16 macroblock (`y = [Y0, Y1, Y2, Y3]`, `cr`, `cb`) into
/// raster-ordered 16×16 RGB24 (`768` bytes).
pub fn convert_macroblock_rgb24(
    y: &[[i16; 64]; 4],
    cr: &[i16; 64],
    cb: &[i16; 64],
    signed: bool,
    out: &mut [u8; 768],
) {
    convert_macroblock_rgb24_strided(y, cr, cb, signed, out, 16 * 3);
}

/// Converts a full 16×16 macroblock directly into a strided RGBA32 raster buffer
/// (`stride` bytes between rows, alpha = `0xFF`).
#[inline]
pub(crate) fn convert_macroblock_rgba32_strided(
    y: &[[i16; 64]; 4],
    cr: &[i16; 64],
    cb: &[i16; 64],
    signed: bool,
    out: &mut [u8],
    stride: usize,
) {
    for by in 0..2 {
        for bx in 0..2 {
            let y_blk = &y[by * 2 + bx];
            for qy in 0..4 {
                let cy = by * 4 + qy;
                let row0 = by * 8 + qy * 2;
                let row1 = row0 + 1;
                let ly0 = qy * 2;
                let ly1 = ly0 + 1;
                for qx in 0..4 {
                    let cx = bx * 4 + qx;
                    let (r_off, g_off, b_off) = chroma_offsets(cr[cy * 8 + cx], cb[cy * 8 + cx]);
                    let col0 = bx * 8 + qx * 2;
                    let lx0 = qx * 2;

                    let off0 = row0 * stride + col0 * 4;
                    let y00 = i32::from(y_blk[ly0 * 8 + lx0]);
                    let y01 = i32::from(y_blk[ly0 * 8 + lx0 + 1]);
                    out[off0] = clamp_channel(y00 + r_off, signed);
                    out[off0 + 1] = clamp_channel(y00 + g_off, signed);
                    out[off0 + 2] = clamp_channel(y00 + b_off, signed);
                    out[off0 + 3] = 0xFF;
                    out[off0 + 4] = clamp_channel(y01 + r_off, signed);
                    out[off0 + 5] = clamp_channel(y01 + g_off, signed);
                    out[off0 + 6] = clamp_channel(y01 + b_off, signed);
                    out[off0 + 7] = 0xFF;

                    let off1 = row1 * stride + col0 * 4;
                    let y10 = i32::from(y_blk[ly1 * 8 + lx0]);
                    let y11 = i32::from(y_blk[ly1 * 8 + lx0 + 1]);
                    out[off1] = clamp_channel(y10 + r_off, signed);
                    out[off1 + 1] = clamp_channel(y10 + g_off, signed);
                    out[off1 + 2] = clamp_channel(y10 + b_off, signed);
                    out[off1 + 3] = 0xFF;
                    out[off1 + 4] = clamp_channel(y11 + r_off, signed);
                    out[off1 + 5] = clamp_channel(y11 + g_off, signed);
                    out[off1 + 6] = clamp_channel(y11 + b_off, signed);
                    out[off1 + 7] = 0xFF;
                }
            }
        }
    }
}

/// Converts a full 16×16 macroblock into raster-ordered 16×16 RGBA32 (`1024` bytes,
/// alpha = `0xFF`).
pub fn convert_macroblock_rgba32(
    y: &[[i16; 64]; 4],
    cr: &[i16; 64],
    cb: &[i16; 64],
    signed: bool,
    out: &mut [u8; 1024],
) {
    convert_macroblock_rgba32_strided(y, cr, cb, signed, out, 16 * 4);
}

/// Converts a full 16×16 macroblock directly into a strided little-endian RGB15
/// raster buffer (`stride` bytes between rows).
#[inline]
pub(crate) fn convert_macroblock_rgb15_le_strided(
    y: &[[i16; 64]; 4],
    cr: &[i16; 64],
    cb: &[i16; 64],
    signed: bool,
    stp: bool,
    out: &mut [u8],
    stride: usize,
) {
    for by in 0..2 {
        for bx in 0..2 {
            let y_blk = &y[by * 2 + bx];
            for qy in 0..4 {
                let cy = by * 4 + qy;
                let row0 = by * 8 + qy * 2;
                let row1 = row0 + 1;
                let ly0 = qy * 2;
                let ly1 = ly0 + 1;
                for qx in 0..4 {
                    let cx = bx * 4 + qx;
                    let (r_off, g_off, b_off) = chroma_offsets(cr[cy * 8 + cx], cb[cy * 8 + cx]);
                    let col0 = bx * 8 + qx * 2;
                    let lx0 = qx * 2;

                    let off0 = row0 * stride + col0 * 2;
                    let y00 = i32::from(y_blk[ly0 * 8 + lx0]);
                    let y01 = i32::from(y_blk[ly0 * 8 + lx0 + 1]);
                    let p00 = rgb8_to_rgb15(
                        clamp_channel(y00 + r_off, signed),
                        clamp_channel(y00 + g_off, signed),
                        clamp_channel(y00 + b_off, signed),
                        stp,
                    )
                    .to_le_bytes();
                    let p01 = rgb8_to_rgb15(
                        clamp_channel(y01 + r_off, signed),
                        clamp_channel(y01 + g_off, signed),
                        clamp_channel(y01 + b_off, signed),
                        stp,
                    )
                    .to_le_bytes();
                    out[off0] = p00[0];
                    out[off0 + 1] = p00[1];
                    out[off0 + 2] = p01[0];
                    out[off0 + 3] = p01[1];

                    let off1 = row1 * stride + col0 * 2;
                    let y10 = i32::from(y_blk[ly1 * 8 + lx0]);
                    let y11 = i32::from(y_blk[ly1 * 8 + lx0 + 1]);
                    let p10 = rgb8_to_rgb15(
                        clamp_channel(y10 + r_off, signed),
                        clamp_channel(y10 + g_off, signed),
                        clamp_channel(y10 + b_off, signed),
                        stp,
                    )
                    .to_le_bytes();
                    let p11 = rgb8_to_rgb15(
                        clamp_channel(y11 + r_off, signed),
                        clamp_channel(y11 + g_off, signed),
                        clamp_channel(y11 + b_off, signed),
                        stp,
                    )
                    .to_le_bytes();
                    out[off1] = p10[0];
                    out[off1 + 1] = p10[1];
                    out[off1 + 2] = p11[0];
                    out[off1 + 3] = p11[1];
                }
            }
        }
    }
}

/// Converts a full 16×16 macroblock into raster-ordered 16×16 RGB15 (`256` `u16`s).
pub fn convert_macroblock_rgb15(
    y: &[[i16; 64]; 4],
    cr: &[i16; 64],
    cb: &[i16; 64],
    signed: bool,
    stp: bool,
    out: &mut [u16; 256],
) {
    for by in 0..2 {
        for bx in 0..2 {
            let y_blk = &y[by * 2 + bx];
            for qy in 0..4 {
                let cy = by * 4 + qy;
                let row0 = by * 8 + qy * 2;
                let row1 = row0 + 1;
                let ly0 = qy * 2;
                let ly1 = ly0 + 1;
                for qx in 0..4 {
                    let cx = bx * 4 + qx;
                    let (r_off, g_off, b_off) = chroma_offsets(cr[cy * 8 + cx], cb[cy * 8 + cx]);
                    let col0 = bx * 8 + qx * 2;
                    let lx0 = qx * 2;

                    let y00 = i32::from(y_blk[ly0 * 8 + lx0]);
                    let y01 = i32::from(y_blk[ly0 * 8 + lx0 + 1]);
                    out[row0 * 16 + col0] = rgb8_to_rgb15(
                        clamp_channel(y00 + r_off, signed),
                        clamp_channel(y00 + g_off, signed),
                        clamp_channel(y00 + b_off, signed),
                        stp,
                    );
                    out[row0 * 16 + col0 + 1] = rgb8_to_rgb15(
                        clamp_channel(y01 + r_off, signed),
                        clamp_channel(y01 + g_off, signed),
                        clamp_channel(y01 + b_off, signed),
                        stp,
                    );

                    let y10 = i32::from(y_blk[ly1 * 8 + lx0]);
                    let y11 = i32::from(y_blk[ly1 * 8 + lx0 + 1]);
                    out[row1 * 16 + col0] = rgb8_to_rgb15(
                        clamp_channel(y10 + r_off, signed),
                        clamp_channel(y10 + g_off, signed),
                        clamp_channel(y10 + b_off, signed),
                        stp,
                    );
                    out[row1 * 16 + col0 + 1] = rgb8_to_rgb15(
                        clamp_channel(y11 + r_off, signed),
                        clamp_channel(y11 + g_off, signed),
                        clamp_channel(y11 + b_off, signed),
                        stp,
                    );
                }
            }
        }
    }
}
