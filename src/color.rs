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

/// 1 KiB branchless clamp LUT covering `-512..=511 -> 0..=255`.
///
/// For any IDCT output `(Y, Cr, Cb)` in `-128..=127`, `Y + chroma_offset + 128`
/// lies in `-227..=480`, strictly inside `-512..=511`.
static CLAMP_U8_LUT: [u8; 1024] = {
    let mut t = [0u8; 1024];
    let mut i = 0;
    while i < 1024 {
        let v = (i as i32) - 512;
        t[i] = if v < 0 {
            0
        } else if v > 255 {
            255
        } else {
            v as u8
        };
        i += 1;
    }
    t
};

/// 1 KiB branchless clamp + hardware 5-bit rounding LUT (`((u + 4) >> 3).min(31)`)
/// covering `-512..=511 -> 0..=31` for unsigned RGB15 output.
static CLAMP_U5_LUT: [u8; 1024] = {
    let mut t = [0u8; 1024];
    let mut i = 0;
    while i < 1024 {
        let u = CLAMP_U8_LUT[i] as u16;
        let q = (u + 4) >> 3;
        t[i] = if q > 31 { 31 } else { q as u8 };
        i += 1;
    }
    t
};

/// Branchless clamp of `v` (assumed in `-512..=511`) to `0..=255` (or signed
/// two's complement via `^ 0x80` when `SIGNED` is `true`).
#[inline(always)]
fn clamp_u8_lut<const SIGNED: bool>(v: i16) -> u8 {
    let u = CLAMP_U8_LUT[((i32::from(v) + 512) as usize) & 1023];
    if SIGNED {
        u ^ 0x80
    } else {
        u
    }
}

/// Branchless clamp + 5-bit rounding of `v` (assumed in `-512..=511`).
#[inline(always)]
fn clamp_u5_lut<const SIGNED: bool>(v: i16) -> u16 {
    if SIGNED {
        let u = CLAMP_U8_LUT[((i32::from(v) + 512) as usize) & 1023] ^ 0x80;
        ((u16::from(u) + 4) >> 3).min(31)
    } else {
        u16::from(CLAMP_U5_LUT[((i32::from(v) + 512) as usize) & 1023])
    }
}

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

/// Computes 4 chroma `(R, G, B)` offsets biased by `+128` and duplicates each
/// pair to 8 horizontal lanes.
#[inline(always)]
fn chroma_offsets_8lanes(cr_row: &[i16], cb_row: &[i16]) -> ([i16; 8], [i16; 8], [i16; 8]) {
    let mut r_off = [0i16; 8];
    let mut g_off = [0i16; 8];
    let mut b_off = [0i16; 8];
    for qx in 0..4 {
        let crv = i32::from(cr_row[qx]);
        let cbv = i32::from(cb_row[qx]);
        let ro = ((359 * crv) >> 8) as i16 + 128;
        let go = ((-88 * cbv - 183 * crv) >> 8) as i16 + 128;
        let bo = ((454 * cbv) >> 8) as i16 + 128;
        r_off[qx * 2] = ro;
        r_off[qx * 2 + 1] = ro;
        g_off[qx * 2] = go;
        g_off[qx * 2 + 1] = go;
        b_off[qx * 2] = bo;
        b_off[qx * 2 + 1] = bo;
    }
    (r_off, g_off, b_off)
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
    for (i, slot) in out.iter_mut().enumerate() {
        let lo = y_to_gray4_nibble(y[i * 2], signed);
        let hi = y_to_gray4_nibble(y[i * 2 + 1], signed);
        *slot = (hi << 4) | lo;
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
        for (col, slot) in dst.iter_mut().enumerate() {
            let lo = y_to_gray4_nibble(y_row[col * 2], signed);
            let hi = y_to_gray4_nibble(y_row[col * 2 + 1], signed);
            *slot = (hi << 4) | lo;
        }
    }
}

/// Converts an 8×8 monochrome block into 64 8-bit grayscale bytes.
pub fn convert_mono_8bit(y: &[i16; 64], signed: bool, out: &mut [u8; 64]) {
    for (slot, &yv) in out.iter_mut().zip(y.iter()) {
        *slot = y_to_gray8(yv, signed);
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
        for (slot, &yv) in dst.iter_mut().zip(y_row.iter()) {
            *slot = y_to_gray8(yv, signed);
        }
    }
}

#[inline]
fn convert_subblock_rgb15_impl<const SIGNED: bool>(
    y_blk: &[i16; 64],
    cr: &[i16; 64],
    cb: &[i16; 64],
    quad_x: usize,
    quad_y: usize,
    stp: bool,
    out: &mut [u16; 64],
) {
    let stp_bit: u16 = if stp { 0x8000 } else { 0 };
    for qy in 0..4 {
        let cy = quad_y + qy;
        let cr_row = &cr[cy * 8 + quad_x..cy * 8 + quad_x + 4];
        let cb_row = &cb[cy * 8 + quad_x..cy * 8 + quad_x + 4];
        let (r_off, g_off, b_off) = chroma_offsets_8lanes(cr_row, cb_row);

        for dr in 0..2 {
            let row = qy * 2 + dr;
            let y_row = &y_blk[row * 8..row * 8 + 8];
            let dst = &mut out[row * 8..row * 8 + 8];
            for i in 0..8 {
                let yv = y_row[i];
                let r5 = clamp_u5_lut::<SIGNED>(yv + r_off[i]);
                let g5 = clamp_u5_lut::<SIGNED>(yv + g_off[i]);
                let b5 = clamp_u5_lut::<SIGNED>(yv + b_off[i]);
                dst[i] = stp_bit | (b5 << 10) | (g5 << 5) | r5;
            }
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
    if signed {
        convert_subblock_rgb15_impl::<true>(y_blk, cr, cb, quad_x, quad_y, stp, out);
    } else {
        convert_subblock_rgb15_impl::<false>(y_blk, cr, cb, quad_x, quad_y, stp, out);
    }
}

#[inline]
fn convert_subblock_rgb24_impl<const SIGNED: bool>(
    y_blk: &[i16; 64],
    cr: &[i16; 64],
    cb: &[i16; 64],
    quad_x: usize,
    quad_y: usize,
    out: &mut [u8; 192],
) {
    for qy in 0..4 {
        let cy = quad_y + qy;
        let cr_row = &cr[cy * 8 + quad_x..cy * 8 + quad_x + 4];
        let cb_row = &cb[cy * 8 + quad_x..cy * 8 + quad_x + 4];
        let (r_off, g_off, b_off) = chroma_offsets_8lanes(cr_row, cb_row);

        for dr in 0..2 {
            let row = qy * 2 + dr;
            let y_row = &y_blk[row * 8..row * 8 + 8];
            let dst = &mut out[row * 24..row * 24 + 24];
            for i in 0..8 {
                let yv = y_row[i];
                dst[i * 3] = clamp_u8_lut::<SIGNED>(yv + r_off[i]);
                dst[i * 3 + 1] = clamp_u8_lut::<SIGNED>(yv + g_off[i]);
                dst[i * 3 + 2] = clamp_u8_lut::<SIGNED>(yv + b_off[i]);
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
    if signed {
        convert_subblock_rgb24_impl::<true>(y_blk, cr, cb, quad_x, quad_y, out);
    } else {
        convert_subblock_rgb24_impl::<false>(y_blk, cr, cb, quad_x, quad_y, out);
    }
}

#[inline]
fn convert_macroblock_rgb24_strided_impl<const SIGNED: bool>(
    y: &[[i16; 64]; 4],
    cr: &[i16; 64],
    cb: &[i16; 64],
    out: &mut [u8],
    stride: usize,
) {
    for by in 0..2 {
        for bx in 0..2 {
            let y_blk = &y[by * 2 + bx];
            for qy in 0..4 {
                let cy = by * 4 + qy;
                let cr_row = &cr[cy * 8 + bx * 4..cy * 8 + bx * 4 + 4];
                let cb_row = &cb[cy * 8 + bx * 4..cy * 8 + bx * 4 + 4];
                let (r_off, g_off, b_off) = chroma_offsets_8lanes(cr_row, cb_row);

                for dr in 0..2 {
                    let ly = qy * 2 + dr;
                    let y_row = &y_blk[ly * 8..ly * 8 + 8];
                    let mut r8 = [0u8; 8];
                    let mut g8 = [0u8; 8];
                    let mut b8 = [0u8; 8];
                    for i in 0..8 {
                        let yv = y_row[i];
                        r8[i] = clamp_u8_lut::<SIGNED>(yv + r_off[i]);
                        g8[i] = clamp_u8_lut::<SIGNED>(yv + g_off[i]);
                        b8[i] = clamp_u8_lut::<SIGNED>(yv + b_off[i]);
                    }
                    let row = by * 8 + ly;
                    let col_byte = bx * 24;
                    let dst = &mut out[row * stride + col_byte..row * stride + col_byte + 24];
                    for i in 0..8 {
                        dst[i * 3] = r8[i];
                        dst[i * 3 + 1] = g8[i];
                        dst[i * 3 + 2] = b8[i];
                    }
                }
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
    if signed {
        convert_macroblock_rgb24_strided_impl::<true>(y, cr, cb, out, stride);
    } else {
        convert_macroblock_rgb24_strided_impl::<false>(y, cr, cb, out, stride);
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

#[inline]
fn convert_macroblock_rgba32_strided_impl<const SIGNED: bool>(
    y: &[[i16; 64]; 4],
    cr: &[i16; 64],
    cb: &[i16; 64],
    out: &mut [u8],
    stride: usize,
) {
    for by in 0..2 {
        for bx in 0..2 {
            let y_blk = &y[by * 2 + bx];
            for qy in 0..4 {
                let cy = by * 4 + qy;
                let cr_row = &cr[cy * 8 + bx * 4..cy * 8 + bx * 4 + 4];
                let cb_row = &cb[cy * 8 + bx * 4..cy * 8 + bx * 4 + 4];
                let (r_off, g_off, b_off) = chroma_offsets_8lanes(cr_row, cb_row);

                for dr in 0..2 {
                    let ly = qy * 2 + dr;
                    let y_row = &y_blk[ly * 8..ly * 8 + 8];
                    let mut r8 = [0u8; 8];
                    let mut g8 = [0u8; 8];
                    let mut b8 = [0u8; 8];
                    for i in 0..8 {
                        let yv = y_row[i];
                        r8[i] = clamp_u8_lut::<SIGNED>(yv + r_off[i]);
                        g8[i] = clamp_u8_lut::<SIGNED>(yv + g_off[i]);
                        b8[i] = clamp_u8_lut::<SIGNED>(yv + b_off[i]);
                    }
                    let row = by * 8 + ly;
                    let col_byte = bx * 32;
                    let dst = &mut out[row * stride + col_byte..row * stride + col_byte + 32];
                    for i in 0..8 {
                        dst[i * 4] = r8[i];
                        dst[i * 4 + 1] = g8[i];
                        dst[i * 4 + 2] = b8[i];
                        dst[i * 4 + 3] = 0xFF;
                    }
                }
            }
        }
    }
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
    if signed {
        convert_macroblock_rgba32_strided_impl::<true>(y, cr, cb, out, stride);
    } else {
        convert_macroblock_rgba32_strided_impl::<false>(y, cr, cb, out, stride);
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

#[inline]
fn convert_macroblock_rgb15_le_strided_impl<const SIGNED: bool>(
    y: &[[i16; 64]; 4],
    cr: &[i16; 64],
    cb: &[i16; 64],
    stp: bool,
    out: &mut [u8],
    stride: usize,
) {
    let stp_bit: u16 = if stp { 0x8000 } else { 0 };
    for by in 0..2 {
        for bx in 0..2 {
            let y_blk = &y[by * 2 + bx];
            for qy in 0..4 {
                let cy = by * 4 + qy;
                let cr_row = &cr[cy * 8 + bx * 4..cy * 8 + bx * 4 + 4];
                let cb_row = &cb[cy * 8 + bx * 4..cy * 8 + bx * 4 + 4];
                let (r_off, g_off, b_off) = chroma_offsets_8lanes(cr_row, cb_row);

                for dr in 0..2 {
                    let ly = qy * 2 + dr;
                    let y_row = &y_blk[ly * 8..ly * 8 + 8];
                    let mut px = [0u16; 8];
                    for i in 0..8 {
                        let yv = y_row[i];
                        let r5 = clamp_u5_lut::<SIGNED>(yv + r_off[i]);
                        let g5 = clamp_u5_lut::<SIGNED>(yv + g_off[i]);
                        let b5 = clamp_u5_lut::<SIGNED>(yv + b_off[i]);
                        px[i] = stp_bit | (b5 << 10) | (g5 << 5) | r5;
                    }
                    let row = by * 8 + ly;
                    let col_byte = bx * 16;
                    let dst = &mut out[row * stride + col_byte..row * stride + col_byte + 16];
                    for i in 0..8 {
                        let b = px[i].to_le_bytes();
                        dst[i * 2] = b[0];
                        dst[i * 2 + 1] = b[1];
                    }
                }
            }
        }
    }
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
    if signed {
        convert_macroblock_rgb15_le_strided_impl::<true>(y, cr, cb, stp, out, stride);
    } else {
        convert_macroblock_rgb15_le_strided_impl::<false>(y, cr, cb, stp, out, stride);
    }
}

#[inline]
fn convert_macroblock_rgb15_impl<const SIGNED: bool>(
    y: &[[i16; 64]; 4],
    cr: &[i16; 64],
    cb: &[i16; 64],
    stp: bool,
    out: &mut [u16; 256],
) {
    let stp_bit: u16 = if stp { 0x8000 } else { 0 };
    for by in 0..2 {
        for bx in 0..2 {
            let y_blk = &y[by * 2 + bx];
            for qy in 0..4 {
                let cy = by * 4 + qy;
                let cr_row = &cr[cy * 8 + bx * 4..cy * 8 + bx * 4 + 4];
                let cb_row = &cb[cy * 8 + bx * 4..cy * 8 + bx * 4 + 4];
                let (r_off, g_off, b_off) = chroma_offsets_8lanes(cr_row, cb_row);

                for dr in 0..2 {
                    let ly = qy * 2 + dr;
                    let y_row = &y_blk[ly * 8..ly * 8 + 8];
                    let row = by * 8 + ly;
                    let col = bx * 8;
                    let dst = &mut out[row * 16 + col..row * 16 + col + 8];
                    for i in 0..8 {
                        let yv = y_row[i];
                        let r5 = clamp_u5_lut::<SIGNED>(yv + r_off[i]);
                        let g5 = clamp_u5_lut::<SIGNED>(yv + g_off[i]);
                        let b5 = clamp_u5_lut::<SIGNED>(yv + b_off[i]);
                        dst[i] = stp_bit | (b5 << 10) | (g5 << 5) | r5;
                    }
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
    if signed {
        convert_macroblock_rgb15_impl::<true>(y, cr, cb, stp, out);
    } else {
        convert_macroblock_rgb15_impl::<false>(y, cr, cb, stp, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_clamps_match_reference_over_full_ycbcr_range() {
        for v in -300i16..=300 {
            let biased = v + 128;
            assert_eq!(
                clamp_u8_lut::<false>(biased),
                clamp_channel(i32::from(v), false)
            );
            assert_eq!(
                clamp_u8_lut::<true>(biased),
                clamp_channel(i32::from(v), true)
            );

            let u_unsigned = clamp_channel(i32::from(v), false);
            assert_eq!(
                clamp_u5_lut::<false>(biased),
                ((u16::from(u_unsigned) + 4) >> 3).min(31)
            );
            let u_signed = clamp_channel(i32::from(v), true);
            assert_eq!(
                clamp_u5_lut::<true>(biased),
                ((u16::from(u_signed) + 4) >> 3).min(31)
            );
        }
    }

    #[test]
    fn macroblock_converters_match_per_pixel_reference() {
        let mut cr = [0i16; 64];
        let mut cb = [0i16; 64];
        let mut y = [[0i16; 64]; 4];
        let mut state: u32 = 0xCAFE_BABE;
        let mut next_sample = || -> i16 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (((state >> 24) as u8) as i8) as i16
        };
        for i in 0..64 {
            cr[i] = next_sample();
            cb[i] = next_sample();
            for blk in &mut y {
                blk[i] = next_sample();
            }
        }

        for signed in [false, true] {
            let mut rgb24 = [0u8; 768];
            let mut rgba32 = [0u8; 1024];
            let mut rgb15 = [0u16; 256];
            let mut rgb15_le = [0u8; 512];
            convert_macroblock_rgb24(&y, &cr, &cb, signed, &mut rgb24);
            convert_macroblock_rgba32(&y, &cr, &cb, signed, &mut rgba32);
            convert_macroblock_rgb15(&y, &cr, &cb, signed, true, &mut rgb15);
            convert_macroblock_rgb15_le_strided(&y, &cr, &cb, signed, true, &mut rgb15_le, 32);

            for row in 0..16 {
                for col in 0..16 {
                    let by = row / 8;
                    let bx = col / 8;
                    let yv = y[by * 2 + bx][(row % 8) * 8 + (col % 8)];
                    let c_idx = (row / 2) * 8 + (col / 2);
                    let [r, g, b] = ycbcr_to_rgb(yv, cr[c_idx], cb[c_idx], signed);
                    let p = row * 16 + col;
                    assert_eq!(&rgb24[p * 3..p * 3 + 3], &[r, g, b]);
                    assert_eq!(&rgba32[p * 4..p * 4 + 4], &[r, g, b, 0xFF]);
                    let p15 = rgb8_to_rgb15(r, g, b, true);
                    assert_eq!(rgb15[p], p15);
                    assert_eq!(&rgb15_le[p * 2..p * 2 + 2], &p15.to_le_bytes());
                }
            }
        }
    }
}
