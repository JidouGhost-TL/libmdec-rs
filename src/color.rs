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

/// Converts a single `(Y, Cr, Cb)` pixel (each in `-128..=127`) to `[R, G, B]`.
///
/// Uses the PS1 MDEC fixed-point coefficients:
/// - `R = Y + ((359 * Cr) >> 8)` (`1.40234375 * Cr`)
/// - `G = Y + ((-88 * Cb - 183 * Cr) >> 8)` (`-0.34375 * Cb - 0.71484375 * Cr`)
/// - `B = Y + ((454 * Cb) >> 8)` (`1.7734375 * Cb`)
#[inline]
pub const fn ycbcr_to_rgb(y: i16, cr: i16, cb: i16, signed: bool) -> [u8; 3] {
    let y = y as i32;
    let cr = cr as i32;
    let cb = cb as i32;

    let r = y + ((359 * cr) >> 8);
    let g = y + ((-88 * cb - 183 * cr) >> 8);
    let b = y + ((454 * cb) >> 8);

    [
        clamp_channel(r, signed),
        clamp_channel(g, signed),
        clamp_channel(b, signed),
    ]
}

/// Packs 8-bit `[R, G, B]` channels into PS1 15-bit `RGB555` (`0bbbbbgggggrrrrr`),
/// rounding to the nearest 5-bit level (matching retail PS1 hardware) and
/// setting bit 15 when `stp` is `true`.
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
#[inline]
pub const fn y_to_gray8(y: i16, signed: bool) -> u8 {
    clamp_channel(y as i32, signed)
}

/// Converts a single 4-bit monochrome sample (`-128..=127`) to a 4-bit nibble (`0..=15`).
///
/// Retail PS1 hardware rounds to the nearest 4-bit level (`(u + 8) >> 4`, clamped to 15)
/// in unsigned mode, or arithmetic-shifts in signed mode.
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

/// Converts an 8×8 monochrome block into 64 8-bit grayscale bytes.
pub fn convert_mono_8bit(y: &[i16; 64], signed: bool, out: &mut [u8; 64]) {
    for i in 0..64 {
        out[i] = y_to_gray8(y[i], signed);
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
    for row in 0..8 {
        let cy = quad_y + (row >> 1);
        for col in 0..8 {
            let cx = quad_x + (col >> 1);
            let y_val = y_blk[row * 8 + col];
            let cr_val = cr[cy * 8 + cx];
            let cb_val = cb[cy * 8 + cx];
            let [r, g, b] = ycbcr_to_rgb(y_val, cr_val, cb_val, signed);
            out[row * 8 + col] = rgb8_to_rgb15(r, g, b, stp);
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
    for row in 0..8 {
        let cy = quad_y + (row >> 1);
        for col in 0..8 {
            let cx = quad_x + (col >> 1);
            let y_val = y_blk[row * 8 + col];
            let cr_val = cr[cy * 8 + cx];
            let cb_val = cb[cy * 8 + cx];
            let [r, g, b] = ycbcr_to_rgb(y_val, cr_val, cb_val, signed);
            let off = (row * 8 + col) * 3;
            out[off] = r;
            out[off + 1] = g;
            out[off + 2] = b;
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
    for row in 0..16 {
        let by = row >> 3;
        let ly = row & 7;
        let cy = row >> 1;
        for col in 0..16 {
            let bx = col >> 3;
            let lx = col & 7;
            let cx = col >> 1;
            let y_val = y[by * 2 + bx][ly * 8 + lx];
            let [r, g, b] = ycbcr_to_rgb(y_val, cr[cy * 8 + cx], cb[cy * 8 + cx], signed);
            let off = (row * 16 + col) * 3;
            out[off] = r;
            out[off + 1] = g;
            out[off + 2] = b;
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
    for row in 0..16 {
        let by = row >> 3;
        let ly = row & 7;
        let cy = row >> 1;
        for col in 0..16 {
            let bx = col >> 3;
            let lx = col & 7;
            let cx = col >> 1;
            let y_val = y[by * 2 + bx][ly * 8 + lx];
            let [r, g, b] = ycbcr_to_rgb(y_val, cr[cy * 8 + cx], cb[cy * 8 + cx], signed);
            let off = (row * 16 + col) * 4;
            out[off] = r;
            out[off + 1] = g;
            out[off + 2] = b;
            out[off + 3] = 0xFF;
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
    for row in 0..16 {
        let by = row >> 3;
        let ly = row & 7;
        let cy = row >> 1;
        for col in 0..16 {
            let bx = col >> 3;
            let lx = col & 7;
            let cx = col >> 1;
            let y_val = y[by * 2 + bx][ly * 8 + lx];
            let [r, g, b] = ycbcr_to_rgb(y_val, cr[cy * 8 + cx], cb[cy * 8 + cx], signed);
            out[row * 16 + col] = rgb8_to_rgb15(r, g, b, stp);
        }
    }
}
