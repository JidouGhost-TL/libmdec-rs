//! Hardware 16-bit RLE decoder and dequantizer.
//!
//! Implements the PS1 MDEC RLE halfword stream format:
//! - Leading `0xFE00` padding halfwords are skipped before each block.
//! - First non-`0xFE00` halfword encodes `(q_scale: u6, dc: i10)`.
//! - Subsequent halfwords encode `(run: u6, ac: i10)` until `0xFE00` (EOB)
//!   or coefficient index `k >= 63`.
//! - When `q_scale == 0`, coefficients are unquantized (`val * 2`) and placed
//!   in linear order `0..64`.
//! - When `q_scale` is `1..=63`, coefficients are dequantized with the zig-zag
//!   quantization table and placed in zig-zag order (`ZIGZAG[k]`).
//! - All dequantized coefficients are clamped to signed 11-bit (`-1024..=1023`).

use crate::tables::ZIGZAG;

/// End-of-block / inter-block padding halfword (`0xFE00`).
pub const EOB_HALFWORD: u16 = 0xFE00;

/// Sign-extends a 10-bit two's-complement integer (`0..=1023`) to `i32` (`-512..=511`).
#[inline]
pub const fn sign_extend_10(raw: u16) -> i32 {
    let v = (raw & 0x3FF) as i32;
    if v >= 512 {
        v - 1024
    } else {
        v
    }
}

/// Clamps a dequantized coefficient to the hardware signed 11-bit range (`-1024..=1023`).
#[inline]
const fn clamp_i11(v: i32) -> i16 {
    if v < -1024 {
        -1024
    } else if v > 1023 {
        1023
    } else {
        v as i16
    }
}

/// Hardware AC dequantization rounding: `(val * qt * q_scale + 4) >> 3`,
/// clamped to signed 11-bit (`-1024..=1023`).
#[inline]
const fn dequant_ac(level: i32, qt: u8, q_scale: i32) -> i16 {
    let prod = level * (qt as i32) * q_scale;
    let val = (prod + 4) >> 3;
    clamp_i11(val)
}

/// Decodes a single 8×8 block from a slice of 16-bit RLE halfwords into `out`
/// (natural row-major order), using `quant_zigzag` (in zig-zag scan order).
///
/// Returns `Some(consumed)` with the number of halfwords consumed from `rle`,
/// or `None` if `rle` contained only `0xFE00` padding (or was empty).
/// Zero heap allocations.
pub fn decode_rle_block(
    rle: &[u16],
    quant_zigzag: &[u8; 64],
    out: &mut [i16; 64],
) -> Option<usize> {
    out.fill(0);

    let mut pos = 0;
    while pos < rle.len() && rle[pos] == EOB_HALFWORD {
        pos += 1;
    }
    if pos >= rle.len() {
        return None;
    }

    let first = rle[pos];
    pos += 1;

    let q_scale = i32::from((first >> 10) & 0x3F);
    let dc_raw = sign_extend_10(first);

    let dc = if q_scale == 0 {
        clamp_i11(dc_raw * 2)
    } else {
        clamp_i11(dc_raw * i32::from(quant_zigzag[0]))
    };
    out[0] = dc;

    let mut k: usize = 0;
    while k < 63 && pos < rle.len() {
        let hw = rle[pos];
        pos += 1;
        if hw == EOB_HALFWORD {
            break;
        }
        let run = usize::from((hw >> 10) & 0x3F);
        k += run + 1;
        if k >= 64 {
            break;
        }
        let ac_raw = sign_extend_10(hw);
        if q_scale == 0 {
            out[k] = clamp_i11(ac_raw * 2);
        } else {
            out[ZIGZAG[k]] = dequant_ac(ac_raw, quant_zigzag[k], q_scale);
        }
    }

    Some(pos)
}

/// Zero-allocation streaming RLE decoder for one 8×8 block, used by [`crate::MdecDevice`].
#[derive(Debug, Clone)]
pub struct RleBlockDecoder {
    in_block: bool,
    q_scale: i32,
    k: usize,
    coeffs: [i16; 64],
}

impl Default for RleBlockDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl RleBlockDecoder {
    /// Creates a fresh idle RLE block decoder.
    pub const fn new() -> Self {
        Self {
            in_block: false,
            q_scale: 0,
            k: 0,
            coeffs: [0; 64],
        }
    }

    /// Resets the decoder to the start of a new block.
    #[inline]
    pub fn reset(&mut self) {
        self.in_block = false;
        self.q_scale = 0;
        self.k = 0;
        self.coeffs.fill(0);
    }

    /// Returns `true` if the decoder has started a block (consumed a DC halfword).
    #[inline]
    pub const fn is_in_block(&self) -> bool {
        self.in_block
    }

    /// Returns a reference to the current 8×8 dequantized coefficient buffer.
    #[inline]
    pub const fn coeffs(&self) -> &[i16; 64] {
        &self.coeffs
    }

    /// Feeds one 16-bit RLE halfword.
    ///
    /// Returns `true` when the block is complete (either via `0xFE00` EOB or
    /// reaching `k >= 63`). When `true` is returned, [`Self::coeffs`] holds
    /// the finished 8×8 coefficient block, and the decoder is reset for the
    /// next block on the following call (after caller reads [`Self::coeffs`]).
    pub fn push_halfword(&mut self, hw: u16, quant_zigzag: &[u8; 64]) -> bool {
        if !self.in_block {
            if hw == EOB_HALFWORD {
                return false;
            }
            self.coeffs.fill(0);
            self.in_block = true;
            self.k = 0;
            self.q_scale = i32::from((hw >> 10) & 0x3F);
            let dc_raw = sign_extend_10(hw);
            self.coeffs[0] = if self.q_scale == 0 {
                clamp_i11(dc_raw * 2)
            } else {
                clamp_i11(dc_raw * i32::from(quant_zigzag[0]))
            };
            return false;
        }

        if hw == EOB_HALFWORD {
            self.in_block = false;
            return true;
        }

        let run = usize::from((hw >> 10) & 0x3F);
        self.k += run + 1;
        if self.k >= 64 {
            self.in_block = false;
            return true;
        }

        let ac_raw = sign_extend_10(hw);
        if self.q_scale == 0 {
            self.coeffs[self.k] = clamp_i11(ac_raw * 2);
        } else {
            self.coeffs[ZIGZAG[self.k]] = dequant_ac(ac_raw, quant_zigzag[self.k], self.q_scale);
        }

        if self.k == 63 {
            self.in_block = false;
            return true;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::DEFAULT_QUANT_LUMA;

    #[test]
    fn sign_extend_10_bounds() {
        assert_eq!(sign_extend_10(0), 0);
        assert_eq!(sign_extend_10(511), 511);
        assert_eq!(sign_extend_10(512), -512);
        assert_eq!(sign_extend_10(1023), -1);
    }

    #[test]
    fn skips_leading_eob_and_decodes_q0_linear() {
        // q_scale = 0, dc = 10, then run = 0, ac = -5 at linear index 1, then EOB.
        let rle = [
            EOB_HALFWORD,
            EOB_HALFWORD,
            10u16,
            (-5i16 as u16) & 0x3FF,
            EOB_HALFWORD,
        ];
        let mut blk = [0i16; 64];
        let consumed = decode_rle_block(&rle, &DEFAULT_QUANT_LUMA, &mut blk).unwrap();
        assert_eq!(consumed, 5);
        assert_eq!(blk[0], 20);
        assert_eq!(blk[1], -10);
        assert!(blk[2..].iter().all(|&v| v == 0));
    }

    #[test]
    fn clamps_to_signed_11_bit() {
        // q_scale = 63, dc = 511 -> 511 * 2 = 1022; ac = 511 at k=1 (qt=16) -> clamped to 1023.
        let dc = (63u16 << 10) | 511;
        let ac_pos = 511u16;
        let ac_neg = 512u16; // -512
        let rle = [dc, ac_pos, ac_neg, EOB_HALFWORD];
        let mut blk = [0i16; 64];
        decode_rle_block(&rle, &DEFAULT_QUANT_LUMA, &mut blk).unwrap();
        assert_eq!(blk[0], 1022);
        assert_eq!(blk[ZIGZAG[1]], 1023);
        assert_eq!(blk[ZIGZAG[2]], -1024);
    }
}
