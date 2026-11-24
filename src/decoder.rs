//! High-level ergonomic PS1 MDEC macroblock and frame decoder.
//!
//! Outputs raster-ordered 8×8 monochrome blocks, 16×16 color macroblocks, and
//! full `width × height` frames (decoded from PS1 column-major macroblock order)
//! with zero heap allocations in all RLE decode loops.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::color::{
    convert_macroblock_rgb15_le_strided, convert_macroblock_rgb24_strided,
    convert_macroblock_rgba32_strided, convert_mono_4bit_strided, convert_mono_8bit_strided,
};
use crate::idct::idct_2d;
use crate::rle::decode_rle_block;
use crate::tables::{DEFAULT_QUANT_CHROMA, DEFAULT_QUANT_LUMA, DEFAULT_SCALE_TABLE};
#[cfg(feature = "alloc")]
use crate::vlc::{
    vlc_ac_to_rle_with_headers, vlc_to_rle, vlc_to_rle_with_block_headers, AcCodebook, DcCoding,
};
use crate::Error;

/// Pixel output format for [`Decoder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputDepth {
    /// 4-bit monochrome (2 pixels per byte, 8×8 block = 32 bytes).
    Bit4,
    /// 8-bit monochrome (1 byte per pixel, 8×8 block = 64 bytes).
    Bit8,
    /// 15-bit PS1 RGB555 (2 bytes per pixel little-endian, optional STP bit 15).
    Bit15 {
        /// Value written to bit 15 (semi-transparency bit) of each pixel.
        stp: bool,
    },
    /// 24-bit RGB888 (3 bytes per pixel).
    Bit24,
    /// 32-bit RGBA8888 (4 bytes per pixel, alpha = `0xFF`).
    Rgba32,
}

impl OutputDepth {
    /// Returns the number of bytes required for a single 16×16 color macroblock,
    /// or `None` for monochrome modes (`Bit4` / `Bit8`).
    #[must_use]
    #[inline]
    pub const fn macroblock_bytes(self) -> Option<usize> {
        match self {
            Self::Bit4 | Self::Bit8 => None,
            Self::Bit15 { .. } => Some(16 * 16 * 2),
            Self::Bit24 => Some(16 * 16 * 3),
            Self::Rgba32 => Some(16 * 16 * 4),
        }
    }

    /// Returns the number of bytes required for a single 8×8 monochrome block,
    /// or `None` for color modes.
    #[must_use]
    #[inline]
    pub const fn mono_block_bytes(self) -> Option<usize> {
        match self {
            Self::Bit4 => Some(32),
            Self::Bit8 => Some(64),
            Self::Bit15 { .. } | Self::Bit24 | Self::Rgba32 => None,
        }
    }

    /// Returns the number of bytes required for a `width × height` raster frame,
    /// or `None` if the calculation overflows `usize`.
    #[must_use]
    #[inline]
    pub const fn checked_frame_bytes(self, width: usize, height: usize) -> Option<usize> {
        let pixels = match width.checked_mul(height) {
            Some(p) => p,
            None => return None,
        };
        match self {
            Self::Bit4 => Some(pixels / 2),
            Self::Bit8 => Some(pixels),
            Self::Bit15 { .. } => pixels.checked_mul(2),
            Self::Bit24 => pixels.checked_mul(3),
            Self::Rgba32 => pixels.checked_mul(4),
        }
    }

    /// Returns the number of bytes required for a `width × height` raster frame.
    ///
    /// # Panics
    /// Panics if `width * height * bytes_per_pixel` overflows `usize`. Prefer
    /// [`Self::checked_frame_bytes`] for untrusted dimensions.
    #[must_use]
    #[inline]
    pub const fn frame_bytes(self, width: usize, height: usize) -> usize {
        match self.checked_frame_bytes(width, height) {
            Some(n) => n,
            None => panic!("frame byte size overflow"),
        }
    }
}

/// High-level ergonomic PS1 MDEC decoder.
///
/// Configured with output depth, signedness, quantization tables (zig-zag order),
/// and 64×`i16` IDCT scale table. All RLE block/macroblock/frame decode methods
/// perform zero heap allocations.
#[derive(Debug, Clone)]
pub struct Decoder {
    output_depth: OutputDepth,
    signed: bool,
    quant_luma: [u8; 64],
    quant_chroma: [u8; 64],
    scale_table: [i16; 64],
}

impl Decoder {
    /// Creates a new [`Decoder`] with standard PS1 quantization and scale tables.
    #[must_use]
    pub const fn new(output_depth: OutputDepth) -> Self {
        Self {
            output_depth,
            signed: false,
            quant_luma: DEFAULT_QUANT_LUMA,
            quant_chroma: DEFAULT_QUANT_CHROMA,
            scale_table: DEFAULT_SCALE_TABLE,
        }
    }

    /// Enables or disables signed output mode (`-128..=127` instead of `0..=255`).
    #[must_use]
    pub const fn with_signed(mut self, signed: bool) -> Self {
        self.signed = signed;
        self
    }

    /// Configures custom luminance and chrominance quantization tables (zig-zag order).
    #[must_use]
    pub const fn with_quant_tables(mut self, quant_luma: [u8; 64], quant_chroma: [u8; 64]) -> Self {
        self.quant_luma = quant_luma;
        self.quant_chroma = quant_chroma;
        self
    }

    /// Configures a custom 64×`i16` IDCT scale table.
    #[must_use]
    pub const fn with_scale_table(mut self, scale_table: [i16; 64]) -> Self {
        self.scale_table = scale_table;
        self
    }

    /// Returns the configured [`OutputDepth`].
    #[must_use]
    #[inline]
    pub const fn output_depth(&self) -> OutputDepth {
        self.output_depth
    }

    /// Updates the output depth.
    #[inline]
    pub fn set_output_depth(&mut self, output_depth: OutputDepth) {
        self.output_depth = output_depth;
    }

    /// Returns whether signed output mode is enabled.
    #[must_use]
    #[inline]
    pub const fn is_signed(&self) -> bool {
        self.signed
    }

    /// Enables or disables signed output mode.
    #[inline]
    pub fn set_signed(&mut self, signed: bool) {
        self.signed = signed;
    }

    /// Updates the luminance and chrominance quantization tables (zig-zag order).
    #[inline]
    pub fn set_quant_tables(&mut self, quant_luma: [u8; 64], quant_chroma: [u8; 64]) {
        self.quant_luma = quant_luma;
        self.quant_chroma = quant_chroma;
    }

    /// Updates the 64×`i16` IDCT scale table.
    #[inline]
    pub fn set_scale_table(&mut self, scale_table: [i16; 64]) {
        self.scale_table = scale_table;
    }

    /// Decodes one 8×8 monochrome block from `rle` directly into strided `out`.
    fn decode_rle_mono_block_strided(
        &self,
        rle: &[u16],
        out: &mut [u8],
        stride: usize,
    ) -> Result<usize, Error> {
        let mut coeffs = [0i16; 64];
        let mut spatial = [0i16; 64];
        let consumed =
            decode_rle_block(rle, &self.quant_luma, &mut coeffs).ok_or(Error::UnexpectedEof)?;
        idct_2d(&coeffs, &self.scale_table, &mut spatial);

        match self.output_depth {
            OutputDepth::Bit4 => convert_mono_4bit_strided(&spatial, self.signed, out, stride),
            OutputDepth::Bit8 => convert_mono_8bit_strided(&spatial, self.signed, out, stride),
            _ => unreachable!(),
        }

        Ok(consumed)
    }

    /// Decodes one 16×16 color macroblock from `rle` directly into strided `out`.
    fn decode_rle_macroblock_strided(
        &self,
        rle: &[u16],
        out: &mut [u8],
        stride: usize,
    ) -> Result<usize, Error> {
        let mut pos = 0;
        let mut coeffs = [0i16; 64];
        let mut cr = [0i16; 64];
        let mut cb = [0i16; 64];
        let mut y = [[0i16; 64]; 4];

        let c = decode_rle_block(&rle[pos..], &self.quant_chroma, &mut coeffs)
            .ok_or(Error::UnexpectedEof)?;
        pos += c;
        idct_2d(&coeffs, &self.scale_table, &mut cr);

        let c = decode_rle_block(&rle[pos..], &self.quant_chroma, &mut coeffs)
            .ok_or(Error::UnexpectedEof)?;
        pos += c;
        idct_2d(&coeffs, &self.scale_table, &mut cb);

        for y_blk in &mut y {
            let c = decode_rle_block(&rle[pos..], &self.quant_luma, &mut coeffs)
                .ok_or(Error::UnexpectedEof)?;
            pos += c;
            idct_2d(&coeffs, &self.scale_table, y_blk);
        }

        match self.output_depth {
            OutputDepth::Bit15 { stp } => {
                convert_macroblock_rgb15_le_strided(&y, &cr, &cb, self.signed, stp, out, stride);
            }
            OutputDepth::Bit24 => {
                convert_macroblock_rgb24_strided(&y, &cr, &cb, self.signed, out, stride);
            }
            OutputDepth::Rgba32 => {
                convert_macroblock_rgba32_strided(&y, &cr, &cb, self.signed, out, stride);
            }
            OutputDepth::Bit4 | OutputDepth::Bit8 => unreachable!(),
        }

        Ok(pos)
    }

    /// Decodes a single 8×8 monochrome block (`OutputDepth::Bit4` = 32 bytes,
    /// `OutputDepth::Bit8` = 64 bytes) from `rle` into `out`.
    ///
    /// Returns the number of RLE halfwords consumed. Zero heap allocations.
    pub fn decode_rle_mono_block(&self, rle: &[u16], out: &mut [u8]) -> Result<usize, Error> {
        let needed = self
            .output_depth
            .mono_block_bytes()
            .ok_or(Error::InvalidParameter("output depth is not monochrome"))?;
        if out.len() < needed {
            return Err(Error::BufferTooSmall {
                needed,
                actual: out.len(),
            });
        }
        let stride = if self.output_depth == OutputDepth::Bit4 {
            4
        } else {
            8
        };
        self.decode_rle_mono_block_strided(rle, out, stride)
    }

    /// Decodes a single 16×16 color macroblock (`Cr, Cb, Y0, Y1, Y2, Y3`) from `rle`
    /// into raster-ordered `out` (`Bit15` = 512B, `Bit24` = 768B, `Rgba32` = 1024B).
    ///
    /// Returns the number of RLE halfwords consumed. Zero heap allocations.
    pub fn decode_rle_macroblock(&self, rle: &[u16], out: &mut [u8]) -> Result<usize, Error> {
        let needed = self
            .output_depth
            .macroblock_bytes()
            .ok_or(Error::InvalidParameter(
                "output depth is not a color format",
            ))?;
        if out.len() < needed {
            return Err(Error::BufferTooSmall {
                needed,
                actual: out.len(),
            });
        }
        let stride = needed / 16;
        self.decode_rle_macroblock_strided(rle, out, stride)
    }

    /// Decodes a full `width × height` raster frame from PS1 column-major MDEC RLE
    /// halfwords into `out`.
    ///
    /// - For color depths (`Bit15`, `Bit24`, `Rgba32`), `width` and `height` must be
    ///   non-zero multiples of 16, and macroblocks are arranged column-by-column
    ///   (top-to-bottom within each 16-pixel stripe, left-to-right across stripes).
    /// - For monochrome depths (`Bit4`, `Bit8`), `width` and `height` must be
    ///   non-zero multiples of 8, arranged column-by-column in 8×8 blocks.
    ///
    /// Zero heap allocations.
    pub fn decode_rle_frame(
        &self,
        rle: &[u16],
        width: usize,
        height: usize,
        out: &mut [u8],
    ) -> Result<(), Error> {
        match self.output_depth {
            OutputDepth::Bit4 | OutputDepth::Bit8 => {
                if width == 0 || height == 0 || width % 8 != 0 || height % 8 != 0 {
                    return Err(Error::InvalidDimensions { width, height });
                }
                let needed = self
                    .output_depth
                    .checked_frame_bytes(width, height)
                    .ok_or(Error::InvalidDimensions { width, height })?;
                if out.len() < needed {
                    return Err(Error::BufferTooSmall {
                        needed,
                        actual: out.len(),
                    });
                }

                let cols = width / 8;
                let rows = height / 8;
                let row_bytes_block = if self.output_depth == OutputDepth::Bit4 {
                    4
                } else {
                    8
                };
                let stride = if self.output_depth == OutputDepth::Bit4 {
                    width / 2
                } else {
                    width
                };
                let mut pos = 0;
                for bx in 0..cols {
                    for by in 0..rows {
                        let base = (by * 8) * stride + bx * row_bytes_block;
                        let consumed = self.decode_rle_mono_block_strided(
                            &rle[pos..],
                            &mut out[base..],
                            stride,
                        )?;
                        pos += consumed;
                    }
                }
            }
            OutputDepth::Bit15 { .. } | OutputDepth::Bit24 | OutputDepth::Rgba32 => {
                if width == 0 || height == 0 || width % 16 != 0 || height % 16 != 0 {
                    return Err(Error::InvalidDimensions { width, height });
                }
                let needed = self
                    .output_depth
                    .checked_frame_bytes(width, height)
                    .ok_or(Error::InvalidDimensions { width, height })?;
                if out.len() < needed {
                    return Err(Error::BufferTooSmall {
                        needed,
                        actual: out.len(),
                    });
                }

                let mb_cols = width / 16;
                let mb_rows = height / 16;
                let bpp = match self.output_depth {
                    OutputDepth::Bit15 { .. } => 2,
                    OutputDepth::Bit24 => 3,
                    OutputDepth::Rgba32 => 4,
                    _ => unreachable!(),
                };
                let stride = width * bpp;
                let mb_row_bytes = 16 * bpp;
                let mut pos = 0;
                for mb_x in 0..mb_cols {
                    for mb_y in 0..mb_rows {
                        let base = (mb_y * 16) * stride + mb_x * mb_row_bytes;
                        let consumed = self.decode_rle_macroblock_strided(
                            &rle[pos..],
                            &mut out[base..],
                            stride,
                        )?;
                        pos += consumed;
                    }
                }
            }
        }

        Ok(())
    }

    /// Validates color output depth, `width × height` dimensions, and `out.len()`
    /// upfront before allocating or decoding a BS frame. Returns the macroblock count.
    #[cfg(feature = "alloc")]
    fn validate_bs_color_frame(
        &self,
        width: usize,
        height: usize,
        out_len: usize,
    ) -> Result<usize, Error> {
        if self.output_depth.macroblock_bytes().is_none() {
            return Err(Error::InvalidParameter(
                "output depth is not a color format",
            ));
        }
        if width == 0 || height == 0 || width % 16 != 0 || height % 16 != 0 {
            return Err(Error::InvalidDimensions { width, height });
        }
        let needed = self
            .output_depth
            .checked_frame_bytes(width, height)
            .ok_or(Error::InvalidDimensions { width, height })?;
        if out_len < needed {
            return Err(Error::BufferTooSmall {
                needed,
                actual: out_len,
            });
        }
        Ok((width / 16) * (height / 16))
    }

    /// Decodes a PS1 BS VLC bitstream frame (with frame-level `q_scale` and
    /// [`DcCoding`]) using a caller-supplied [`AcCodebook`] (such as a custom
    /// `VLC0` table) into a `width × height` raster buffer `out`.
    #[cfg(feature = "alloc")]
    #[allow(clippy::too_many_arguments)]
    pub fn decode_bs_frame_with_codebook(
        &self,
        bitstream: &[u8],
        width: usize,
        height: usize,
        q_scale: u16,
        dc_coding: DcCoding,
        codebook: &AcCodebook,
        out: &mut [u8],
    ) -> Result<(), Error> {
        let macroblocks = self.validate_bs_color_frame(width, height, out.len())?;
        let mut rle = Vec::with_capacity(macroblocks * 6 * 8);
        vlc_to_rle(
            bitstream,
            q_scale,
            dc_coding,
            codebook,
            macroblocks,
            &mut rle,
        )?;
        self.decode_rle_frame(&rle, width, height, out)
    }

    /// Decodes a PS1 BS VLC bitstream frame (with frame-level `q_scale` and
    /// [`DcCoding`]) using the standard PS1 AC codebook into a `width × height`
    /// raster buffer `out`.
    #[cfg(feature = "alloc")]
    pub fn decode_bs_frame(
        &self,
        bitstream: &[u8],
        width: usize,
        height: usize,
        q_scale: u16,
        dc_coding: DcCoding,
        out: &mut [u8],
    ) -> Result<(), Error> {
        self.decode_bs_frame_with_codebook(
            bitstream,
            width,
            height,
            q_scale,
            dc_coding,
            AcCodebook::standard(),
            out,
        )
    }

    /// Decodes a PS1 BS VLC bitstream frame where each block begins with a 16-bit
    /// `(q_scale: 6 bits, dc: 10 signed bits)` header into a `width × height`
    /// raster buffer `out`.
    #[cfg(feature = "alloc")]
    pub fn decode_bs_frame_with_block_headers(
        &self,
        bitstream: &[u8],
        width: usize,
        height: usize,
        out: &mut [u8],
    ) -> Result<(), Error> {
        let macroblocks = self.validate_bs_color_frame(width, height, out.len())?;
        let mut rle = Vec::with_capacity(macroblocks * 6 * 8);
        vlc_to_rle_with_block_headers(bitstream, AcCodebook::standard(), macroblocks, &mut rle)?;
        self.decode_rle_frame(&rle, width, height, out)
    }

    /// Decodes a `width × height` color frame from an AC-only BS bitstream paired
    /// with out-of-band `(q_scale, dc)` per-block headers (such as `Iki` frames)
    /// and a caller-supplied [`AcCodebook`].
    ///
    /// `block_headers.len()` must equal `(width / 16) * (height / 16) * 6`.
    #[cfg(feature = "alloc")]
    pub fn decode_ac_frame_with_headers(
        &self,
        ac_bitstream: &[u8],
        block_headers: &[(u16, i32)],
        width: usize,
        height: usize,
        codebook: &AcCodebook,
        out: &mut [u8],
    ) -> Result<(), Error> {
        let macroblocks = self.validate_bs_color_frame(width, height, out.len())?;
        let expected_blocks = macroblocks * 6;
        if block_headers.len() != expected_blocks {
            return Err(Error::InvalidParameter(
                "block_headers length must equal (width / 16) * (height / 16) * 6",
            ));
        }
        let mut rle = Vec::with_capacity(expected_blocks * 8);
        vlc_ac_to_rle_with_headers(ac_bitstream, block_headers, codebook, &mut rle)?;
        self.decode_rle_frame(&rle, width, height, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_frame_bytes_handles_overflow() {
        assert_eq!(OutputDepth::Bit24.checked_frame_bytes(usize::MAX, 16), None);
        assert_eq!(
            OutputDepth::Rgba32.checked_frame_bytes(usize::MAX / 2, 16),
            None
        );
        assert_eq!(
            OutputDepth::Bit24.checked_frame_bytes(320, 240),
            Some(320 * 240 * 3)
        );
    }

    #[test]
    fn decode_rle_frame_validates_dimensions_before_buffer_size() {
        let dec = Decoder::new(OutputDepth::Bit24);
        let mut small = [0u8; 4];
        assert_eq!(
            dec.decode_rle_frame(&[], 10, 16, &mut small),
            Err(Error::InvalidDimensions {
                width: 10,
                height: 16
            })
        );
        assert_eq!(
            dec.decode_rle_frame(&[], usize::MAX - 15, 16, &mut small),
            Err(Error::InvalidDimensions {
                width: usize::MAX - 15,
                height: 16
            })
        );
        assert_eq!(
            dec.decode_rle_frame(&[], 16, 16, &mut small),
            Err(Error::BufferTooSmall {
                needed: 16 * 16 * 3,
                actual: 4
            })
        );
    }

    #[cfg(feature = "alloc")]
    mod alloc_tests {
        use super::*;
        use crate::vlc::AcCodeEntry;
        use alloc::string::String;

        fn pack_bits(bits: &str) -> Vec<u8> {
            let mut out = Vec::new();
            let mut acc = 0u32;
            let mut n = 0u32;
            for c in bits.chars() {
                if c == '0' || c == '1' {
                    acc = (acc << 1) | u32::from(c == '1');
                    n += 1;
                    if n == 16 {
                        out.extend_from_slice(&(acc as u16).to_le_bytes());
                        acc = 0;
                        n = 0;
                    }
                }
            }
            if n > 0 {
                out.extend_from_slice(&((acc << (16 - n)) as u16).to_le_bytes());
            }
            out
        }

        #[test]
        fn bs_frame_rejects_mono_depth_and_short_buffer_before_decoding() {
            let mono = Decoder::new(OutputDepth::Bit8);
            let mut buf = alloc::vec![0u8; 16 * 16];
            assert!(matches!(
                mono.decode_bs_frame(&[], 16, 16, 1, DcCoding::FixedLength, &mut buf),
                Err(Error::InvalidParameter(_))
            ));

            let color = Decoder::new(OutputDepth::Bit24);
            let mut tiny = [0u8; 8];
            assert_eq!(
                color.decode_bs_frame(&[], 16, 16, 1, DcCoding::FixedLength, &mut tiny),
                Err(Error::BufferTooSmall {
                    needed: 16 * 16 * 3,
                    actual: 8
                })
            );
        }

        #[test]
        fn decode_bs_frame_with_custom_codebook_and_ac_frame_with_headers() {
            // Custom codebook: EOB = "0", single entry "1" -> (run=0, level=1).
            let custom_cb = AcCodebook::try_new(
                (0b0, 1),
                None,
                &[AcCodeEntry {
                    code: 0b1,
                    len: 1,
                    run: 0,
                    level: 1,
                }],
            )
            .unwrap();

            // 1 macroblock (6 blocks): Cr=0, Cb=0, Y0..Y3=+64 (+16 spatial -> 144).
            let headers = [(1u16, 0i32), (1, 0), (1, 64), (1, 64), (1, 64), (1, 64)];
            let mut bs_bits = String::new();
            let mut ac_bits = String::new();
            for &(_, dc) in &headers {
                let dc10 = (dc as u32) & 0x3FF;
                for shift in (0..10).rev() {
                    bs_bits.push(if ((dc10 >> shift) & 1) != 0 { '1' } else { '0' });
                }
                bs_bits.push('0'); // custom EOB
                ac_bits.push('0'); // custom EOB
            }
            let bs_bytes = pack_bits(&bs_bits);
            let ac_bytes = pack_bits(&ac_bits);

            let dec = Decoder::new(OutputDepth::Bit24);
            let mut out_bs = alloc::vec![0u8; 16 * 16 * 3];
            let mut out_ac = alloc::vec![0u8; 16 * 16 * 3];

            dec.decode_bs_frame_with_codebook(
                &bs_bytes,
                16,
                16,
                1,
                DcCoding::FixedLength,
                &custom_cb,
                &mut out_bs,
            )
            .unwrap();
            dec.decode_ac_frame_with_headers(&ac_bytes, &headers, 16, 16, &custom_cb, &mut out_ac)
                .unwrap();

            assert_eq!(out_bs, out_ac);
            assert!(out_bs.iter().all(|&b| b == 144));
        }
    }
}
