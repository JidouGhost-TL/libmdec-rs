//! High-level ergonomic PS1 MDEC macroblock and frame decoder.
//!
//! Outputs raster-ordered 8×8 monochrome blocks, 16×16 color macroblocks, and
//! full `width × height` frames (decoded from PS1 column-major macroblock order)
//! with zero heap allocations in all RLE decode loops.

use crate::color::{
    convert_macroblock_rgb15, convert_macroblock_rgb24, convert_macroblock_rgba32,
    convert_mono_4bit, convert_mono_8bit,
};
use crate::idct::idct_2d;
use crate::rle::decode_rle_block;
use crate::tables::{DEFAULT_QUANT_CHROMA, DEFAULT_QUANT_LUMA, DEFAULT_SCALE_TABLE};
use crate::vlc::{vlc_to_rle, vlc_to_rle_with_block_headers, AcCodebook, DcCoding};
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
    #[inline]
    pub const fn mono_block_bytes(self) -> Option<usize> {
        match self {
            Self::Bit4 => Some(32),
            Self::Bit8 => Some(64),
            Self::Bit15 { .. } | Self::Bit24 | Self::Rgba32 => None,
        }
    }

    /// Returns the number of bytes required for a `width × height` raster frame.
    pub const fn frame_bytes(self, width: usize, height: usize) -> usize {
        match self {
            Self::Bit4 => (width * height) / 2,
            Self::Bit8 => width * height,
            Self::Bit15 { .. } => width * height * 2,
            Self::Bit24 => width * height * 3,
            Self::Rgba32 => width * height * 4,
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
    pub const fn with_signed(mut self, signed: bool) -> Self {
        self.signed = signed;
        self
    }

    /// Configures custom luminance and chrominance quantization tables (zig-zag order).
    pub const fn with_quant_tables(mut self, quant_luma: [u8; 64], quant_chroma: [u8; 64]) -> Self {
        self.quant_luma = quant_luma;
        self.quant_chroma = quant_chroma;
        self
    }

    /// Configures a custom 64×`i16` IDCT scale table.
    pub const fn with_scale_table(mut self, scale_table: [i16; 64]) -> Self {
        self.scale_table = scale_table;
        self
    }

    /// Returns the configured [`OutputDepth`].
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

        let mut coeffs = [0i16; 64];
        let mut spatial = [0i16; 64];
        let consumed =
            decode_rle_block(rle, &self.quant_luma, &mut coeffs).ok_or(Error::UnexpectedEof)?;
        idct_2d(&coeffs, &self.scale_table, &mut spatial);

        match self.output_depth {
            OutputDepth::Bit4 => {
                let mut buf = [0u8; 32];
                convert_mono_4bit(&spatial, self.signed, &mut buf);
                out[..32].copy_from_slice(&buf);
            }
            OutputDepth::Bit8 => {
                let mut buf = [0u8; 64];
                convert_mono_8bit(&spatial, self.signed, &mut buf);
                out[..64].copy_from_slice(&buf);
            }
            _ => unreachable!(),
        }

        Ok(consumed)
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
                let mut pixels = [0u16; 256];
                convert_macroblock_rgb15(&y, &cr, &cb, self.signed, stp, &mut pixels);
                for (i, &px) in pixels.iter().enumerate() {
                    let b = px.to_le_bytes();
                    out[i * 2] = b[0];
                    out[i * 2 + 1] = b[1];
                }
            }
            OutputDepth::Bit24 => {
                let mut buf = [0u8; 768];
                convert_macroblock_rgb24(&y, &cr, &cb, self.signed, &mut buf);
                out[..768].copy_from_slice(&buf);
            }
            OutputDepth::Rgba32 => {
                let mut buf = [0u8; 1024];
                convert_macroblock_rgba32(&y, &cr, &cb, self.signed, &mut buf);
                out[..1024].copy_from_slice(&buf);
            }
            OutputDepth::Bit4 | OutputDepth::Bit8 => unreachable!(),
        }

        Ok(pos)
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
        let needed = self.output_depth.frame_bytes(width, height);
        if out.len() < needed {
            return Err(Error::BufferTooSmall {
                needed,
                actual: out.len(),
            });
        }

        match self.output_depth {
            OutputDepth::Bit4 | OutputDepth::Bit8 => {
                if width == 0 || height == 0 || width % 8 != 0 || height % 8 != 0 {
                    return Err(Error::InvalidDimensions { width, height });
                }
                let cols = width / 8;
                let rows = height / 8;
                let mut pos = 0;
                let mut blk = [0u8; 64];
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
                for bx in 0..cols {
                    for by in 0..rows {
                        let consumed = self.decode_rle_mono_block(&rle[pos..], &mut blk)?;
                        pos += consumed;
                        for ly in 0..8 {
                            let dst_off = (by * 8 + ly) * stride + bx * row_bytes_block;
                            let src_off = ly * row_bytes_block;
                            out[dst_off..dst_off + row_bytes_block]
                                .copy_from_slice(&blk[src_off..src_off + row_bytes_block]);
                        }
                    }
                }
            }
            OutputDepth::Bit15 { .. } | OutputDepth::Bit24 | OutputDepth::Rgba32 => {
                if width == 0 || height == 0 || width % 16 != 0 || height % 16 != 0 {
                    return Err(Error::InvalidDimensions { width, height });
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
                let mut mb_buf = [0u8; 1024];

                for mb_x in 0..mb_cols {
                    for mb_y in 0..mb_rows {
                        let consumed = self.decode_rle_macroblock(&rle[pos..], &mut mb_buf)?;
                        pos += consumed;
                        for ly in 0..16 {
                            let dst_off = (mb_y * 16 + ly) * stride + mb_x * mb_row_bytes;
                            let src_off = ly * mb_row_bytes;
                            out[dst_off..dst_off + mb_row_bytes]
                                .copy_from_slice(&mb_buf[src_off..src_off + mb_row_bytes]);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Decodes a PS1 BS VLC bitstream frame (with frame-level `q_scale` and
    /// [`DcCoding`]) into a `width × height` raster buffer `out`.
    pub fn decode_bs_frame(
        &self,
        bitstream: &[u8],
        width: usize,
        height: usize,
        q_scale: u16,
        dc_coding: DcCoding,
        out: &mut [u8],
    ) -> Result<(), Error> {
        if width == 0 || height == 0 || width % 16 != 0 || height % 16 != 0 {
            return Err(Error::InvalidDimensions { width, height });
        }
        let macroblocks = (width / 16) * (height / 16);
        let mut rle = Vec::with_capacity(macroblocks * 6 * 8);
        vlc_to_rle(
            bitstream,
            q_scale,
            dc_coding,
            AcCodebook::standard(),
            macroblocks,
            &mut rle,
        )?;
        self.decode_rle_frame(&rle, width, height, out)
    }

    /// Decodes a PS1 BS VLC bitstream frame where each block begins with a 16-bit
    /// `(q_scale: 6 bits, dc: 10 signed bits)` header into a `width × height`
    /// raster buffer `out`.
    pub fn decode_bs_frame_with_block_headers(
        &self,
        bitstream: &[u8],
        width: usize,
        height: usize,
        out: &mut [u8],
    ) -> Result<(), Error> {
        if width == 0 || height == 0 || width % 16 != 0 || height % 16 != 0 {
            return Err(Error::InvalidDimensions { width, height });
        }
        let macroblocks = (width / 16) * (height / 16);
        let mut rle = Vec::with_capacity(macroblocks * 6 * 8);
        vlc_to_rle_with_block_headers(bitstream, AcCodebook::standard(), macroblocks, &mut rle)?;
        self.decode_rle_frame(&rle, width, height, out)
    }
}
