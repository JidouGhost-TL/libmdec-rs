//! Standalone pure-Rust PlayStation 1 MDEC hardware video decoder and BS bitstream library.
//!
//! `libmdec` implements the PlayStation 1 Macroblock Decoder (MDEC) silicon
//! specification with zero runtime dependencies, `#![forbid(unsafe_code)]`,
//! deterministic fixed-point integer arithmetic (zero `f64`), and zero heap
//! allocations in block/macroblock decode loops.
//!
//! # Architecture
//! - [`device::MdecDevice`]: low-level hardware-faithful MMIO/DMA device
//!   (`MDEC0` / `MDEC1`, `MDECSTAT`, `MDECCTRL`, `MDEC(0..=7)`).
//! - [`decoder::Decoder`]: high-level ergonomic macroblock & frame decoder
//!   supporting 4-bit, 8-bit, 15-bit (RGB555 + STP), 24-bit (RGB888), and
//!   32-bit (RGBA8888) outputs.
//! - [`vlc`]: pure PS1 BS VLC-to-RLE frontend (`BitReader`, validated prefix-free
//!   `AcCodebook`, BS v1/v2 fixed-length DC and BS v3 predicted VLC DC).
//! - [`rle`], [`idct`], [`color`], [`tables`], [`registers`]: composable
//!   building blocks for emulators, tools, and decoders.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod color;
pub mod decoder;
pub mod device;
pub mod idct;
pub mod registers;
pub mod rle;
pub mod tables;
pub mod vlc;

use std::fmt;

pub use decoder::{Decoder, OutputDepth};
pub use device::MdecDevice;
pub use registers::{BlockId, CommandOpcode, HardwareDepth};
pub use vlc::{
    vlc0_bitcodes, vlc_to_rle, vlc_to_rle_with_block_headers, AcCodeEntry, AcCodebook, AcEscape,
    AcEvent, BitReader, DcCoding,
};

/// Error type returned by fallible `libmdec` operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Input bitstream or RLE slice ended before the expected block/frame completed.
    UnexpectedEof,
    /// Variable-length bitstream did not match any valid Huffman code or overflowed a block.
    BitstreamDesync,
    /// Custom AC codebook failed validation (invalid bit width, duplicate, or prefix collision).
    InvalidCodebook(&'static str),
    /// Invalid frame dimensions (must be non-zero multiples of 8 for mono or 16 for color).
    InvalidDimensions {
        /// Requested width in pixels.
        width: usize,
        /// Requested height in pixels.
        height: usize,
    },
    /// Caller-supplied output buffer is smaller than required.
    BufferTooSmall {
        /// Minimum number of bytes required.
        needed: usize,
        /// Actual buffer length provided.
        actual: usize,
    },
    /// Invalid parameter value supplied by caller.
    InvalidParameter(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of MDEC input stream"),
            Self::BitstreamDesync => write!(f, "BS VLC bitstream desynchronization"),
            Self::InvalidCodebook(msg) => write!(f, "invalid AC codebook: {msg}"),
            Self::InvalidDimensions { width, height } => {
                write!(f, "invalid frame dimensions: {width}x{height}")
            }
            Self::BufferTooSmall { needed, actual } => {
                write!(
                    f,
                    "output buffer too small: needed {needed} bytes, got {actual}"
                )
            }
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn public_types_are_send_and_sync() {
        assert_send_sync::<MdecDevice>();
        assert_send_sync::<Decoder>();
        assert_send_sync::<AcCodebook>();
        assert_send_sync::<OutputDepth>();
        assert_send_sync::<Error>();
    }
}
