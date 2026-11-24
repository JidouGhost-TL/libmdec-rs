//! Low-level hardware-faithful PS1 MDEC device (`MDEC0` / `MDEC1` MMIO & DMA).
//!
//! Emulates the silicon FIFO state machine, `MDECSTAT` status register,
//! `MDECCTRL` reset/DMA-enable bits, and commands `MDEC(0)`..`MDEC(7)`.
//!
//! Zero heap allocations: all FIFOs, quantization tables, scale table, and
//! working 8×8 block buffers live inline inside [`MdecDevice`].

use crate::color::{
    convert_mono_4bit, convert_mono_8bit, convert_subblock_rgb15, convert_subblock_rgb24,
};
use crate::idct::idct_2d;
use crate::registers::{
    BlockId, CommandOpcode, HardwareDepth, CMD_DEPTH_SHIFT, CMD_QUANT_COLOR_BIT, CMD_SET_BIT15_BIT,
    CMD_SIGNED_BIT, CTRL_ENABLE_DMA_IN, CTRL_ENABLE_DMA_OUT, CTRL_RESET, STAT_CMD_BUSY,
    STAT_CURRENT_BLOCK_SHIFT, STAT_DATA_IN_FIFO_FULL, STAT_DATA_IN_REQ, STAT_DATA_OUT_FIFO_EMPTY,
    STAT_DATA_OUT_REQ, STAT_OUTPUT_DEPTH_SHIFT, STAT_OUTPUT_SIGNED, STAT_REMAINING_MASK,
    STAT_SET_BIT15,
};
use crate::rle::RleBlockDecoder;
use crate::tables::{DEFAULT_QUANT_CHROMA, DEFAULT_QUANT_LUMA, DEFAULT_SCALE_TABLE};

/// Hardware Data-In FIFO capacity in 32-bit words (32 words = 128 bytes).
pub const DATA_IN_FIFO_CAPACITY: usize = 32;

/// Data-Out FIFO capacity in 32-bit words (64 words holds up to a full 48-word
/// 24-bit 8×8 sub-block or two 32-word 15-bit 8×8 sub-blocks).
pub const DATA_OUT_FIFO_CAPACITY: usize = 64;

/// Fixed-capacity ring buffer of `u32` words (zero heap allocation).
#[derive(Debug, Clone)]
struct WordFifo<const N: usize> {
    buf: [u32; N],
    head: usize,
    len: usize,
}

impl<const N: usize> WordFifo<N> {
    const fn new() -> Self {
        Self {
            buf: [0; N],
            head: 0,
            len: 0,
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    #[inline]
    const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    const fn is_full(&self) -> bool {
        self.len == N
    }

    #[inline]
    const fn free_space(&self) -> usize {
        N - self.len
    }

    #[inline]
    fn push(&mut self, word: u32) -> bool {
        if self.len == N {
            return false;
        }
        let tail = (self.head + self.len) % N;
        self.buf[tail] = word;
        self.len += 1;
        true
    }

    #[inline]
    fn pop(&mut self) -> Option<u32> {
        if self.len == 0 {
            return None;
        }
        let w = self.buf[self.head];
        self.head = (self.head + 1) % N;
        self.len -= 1;
        Some(w)
    }
}

/// Active MDEC command state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveCommand {
    Idle,
    DecodeMacroblock,
    SetQuantTable { color: bool, index: usize },
    SetScaleTable { index: usize },
}

/// Hardware-faithful PS1 MDEC device.
///
/// Matches retail PS1 silicon behavior for `MDEC0` (`0x1F801820`) and `MDEC1`
/// (`0x1F801824`), including 8×8 sub-block streaming output for 15-bit/24-bit
/// color modes and 8×8 mono output for 4-bit/8-bit modes.
#[derive(Debug, Clone)]
pub struct MdecDevice {
    // Loaded tables (preserved across MDECCTRL reset, matching hardware).
    quant_luma: [u8; 64],
    quant_chroma: [u8; 64],
    scale_table: [i16; 64],

    // Control / status flags.
    dma_in_enable: bool,
    dma_out_enable: bool,
    output_depth: HardwareDepth,
    output_signed: bool,
    set_bit15: bool,

    // Active command tracking.
    active_cmd: ActiveCommand,
    words_remaining: u32,
    current_block: BlockId,
    status_block: BlockId,

    // Hardware FIFOs.
    in_fifo: WordFifo<DATA_IN_FIFO_CAPACITY>,
    out_fifo: WordFifo<DATA_OUT_FIFO_CAPACITY>,

    // Halfword unpacking state for in_fifo -> RLE decoder.
    pending_hi_halfword: Option<u16>,

    // Block decode pipeline state.
    rle: RleBlockDecoder,
    cr_block: [i16; 64],
    cb_block: [i16; 64],
    y_block: [i16; 64],
}

impl Default for MdecDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl MdecDevice {
    /// Creates a new [`MdecDevice`] in its power-on / reset state, preloaded
    /// with the standard PS1 quantization and IDCT scale tables.
    pub fn new() -> Self {
        Self {
            quant_luma: DEFAULT_QUANT_LUMA,
            quant_chroma: DEFAULT_QUANT_CHROMA,
            scale_table: DEFAULT_SCALE_TABLE,
            dma_in_enable: false,
            dma_out_enable: false,
            output_depth: HardwareDepth::Bit4,
            output_signed: false,
            set_bit15: false,
            active_cmd: ActiveCommand::Idle,
            words_remaining: 0,
            current_block: BlockId::Cr,
            status_block: BlockId::Cr,
            in_fifo: WordFifo::new(),
            out_fifo: WordFifo::new(),
            pending_hi_halfword: None,
            rle: RleBlockDecoder::new(),
            cr_block: [0; 64],
            cb_block: [0; 64],
            y_block: [0; 64],
        }
    }

    /// Performs a hardware reset (equivalent to writing `CTRL_RESET` to `MDEC1`).
    ///
    /// Aborts any active command, clears both FIFOs, disables DMA requests,
    /// and resets `MDECSTAT` to `0x8004_0000`. Quantization and scale tables
    /// are preserved, matching PS1 hardware.
    pub fn reset(&mut self) {
        self.dma_in_enable = false;
        self.dma_out_enable = false;
        self.output_depth = HardwareDepth::Bit4;
        self.output_signed = false;
        self.set_bit15 = false;
        self.active_cmd = ActiveCommand::Idle;
        self.words_remaining = 0;
        self.current_block = BlockId::Cr;
        self.status_block = BlockId::Cr;
        self.in_fifo.clear();
        self.out_fifo.clear();
        self.pending_hi_halfword = None;
        self.rle.reset();
    }

    /// Returns the currently loaded luminance quantization table (zig-zag order).
    pub const fn quant_luma(&self) -> &[u8; 64] {
        &self.quant_luma
    }

    /// Returns the currently loaded chrominance quantization table (zig-zag order).
    pub const fn quant_chroma(&self) -> &[u8; 64] {
        &self.quant_chroma
    }

    /// Returns the currently loaded 64×`i16` IDCT scale table.
    pub const fn scale_table(&self) -> &[i16; 64] {
        &self.scale_table
    }

    /// Writes to `MDEC1` (`0x1F801824` — `MDECCTRL`).
    pub fn write_control(&mut self, value: u32) {
        if (value & CTRL_RESET) != 0 {
            self.reset();
        }
        self.dma_in_enable = (value & CTRL_ENABLE_DMA_IN) != 0;
        self.dma_out_enable = (value & CTRL_ENABLE_DMA_OUT) != 0;
    }

    /// Reads `MDEC1` (`0x1F801824` — `MDECSTAT`).
    pub fn read_status(&self) -> u32 {
        let mut stat = 0u32;

        if self.out_fifo.is_empty() {
            stat |= STAT_DATA_OUT_FIFO_EMPTY;
        }
        if self.in_fifo.is_full() {
            stat |= STAT_DATA_IN_FIFO_FULL;
        }
        if self.is_cmd_busy() {
            stat |= STAT_CMD_BUSY;
        }
        if self.dma_in_ready() {
            stat |= STAT_DATA_IN_REQ;
        }
        if self.dma_out_ready() {
            stat |= STAT_DATA_OUT_REQ;
        }

        stat |= self.output_depth.as_bits() << STAT_OUTPUT_DEPTH_SHIFT;
        if self.output_signed {
            stat |= STAT_OUTPUT_SIGNED;
        }
        if self.set_bit15 {
            stat |= STAT_SET_BIT15;
        }
        let reported_block = if self.active_cmd == ActiveCommand::Idle {
            BlockId::Cr
        } else {
            self.status_block
        };
        stat |= reported_block.as_bits() << STAT_CURRENT_BLOCK_SHIFT;

        let rem = if self.words_remaining == 0 {
            if self.active_cmd == ActiveCommand::Idle
                && self.out_fifo.is_empty()
                && self.output_depth == HardwareDepth::Bit4
                && !self.output_signed
                && !self.set_bit15
                && !self.dma_in_enable
                && !self.dma_out_enable
            {
                0
            } else {
                0xFFFF
            }
        } else {
            (self.words_remaining - 1) & STAT_REMAINING_MASK
        };
        stat |= rem;

        stat
    }

    /// Returns `true` if a command is currently executing or decoded output
    /// remains unread in the Data-Out FIFO.
    #[inline]
    pub fn is_cmd_busy(&self) -> bool {
        self.active_cmd != ActiveCommand::Idle || !self.out_fifo.is_empty()
    }

    /// Returns `true` when DMA0 (MDECin) request is asserted.
    #[inline]
    pub fn dma_in_ready(&self) -> bool {
        self.dma_in_enable && self.words_remaining > 0 && !self.in_fifo.is_full()
    }

    /// Returns `true` when DMA1 (MDECout) request is asserted.
    #[inline]
    pub fn dma_out_ready(&self) -> bool {
        self.dma_out_enable && !self.out_fifo.is_empty()
    }

    /// Writes a 32-bit command or parameter word to `MDEC0` (`0x1F801820`).
    pub fn write_command(&mut self, word: u32) {
        if self.active_cmd == ActiveCommand::Idle && self.out_fifo.is_empty() {
            self.start_command(word);
            return;
        }

        match self.active_cmd {
            ActiveCommand::Idle => {
                // Command finished input, only draining output FIFO; ignore stray writes.
            }
            ActiveCommand::SetQuantTable { color, mut index } => {
                let bytes = word.to_le_bytes();
                for &b in &bytes {
                    if index < 64 {
                        self.quant_luma[index] = b;
                    } else if color && index < 128 {
                        self.quant_chroma[index - 64] = b;
                    }
                    index += 1;
                }
                self.words_remaining = self.words_remaining.saturating_sub(1);
                if self.words_remaining == 0 {
                    self.active_cmd = ActiveCommand::Idle;
                } else {
                    self.active_cmd = ActiveCommand::SetQuantTable { color, index };
                }
            }
            ActiveCommand::SetScaleTable { mut index } => {
                let lo = (word & 0xFFFF) as i16;
                let hi = (word >> 16) as i16;
                if index < 64 {
                    self.scale_table[index] = lo;
                    index += 1;
                }
                if index < 64 {
                    self.scale_table[index] = hi;
                    index += 1;
                }
                self.words_remaining = self.words_remaining.saturating_sub(1);
                if self.words_remaining == 0 {
                    self.active_cmd = ActiveCommand::Idle;
                } else {
                    self.active_cmd = ActiveCommand::SetScaleTable { index };
                }
            }
            ActiveCommand::DecodeMacroblock => {
                if self.words_remaining == 0 {
                    return;
                }
                if self.in_fifo.push(word) {
                    self.words_remaining -= 1;
                }
                self.step_decode();
                if self.words_remaining == 0
                    && self.in_fifo.is_empty()
                    && self.pending_hi_halfword.is_none()
                {
                    self.active_cmd = ActiveCommand::Idle;
                }
            }
        }
    }

    /// Reads one 32-bit decoded data word from `MDEC0` (`0x1F801820`).
    ///
    /// Returns `0` if the Data-Out FIFO is empty.
    pub fn read_data(&mut self) -> u32 {
        let word = self.out_fifo.pop().unwrap_or(0);
        self.step_decode();
        if self.active_cmd == ActiveCommand::DecodeMacroblock
            && self.words_remaining == 0
            && self.in_fifo.is_empty()
            && self.pending_hi_halfword.is_none()
        {
            self.active_cmd = ActiveCommand::Idle;
        }
        word
    }

    /// Feeds up to `words.len()` parameter words via DMA0 (MDECin), stopping
    /// if the Data-In FIFO becomes full or the command no longer accepts input.
    ///
    /// Returns the number of words consumed.
    pub fn dma_in(&mut self, words: &[u32]) -> usize {
        let mut written = 0;
        for &w in words {
            if self.words_remaining == 0 || self.in_fifo.is_full() {
                break;
            }
            self.write_command(w);
            written += 1;
        }
        written
    }

    /// Reads up to `out.len()` decoded words via DMA1 (MDECout), stopping when
    /// the Data-Out FIFO becomes empty.
    ///
    /// Returns the number of words written into `out`.
    pub fn dma_out(&mut self, out: &mut [u32]) -> usize {
        let mut read = 0;
        for slot in out.iter_mut() {
            if self.out_fifo.is_empty() {
                break;
            }
            *slot = self.read_data();
            read += 1;
        }
        read
    }

    /// Starts a new command written to `MDEC0`.
    fn start_command(&mut self, word: u32) {
        // All commands latch depth/signed/stp from bits 28..25 into MDECSTAT bits 26..23.
        self.output_depth = HardwareDepth::from_bits(word >> CMD_DEPTH_SHIFT);
        self.output_signed = (word & CMD_SIGNED_BIT) != 0;
        self.set_bit15 = (word & CMD_SET_BIT15_BIT) != 0;

        match CommandOpcode::from_word(word) {
            CommandOpcode::DecodeMacroblock => {
                let count = word & 0xFFFF;
                self.words_remaining = if count == 0 { 0x1_0000 } else { count };
                self.active_cmd = ActiveCommand::DecodeMacroblock;
                self.current_block = BlockId::Cr;
                self.status_block = BlockId::Cr;
                self.in_fifo.clear();
                self.out_fifo.clear();
                self.pending_hi_halfword = None;
                self.rle.reset();
            }
            CommandOpcode::SetQuantTable => {
                let color = (word & CMD_QUANT_COLOR_BIT) != 0;
                self.words_remaining = if color { 32 } else { 16 };
                self.active_cmd = ActiveCommand::SetQuantTable { color, index: 0 };
            }
            CommandOpcode::SetScaleTable => {
                self.words_remaining = 32;
                self.active_cmd = ActiveCommand::SetScaleTable { index: 0 };
            }
            CommandOpcode::Nop0
            | CommandOpcode::Nop4
            | CommandOpcode::Nop5
            | CommandOpcode::Nop6
            | CommandOpcode::Nop7 => {
                self.words_remaining = 0;
                self.active_cmd = ActiveCommand::Idle;
            }
        }
    }

    /// Number of 32-bit output words produced by one completed output block.
    #[inline]
    const fn words_per_output_block(&self) -> usize {
        match self.output_depth {
            HardwareDepth::Bit4 => 8,
            HardwareDepth::Bit8 => 16,
            HardwareDepth::Bit15 => 32,
            HardwareDepth::Bit24 => 48,
        }
    }

    /// Advances the internal RLE -> IDCT -> Color pipeline as far as available
    /// input halfwords and output FIFO capacity allow.
    fn step_decode(&mut self) {
        if self.active_cmd != ActiveCommand::DecodeMacroblock {
            return;
        }

        loop {
            // Check whether the next completed block will produce output, and if so,
            // ensure the Data-Out FIFO has room before consuming input.
            let produces_output = self.output_depth.is_mono()
                || !matches!(self.current_block, BlockId::Cr | BlockId::Cb);
            if produces_output && self.out_fifo.free_space() < self.words_per_output_block() {
                break;
            }

            let hw = match self.pop_in_halfword() {
                Some(h) => h,
                None => break,
            };

            let qt = match (self.output_depth.is_mono(), self.current_block) {
                (true, _) => &self.quant_luma,
                (false, BlockId::Cr | BlockId::Cb) => &self.quant_chroma,
                (false, _) => &self.quant_luma,
            };

            if self.rle.push_halfword(hw, qt) {
                self.finish_current_block();
            }
        }
    }

    #[inline]
    fn pop_in_halfword(&mut self) -> Option<u16> {
        if let Some(hi) = self.pending_hi_halfword.take() {
            return Some(hi);
        }
        let word = self.in_fifo.pop()?;
        let lo = (word & 0xFFFF) as u16;
        let hi = (word >> 16) as u16;
        self.pending_hi_halfword = Some(hi);
        Some(lo)
    }

    fn finish_current_block(&mut self) {
        match self.output_depth {
            HardwareDepth::Bit4 => {
                idct_2d(self.rle.coeffs(), &self.scale_table, &mut self.y_block);
                let mut bytes = [0u8; 32];
                convert_mono_4bit(&self.y_block, self.output_signed, &mut bytes);
                for chunk in bytes.chunks_exact(4) {
                    let w = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    self.out_fifo.push(w);
                }
                self.current_block = BlockId::Cr;
            }
            HardwareDepth::Bit8 => {
                idct_2d(self.rle.coeffs(), &self.scale_table, &mut self.y_block);
                let mut bytes = [0u8; 64];
                convert_mono_8bit(&self.y_block, self.output_signed, &mut bytes);
                for chunk in bytes.chunks_exact(4) {
                    let w = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    self.out_fifo.push(w);
                }
                self.current_block = BlockId::Cr;
            }
            HardwareDepth::Bit15 | HardwareDepth::Bit24 => match self.current_block {
                BlockId::Cr => {
                    idct_2d(self.rle.coeffs(), &self.scale_table, &mut self.cr_block);
                    self.current_block = BlockId::Cb;
                }
                BlockId::Cb => {
                    idct_2d(self.rle.coeffs(), &self.scale_table, &mut self.cb_block);
                    self.current_block = BlockId::Y1;
                }
                BlockId::Y1 => {
                    self.emit_color_subblock(0, 0);
                    self.current_block = BlockId::Y2;
                    self.status_block = BlockId::Y2;
                }
                BlockId::Y2 => {
                    self.emit_color_subblock(4, 0);
                    self.current_block = BlockId::Y3;
                    self.status_block = BlockId::Y3;
                }
                BlockId::Y3 => {
                    self.emit_color_subblock(0, 4);
                    self.current_block = BlockId::Y4;
                    self.status_block = BlockId::Y4;
                }
                BlockId::Y4 => {
                    self.emit_color_subblock(4, 4);
                    self.current_block = BlockId::Cr;
                    self.status_block = BlockId::Y1;
                }
            },
        }
    }

    fn emit_color_subblock(&mut self, quad_x: usize, quad_y: usize) {
        idct_2d(self.rle.coeffs(), &self.scale_table, &mut self.y_block);
        match self.output_depth {
            HardwareDepth::Bit15 => {
                let mut pixels = [0u16; 64];
                convert_subblock_rgb15(
                    &self.y_block,
                    &self.cr_block,
                    &self.cb_block,
                    quad_x,
                    quad_y,
                    self.output_signed,
                    self.set_bit15,
                    &mut pixels,
                );
                for pair in pixels.chunks_exact(2) {
                    let w = u32::from(pair[0]) | (u32::from(pair[1]) << 16);
                    self.out_fifo.push(w);
                }
            }
            HardwareDepth::Bit24 => {
                let mut bytes = [0u8; 192];
                convert_subblock_rgb24(
                    &self.y_block,
                    &self.cr_block,
                    &self.cb_block,
                    quad_x,
                    quad_y,
                    self.output_signed,
                    &mut bytes,
                );
                for chunk in bytes.chunks_exact(4) {
                    let w = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    self.out_fifo.push(w);
                }
            }
            HardwareDepth::Bit4 | HardwareDepth::Bit8 => {}
        }
    }
}
