//! PS1 MDEC MMIO register addresses, bitmasks, command opcodes, and hardware enums.
//!
//! Reference: Nocash PSX-SPX MDEC I/O Ports (`0x1F801820` and `0x1F801824`).

/// `MDEC0` register address: Command / Parameter Write, Data / Response Read.
pub const MDEC0_ADDR: u32 = 0x1F80_1820;

/// `MDEC1` register address: Control / Reset Write, Status Read.
pub const MDEC1_ADDR: u32 = 0x1F80_1824;

// --- MDECSTAT (0x1F801824 Read) ---

/// Bit 31: Data-Out FIFO Empty (`1` = empty, `0` = has data).
pub const STAT_DATA_OUT_FIFO_EMPTY: u32 = 1 << 31;

/// Bit 30: Data-In FIFO Full (`1` = full, `0` = can accept words).
pub const STAT_DATA_IN_FIFO_FULL: u32 = 1 << 30;

/// Bit 29: Command Busy (`1` = executing command, `0` = ready for new command).
pub const STAT_CMD_BUSY: u32 = 1 << 29;

/// Bit 28: Data-In Request (DMA0) (`1` = DMA0 enabled and ready to receive input).
pub const STAT_DATA_IN_REQ: u32 = 1 << 28;

/// Bit 27: Data-Out Request (DMA1) (`1` = DMA1 enabled and ready to output data).
pub const STAT_DATA_OUT_REQ: u32 = 1 << 27;

/// Bits 26..25 shift: Data Output Depth (`0`=4-bit, `1`=8-bit, `2`=24-bit, `3`=15-bit).
pub const STAT_OUTPUT_DEPTH_SHIFT: u32 = 25;

/// Bits 26..25 mask: Data Output Depth.
pub const STAT_OUTPUT_DEPTH_MASK: u32 = 0b11 << STAT_OUTPUT_DEPTH_SHIFT;

/// Bit 24: Data Output Signed (`0` = unsigned `0..=255`, `1` = signed `-128..=127`).
pub const STAT_OUTPUT_SIGNED: u32 = 1 << 24;

/// Bit 23: Data Output Bit 15 (STP flag for 15-bit mode).
pub const STAT_SET_BIT15: u32 = 1 << 23;

/// Bits 18..16 shift: Current Block (`0..=3` = Y1..Y4, `4` = Cr, `5` = Cb).
pub const STAT_CURRENT_BLOCK_SHIFT: u32 = 16;

/// Bits 18..16 mask: Current Block.
pub const STAT_CURRENT_BLOCK_MASK: u32 = 0b111 << STAT_CURRENT_BLOCK_SHIFT;

/// Bits 15..0 mask: Number of parameter words remaining minus 1 (`0xFFFF` = none).
pub const STAT_REMAINING_MASK: u32 = 0xFFFF;

/// Value of `MDECSTAT` immediately after hardware reset (`MDECCTRL` bit 31).
pub const STAT_RESET_VALUE: u32 = STAT_DATA_OUT_FIFO_EMPTY | (4 << STAT_CURRENT_BLOCK_SHIFT);

// --- MDECCTRL (0x1F801824 Write) ---

/// Bit 31: Reset MDEC (aborts command, empties FIFOs, sets `MDECSTAT = 0x80040000`).
pub const CTRL_RESET: u32 = 1 << 31;

/// Bit 30: Enable Data-In Request (DMA0).
pub const CTRL_ENABLE_DMA_IN: u32 = 1 << 30;

/// Bit 29: Enable Data-Out Request (DMA1).
pub const CTRL_ENABLE_DMA_OUT: u32 = 1 << 29;

// --- MDEC0 Command Word ---

/// Bits 31..29 shift: Command opcode (`0..=7`).
pub const CMD_OPCODE_SHIFT: u32 = 29;

/// Bits 31..29 mask (after shifting): 3-bit command opcode.
pub const CMD_OPCODE_MASK: u32 = 0b111;

/// Bits 28..27 shift: Output depth in command word.
pub const CMD_DEPTH_SHIFT: u32 = 27;

/// Bit 26: Output signed flag in command word.
pub const CMD_SIGNED_BIT: u32 = 1 << 26;

/// Bit 25: Set bit 15 (STP) flag in command word.
pub const CMD_SET_BIT15_BIT: u32 = 1 << 25;

/// Bit 0 of `MDEC(2)`: `0` = Luma only (64 bytes), `1` = Luma + Chroma (128 bytes).
pub const CMD_QUANT_COLOR_BIT: u32 = 1 << 0;

/// MDEC command opcode (bits 31..29 of the `MDEC0` command word).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CommandOpcode {
    /// Opcode 0: No function (NOP).
    Nop0 = 0,
    /// Opcode 1: Decode Macroblock(s).
    DecodeMacroblock = 1,
    /// Opcode 2: Set Quant Table(s).
    SetQuantTable = 2,
    /// Opcode 3: Set Scale Table (IDCT matrix).
    SetScaleTable = 3,
    /// Opcode 4: No function (NOP).
    Nop4 = 4,
    /// Opcode 5: No function (NOP).
    Nop5 = 5,
    /// Opcode 6: No function (NOP).
    Nop6 = 6,
    /// Opcode 7: No function (NOP).
    Nop7 = 7,
}

impl CommandOpcode {
    /// Extracts the 3-bit command opcode from an `MDEC0` command word.
    #[inline]
    pub const fn from_word(word: u32) -> Self {
        match (word >> CMD_OPCODE_SHIFT) & CMD_OPCODE_MASK {
            0 => Self::Nop0,
            1 => Self::DecodeMacroblock,
            2 => Self::SetQuantTable,
            3 => Self::SetScaleTable,
            4 => Self::Nop4,
            5 => Self::Nop5,
            6 => Self::Nop6,
            _ => Self::Nop7,
        }
    }

    /// Returns the numeric 3-bit opcode (`0..=7`).
    #[inline]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Hardware pixel output depth encoded in `MDEC(1)` and `MDECSTAT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HardwareDepth {
    /// 4-bit monochrome (8×8 block = 32 bytes = 8 words).
    Bit4 = 0,
    /// 8-bit monochrome (8×8 block = 64 bytes = 16 words).
    Bit8 = 1,
    /// 24-bit RGB (16×16 macroblock = four 8×8 sub-blocks, 192 bytes / 48 words each).
    Bit24 = 2,
    /// 15-bit RGB555 (16×16 macroblock = four 8×8 sub-blocks, 128 bytes / 32 words each).
    Bit15 = 3,
}

impl HardwareDepth {
    /// Decodes the 2-bit depth field (`0..=3`).
    #[inline]
    pub const fn from_bits(bits: u32) -> Self {
        match bits & 0b11 {
            0 => Self::Bit4,
            1 => Self::Bit8,
            2 => Self::Bit24,
            _ => Self::Bit15,
        }
    }

    /// Returns the 2-bit encoding (`0..=3`).
    #[inline]
    pub const fn as_bits(self) -> u32 {
        self as u32
    }

    /// Returns `true` for monochrome modes (`Bit4` and `Bit8`).
    #[inline]
    pub const fn is_mono(self) -> bool {
        matches!(self, Self::Bit4 | Self::Bit8)
    }
}

/// Current block index reported in `MDECSTAT` bits 18..16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BlockId {
    /// Top-left luminance 8×8 block (Y1).
    Y1 = 0,
    /// Top-right luminance 8×8 block (Y2).
    Y2 = 1,
    /// Bottom-left luminance 8×8 block (Y3).
    Y3 = 2,
    /// Bottom-right luminance 8×8 block (Y4).
    Y4 = 3,
    /// Chrominance red 8×8 block (Cr), also reported in monochrome modes.
    Cr = 4,
    /// Chrominance blue 8×8 block (Cb).
    Cb = 5,
}

impl BlockId {
    /// Returns the 3-bit status register encoding (`0..=5`).
    #[inline]
    pub const fn as_bits(self) -> u32 {
        self as u32
    }
}
