//! Pure PS1 BS bitstream VLC-to-RLE frontend.
//!
//! Provides:
//! - [`BitReader`]: MSB-first reader over PS1 little-endian 16-bit halfword streams
//! - [`AcCodebook`]: validated, prefix-free AC run/level Huffman codebook
//! - [`DcCoding`]: fixed-length 10-bit DC (BS v1/v2) and predicted VLC DC (BS v3)
//! - [`vlc_to_rle`], [`vlc_to_rle_with_block_headers`], and [`vlc_ac_to_rle_with_headers`]
//! - [`vlc0_bitcodes`]: returns the 224 standard Sony `VLC0` bitcodes in table order

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use crate::rle::EOB_HALFWORD;
use crate::rle::{clamp_i11, dequant_ac};
use crate::tables::{
    Component, DcCode, AC_EOB_CODE, AC_EOB_LEN, AC_ESCAPE_CODE, AC_ESCAPE_LEN, DC_CHROMA, DC_LUMA,
    STANDARD_AC_ENTRIES, ZIGZAG,
};
use crate::Error;

/// MSB-first bit reader over a PS1 BS bitstream stored as little-endian 16-bit halfwords.
///
/// Valid bits are kept MSB-aligned at the top of `buf` (bits `63` down to
/// `64 - bits_in_buf`), with all lower bits guaranteed to be zero.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    buf: u64,
    bits_in_buf: u32,
}

impl<'a> BitReader<'a> {
    /// Creates a new [`BitReader`] over `data`.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            buf: 0,
            bits_in_buf: 0,
        }
    }

    /// Returns the total number of unread bits remaining in the stream.
    #[must_use]
    pub fn bits_remaining(&self) -> usize {
        let remaining_halfwords = (self.data.len().saturating_sub(self.byte_pos)) / 2;
        self.bits_in_buf as usize + remaining_halfwords * 16
    }

    /// Refills the internal MSB-aligned `u64` buffer with a single 32-bit load
    /// (two little-endian 16-bit halfwords) when at least 4 bytes remain.
    #[inline(always)]
    fn refill(&mut self) {
        if self.bits_in_buf <= 32 {
            if let Some(chunk) = self.data.get(self.byte_pos..self.byte_pos + 4) {
                let w = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                let swapped = ((w & 0x00FF_00FF) << 8) | ((w >> 8) & 0x00FF_00FF);
                self.buf |= u64::from(swapped) << (32 - self.bits_in_buf);
                self.bits_in_buf += 32;
                self.byte_pos += 4;
            } else {
                self.refill_slow();
            }
        }
    }

    #[cold]
    fn refill_slow(&mut self) {
        while self.bits_in_buf <= 48 && self.byte_pos + 1 < self.data.len() {
            let hw = u16::from_le_bytes([self.data[self.byte_pos], self.data[self.byte_pos + 1]]);
            self.byte_pos += 2;
            self.buf |= u64::from(hw) << (48 - self.bits_in_buf);
            self.bits_in_buf += 16;
        }
    }

    /// Peeks up to `n` (`1..=32`) bits MSB-first without consuming them.
    ///
    /// Returns `(bits, count)` where `count <= n` is the actual number of bits
    /// available and `bits` is right-aligned to `count` bits.
    #[must_use]
    pub fn peek_upto(&mut self, n: u32) -> (u32, u32) {
        debug_assert!((1..=32).contains(&n));
        if self.bits_in_buf < n {
            self.refill();
        }
        let avail = self.bits_in_buf.min(n);
        if avail == 0 {
            return (0, 0);
        }
        ((self.buf >> (64 - avail)) as u32, avail)
    }

    /// Consumes `n` bits that were previously peeked.
    #[inline(always)]
    pub fn consume(&mut self, n: u32) {
        debug_assert!(n <= self.bits_in_buf);
        self.buf = self.buf.checked_shl(n).unwrap_or(0);
        self.bits_in_buf -= n;
    }

    /// Fast internal consume for `1..=32` bits where `n <= self.bits_in_buf`.
    #[inline(always)]
    fn consume_fast(&mut self, n: u32) {
        self.buf <<= n;
        self.bits_in_buf -= n;
    }

    /// If the next `len` bits equal `code`, consumes them and returns `true`.
    pub fn try_match(&mut self, code: u32, len: u32) -> bool {
        let (bits, got) = self.peek_upto(len);
        if got == len && bits == code {
            self.consume(len);
            true
        } else {
            false
        }
    }

    /// Reads `n` (`1..=32`) unsigned bits MSB-first.
    #[inline]
    pub fn read(&mut self, n: u32) -> Result<u32, Error> {
        if !(1..=32).contains(&n) {
            return Err(Error::InvalidParameter("bit count out of range 1..=32"));
        }
        if self.bits_in_buf < n {
            self.refill();
            if self.bits_in_buf < n {
                return Err(Error::UnexpectedEof);
            }
        }
        let bits = (self.buf >> (64 - n)) as u32;
        self.consume_fast(n);
        Ok(bits)
    }

    /// Reads `n` (`1..=32`) two's-complement signed bits MSB-first.
    #[inline]
    pub fn read_signed(&mut self, n: u32) -> Result<i32, Error> {
        if !(1..=32).contains(&n) {
            return Err(Error::InvalidParameter("bit count out of range 1..=32"));
        }
        if self.bits_in_buf < n {
            self.refill();
            if self.bits_in_buf < n {
                return Err(Error::UnexpectedEof);
            }
        }
        let val = ((self.buf as i64) >> (64 - n)) as i32;
        self.consume_fast(n);
        Ok(val)
    }

    /// Fast specialized 10-bit signed DC read for BS v1/v2 blocks.
    #[inline(always)]
    pub(crate) fn read_signed_10(&mut self) -> Result<i32, Error> {
        if self.bits_in_buf < 10 {
            self.refill();
            if self.bits_in_buf < 10 {
                return Err(Error::UnexpectedEof);
            }
        }
        let val = ((self.buf as i64) >> 54) as i32;
        self.consume_fast(10);
        Ok(val)
    }
}

/// One variable-length AC run/level entry in an [`AcCodebook`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcCodeEntry {
    /// Right-aligned MSB-first bit pattern.
    pub code: u32,
    /// Number of valid bits in `code` (`1..=32`).
    pub len: u32,
    /// Number of zero AC coefficients preceding `level`.
    pub run: u32,
    /// Signed non-zero AC coefficient level.
    pub level: i32,
}

/// Escape prefix descriptor for fixed-length `(run, level)` pairs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcEscape {
    /// Right-aligned MSB-first escape prefix bit pattern.
    pub code: u32,
    /// Number of valid bits in `code` (`1..=32`).
    pub len: u32,
    /// Bit width of the unsigned run field following the escape prefix (`1..=32`).
    pub run_bits: u32,
    /// Bit width of the two's-complement signed level field (`1..=32`).
    pub level_bits: u32,
}

impl AcEscape {
    /// Standard PS1 MDEC escape: `000001` (6 bits) followed by 6-bit run and
    /// 10-bit signed level.
    pub const STANDARD: Self = Self {
        code: AC_ESCAPE_CODE,
        len: AC_ESCAPE_LEN,
        run_bits: 6,
        level_bits: 10,
    };
}

/// One decoded AC run/level event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcEvent {
    /// End of the current 8×8 block.
    Eob,
    /// A run of `run` zero coefficients followed by `level`.
    RunLevel {
        /// Number of preceding zero AC coefficients.
        run: u32,
        /// Signed AC coefficient level.
        level: i32,
    },
}

// --- Compact 9-bit primary LUT + O(1) LZCNT standard table / secondary trie ---
//
// Each 32-bit slot in `primary: [u32; 512]` and `TrieNode::child: [u32; 2]`
// uses the top 3 bits as a tag:
// - `0`: invalid / unassigned (`entry == 0`)
// - `1`: EOB; bits 4..0 = code length (1..=9 for primary, 0 for trie leaf)
// - `2`: Escape; bits 4..0 = code length (1..=9 for primary, 0 for trie leaf)
// - `3`: Inline RunLevel; bits 28..24 = len, bits 23..16 = run (0..=255),
//        bits 15..0 = level as i16 (-32768..=32767)
// - `4`: Secondary trie branch (custom codebooks); bits 28..0 = node index
// - `5`: Out-of-range RunLevel (custom codebooks); bits 28..24 = len,
//        bits 23..0 = index in `extra`
// - `6`: Standard long code (> 9 bits); resolved in O(1) via `leading_zeros()`
//        into the 176-entry (704-byte) `STANDARD_LONG_TABLE`

const PRIMARY_BITS: u32 = 9;
const PRIMARY_SIZE: usize = 1 << PRIMARY_BITS;
const STANDARD_LONG_SIZE: usize = 176; // 16 (len=11) + 5 * 32 (len=13..=17) = 704 bytes

const TAG_SHIFT: u32 = 29;
const TAG_INVALID: u32 = 0;
const TAG_EOB: u32 = 1;
const TAG_ESCAPE: u32 = 2;
const TAG_INLINE: u32 = 3;
#[cfg(feature = "alloc")]
const TAG_BRANCH: u32 = 4;
#[cfg(feature = "alloc")]
const TAG_EXTRA: u32 = 5;
const TAG_LONG: u32 = 6;

#[cfg(feature = "alloc")]
const PAYLOAD_MASK: u32 = (1 << TAG_SHIFT) - 1;

#[inline]
const fn pack_eob(len: u32) -> u32 {
    (TAG_EOB << TAG_SHIFT) | (len & 0x1F)
}

#[inline]
const fn pack_escape(len: u32) -> u32 {
    (TAG_ESCAPE << TAG_SHIFT) | (len & 0x1F)
}

#[inline]
const fn pack_inline(len: u32, run: u32, level: i16) -> u32 {
    (TAG_INLINE << TAG_SHIFT)
        | ((len & 0x1F) << 24)
        | ((run & 0xFF) << 16)
        | ((level as u16) as u32)
}

#[cfg(feature = "alloc")]
#[inline]
const fn pack_branch(node_idx: usize) -> u32 {
    (TAG_BRANCH << TAG_SHIFT) | ((node_idx as u32) & PAYLOAD_MASK)
}

#[cfg(feature = "alloc")]
#[inline]
const fn pack_extra(len: u32, extra_idx: usize) -> u32 {
    (TAG_EXTRA << TAG_SHIFT) | ((len & 0x1F) << 24) | ((extra_idx as u32) & 0x00FF_FFFF)
}

#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TrieNode {
    child: [u32; 2],
}

#[cfg(feature = "alloc")]
impl TrieNode {
    const EMPTY: Self = Self { child: [0, 0] };
}

#[derive(Debug, Clone)]
enum SecondaryTable {
    /// O(1) `leading_zeros()` flat table for the standard PS1 AC codebook (704 bytes).
    StandardLong(&'static [u32; STANDARD_LONG_SIZE]),
    /// Compact binary trie for custom codebooks built via [`AcCodebook::try_new`].
    #[cfg(feature = "alloc")]
    Trie(Vec<TrieNode>),
}

/// Validated, prefix-free AC run/level variable-length codebook.
///
/// Uses a 2 KiB 9-bit primary flat lookup table backed by an O(1) 704-byte
/// `leading_zeros()` table for the standard PS1 codebook (or a compact flat
/// binary trie for custom codebooks). Zero hashing on the decode hot path.
#[derive(Debug, Clone)]
pub struct AcCodebook {
    primary: [u32; PRIMARY_SIZE],
    secondary: SecondaryTable,
    escape: Option<AcEscape>,
    #[cfg(feature = "alloc")]
    extra: Vec<(u32, i32)>,
}

#[cfg(feature = "alloc")]
fn check_code_bits(code: u32, len: u32) -> Result<(), Error> {
    if !(1..=32).contains(&len) {
        return Err(Error::InvalidCodebook("code length must be in 1..=32"));
    }
    if len < 32 && (code >> len) != 0 {
        return Err(Error::InvalidCodebook("code has bits set above its length"));
    }
    Ok(())
}

/// Parses a `'0'`/`'1'` bit string into a right-aligned `(code, len)` pair.
const fn parse_bits(s: &str) -> (u32, u32) {
    let bytes = s.as_bytes();
    let mut code = 0u32;
    let mut i = 0;
    while i < bytes.len() {
        code = (code << 1) | if bytes[i] == b'1' { 1 } else { 0 };
        i += 1;
    }
    (code, bytes.len() as u32)
}

// Compile-time construction of the standard PS1 MDEC AC codebook (224 codes):
// 2 KiB 9-bit primary table + 704-byte O(1) LZCNT long table.
const fn build_standard_tables() -> ([u32; PRIMARY_SIZE], [u32; STANDARD_LONG_SIZE]) {
    let mut primary = [0u32; PRIMARY_SIZE];
    let mut long_tbl = [0u32; STANDARD_LONG_SIZE];

    // Insert EOB (`10`, len 2) into primary LUT.
    {
        let shift = PRIMARY_BITS - AC_EOB_LEN;
        let base = (AC_EOB_CODE << shift) as usize;
        let count = 1usize << shift;
        let val = pack_eob(AC_EOB_LEN);
        let mut j = 0;
        while j < count {
            primary[base + j] = val;
            j += 1;
        }
    }

    // Insert Escape (`000001`, len 6) into primary LUT.
    {
        let shift = PRIMARY_BITS - AC_ESCAPE_LEN;
        let base = (AC_ESCAPE_CODE << shift) as usize;
        let count = 1usize << shift;
        let val = pack_escape(AC_ESCAPE_LEN);
        let mut j = 0;
        while j < count {
            primary[base + j] = val;
            j += 1;
        }
    }

    // Insert all 222 standard AC entries.
    let mut i = 0;
    while i < STANDARD_AC_ENTRIES.len() {
        let (bits, run, level) = STANDARD_AC_ENTRIES[i];
        let (code, len) = parse_bits(bits);
        let packed = pack_inline(len, run, level as i16);
        if len <= PRIMARY_BITS {
            let shift = PRIMARY_BITS - len;
            let base = (code << shift) as usize;
            let count = 1usize << shift;
            let mut j = 0;
            while j < count {
                primary[base + j] = packed;
                j += 1;
            }
        } else {
            let top9 = (code >> (len - PRIMARY_BITS)) as usize;
            primary[top9] = TAG_LONG << TAG_SHIFT;
            // All standard PS1 AC codes with len > 9 have the form 0^(lz) 1 <payload>:
            //   len=11 -> lz=6, 4 payload bits -> offset 0..16
            //   len=13 -> lz=7, 5 payload bits -> offset 16..48
            //   len=14 -> lz=8, 5 payload bits -> offset 48..80
            //   len=15 -> lz=9, 5 payload bits -> offset 80..112
            //   len=16 -> lz=10, 5 payload bits -> offset 112..144
            //   len=17 -> lz=11, 5 payload bits -> offset 144..176
            let idx = if len == 11 {
                assert!(code >> 4 == 1);
                (code & 0xF) as usize
            } else {
                assert!(len >= 13 && len <= 17);
                assert!(code >> 5 == 1);
                let bucket = (len - 13) as usize;
                16 + bucket * 32 + ((code & 0x1F) as usize)
            };
            assert!(long_tbl[idx] == 0);
            long_tbl[idx] = packed;
        }
        i += 1;
    }

    (primary, long_tbl)
}

const STANDARD_RAW: ([u32; PRIMARY_SIZE], [u32; STANDARD_LONG_SIZE]) = build_standard_tables();

static STANDARD_AC_CODEBOOK: AcCodebook = AcCodebook {
    primary: STANDARD_RAW.0,
    secondary: SecondaryTable::StandardLong(&STANDARD_RAW.1),
    escape: Some(AcEscape::STANDARD),
    #[cfg(feature = "alloc")]
    extra: Vec::new(),
};

/// Resolves a standard PS1 AC code with `len > 9` in O(1) via `leading_zeros()`.
#[inline(always)]
fn decode_standard_long(
    br: &mut BitReader<'_>,
    long_tbl: &[u32; STANDARD_LONG_SIZE],
) -> Result<(u32, i32), Error> {
    let lz = br.buf.leading_zeros();
    if !(6..=11).contains(&lz) {
        return Err(Error::BitstreamDesync);
    }
    let (len, idx) = if lz == 6 {
        (11u32, ((br.buf >> (64 - 11)) & 0xF) as usize)
    } else {
        let len = lz + 6;
        let bucket = (lz - 7) as usize;
        (
            len,
            16 + bucket * 32 + (((br.buf >> (64 - len)) & 0x1F) as usize),
        )
    };
    if br.bits_in_buf < len {
        return Err(Error::BitstreamDesync);
    }
    let lentry = long_tbl[idx];
    if lentry == 0 {
        return Err(Error::BitstreamDesync);
    }
    br.consume_fast(len);
    Ok(((lentry >> 16) & 0xFF, i32::from((lentry & 0xFFFF) as i16)))
}

/// Fused 22-bit standard escape decoder: `000001` (6) + `run` (6) + `level` (signed 10).
#[inline(always)]
fn decode_standard_escape_fused(br: &mut BitReader<'_>) -> Result<(u32, i32), Error> {
    if br.bits_in_buf < AC_ESCAPE_LEN {
        return Err(Error::BitstreamDesync);
    }
    if br.bits_in_buf < 22 {
        br.refill();
        if br.bits_in_buf < 22 {
            return Err(Error::UnexpectedEof);
        }
    }
    let run = ((br.buf >> (64 - 12)) & 0x3F) as u32;
    let level = (((br.buf << 12) as i64) >> 54) as i32;
    br.consume_fast(22);
    Ok((run, level))
}

impl AcCodebook {
    /// Returns the standard PS1 MDEC AC codebook (precomputed at compile time,
    /// zero runtime allocation and `#![no_std]`-compatible).
    #[must_use]
    #[inline]
    pub fn standard() -> &'static Self {
        &STANDARD_AC_CODEBOOK
    }

    /// Returns `true` if this codebook is the standard PS1 MDEC AC codebook.
    #[inline]
    pub(crate) fn is_standard(&self) -> bool {
        matches!(self.secondary, SecondaryTable::StandardLong(_))
    }

    /// Constructs a custom AC codebook, validating bit widths, uniqueness, and
    /// prefix-freeness across `eob`, `escape`, and `entries`.
    #[cfg(feature = "alloc")]
    pub fn try_new(
        eob: (u32, u32),
        escape: Option<AcEscape>,
        entries: &[AcCodeEntry],
    ) -> Result<Self, Error> {
        check_code_bits(eob.0, eob.1)?;

        // Collect (len, code, leaf_value_without_len) so we can insert shortest-first
        // and detect every duplicate or prefix collision in a single pass.
        let mut items: Vec<(u32, u32, u32)> = Vec::with_capacity(entries.len() + 2);
        items.push((eob.1, eob.0, pack_eob(0)));

        if let Some(esc) = escape {
            check_code_bits(esc.code, esc.len)?;
            if !(1..=32).contains(&esc.run_bits) || !(1..=32).contains(&esc.level_bits) {
                return Err(Error::InvalidCodebook(
                    "escape run/level bit widths must be in 1..=32",
                ));
            }
            items.push((esc.len, esc.code, pack_escape(0)));
        }

        let mut extra: Vec<(u32, i32)> = Vec::new();
        for entry in entries {
            check_code_bits(entry.code, entry.len)?;
            let leaf = if entry.run <= u32::from(u8::MAX)
                && (i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&entry.level)
            {
                pack_inline(0, entry.run, entry.level as i16)
            } else {
                let idx = extra.len();
                if idx > 0x00FF_FFFF {
                    return Err(Error::InvalidCodebook("too many extended codebook entries"));
                }
                extra.push((entry.run, entry.level));
                pack_extra(0, idx)
            };
            items.push((entry.len, entry.code, leaf));
        }

        items.sort_unstable_by_key(|&(len, code, _)| (len, code));

        let mut primary = [0u32; PRIMARY_SIZE];
        let mut trie: Vec<TrieNode> = Vec::new();

        for (len, code, leaf) in items {
            if len <= PRIMARY_BITS {
                let primary_val = match leaf >> TAG_SHIFT {
                    TAG_EOB => pack_eob(len),
                    TAG_ESCAPE => pack_escape(len),
                    TAG_INLINE | TAG_EXTRA => leaf | ((len & 0x1F) << 24),
                    _ => unreachable!(),
                };
                let shift = PRIMARY_BITS - len;
                let base = (code << shift) as usize;
                let count = 1usize << shift;
                for slot in &mut primary[base..base + count] {
                    if *slot != 0 {
                        return Err(Error::InvalidCodebook(
                            "AC codebook is not prefix-free (duplicate or prefix collision)",
                        ));
                    }
                    *slot = primary_val;
                }
            } else {
                let top9 = (code >> (len - PRIMARY_BITS)) as usize;
                let mut node_idx = match primary[top9] >> TAG_SHIFT {
                    TAG_INVALID => {
                        let idx = trie.len();
                        trie.push(TrieNode::EMPTY);
                        primary[top9] = pack_branch(idx);
                        idx
                    }
                    TAG_BRANCH => (primary[top9] & PAYLOAD_MASK) as usize,
                    _ => {
                        return Err(Error::InvalidCodebook(
                            "AC codebook is not prefix-free (duplicate or prefix collision)",
                        ));
                    }
                };

                let mut bit_pos = len - PRIMARY_BITS - 1;
                loop {
                    let bit = ((code >> bit_pos) & 1) as usize;
                    if bit_pos == 0 {
                        if trie[node_idx].child[bit] != 0 {
                            return Err(Error::InvalidCodebook(
                                "AC codebook is not prefix-free (duplicate or prefix collision)",
                            ));
                        }
                        trie[node_idx].child[bit] = leaf;
                        break;
                    }
                    let child = trie[node_idx].child[bit];
                    match child >> TAG_SHIFT {
                        TAG_INVALID => {
                            let idx = trie.len();
                            trie.push(TrieNode::EMPTY);
                            trie[node_idx].child[bit] = pack_branch(idx);
                            node_idx = idx;
                        }
                        TAG_BRANCH => {
                            node_idx = (child & PAYLOAD_MASK) as usize;
                        }
                        _ => {
                            return Err(Error::InvalidCodebook(
                                "AC codebook is not prefix-free (duplicate or prefix collision)",
                            ));
                        }
                    }
                    bit_pos -= 1;
                }
            }
        }

        Ok(Self {
            primary,
            secondary: SecondaryTable::Trie(trie),
            escape,
            extra,
        })
    }

    #[inline]
    fn decode_escape(&self, br: &mut BitReader<'_>) -> Result<AcEvent, Error> {
        let esc = self.escape.ok_or(Error::BitstreamDesync)?;
        let run = br.read(esc.run_bits)?;
        let level = br.read_signed(esc.level_bits)?;
        Ok(AcEvent::RunLevel { run, level })
    }

    #[cfg(feature = "alloc")]
    #[inline(never)]
    fn decode_trie(&self, br: &mut BitReader<'_>, mut node_idx: usize) -> Result<AcEvent, Error> {
        let SecondaryTable::Trie(ref nodes) = self.secondary else {
            return Err(Error::BitstreamDesync);
        };
        loop {
            if br.bits_in_buf == 0 {
                br.refill();
                if br.bits_in_buf == 0 {
                    return Err(Error::BitstreamDesync);
                }
            }
            let bit = (br.buf >> 63) as usize;
            br.buf <<= 1;
            br.bits_in_buf -= 1;

            let entry = nodes[node_idx].child[bit];
            match entry >> TAG_SHIFT {
                TAG_INVALID => return Err(Error::BitstreamDesync),
                TAG_EOB => return Ok(AcEvent::Eob),
                TAG_ESCAPE => return self.decode_escape(br),
                TAG_INLINE => {
                    let run = (entry >> 16) & 0xFF;
                    let level = i32::from((entry & 0xFFFF) as i16);
                    return Ok(AcEvent::RunLevel { run, level });
                }
                TAG_EXTRA => {
                    let idx = (entry & 0x00FF_FFFF) as usize;
                    let (run, level) = self.extra[idx];
                    return Ok(AcEvent::RunLevel { run, level });
                }
                TAG_BRANCH => {
                    node_idx = (entry & PAYLOAD_MASK) as usize;
                }
                _ => return Err(Error::BitstreamDesync),
            }
        }
    }

    /// Decodes the next AC event from `br`, resolving escapes to `RunLevel`.
    #[inline]
    pub fn decode(&self, br: &mut BitReader<'_>) -> Result<AcEvent, Error> {
        if br.bits_in_buf < 17 {
            br.refill();
            if br.bits_in_buf == 0 {
                return Err(Error::BitstreamDesync);
            }
        }

        let idx = (br.buf >> (64 - PRIMARY_BITS)) as usize;
        let entry = self.primary[idx];
        match entry >> TAG_SHIFT {
            TAG_INVALID => Err(Error::BitstreamDesync),
            TAG_INLINE => {
                let len = (entry >> 24) & 0x1F;
                if br.bits_in_buf < len {
                    return Err(Error::BitstreamDesync);
                }
                br.consume_fast(len);
                let run = (entry >> 16) & 0xFF;
                let level = i32::from((entry & 0xFFFF) as i16);
                Ok(AcEvent::RunLevel { run, level })
            }
            TAG_EOB => {
                let len = entry & 0x1F;
                if br.bits_in_buf < len {
                    return Err(Error::BitstreamDesync);
                }
                br.consume_fast(len);
                Ok(AcEvent::Eob)
            }
            TAG_LONG => {
                let long_tbl = match self.secondary {
                    SecondaryTable::StandardLong(tbl) => tbl,
                    #[cfg(feature = "alloc")]
                    SecondaryTable::Trie(_) => return Err(Error::BitstreamDesync),
                };
                let (run, level) = decode_standard_long(br, long_tbl)?;
                Ok(AcEvent::RunLevel { run, level })
            }
            TAG_ESCAPE => {
                if self.escape == Some(AcEscape::STANDARD) {
                    let (run, level) = decode_standard_escape_fused(br)?;
                    return Ok(AcEvent::RunLevel { run, level });
                }
                let len = entry & 0x1F;
                if br.bits_in_buf < len {
                    return Err(Error::BitstreamDesync);
                }
                br.consume_fast(len);
                self.decode_escape(br)
            }
            #[cfg(feature = "alloc")]
            TAG_EXTRA => {
                let len = (entry >> 24) & 0x1F;
                if br.bits_in_buf < len {
                    return Err(Error::BitstreamDesync);
                }
                br.consume_fast(len);
                let extra_idx = (entry & 0x00FF_FFFF) as usize;
                let (run, level) = self.extra[extra_idx];
                Ok(AcEvent::RunLevel { run, level })
            }
            #[cfg(feature = "alloc")]
            TAG_BRANCH => {
                if br.bits_in_buf < PRIMARY_BITS {
                    return Err(Error::BitstreamDesync);
                }
                br.consume_fast(PRIMARY_BITS);
                self.decode_trie(br, (entry & PAYLOAD_MASK) as usize)
            }
            _ => Err(Error::BitstreamDesync),
        }
    }
}

/// Returns the 224 standard PS1 MDEC AC bitcodes in Sony `VLC0` table order:
/// - index `0`: 2-bit EOB (`10`)
/// - indices `1..=22`: 3-to-7-bit run/level codes
/// - index `23`: 6-bit escape prefix (`000001`)
/// - indices `24..=223`: 8-to-17-bit run/level codes
#[must_use]
pub fn vlc0_bitcodes() -> [(u32, u32); 224] {
    let mut out = [(0u32, 0u32); 224];
    out[0] = (AC_EOB_CODE, AC_EOB_LEN);
    for (slot, &(bits, _, _)) in out[1..23].iter_mut().zip(&STANDARD_AC_ENTRIES[..22]) {
        *slot = parse_bits(bits);
    }
    out[23] = (AC_ESCAPE_CODE, AC_ESCAPE_LEN);
    for (slot, &(bits, _, _)) in out[24..224].iter_mut().zip(&STANDARD_AC_ENTRIES[22..]) {
        *slot = parse_bits(bits);
    }
    out
}

/// DC coefficient coding mode for PS1 BS bitstreams.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DcCoding {
    /// Fixed-length 10-bit signed DC per block (BS v1 / v2).
    FixedLength,
    /// Differential Huffman-coded DC with per-component predictors (BS v3).
    PredictedVlc,
}

/// Builds a 256-entry 8-bit lookup table for a BS v3 DC-difference Huffman table.
///
/// Each byte encodes `(len << 4) | width` (or `0` for invalid prefixes).
const fn build_dc_lut(table: &[DcCode; 9]) -> [u8; 256] {
    let mut lut = [0u8; 256];
    let mut i = 0;
    while i < table.len() {
        let entry = table[i];
        let shift = 8 - entry.len;
        let base = (entry.code << shift) as usize;
        let count = 1usize << shift;
        let packed = ((entry.len as u8) << 4) | (entry.width as u8);
        let mut j = 0;
        while j < count {
            lut[base + j] = packed;
            j += 1;
        }
        i += 1;
    }
    lut
}

const DC_LUMA_LUT: [u8; 256] = build_dc_lut(&DC_LUMA);
const DC_CHROMA_LUT: [u8; 256] = build_dc_lut(&DC_CHROMA);

/// Decodes a BS v3 differential DC value using a 256-entry lookup table
/// with a single combined refill check (max 8-bit code + 8-bit magnitude = 16 bits).
#[inline]
pub(crate) fn decode_dc_diff(br: &mut BitReader<'_>, component: Component) -> Result<i32, Error> {
    let lut = match component {
        Component::Luma => &DC_LUMA_LUT,
        Component::Chroma => &DC_CHROMA_LUT,
    };
    if br.bits_in_buf < 16 {
        br.refill();
        if br.bits_in_buf == 0 {
            return Err(Error::BitstreamDesync);
        }
    }
    let packed = lut[(br.buf >> 56) as usize];
    if packed == 0 {
        return Err(Error::BitstreamDesync);
    }
    let len = u32::from(packed >> 4);
    if br.bits_in_buf < len {
        return Err(Error::BitstreamDesync);
    }
    br.consume_fast(len);

    let width = u32::from(packed & 0x0F);
    if width == 0 {
        return Ok(0);
    }
    if br.bits_in_buf < width {
        return Err(Error::UnexpectedEof);
    }
    let raw = (br.buf >> (64 - width)) as u32;
    br.consume_fast(width);
    let half = 1u32 << (width - 1);
    let diff = if raw < half {
        raw as i32 - ((1i32 << width) - 1)
    } else {
        raw as i32
    };
    Ok(diff)
}

/// Fused streaming VLC -> dequantization for one 8×8 block using the standard
/// PS1 AC codebook, writing directly into `coeffs` with zero heap allocation.
///
/// Returns `true` if any AC coefficient was decoded.
#[inline]
pub(crate) fn decode_block_dequant_standard(
    br: &mut BitReader<'_>,
    q_scale: i32,
    dc: i32,
    quant_zigzag: &[u8; 64],
    coeffs: &mut [i16; 64],
) -> Result<bool, Error> {
    coeffs.fill(0);
    coeffs[0] = if q_scale == 0 {
        clamp_i11(dc * 2)
    } else {
        clamp_i11(dc * i32::from(quant_zigzag[0]))
    };

    let primary = &STANDARD_RAW.0;
    let long_tbl = &STANDARD_RAW.1;
    let mut k = 0usize;
    let mut has_ac = false;

    loop {
        if br.bits_in_buf < 17 {
            br.refill();
            if br.bits_in_buf == 0 {
                return Err(Error::BitstreamDesync);
            }
        }
        let top9 = (br.buf >> (64 - PRIMARY_BITS)) as usize;
        let entry = primary[top9];
        let tag = entry >> TAG_SHIFT;

        let (run, level) = if tag == TAG_INLINE {
            let len = (entry >> 24) & 0x1F;
            if br.bits_in_buf < len {
                return Err(Error::BitstreamDesync);
            }
            br.consume_fast(len);
            (
                ((entry >> 16) & 0xFF) as usize,
                i32::from((entry & 0xFFFF) as i16),
            )
        } else if tag == TAG_EOB {
            let len = entry & 0x1F;
            if br.bits_in_buf < len {
                return Err(Error::BitstreamDesync);
            }
            br.consume_fast(len);
            break;
        } else if tag == TAG_LONG {
            let (r, l) = decode_standard_long(br, long_tbl)?;
            (r as usize, l)
        } else if tag == TAG_ESCAPE {
            let (r, l) = decode_standard_escape_fused(br)?;
            (r as usize, l)
        } else {
            return Err(Error::BitstreamDesync);
        };

        k += run + 1;
        if k >= 64 {
            return Err(Error::BitstreamDesync);
        }
        has_ac = true;
        if q_scale == 0 {
            coeffs[k] = clamp_i11(level * 2);
        } else {
            coeffs[usize::from(ZIGZAG[k])] = dequant_ac(level, quant_zigzag[k], q_scale);
        }
    }

    Ok(has_ac)
}

/// Fused streaming VLC -> dequantization for one 8×8 block using an arbitrary
/// caller-supplied [`AcCodebook`], writing directly into `coeffs` with zero
/// heap allocation.
///
/// Masks and sign-extends `level` to 10-bit signed for 100% bit-exact parity
/// with the two-stage `vlc_to_rle` -> `decode_rle_block` path.
#[inline]
pub(crate) fn decode_block_dequant_custom(
    br: &mut BitReader<'_>,
    codebook: &AcCodebook,
    q_scale: i32,
    dc: i32,
    quant_zigzag: &[u8; 64],
    coeffs: &mut [i16; 64],
) -> Result<bool, Error> {
    coeffs.fill(0);
    coeffs[0] = if q_scale == 0 {
        clamp_i11(dc * 2)
    } else {
        clamp_i11(dc * i32::from(quant_zigzag[0]))
    };

    let mut k = 0usize;
    let mut has_ac = false;

    loop {
        match codebook.decode(br)? {
            AcEvent::Eob => break,
            AcEvent::RunLevel { run, level } => {
                if run > 63 {
                    return Err(Error::BitstreamDesync);
                }
                k += (run as usize) + 1;
                if k >= 64 {
                    return Err(Error::BitstreamDesync);
                }
                has_ac = true;
                // Mask and sign-extend to 10-bit signed (-512..=511) to match
                // hardware 16-bit RLE halfword packing.
                let level10 = (level << 22) >> 22;
                if q_scale == 0 {
                    coeffs[k] = clamp_i11(level10 * 2);
                } else {
                    coeffs[usize::from(ZIGZAG[k])] = dequant_ac(level10, quant_zigzag[k], q_scale);
                }
            }
        }
    }

    Ok(has_ac)
}

/// Packs `(top6, signed10)` into a hardware MDEC 16-bit RLE halfword.
#[cfg(feature = "alloc")]
#[inline]
const fn pack_rle_halfword(top6: u16, val10: i32) -> u16 {
    ((top6 & 0x3F) << 10) | ((val10 as u16) & 0x3FF)
}

/// Decodes a PS1 BS VLC bitstream into hardware MDEC 16-bit RLE halfwords.
///
/// Decodes `macroblocks` 16×16 macroblocks (`6` blocks per macroblock:
/// `Cr, Cb, Y0, Y1, Y2, Y3`) using frame-level `q_scale` (`0..=63`) and
/// appends the resulting RLE halfwords to `out`.
#[cfg(feature = "alloc")]
pub fn vlc_to_rle(
    bitstream: &[u8],
    q_scale: u16,
    dc_coding: DcCoding,
    codebook: &AcCodebook,
    macroblocks: usize,
    out: &mut Vec<u16>,
) -> Result<(), Error> {
    if q_scale > 63 {
        return Err(Error::InvalidParameter("q_scale must be in 0..=63"));
    }

    let mut br = BitReader::new(bitstream);
    let mut pred_cr = 0i32;
    let mut pred_cb = 0i32;
    let mut pred_y = 0i32;

    for _ in 0..macroblocks {
        for block_idx in 0..6 {
            let component = if block_idx < 2 {
                Component::Chroma
            } else {
                Component::Luma
            };

            let dc = match dc_coding {
                DcCoding::FixedLength => br.read_signed_10()?,
                DcCoding::PredictedVlc => {
                    // BS v3 DC diffs are in spatial units; pre-qt[0] RLE DC
                    // scales by 4 so `diff * 4 * qt[0](2) / 8 == diff`.
                    let delta = decode_dc_diff(&mut br, component)? * 4;
                    let pred = match block_idx {
                        0 => {
                            pred_cr += delta;
                            pred_cr
                        }
                        1 => {
                            pred_cb += delta;
                            pred_cb
                        }
                        _ => {
                            pred_y += delta;
                            pred_y
                        }
                    };
                    pred.clamp(-512, 511)
                }
            };

            out.push(pack_rle_halfword(q_scale, dc));
            decode_ac_block(&mut br, codebook, out)?;
        }
    }

    Ok(())
}

/// Decodes a PS1 BS VLC bitstream where every block begins with a 16-bit
/// `(q_scale: 6 bits, dc: 10 signed bits)` header into hardware MDEC 16-bit
/// RLE halfwords.
#[cfg(feature = "alloc")]
pub fn vlc_to_rle_with_block_headers(
    bitstream: &[u8],
    codebook: &AcCodebook,
    macroblocks: usize,
    out: &mut Vec<u16>,
) -> Result<(), Error> {
    let mut br = BitReader::new(bitstream);

    for _ in 0..macroblocks {
        for _ in 0..6 {
            let q_scale = br.read(6)? as u16;
            let dc = br.read_signed_10()?;
            out.push(pack_rle_halfword(q_scale, dc));
            decode_ac_block(&mut br, codebook, out)?;
        }
    }

    Ok(())
}

/// Decodes an AC-only PS1 BS bitstream paired with out-of-band `(q_scale, dc)`
/// per-block headers (such as `Iki` frames) into hardware MDEC 16-bit RLE halfwords.
///
/// Each `(q_scale, dc)` entry supplies the 6-bit quantization scale (`0..=63`)
/// and 10-bit signed DC value for one 8×8 block, followed by AC run/level
/// symbols decoded from `ac_bitstream` using `codebook`.
#[cfg(feature = "alloc")]
pub fn vlc_ac_to_rle_with_headers(
    ac_bitstream: &[u8],
    block_headers: &[(u16, i32)],
    codebook: &AcCodebook,
    out: &mut Vec<u16>,
) -> Result<(), Error> {
    let mut br = BitReader::new(ac_bitstream);

    for &(q_scale, dc) in block_headers {
        if q_scale > 63 {
            return Err(Error::InvalidParameter("q_scale must be in 0..=63"));
        }
        out.push(pack_rle_halfword(q_scale, dc.clamp(-512, 511)));
        decode_ac_block(&mut br, codebook, out)?;
    }

    Ok(())
}

#[cfg(feature = "alloc")]
fn decode_ac_block(
    br: &mut BitReader<'_>,
    codebook: &AcCodebook,
    out: &mut Vec<u16>,
) -> Result<(), Error> {
    let mut k = 0u32;
    loop {
        match codebook.decode(br)? {
            AcEvent::Eob => {
                out.push(EOB_HALFWORD);
                break;
            }
            AcEvent::RunLevel { run, level } => {
                if run > 63 {
                    return Err(Error::BitstreamDesync);
                }
                k += run + 1;
                if k >= 64 {
                    return Err(Error::BitstreamDesync);
                }
                out.push(pack_rle_halfword(run as u16, level));
            }
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;
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
            let hw = (acc << (16 - n)) as u16;
            out.extend_from_slice(&hw.to_le_bytes());
        }
        out
    }

    #[test]
    fn vlc0_bitcodes_has_224_entries_with_eob_and_escape() {
        let codes = vlc0_bitcodes();
        assert_eq!(codes.len(), 224);
        assert_eq!(codes[0], (AC_EOB_CODE, AC_EOB_LEN));
        assert_eq!(codes[23], (AC_ESCAPE_CODE, AC_ESCAPE_LEN));
    }

    #[test]
    fn standard_codebook_decodes_eob_entry_and_escape() {
        let cb = AcCodebook::standard();
        // 110 -> (0, 1), 000001 + 000101 (5) + 1111111101 (-3), 10 -> EOB
        let data = pack_bits("110 000001 000101 1111111101 10");
        let mut br = BitReader::new(&data);
        assert_eq!(
            cb.decode(&mut br).unwrap(),
            AcEvent::RunLevel { run: 0, level: 1 }
        );
        assert_eq!(
            cb.decode(&mut br).unwrap(),
            AcEvent::RunLevel { run: 5, level: -3 }
        );
        assert_eq!(cb.decode(&mut br).unwrap(), AcEvent::Eob);
    }

    #[test]
    fn standard_codebook_matches_dynamic_try_new_for_all_224_codes() {
        let entries: Vec<AcCodeEntry> = STANDARD_AC_ENTRIES
            .iter()
            .map(|&(bits, run, level)| {
                let (code, len) = parse_bits(bits);
                AcCodeEntry {
                    code,
                    len,
                    run,
                    level,
                }
            })
            .collect();
        let dyn_cb = AcCodebook::try_new(
            (AC_EOB_CODE, AC_EOB_LEN),
            Some(AcEscape::STANDARD),
            &entries,
        )
        .unwrap();
        let std_cb = AcCodebook::standard();

        for &(bits, run, level) in &STANDARD_AC_ENTRIES {
            let mut stream = String::from(bits);
            stream.push_str("10");
            let data = pack_bits(&stream);

            let mut br_std = BitReader::new(&data);
            let mut br_dyn = BitReader::new(&data);
            assert_eq!(
                std_cb.decode(&mut br_std).unwrap(),
                AcEvent::RunLevel { run, level }
            );
            assert_eq!(
                dyn_cb.decode(&mut br_dyn).unwrap(),
                AcEvent::RunLevel { run, level }
            );
            assert_eq!(std_cb.decode(&mut br_std).unwrap(), AcEvent::Eob);
            assert_eq!(dyn_cb.decode(&mut br_dyn).unwrap(), AcEvent::Eob);
        }
    }

    #[test]
    fn custom_codebook_rejects_prefix_collision() {
        let res = AcCodebook::try_new(
            (0b0, 1),
            None,
            &[AcCodeEntry {
                code: 0b01,
                len: 2,
                run: 0,
                level: 1,
            }],
        );
        assert!(res.is_err());
    }

    #[test]
    fn vlc_to_rle_fixed_and_predicted_dc() {
        // 1 macroblock = 6 blocks, each with DC=0 and immediate EOB (`10`).
        // FixedLength: 6 * (10 zero bits + "10") = 72 bits.
        let mut bits = String::new();
        for _ in 0..6 {
            bits.push_str("0000000000 10");
        }
        let data = pack_bits(&bits);
        let mut rle = Vec::new();
        vlc_to_rle(
            &data,
            4,
            DcCoding::FixedLength,
            AcCodebook::standard(),
            1,
            &mut rle,
        )
        .unwrap();
        assert_eq!(rle.len(), 12);
        for pair in rle.chunks_exact(2) {
            assert_eq!(pair[0], 4 << 10);
            assert_eq!(pair[1], EOB_HALFWORD);
        }

        // PredictedVlc (BS v3): diffs are scaled by 4 in the pre-qt[0] RLE DC domain.
        // Cr diff=0 ("00"), Cb diff=-1 ("01" width=1, "0"),
        // Y0 diff=+1 ("00" width=1, "1"), Y1 diff=0 ("100"),
        // Y2 diff=+2 ("01" width=2, "10"), Y3 diff=0 ("100"), each followed by EOB ("10").
        let bits_v3 = "00 10  01 0 10  00 1 10  100 10  01 10 10  100 10";
        let data_v3 = pack_bits(bits_v3);
        let mut rle_v3 = Vec::new();
        vlc_to_rle(
            &data_v3,
            2,
            DcCoding::PredictedVlc,
            AcCodebook::standard(),
            1,
            &mut rle_v3,
        )
        .unwrap();
        assert_eq!(rle_v3[0], 2 << 10); // Cr DC = 0
        assert_eq!(rle_v3[2], (2 << 10) | ((-4i32 as u16) & 0x3FF)); // Cb DC = -1 * 4 = -4
        assert_eq!(rle_v3[4], (2 << 10) | 4); // Y0 DC = +1 * 4 = +4
        assert_eq!(rle_v3[6], (2 << 10) | 4); // Y1 DC = +4 (predicted)
        assert_eq!(rle_v3[8], (2 << 10) | 12); // Y2 DC = +4 + 2*4 = +12
        assert_eq!(rle_v3[10], (2 << 10) | 12); // Y3 DC = +12 (predicted)
    }

    #[test]
    fn vlc_ac_to_rle_with_headers_decodes_out_of_band_headers() {
        // Block 0: AC (0, 1) -> "110", EOB -> "10"
        // Block 1: immediate EOB -> "10"
        let ac_data = pack_bits("110 10 10");
        let headers = [(3u16, 16i32), (5u16, -8i32)];
        let mut rle = Vec::new();
        vlc_ac_to_rle_with_headers(&ac_data, &headers, AcCodebook::standard(), &mut rle).unwrap();
        assert_eq!(
            rle,
            alloc::vec![
                (3 << 10) | 16,
                1,
                EOB_HALFWORD,
                (5 << 10) | ((-8i32 as u16) & 0x3FF),
                EOB_HALFWORD,
            ]
        );

        // Invalid q_scale > 63 is rejected.
        let err =
            vlc_ac_to_rle_with_headers(&ac_data, &[(64, 0)], AcCodebook::standard(), &mut rle)
                .unwrap_err();
        assert!(matches!(err, Error::InvalidParameter(_)));
    }
}
