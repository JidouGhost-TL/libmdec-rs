//! Pure PS1 BS bitstream VLC-to-RLE frontend.
//!
//! Provides:
//! - [`BitReader`]: MSB-first reader over PS1 little-endian 16-bit halfword streams
//! - [`AcCodebook`]: validated, prefix-free AC run/level Huffman codebook
//! - [`DcCoding`]: fixed-length 10-bit DC (BS v1/v2) and predicted VLC DC (BS v3)
//! - [`vlc_to_rle`] and [`vlc_to_rle_with_block_headers`]
//! - [`vlc0_bitcodes`]: returns the 224 standard Sony `VLC0` bitcodes in table order

use crate::rle::EOB_HALFWORD;
use crate::tables::{
    dc_table, Component, AC_EOB_CODE, AC_EOB_LEN, AC_ESCAPE_CODE, AC_ESCAPE_LEN,
    STANDARD_AC_ENTRIES,
};
use crate::Error;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// MSB-first bit reader over a PS1 BS bitstream stored as little-endian 16-bit halfwords.
#[derive(Debug, Clone)]
pub struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    buf: u64,
    bits_in_buf: u32,
}

impl<'a> BitReader<'a> {
    /// Creates a new [`BitReader`] over `data`.
    pub const fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            buf: 0,
            bits_in_buf: 0,
        }
    }

    /// Returns the total number of unread bits remaining in the stream.
    pub fn bits_remaining(&self) -> usize {
        let remaining_halfwords = (self.data.len().saturating_sub(self.byte_pos)) / 2;
        self.bits_in_buf as usize + remaining_halfwords * 16
    }

    /// Ensures the internal bit buffer holds at least `n` bits (up to 32) if
    /// enough halfwords remain in `self.data`.
    fn refill(&mut self, n: u32) {
        while self.bits_in_buf < n && self.byte_pos + 1 < self.data.len() {
            let hw = u16::from_le_bytes([self.data[self.byte_pos], self.data[self.byte_pos + 1]]);
            self.byte_pos += 2;
            self.buf = (self.buf << 16) | u64::from(hw);
            self.bits_in_buf += 16;
        }
    }

    /// Peeks up to `n` (`1..=32`) bits MSB-first without consuming them.
    ///
    /// Returns `(bits, count)` where `count <= n` is the actual number of bits
    /// available and `bits` is right-aligned to `count` bits.
    pub fn peek_upto(&mut self, n: u32) -> (u32, u32) {
        debug_assert!((1..=32).contains(&n));
        self.refill(n);
        let avail = self.bits_in_buf.min(n);
        if avail == 0 {
            return (0, 0);
        }
        let shift = self.bits_in_buf - avail;
        let mask = if avail == 32 {
            u32::MAX
        } else {
            (1u32 << avail) - 1
        };
        (((self.buf >> shift) as u32) & mask, avail)
    }

    /// Consumes `n` bits that were previously peeked.
    #[inline]
    pub fn consume(&mut self, n: u32) {
        debug_assert!(n <= self.bits_in_buf);
        self.bits_in_buf -= n;
        if self.bits_in_buf == 0 {
            self.buf = 0;
        } else {
            self.buf &= (1u64 << self.bits_in_buf) - 1;
        }
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
    pub fn read(&mut self, n: u32) -> Result<u32, Error> {
        if !(1..=32).contains(&n) {
            return Err(Error::InvalidParameter("bit count out of range 1..=32"));
        }
        let (bits, got) = self.peek_upto(n);
        if got < n {
            return Err(Error::UnexpectedEof);
        }
        self.consume(n);
        Ok(bits)
    }

    /// Reads `n` (`1..=32`) two's-complement signed bits MSB-first.
    pub fn read_signed(&mut self, n: u32) -> Result<i32, Error> {
        let raw = self.read(n)?;
        if n == 32 {
            return Ok(raw as i32);
        }
        let sign_bit = 1u32 << (n - 1);
        if (raw & sign_bit) != 0 {
            let ext = raw | !((1u32 << n) - 1);
            Ok(ext as i32)
        } else {
            Ok(raw as i32)
        }
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

/// Validated, prefix-free AC run/level variable-length codebook.
#[derive(Debug, Clone)]
pub struct AcCodebook {
    eob_code: u32,
    eob_len: u32,
    escape: Option<AcEscape>,
    lengths: Vec<u32>,
    map: HashMap<(u32, u32), (u32, i32)>,
}

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

impl AcCodebook {
    /// Returns the standard PS1 MDEC AC codebook (built once and cached globally).
    pub fn standard() -> &'static Self {
        static STANDARD: OnceLock<AcCodebook> = OnceLock::new();
        STANDARD.get_or_init(|| {
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
            Self::try_new(
                (AC_EOB_CODE, AC_EOB_LEN),
                Some(AcEscape::STANDARD),
                &entries,
            )
            .expect("built-in standard PS1 MDEC AC codebook is valid and prefix-free")
        })
    }

    /// Constructs a custom AC codebook, validating bit widths, uniqueness, and
    /// prefix-freeness across `eob`, `escape`, and `entries`.
    pub fn try_new(
        eob: (u32, u32),
        escape: Option<AcEscape>,
        entries: &[AcCodeEntry],
    ) -> Result<Self, Error> {
        check_code_bits(eob.0, eob.1)?;
        let mut all_codes: Vec<(u32, u32)> = Vec::with_capacity(entries.len() + 2);
        all_codes.push((eob.1, eob.0));

        if let Some(esc) = escape {
            check_code_bits(esc.code, esc.len)?;
            if !(1..=32).contains(&esc.run_bits) || !(1..=32).contains(&esc.level_bits) {
                return Err(Error::InvalidCodebook(
                    "escape run/level bit widths must be in 1..=32",
                ));
            }
            all_codes.push((esc.len, esc.code));
        }

        let mut map = HashMap::with_capacity(entries.len());
        let mut length_set: Vec<u32> = Vec::new();
        for entry in entries {
            check_code_bits(entry.code, entry.len)?;
            if map
                .insert((entry.len, entry.code), (entry.run, entry.level))
                .is_some()
            {
                return Err(Error::InvalidCodebook("duplicate AC codebook entry"));
            }
            if !length_set.contains(&entry.len) {
                length_set.push(entry.len);
            }
            all_codes.push((entry.len, entry.code));
        }

        // Verify prefix-freeness across EOB, escape, and all entries.
        all_codes.sort_unstable();
        let mut seen: HashSet<(u32, u32)> = HashSet::with_capacity(all_codes.len());
        for &(len, code) in &all_codes {
            for prefix_len in 1..=len {
                let prefix = code >> (len - prefix_len);
                if seen.contains(&(prefix_len, prefix)) {
                    return Err(Error::InvalidCodebook(
                        "AC codebook is not prefix-free (duplicate or prefix collision)",
                    ));
                }
            }
            seen.insert((len, code));
        }

        length_set.sort_unstable();
        Ok(Self {
            eob_code: eob.0,
            eob_len: eob.1,
            escape,
            lengths: length_set,
            map,
        })
    }

    /// Decodes the next AC event from `br`, resolving escapes to `RunLevel`.
    pub fn decode(&self, br: &mut BitReader<'_>) -> Result<AcEvent, Error> {
        if br.try_match(self.eob_code, self.eob_len) {
            return Ok(AcEvent::Eob);
        }
        if let Some(esc) = self.escape {
            if br.try_match(esc.code, esc.len) {
                let run = br.read(esc.run_bits)?;
                let level = br.read_signed(esc.level_bits)?;
                return Ok(AcEvent::RunLevel { run, level });
            }
        }
        for &len in &self.lengths {
            let (bits, got) = br.peek_upto(len);
            if got == len {
                if let Some(&(run, level)) = self.map.get(&(len, bits)) {
                    br.consume(len);
                    return Ok(AcEvent::RunLevel { run, level });
                }
            }
        }
        Err(Error::BitstreamDesync)
    }
}

/// Returns the 224 standard PS1 MDEC AC bitcodes in Sony `VLC0` table order:
/// - index `0`: 2-bit EOB (`10`)
/// - indices `1..=22`: 3-to-7-bit run/level codes
/// - index `23`: 6-bit escape prefix (`000001`)
/// - indices `24..=223`: 8-to-17-bit run/level codes
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

/// Decodes a BS v3 differential DC value using the component's Huffman table.
fn decode_dc_diff(br: &mut BitReader<'_>, component: Component) -> Result<i32, Error> {
    let table = dc_table(component);
    for entry in table {
        if br.try_match(entry.code, entry.len) {
            if entry.width == 0 {
                return Ok(0);
            }
            let raw = br.read(entry.width)?;
            let half = 1u32 << (entry.width - 1);
            let diff = if raw < half {
                raw as i32 - ((1i32 << entry.width) - 1)
            } else {
                raw as i32
            };
            return Ok(diff);
        }
    }
    Err(Error::BitstreamDesync)
}

/// Packs `(top6, signed10)` into a hardware MDEC 16-bit RLE halfword.
#[inline]
const fn pack_rle_halfword(top6: u16, val10: i32) -> u16 {
    ((top6 & 0x3F) << 10) | ((val10 as u16) & 0x3FF)
}

/// Decodes a PS1 BS VLC bitstream into hardware MDEC 16-bit RLE halfwords.
///
/// Decodes `macroblocks` 16×16 macroblocks (`6` blocks per macroblock:
/// `Cr, Cb, Y0, Y1, Y2, Y3`) using frame-level `q_scale` (`0..=63`) and
/// appends the resulting RLE halfwords to `out`.
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
                DcCoding::FixedLength => br.read_signed(10)?,
                DcCoding::PredictedVlc => {
                    let diff = decode_dc_diff(&mut br, component)?;
                    let pred = match block_idx {
                        0 => {
                            pred_cr += diff;
                            pred_cr
                        }
                        1 => {
                            pred_cb += diff;
                            pred_cb
                        }
                        _ => {
                            pred_y += diff;
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
            let dc = br.read_signed(10)?;
            out.push(pack_rle_halfword(q_scale, dc));
            decode_ac_block(&mut br, codebook, out)?;
        }
    }

    Ok(())
}

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

#[cfg(test)]
mod tests {
    use super::*;

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

        // PredictedVlc: Cr diff=0 ("00"), Cb diff=0 ("00"), Y0 diff=+1 ("00" width=1, "1"),
        // Y1..Y3 diff=0 ("100"), each followed by EOB ("10").
        let bits_v3 = "00 10  00 10  00 1 10  100 10  100 10  100 10";
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
        assert_eq!(rle_v3[0], 2 << 10); // Cr DC=0
        assert_eq!(rle_v3[2], 2 << 10); // Cb DC=0
        assert_eq!(rle_v3[4], (2 << 10) | 1); // Y0 DC=+1
        assert_eq!(rle_v3[6], (2 << 10) | 1); // Y1 DC=+1 (predicted)
        assert_eq!(rle_v3[8], (2 << 10) | 1); // Y2 DC=+1
        assert_eq!(rle_v3[10], (2 << 10) | 1); // Y3 DC=+1
    }
}
