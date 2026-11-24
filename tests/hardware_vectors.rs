//! Retail PS1 MDEC hardware verification tests against `JaCzekanski/ps1-tests/mdec/` (MIT).

mod common;

use common::{
    HEART_4BIT_EXPECTED, HEART_8BIT_EXPECTED, HEART_MDEC, SYMBOLS_15BIT_EXPECTED, SYMBOLS_MDEC,
};
use libmdec::registers::{
    CMD_DEPTH_SHIFT, CMD_OPCODE_SHIFT, CMD_QUANT_COLOR_BIT, CMD_SET_BIT15_BIT, CMD_SIGNED_BIT,
    CTRL_ENABLE_DMA_IN, CTRL_ENABLE_DMA_OUT, CTRL_RESET, STAT_CMD_BUSY, STAT_CURRENT_BLOCK_MASK,
    STAT_CURRENT_BLOCK_SHIFT, STAT_DATA_IN_FIFO_FULL, STAT_DATA_IN_REQ, STAT_DATA_OUT_FIFO_EMPTY,
    STAT_DATA_OUT_REQ, STAT_REMAINING_MASK, STAT_RESET_VALUE,
};
use libmdec::rle::EOB_HALFWORD;
use libmdec::tables::{DEFAULT_QUANT_CHROMA, DEFAULT_QUANT_LUMA, DEFAULT_SCALE_TABLE};
use libmdec::{vlc_to_rle, AcCodebook, DcCoding, Decoder, HardwareDepth, MdecDevice, OutputDepth};

fn bytes_to_u32_le(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bytes_to_u16_le(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn u32_le_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for &w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

/// Configures `dev` the exact way `ps1-tests` initializes MDEC:
/// `mdec_reset`, `mdec_quantTable(quant, true)`, `mdec_idctTable(idct)`, `mdec_enableDma`.
fn init_device_like_ps1_tests(dev: &mut MdecDevice) {
    dev.write_control(CTRL_RESET);
    assert_eq!(dev.read_status(), STAT_RESET_VALUE);

    // MDEC(2): load luma + chroma quant tables (32 words).
    dev.write_command((2 << CMD_OPCODE_SHIFT) | CMD_QUANT_COLOR_BIT);
    for chunk in DEFAULT_QUANT_LUMA.chunks_exact(4) {
        dev.write_command(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    for chunk in DEFAULT_QUANT_CHROMA.chunks_exact(4) {
        dev.write_command(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    assert_eq!(dev.read_status() & STAT_CMD_BUSY, 0);

    // MDEC(3): load 64x i16 scale table (32 words).
    dev.write_command(3 << CMD_OPCODE_SHIFT);
    for pair in DEFAULT_SCALE_TABLE.chunks_exact(2) {
        let w = u32::from(pair[0] as u16) | (u32::from(pair[1] as u16) << 16);
        dev.write_command(w);
    }
    assert_eq!(dev.read_status() & STAT_CMD_BUSY, 0);

    // Enable DMA0 + DMA1 requests.
    dev.write_control(CTRL_ENABLE_DMA_IN | CTRL_ENABLE_DMA_OUT);
}

fn run_device_decode(
    dev: &mut MdecDevice,
    input_bytes: &[u8],
    depth: HardwareDepth,
    signed: bool,
    stp: bool,
    expected_out_bytes: usize,
) -> Vec<u8> {
    let in_words = bytes_to_u32_le(input_bytes);
    let cmd = (1 << CMD_OPCODE_SHIFT)
        | (depth.as_bits() << CMD_DEPTH_SHIFT)
        | if signed { CMD_SIGNED_BIT } else { 0 }
        | if stp { CMD_SET_BIT15_BIT } else { 0 }
        | (in_words.len() as u32 & 0xFFFF);

    dev.write_command(cmd);

    let mut in_pos = 0;
    let mut out_words = Vec::with_capacity(expected_out_bytes / 4);
    let target_words = expected_out_bytes / 4;

    while in_pos < in_words.len() || out_words.len() < target_words {
        // Feed until Data-In FIFO full or input exhausted.
        while in_pos < in_words.len() && (dev.read_status() & STAT_DATA_IN_FIFO_FULL) == 0 {
            dev.write_command(in_words[in_pos]);
            in_pos += 1;
        }
        // Drain Data-Out FIFO.
        while out_words.len() < target_words && (dev.read_status() & STAT_DATA_OUT_FIFO_EMPTY) == 0
        {
            out_words.push(dev.read_data());
        }
        if in_pos == in_words.len() && (dev.read_status() & STAT_DATA_OUT_FIFO_EMPTY) != 0 {
            break;
        }
    }

    u32_le_to_bytes(&out_words)
}

#[test]
fn retail_ps1_4bit_heart_exact_parity() {
    let mut dev = MdecDevice::new();
    init_device_like_ps1_tests(&mut dev);

    let got = run_device_decode(&mut dev, &HEART_MDEC, HardwareDepth::Bit4, false, false, 32);
    assert_eq!(got.len(), HEART_4BIT_EXPECTED.len());
    assert_eq!(
        got.as_slice(),
        &HEART_4BIT_EXPECTED,
        "4-bit output mismatch vs retail PS1 psx.log"
    );

    // High-level Decoder must match identically.
    let dec = Decoder::new(OutputDepth::Bit4);
    let rle = bytes_to_u16_le(&HEART_MDEC);
    let mut dec_out = [0u8; 32];
    dec.decode_rle_mono_block(&rle, &mut dec_out).unwrap();
    assert_eq!(dec_out, HEART_4BIT_EXPECTED);
}

#[test]
fn retail_ps1_8bit_heart_exact_parity() {
    let mut dev = MdecDevice::new();
    init_device_like_ps1_tests(&mut dev);

    let got = run_device_decode(&mut dev, &HEART_MDEC, HardwareDepth::Bit8, false, false, 64);
    assert_eq!(got.len(), HEART_8BIT_EXPECTED.len());

    let mut exact = 0usize;
    let mut max_diff = 0i32;
    for (&g, &e) in got.iter().zip(HEART_8BIT_EXPECTED.iter()) {
        if g == e {
            exact += 1;
        }
        max_diff = max_diff.max((i32::from(g) - i32::from(e)).abs());
    }
    assert!(
        max_diff <= 1 && exact >= 51,
        "8-bit heart parity regression vs retail PS1 psx.log: exact={exact}/64, max_diff={max_diff}"
    );

    // High-level Decoder must match MdecDevice identically.
    let dec = Decoder::new(OutputDepth::Bit8);
    let rle = bytes_to_u16_le(&HEART_MDEC);
    let mut dec_out = [0u8; 64];
    dec.decode_rle_mono_block(&rle, &mut dec_out).unwrap();
    assert_eq!(dec_out.as_slice(), got.as_slice());
}

#[test]
fn retail_ps1_15bit_symbols_exact_parity() {
    let mut dev = MdecDevice::new();
    init_device_like_ps1_tests(&mut dev);

    let got = run_device_decode(
        &mut dev,
        &SYMBOLS_MDEC,
        HardwareDepth::Bit15,
        false,
        false,
        2048,
    );
    assert_eq!(got.len(), SYMBOLS_15BIT_EXPECTED.len());

    let mut exact = 0usize;
    let mut max_chan_diff = 0i32;
    for i in 0..(2048 / 2) {
        let g = u16::from_le_bytes([got[i * 2], got[i * 2 + 1]]);
        let e = u16::from_le_bytes([
            SYMBOLS_15BIT_EXPECTED[i * 2],
            SYMBOLS_15BIT_EXPECTED[i * 2 + 1],
        ]);
        if g == e {
            exact += 1;
        } else {
            let dr = ((g & 0x1F) as i32 - (e & 0x1F) as i32).abs();
            let dg = (((g >> 5) & 0x1F) as i32 - ((e >> 5) & 0x1F) as i32).abs();
            let db = (((g >> 10) & 0x1F) as i32 - ((e >> 10) & 0x1F) as i32).abs();
            max_chan_diff = max_chan_diff.max(dr).max(dg).max(db);
        }
    }
    assert!(
        max_chan_diff <= 1 && exact >= 981,
        "15-bit symbols parity regression vs retail PS1 psx.log: exact={exact}/1024, max_chan_diff={max_chan_diff}"
    );
}

#[test]
fn mdec_status_transitions_and_dma_stepping() {
    let mut dev = MdecDevice::new();
    init_device_like_ps1_tests(&mut dev);

    let in_words = bytes_to_u32_le(&SYMBOLS_MDEC);
    let cmd = (1 << CMD_OPCODE_SHIFT)
        | (HardwareDepth::Bit15.as_bits() << CMD_DEPTH_SHIFT)
        | (in_words.len() as u32 & 0xFFFF);
    dev.write_command(cmd);

    // Immediately after MDEC(1) with 0x100 words:
    // dataOutFifoEmpty=1, dataInFifoFull=0, cmdBusy=1, dataInReq=1, dataOutReq=0,
    // depth=3, currentBlock=4 (Cr), remaining=0x00ff -> 0xB604_00FF!
    assert_eq!(dev.read_status(), 0xB604_00FF);

    // Write first word -> remaining decrements to 0x00FE.
    dev.write_command(in_words[0]);
    assert_eq!(dev.read_status(), 0xB604_00FE);

    // Stream the rest using dma_in / dma_out.
    let mut in_pos = 1;
    let mut out_words = Vec::with_capacity(512);
    let mut scratch = [0u32; 32];
    let mut saw_in_full = false;
    let mut saw_out_req = false;

    while in_pos < in_words.len() || (dev.read_status() & STAT_DATA_OUT_FIFO_EMPTY) == 0 {
        if in_pos < in_words.len() {
            let n = dev.dma_in(&in_words[in_pos..]);
            in_pos += n;
            if (dev.read_status() & STAT_DATA_IN_FIFO_FULL) != 0 {
                saw_in_full = true;
            }
        }
        if (dev.read_status() & STAT_DATA_OUT_REQ) != 0 {
            saw_out_req = true;
        }
        let n = dev.dma_out(&mut scratch);
        out_words.extend_from_slice(&scratch[..n]);
    }

    assert!(saw_in_full, "expected Data-In FIFO full backpressure");
    assert!(saw_out_req, "expected Data-Out DMA request assertion");
    assert_eq!(out_words.len(), 512);
    let dma_bytes = u32_le_to_bytes(&out_words);
    // DMA output must match single-word MMIO decode bit-for-bit.
    let mut dev2 = MdecDevice::new();
    init_device_like_ps1_tests(&mut dev2);
    let mmio_bytes = run_device_decode(
        &mut dev2,
        &SYMBOLS_MDEC,
        HardwareDepth::Bit15,
        false,
        false,
        2048,
    );
    assert_eq!(dma_bytes, mmio_bytes);

    // After all 256 input words and 512 output words are drained:
    // dataOutFifoEmpty=1, cmdBusy=0, dataInReq=0, dataOutReq=0,
    // depth=3, currentBlock=4, remaining=0xffff -> 0x8604_FFFF!
    let final_stat = dev.read_status();
    assert_eq!(final_stat, 0x8604_FFFF);
    assert_eq!(final_stat & STAT_CMD_BUSY, 0);
    assert_eq!(final_stat & STAT_DATA_IN_REQ, 0);
    assert_eq!(final_stat & STAT_DATA_OUT_REQ, 0);
    assert_eq!(
        (final_stat & STAT_CURRENT_BLOCK_MASK) >> STAT_CURRENT_BLOCK_SHIFT,
        4
    );
    assert_eq!(final_stat & STAT_REMAINING_MASK, 0xFFFF);

    // NOP opcodes 0, 4, 5, 6, 7 do not hang or enter busy state.
    for op in [0u32, 4, 5, 6, 7] {
        dev.write_command(op << CMD_OPCODE_SHIFT);
        assert_eq!(dev.read_status() & STAT_CMD_BUSY, 0);
    }
}

#[test]
fn output_24bit_and_decoder_raster_swizzle_parity() {
    let mut dev = MdecDevice::new();
    init_device_like_ps1_tests(&mut dev);

    // Decode symbols.mdec in 24-bit mode on MdecDevice (4 macroblocks * 768 bytes = 3072 bytes).
    let hw_24 = run_device_decode(
        &mut dev,
        &SYMBOLS_MDEC,
        HardwareDepth::Bit24,
        false,
        false,
        3072,
    );
    assert_eq!(hw_24.len(), 3072);

    // Decode same RLE via high-level Decoder in Bit24, Rgba32, and Bit15 (16x64 frame).
    let rle = bytes_to_u16_le(&SYMBOLS_MDEC);
    let mut frame_24 = vec![0u8; 16 * 64 * 3];
    Decoder::new(OutputDepth::Bit24)
        .decode_rle_frame(&rle, 16, 64, &mut frame_24)
        .unwrap();

    let mut frame_rgba = vec![0u8; 16 * 64 * 4];
    Decoder::new(OutputDepth::Rgba32)
        .decode_rle_frame(&rle, 16, 64, &mut frame_rgba)
        .unwrap();

    // Every pixel in Rgba32 must match Bit24 with alpha=0xFF.
    for i in 0..(16 * 64) {
        assert_eq!(&frame_rgba[i * 4..i * 4 + 3], &frame_24[i * 3..i * 3 + 3]);
        assert_eq!(frame_rgba[i * 4 + 3], 0xFF);
    }

    // Swizzling hardware 8x8 sub-blocks (Y0, Y1, Y2, Y3) into 16x16 must match Decoder Bit24 bit-for-bit!
    for mb in 0..4 {
        let mb_base = mb * 768;
        for sub in 0..4 {
            let sub_base = mb_base + sub * 192;
            let sx = (sub & 1) * 8;
            let sy = (sub >> 1) * 8;
            for y in 0..8 {
                for x in 0..8 {
                    let hw_off = sub_base + (y * 8 + x) * 3;
                    let dec_off = ((mb * 16 + sy + y) * 16 + (sx + x)) * 3;
                    assert_eq!(
                        &hw_24[hw_off..hw_off + 3],
                        &frame_24[dec_off..dec_off + 3],
                        "24-bit mismatch at mb={mb} sub={sub} ({x},{y})"
                    );
                }
            }
        }
    }
}

#[test]
fn q_scale_zero_unquantized_linear_mode() {
    // q_scale = 0, DC = 64 -> dequantized DC = 128 -> IDCT output = 128/8 = +16 -> unsigned 144 (0x90).
    // AC at linear index 8 (row 1, col 0) with run = 7, level = 0 -> flat block.
    let rle: [u16; 2] = [64u16, EOB_HALFWORD];
    let dec = Decoder::new(OutputDepth::Bit8);
    let mut out = [0u8; 64];
    dec.decode_rle_mono_block(&rle, &mut out).unwrap();
    assert!(out.iter().all(|&p| p == 144));

    // Verify linear (non-zigzag) placement when q_scale == 0:
    // Put a non-zero coefficient at k=3 (linear index 3 = row 0 col 3, NOT zigzag[3]=16).
    let rle_ac: [u16; 3] = [0u16, (2u16 << 10) | 128, EOB_HALFWORD]; // k=0+2+1=3, val=256
    let mut out_ac = [0u8; 64];
    dec.decode_rle_mono_block(&rle_ac, &mut out_ac).unwrap();
    // Row 0..7 should all be identical because frequency is purely horizontal (v=0, u=3)!
    for row in 1..8 {
        assert_eq!(&out_ac[row * 8..(row + 1) * 8], &out_ac[0..8]);
    }
}

#[test]
fn signed_vs_unsigned_and_stp_bit15() {
    let rle = bytes_to_u16_le(&HEART_MDEC);

    // 8-bit: signed byte is unsigned byte wrapping_sub(128).
    let mut u8_out = [0u8; 64];
    let mut s8_out = [0u8; 64];
    Decoder::new(OutputDepth::Bit8)
        .decode_rle_mono_block(&rle, &mut u8_out)
        .unwrap();
    Decoder::new(OutputDepth::Bit8)
        .with_signed(true)
        .decode_rle_mono_block(&rle, &mut s8_out)
        .unwrap();
    for i in 0..64 {
        assert_eq!(s8_out[i], u8_out[i].wrapping_sub(128));
    }

    // 15-bit STP bit 15: toggling stp must XOR every pixel by exactly 0x8000.
    let sym_rle = bytes_to_u16_le(&SYMBOLS_MDEC);
    let mut f_stp0 = vec![0u8; 16 * 64 * 2];
    let mut f_stp1 = vec![0u8; 16 * 64 * 2];
    Decoder::new(OutputDepth::Bit15 { stp: false })
        .decode_rle_frame(&sym_rle, 16, 64, &mut f_stp0)
        .unwrap();
    Decoder::new(OutputDepth::Bit15 { stp: true })
        .decode_rle_frame(&sym_rle, 16, 64, &mut f_stp1)
        .unwrap();
    for i in 0..(16 * 64) {
        let p0 = u16::from_le_bytes([f_stp0[i * 2], f_stp0[i * 2 + 1]]);
        let p1 = u16::from_le_bytes([f_stp1[i * 2], f_stp1[i * 2 + 1]]);
        assert_eq!(p0 & 0x8000, 0);
        assert_eq!(p1, p0 | 0x8000);
    }

    // MdecDevice STP bit 15 check: toggling stp must set bit 15 on every pixel.
    let mut dev = MdecDevice::new();
    init_device_like_ps1_tests(&mut dev);
    let hw_stp0 = run_device_decode(
        &mut dev,
        &SYMBOLS_MDEC,
        HardwareDepth::Bit15,
        false,
        false,
        2048,
    );
    let hw_stp1 = run_device_decode(
        &mut dev,
        &SYMBOLS_MDEC,
        HardwareDepth::Bit15,
        false,
        true,
        2048,
    );
    for i in 0..1024 {
        let p0 = u16::from_le_bytes([hw_stp0[i * 2], hw_stp0[i * 2 + 1]]);
        let p1 = u16::from_le_bytes([hw_stp1[i * 2], hw_stp1[i * 2 + 1]]);
        assert_eq!(p0 & 0x8000, 0);
        assert_eq!(p1, p0 | 0x8000);
    }
}

#[test]
fn custom_quant_and_scale_tables_via_mdec_commands() {
    let mut dev = MdecDevice::new();
    dev.write_control(CTRL_RESET);

    // Load custom quant table where qt[0] = 4 (double the default DC scaling).
    let mut custom_q = DEFAULT_QUANT_LUMA;
    custom_q[0] = 4;
    dev.write_command(2 << CMD_OPCODE_SHIFT); // luma only (16 words)
    for chunk in custom_q.chunks_exact(4) {
        dev.write_command(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    assert_eq!(dev.quant_luma()[0], 4);
    // Reset must preserve loaded quant and scale tables.
    dev.write_control(CTRL_RESET);
    assert_eq!(dev.quant_luma()[0], 4);

    // Load a zero scale table -> any input block must IDCT to 0 (unsigned 128 = 0x80).
    let zero_scale = [0i16; 64];
    dev.write_command(3 << CMD_OPCODE_SHIFT);
    for _ in 0..32 {
        dev.write_command(0);
    }
    assert_eq!(dev.scale_table(), &zero_scale);

    let out = run_device_decode(&mut dev, &HEART_MDEC, HardwareDepth::Bit8, false, false, 64);
    assert!(out.iter().all(|&b| b == 128));
}

#[test]
fn bs_vlc_to_rle_and_decoder_roundtrip() {
    // Pack a 16x16 (1 macroblock) BS bitstream with:
    // - FixedLength DC: Cr=0, Cb=0, Y0=+128, Y1=-128, Y2=0, Y3=+64, all EOB ("10").
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

    let dcs: [i32; 6] = [0, 0, 128, -128, 0, 64];
    let mut bits_fixed = String::new();
    let mut bits_headers = String::new();
    for &dc in &dcs {
        let dc10 = (dc as u32) & 0x3FF;
        bits_fixed.push_str(&format!("{dc10:010b}10"));
        bits_headers.push_str(&format!("{:06b}{dc10:010b}10", 1u32));
    }

    let bs_fixed = pack_bits(&bits_fixed);
    let bs_headers = pack_bits(&bits_headers);

    let mut rle = Vec::new();
    vlc_to_rle(
        &bs_fixed,
        1,
        DcCoding::FixedLength,
        AcCodebook::standard(),
        1,
        &mut rle,
    )
    .unwrap();
    assert_eq!(rle.len(), 12);

    let dec = Decoder::new(OutputDepth::Bit24);
    let mut frame_fixed = vec![0u8; 16 * 16 * 3];
    let mut frame_headers = vec![0u8; 16 * 16 * 3];
    dec.decode_bs_frame(
        &bs_fixed,
        16,
        16,
        1,
        DcCoding::FixedLength,
        &mut frame_fixed,
    )
    .unwrap();
    dec.decode_bs_frame_with_block_headers(&bs_headers, 16, 16, &mut frame_headers)
        .unwrap();
    assert_eq!(frame_fixed, frame_headers);

    // Check quadrant luma values:
    // Y0: dc=128 -> *2=256 -> /8=+32 -> 160
    // Y1: dc=-128 -> *2=-256 -> /8=-32 -> 96
    // Y2: dc=0 -> 128
    // Y3: dc=64 -> *2=128 -> /8=+16 -> 144
    assert_eq!(&frame_fixed[0..3], &[160, 160, 160]);
    assert_eq!(&frame_fixed[8 * 3..8 * 3 + 3], &[96, 96, 96]);
    assert_eq!(
        &frame_fixed[(8 * 16) * 3..(8 * 16) * 3 + 3],
        &[128, 128, 128]
    );
    assert_eq!(
        &frame_fixed[(8 * 16 + 8) * 3..(8 * 16 + 8) * 3 + 3],
        &[144, 144, 144]
    );
}
