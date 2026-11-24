# libmdec

Standalone pure-Rust PlayStation 1 MDEC hardware video decoder and BS bitstream library.

## Features

- **Zero runtime dependencies** (`edition = "2021"`, `#![forbid(unsafe_code)]`, `#![warn(missing_docs)]`, `Send + Sync`)
- **Deterministic fixed-point integer arithmetic** (zero `f64`, zero `thread_local!`, zero heap allocations in block/macroblock decode loops)
- **Low-level hardware-faithful [`MdecDevice`]**: `MDEC0`/`MDEC1` MMIO registers, `MDECSTAT`/`MDECCTRL`, `dma_in`/`dma_out`, `MDEC(0..=7)` including `MDEC(1)` Decode Macroblocks, `MDEC(2)` Set Quant Tables, and `MDEC(3)` Set Scale Table
- **High-level ergonomic [`Decoder`]**: 4-bit gray, 8-bit gray, 15-bit RGB555 (with configurable STP bit 15), 24-bit RGB888, and 32-bit RGBA8888 output, signed/unsigned modes, custom quant/scale tables, and raster macroblock/frame decoding
- **PS1 BS VLC-to-RLE frontend**: `BitReader`, validated prefix-free `AcCodebook`, BS v1/v2 fixed-length DC and BS v3 predicted VLC DC, and standard 224-code Sony `VLC0` table
- **Verified against real retail PS1 hardware vectors** from [`JaCzekanski/ps1-tests/mdec/`](https://github.com/JaCzekanski/ps1-tests) (MIT)

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT License](LICENSE-MIT) at your option.
