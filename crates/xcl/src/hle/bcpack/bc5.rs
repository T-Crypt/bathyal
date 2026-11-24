//! BC5 blocks from the 12 lanes of a BCPack BC5 stream (two symbols per 16-byte block).
//!
//! Block `i` reads the symbol `2i` (E) and `2i + 1` (O) of every lane:
//! * lane 0 is a distance `d` back to the block whose endpoints predict this one's, bit
//!   interleaved over E and O: `x` takes O's bits at positions 0, 2, 4, 6 and `E >> 1`'s at
//!   1, 3, 5, and `d` is `255 - x` when E's lowest bit is set, `x` otherwise. Symbol 0 is
//!   "none" (predict 128 for every endpoint); `d <= i` names block `i - d`; a larger `d` names
//!   slot `256 + i - d` of the built-in history ([`Tables::preset`]),
//! * lane 1 is unused; the format notes report it as 0 in every stream, and its symbols are
//!   ignored here (the fixture generator fills it with noise),
//! * lanes 2 and 3 hold the residues of the endpoint pairs of channel 0 and channel 1, E for
//!   the first endpoint and O for the second, 0, -1, 1, -2, 2 … for the symbols 0, 1, 2, 3,
//!   4 …, modulo 256,
//! * lanes 4–7 hold the index pairs of channel 0 and lanes 8–11 those of channel 1: lane
//!   `4 + 4c + k` has E = the indices of pixels `4k` and `4k + 1`, O = pixels `4k + 2` and
//!   `4k + 3`, each symbol the index of one of the 64 possible pairs in a fixed order
//!   ([`Tables::pairs`]).
//!
//! Each channel is a BC4 block: endpoint 0, endpoint 1, then the 16 3-bit indices of the
//! 4 x 4 texels, low texels first, little-endian. The blocks are in the order of the
//! surface's memory, tiled as the console stores it.
//!
//! After the BC5 sections of the public format notes of the SixNG project and the clean-room
//! decoder of leonida-asset-pipeline (wxrrry, MIT), which agree on every symbol; written for
//! XCL on 2026-11-23.

use super::{lane_widths, lanes, Error};

/// The constants of the BC5 rebuild that are not in a stream.
pub struct Tables {
    /// The two 3-bit indices each index symbol codes, the first pixel then the second.
    pub pairs: [(u8, u8); 64],
    /// The endpoint pairs `[channel 0 e0, channel 0 e1, channel 1 e0, channel 1 e1]` of the
    /// blocks before a stream's first: `preset[256 + i - d]` is the block `d - i` before block
    /// 0 when `d > i` (slot 0 is never read). `None`: not known.
    pub preset: [Option<[u8; 4]>; 256],
}

/// The 8-bit symbol whose even bits are `even` and odd bits `odd` (BC3 and BC5 lane 0).
pub(super) fn interleave(even: u8, odd: u8) -> u8 {
    (0..4).map(|k| (even >> k & 1) << (2 * k) | (odd >> k & 1) << (2 * k + 1)).fold(0, |x, y| x | y)
}

/// The distance back to the block whose endpoints predict block `i`, from lane 0's symbols.
pub fn distance(e: u8, o: u8) -> usize {
    let mut x = 0usize;
    for k in 0..4 {
        x |= ((o as usize >> k) & 1) << (2 * k);
    }
    for k in 0..3 {
        x |= ((e as usize >> (k + 1)) & 1) << (2 * k + 1);
    }
    if e & 1 != 0 {
        255 - x
    } else {
        x
    }
}

/// The residue a symbol stands for: 0, -1, 1, -2, 2 … for 0, 1, 2, 3, 4 …
fn zigzag(v: u8) -> i32 {
    if v & 1 == 0 {
        (v >> 1) as i32
    } else {
        -(((v >> 1) + 1) as i32)
    }
}

/// An endpoint from its prediction and residue, modulo 256.
fn endpoint(predicted: u8, symbol: u8) -> u8 {
    (predicted as i32 + zigzag(symbol)) as u8
}

/// Unpack a BC5 stream to its `out_len` bytes of blocks.
pub fn decode(src: &[u8], out_len: usize, tables: &Tables) -> Result<Vec<u8>, Error> {
    if out_len % 16 != 0 {
        return Err(Error::OutputSize);
    }
    let widths = lane_widths(5).unwrap();
    let (symbols, consumed) = lanes::decode(src, widths, out_len / 8)?;
    if src[consumed..].iter().any(|&b| b != 0) {
        return Err(Error::Trailing { consumed });
    }
    assemble(&symbols, tables)
}

/// The blocks of `symbols` (24 per block, two units of 12, in [`super::decode_lanes`]'s order).
pub fn assemble(symbols: &[u8], tables: &Tables) -> Result<Vec<u8>, Error> {
    let blocks = symbols.len() / 24;
    let mut out = Vec::new();
    out.try_reserve_exact(blocks * 16).map_err(|_| Error::OutputSize)?;
    out.resize(blocks * 16, 0u8);
    for i in 0..blocks {
        let s = &symbols[i * 24..i * 24 + 24];
        let d = distance(s[0], s[12]);
        let base = if d == 0 {
            [128; 4]
        } else if d <= i {
            let p = &out[(i - d) * 16..];
            [p[0], p[1], p[8], p[9]]
        } else {
            let slot = 256 + i - d;
            tables.preset[slot].ok_or(Error::History { block: i, slot })?
        };
        let e = [
            endpoint(base[0], s[2]),
            endpoint(base[1], s[14]),
            endpoint(base[2], s[3]),
            endpoint(base[3], s[15]),
        ];
        for ch in 0..2 {
            let mut bits = 0u64;
            for r in 0..4 {
                let lane = 4 + 4 * ch + r;
                let (a0, a1) = tables.pairs[s[lane] as usize];
                let (b0, b1) = tables.pairs[s[12 + lane] as usize];
                bits |= ((a0 as u64) | (a1 as u64) << 3 | (b0 as u64) << 6 | (b1 as u64) << 9) << (12 * r);
            }
            let b = &mut out[i * 16 + 8 * ch..i * 16 + 8 * ch + 8];
            b[0] = e[2 * ch];
            b[1] = e[2 * ch + 1];
            // the 16 indices, 3 bits each, low texels first
            b[2..8].copy_from_slice(&bits.to_le_bytes()[..6]);
        }
    }
    Ok(out)
}
