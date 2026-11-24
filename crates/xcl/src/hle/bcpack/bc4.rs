//! BC4 blocks from the 12 lanes of a BCPack BC4 stream (one symbol per lane and block).
//!
//! For block `i`, in stream order:
//! * lane 0 is a distance `d` back to the block whose endpoints predict this one's, read as in
//!   BC1: symbol 0 is "none" (predict 128 for both), an even symbol `v` is `v / 2`, an odd one
//!   `255 - v / 2`. Before block 0 the prediction comes from a built-in history of 255 endpoint
//!   pairs ([`Tables::history`]),
//! * lane 1 was 0 in every stream seen,
//! * lanes 2 and 3 are the differences of A0 and A1 from the prediction, 0, -1, 1, -2, 2 … for
//!   the symbols 0, 1, 2, 3, 4 …, modulo 256,
//! * lanes 4–11 are the 3-bit selectors two texels at a time: lanes 4 and 5 the left and right
//!   halves of the top row, 6 and 7 the second row, and so on. Each symbol is the index of one
//!   of the 64 possible pairs in a fixed order ([`Tables::pair_codes`]).
//!
//! The rebuild needs both tables. No stream carries them, and XCL has no source for them it can
//! take, so they are a parameter and nothing calls this yet.

use super::{decode_lanes, Error};

/// The constants of the BC4 rebuild that are not in a stream.
pub struct Tables {
    /// The endpoints `[A0, A1]` of the blocks before a stream's first: `history[256 - k]` is
    /// the block `k` before block 0 (k = 1..=255; entry 0 is never read). `None`: not known.
    pub history: [Option<[u8; 2]>; 256],
    /// The selectors of each pair symbol: bits 0–2 the left texel, 3–5 the right one.
    pub pair_codes: [u8; 64],
}

/// Unpack a BC4 stream to its `out_len` bytes of blocks.
pub fn decode(src: &[u8], out_len: usize, tables: &Tables) -> Result<Vec<u8>, Error> {
    assemble(&decode_lanes(4, src, out_len)?, tables)
}

/// The blocks of `symbols` (12 per block, in [`super::decode_lanes`]'s order).
pub fn assemble(symbols: &[u8], tables: &Tables) -> Result<Vec<u8>, Error> {
    let blocks = symbols.len() / 12;
    let mut out = Vec::new();
    out.try_reserve_exact(blocks * 8).map_err(|_| Error::OutputSize)?;
    out.resize(blocks * 8, 0u8);
    for (i, s) in symbols.chunks_exact(12).enumerate() {
        if s[1] != 0 {
            return Err(Error::Lane1 { block: i });
        }
        let predicted = match back_reference(s[0]) {
            0 => [128, 128],
            d if d <= i => [out[(i - d) * 8], out[(i - d) * 8 + 1]],
            d => {
                let slot = 256 + i - d;
                tables.history[slot].ok_or(Error::History { block: i, slot })?
            }
        };
        let b = &mut out[i * 8..i * 8 + 8];
        b[0] = predicted[0].wrapping_add(delta(s[2]));
        b[1] = predicted[1].wrapping_add(delta(s[3]));
        b[2..8].copy_from_slice(&selector_rows(s[4..12].iter().copied(), &tables.pair_codes).to_le_bytes()[..6]);
    }
    Ok(out)
}

/// Lane 0 of BC4, as BC1's: the symbols 0, 1, 2, 3, 4 … are 0, 255, 1,
/// 254, 2 … blocks back.
pub(super) fn back_reference(v: u8) -> usize {
    let v = v as usize;
    if v % 2 == 0 { v / 2 } else { 255 - v / 2 }
}

/// A difference symbol: 0, -1, 1, -2, 2 … as a wrapping 8-bit step.
pub(super) fn delta(v: u8) -> u8 {
    if v % 2 == 0 { v / 2 } else { !(v / 2) }
}

/// The 48 selector bits of a BC4 block (3 bits a texel, the top-left texel lowest) from its
/// eight pair symbols, two per row, left half first.
pub(super) fn selector_rows(halves: impl Iterator<Item = u8>, codes: &[u8; 64]) -> u64 {
    halves.enumerate().map(|(h, s)| (codes[s as usize] as u64) << (6 * h)).fold(0, |a, b| a | b)
}
