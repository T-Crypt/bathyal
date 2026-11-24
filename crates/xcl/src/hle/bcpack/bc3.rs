//! BC3 blocks from the 12 lanes of a BCPack BC3 stream (two symbols per lane and block: units
//! `2i` and `2i + 1`, called `u0` and `u1` here).
//!
//! For block `i`, in stream order:
//! * lane 0 is the back-reference split over both units as in BC5 ([`super::bc5`]). Symbol 0
//!   predicts 128 for both alpha endpoints and 0 for the colour ones, and before block 0 the
//!   prediction comes from a built-in history ([`Tables::history`]),
//! * lane 1 was 0 in every stream seen, in both units,
//! * lane 2 is the difference of A0 (`u0`) and A1 (`u1`) from the prediction, modulo 256,
//! * lanes 3–6 are the alpha selectors, one row each from the top, the left half of the row in
//!   `u0` and the right half in `u1`, through a pair table as BC4's ([`Tables::pair_codes`]),
//! * lanes 7, 8 and 9 are the differences of red, green and blue: of colour 0 in `u0` and of
//!   colour 1 in `u1`, modulo 32, 64 and 32,
//! * lanes 10 and 11 are the colour selectors as BC1's quadrants ([`Tables::quadrants`]):
//!   lane 10 the top-left (`u0`) and top-right (`u1`) 2×2 texels, lane 11 the bottom ones.
//!
//! As for BC4, XCL has no source it can take for the three tables, so nothing calls this yet.

use super::bc4::{back_reference, delta, selector_rows};
use super::bc5::interleave;
use super::{decode_lanes, Error};

/// The constants of the BC3 rebuild that are not in a stream.
pub struct Tables {
    /// The endpoints `[A0, A1, R0, G0, B0, R1, G1, B1]` of the blocks before a stream's first:
    /// `history[256 - k]` is the block `k` before block 0 (k = 1..=255; entry 0 is never read).
    /// `None`: not known.
    pub history: [Option<[u8; 8]>; 256],
    /// The alpha selectors of each pair symbol: bits 0–2 the left texel, 3–5 the right one.
    pub pair_codes: [u8; 64],
    /// The colour selectors of each quadrant symbol, 2 bits a texel: bits 0–1 the top-left
    /// texel, 2–3 top-right, 4–5 bottom-left, 6–7 bottom-right.
    pub quadrants: [u8; 256],
}

/// The range of each endpoint, in the history's order.
const RANGE: [u16; 8] = [256, 256, 32, 64, 32, 32, 64, 32];

/// Unpack a BC3 stream to its `out_len` bytes of blocks.
pub fn decode(src: &[u8], out_len: usize, tables: &Tables) -> Result<Vec<u8>, Error> {
    if out_len % 16 != 0 {
        return Err(Error::OutputSize);
    }
    assemble(&decode_lanes(3, src, out_len)?, tables)
}

/// The endpoints of a rebuilt block, in the history's order.
fn endpoints(block: &[u8]) -> [u8; 8] {
    let c0 = u16::from_le_bytes([block[8], block[9]]);
    let c1 = u16::from_le_bytes([block[10], block[11]]);
    let rgb = |c: u16| [(c >> 11) as u8, (c >> 5 & 63) as u8, (c & 31) as u8];
    let [r0, g0, b0] = rgb(c0);
    let [r1, g1, b1] = rgb(c1);
    [block[0], block[1], r0, g0, b0, r1, g1, b1]
}

/// The blocks of `symbols` (24 per block, in [`super::decode_lanes`]'s order).
pub fn assemble(symbols: &[u8], tables: &Tables) -> Result<Vec<u8>, Error> {
    let blocks = symbols.len() / 24;
    let mut out = Vec::new();
    out.try_reserve_exact(blocks * 16).map_err(|_| Error::OutputSize)?;
    out.resize(blocks * 16, 0u8);
    for (i, s) in symbols.chunks_exact(24).enumerate() {
        let (u0, u1) = s.split_at(12);
        if u0[1] != 0 || u1[1] != 0 {
            return Err(Error::Lane1 { block: i });
        }
        let predicted = match back_reference(interleave(u0[0], u1[0])) {
            0 => [128, 128, 0, 0, 0, 0, 0, 0],
            d if d <= i => endpoints(&out[(i - d) * 16..]),
            d => {
                let slot = 256 + i - d;
                tables.history[slot].ok_or(Error::History { block: i, slot })?
            }
        };
        let diffs = [u0[2], u1[2], u0[7], u0[8], u0[9], u1[7], u1[8], u1[9]];
        let e: [u16; 8] = std::array::from_fn(|k| (predicted[k] as u16 + delta(diffs[k]) as u16) % RANGE[k]);
        let o = &mut out[i * 16..i * 16 + 16];
        o[0] = e[0] as u8;
        o[1] = e[1] as u8;
        let alpha = selector_rows((3..7).flat_map(|l| [u0[l], u1[l]]), &tables.pair_codes);
        o[2..8].copy_from_slice(&alpha.to_le_bytes()[..6]);
        o[8..10].copy_from_slice(&(e[2] << 11 | e[3] << 5 | e[4]).to_le_bytes());
        o[10..12].copy_from_slice(&(e[5] << 11 | e[6] << 5 | e[7]).to_le_bytes());
        let [tl, tr, bl, br] = [u0[10], u1[10], u0[11], u1[11]].map(|q| tables.quadrants[q as usize]);
        // a row of the block is four texels, 2 bits each, the left one lowest
        o[12] = (tl & 15) | (tr & 15) << 4;
        o[13] = (tl >> 4) | (tr >> 4) << 4;
        o[14] = (bl & 15) | (br & 15) << 4;
        o[15] = (bl >> 4) | (br >> 4) << 4;
    }
    Ok(out)
}
