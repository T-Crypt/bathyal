//! Tests of the BCPack decoder on synthetic streams: the fixtures of `tests/data/bcpack`
//! (built by its `generate.py`, an encoder written from the format notes and not from this
//! decoder) and streams of the encoder below. None of them shows what a console would write.

use super::lanes::{self, L};
use super::rans_tables::FREQ;
use super::{bc1, bc3, bc4, bc5, decode_lanes, lane_widths, Error};

/// The tables the fixtures' `.bc1` files were made with: history entry `slot` has the
/// channels `(7 * slot + 13 * k) mod range`, and a quadrant symbol is its own selectors.
pub(crate) fn tables() -> &'static bc1::Tables {
    static T: std::sync::OnceLock<bc1::Tables> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let range = [32usize, 64, 32, 32, 64, 32];
        bc1::Tables {
            history: std::array::from_fn(|slot| Some(std::array::from_fn(|k| ((7 * slot + 13 * k) % range[k]) as u8))),
            quadrants: std::array::from_fn(|s| s as u8),
        }
    })
}

pub(crate) fn data(name: &str) -> Option<Vec<u8>> {
    std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/bcpack").join(name)).ok()
}

/// The BC5 tables the tests use: the index pairs in an order of their own, and a history whose
/// slot `s` holds `(5s, 7s, 11s, 13s)` so a test can name the entry it expects.
pub(crate) fn bc5_tables() -> &'static bc5::Tables {
    static T: std::sync::OnceLock<bc5::Tables> = std::sync::OnceLock::new();
    T.get_or_init(|| bc5::Tables {
        pairs: std::array::from_fn(|s| ((s & 7) as u8, (s >> 3) as u8)),
        preset: std::array::from_fn(|slot| {
            let v = slot as u8;
            Some([v.wrapping_mul(5), v.wrapping_mul(7), v.wrapping_mul(11), v.wrapping_mul(13)])
        }),
    })
}

/// `bc<mode>-<units>-<seed>` → (mode, units)
fn fixture(stem: &str) -> (u8, usize) {
    let mut p = stem.strip_prefix("bc").unwrap().split('-');
    (p.next().unwrap().parse().unwrap(), p.next().unwrap().parse().unwrap())
}

/// A `.lanes` file (all of lane 0, then lane 1 …) in the decoder's order.
fn by_unit(lane_major: &[u8], lanes: usize) -> Vec<u8> {
    let units = lane_major.len() / lanes;
    (0..units * lanes).map(|i| lane_major[(i % lanes) * units + i / lanes]).collect()
}

const FIXTURES: [&str; 17] = [
    "bc1-1-8", "bc1-2-0", "bc1-16-1", "bc1-16-9", "bc1-700-2", "bc1-8192-3", "bc3-16-1", "bc4-16-1", "bc5-16-1", "bc6-16-1", "bc7-2-0", "bc7-16-1",
    "bc7-700-2", "bc7-8192-3", "bc4-300-4", "bc5-600-5", "bc3-600-6",
];

/// The symbols of the two 64 KiB fixtures, which ship without a `.lanes` file (`generate.py`).
pub(crate) fn large_fixture_symbols(mode: u8) -> Vec<u8> {
    let n = lane_widths(mode).unwrap().len();
    let mut s = vec![0u8; 8192 * n];
    if mode == 1 {
        for i in (0..8192).step_by(511) {
            s[i * n + 2] = 30;
            s[i * n + 8] = 5;
        }
        for i in (31..8192).step_by(31) {
            s[i * n] = 2;
        }
    }
    s
}

#[test]
fn fixtures_decode_to_their_lanes() {
    for stem in FIXTURES {
        let (mode, units) = fixture(stem);
        let n = lane_widths(mode).unwrap().len();
        let bin = data(&format!("{stem}.bin")).unwrap();
        let want = match data(&format!("{stem}.lanes")) {
            Some(l) => by_unit(&l, n),
            None => large_fixture_symbols(mode),
        };
        assert_eq!(want.len(), units * n, "{stem}");
        // bc1-1-8 has states cut to fewer bytes than an encoder that starts at `L` can drop,
        // which the hardware's decoder does not take either
        if stem == "bc1-1-8" {
            assert!(decode_lanes(mode, &bin, units * 8).is_err(), "{stem}");
            continue;
        }
        assert_eq!(decode_lanes(mode, &bin, units * 8).as_deref(), Ok(&want[..]), "{stem}");
        // zero padding after the stream is not part of it; anything else is
        let mut padded = bin.clone();
        padded.extend([0; 37]);
        assert_eq!(decode_lanes(mode, &padded, units * 8).as_deref(), Ok(&want[..]), "{stem} padded");
        padded.push(1);
        assert_eq!(decode_lanes(mode, &padded, units * 8), Err(Error::Trailing { consumed: bin.len() }), "{stem}");
    }
}

#[test]
fn bc1_fixtures_decode_to_their_blocks() {
    for stem in FIXTURES.iter().filter(|s| s.starts_with("bc1-") && **s != "bc1-1-8") {
        let (_, units) = fixture(stem);
        let bin = data(&format!("{stem}.bin")).unwrap();
        let blocks = bc1::decode(&bin, units * 8, tables()).unwrap();
        match data(&format!("{stem}.bc1")) {
            Some(want) => assert_eq!(blocks, want, "{stem}"),
            None => assert_eq!(blocks, bc1::assemble(&large_fixture_symbols(1), tables()).unwrap(), "{stem}"),
        }
        assert_eq!(super::decode(1, &bin, units * 8).as_deref(), Ok(&blocks[..]), "{stem}");
    }
    // the modes without a block rebuild
    for mode in [0u8, 2, 3, 4, 6, 7, 8, 15] {
        assert!(!super::can_decode(mode));
        assert_eq!(super::decode(mode, &[], 16), Err(Error::Unsupported));
    }
    assert!(super::can_decode(5));
}

/// One BC1 block by hand: every field of the rebuild, without the entropy stage.
#[test]
fn bc1_block_rebuild() {
    let t = tables();
    // block 0: no reference; R0 +3, G0 -1, B0 0, R1 -2, G1 +5, B1 +1; the four quadrants
    let b0 = [0u8, 0, 6, 1, 0, 3, 10, 2, 0b11_10_01_00, 0b00_00_00_01, 0b10_10_10_10, 0b11_00_11_00];
    // block 1: one block back; R0 -4 (wraps to 31), the rest unchanged
    let b1 = [2u8, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    // block 2: 255 back, which is before the stream: history entry 256 + 2 - 255 = 3
    let b2 = [1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let out = bc1::assemble(&[b0, b1, b2].concat(), t).unwrap();
    let c = |r: u16, g: u16, b: u16| (r << 11 | g << 5 | b).to_le_bytes();
    assert_eq!(out[0..2], c(3, 63, 0));
    assert_eq!(out[2..4], c(30, 5, 1));
    // rows: left quadrant's two texels in the low nibble, right quadrant's in the high one
    assert_eq!(out[4..8], [0b00_01_01_00, 0b00_00_11_10, 0b11_00_10_10, 0b11_00_10_10]);
    assert_eq!(out[8..10], c(31, 63, 0));
    assert_eq!(out[10..12], c(30, 5, 1));
    assert_eq!(out[12..16], [0; 4]);
    let h = t.history[3].unwrap().map(|v| v as u16);
    assert_eq!(out[16..20], [c(h[0], h[1], h[2]), c(h[3], h[4], h[5])].concat()[..]);

    // lane 1 is not understood beyond "0"
    let mut odd = b0;
    odd[1] = 1;
    assert_eq!(bc1::assemble(&odd, t), Err(Error::Lane1 { block: 0 }));
    // a missing history entry is read as zeros, as its source decodes an unfitted slot: the
    // block that read it is zeros here (nothing written before it), the rest is unchanged
    let mut holes = bc1::Tables { history: t.history, quadrants: t.quadrants };
    holes.history[3] = None;
    let (hole_out, unfitted) = bc1::assemble_counted(&[b0, b1, b2].concat(), &holes).unwrap();
    assert_eq!(unfitted, 1);
    assert_eq!(hole_out[..16], out[..16]);
    assert_eq!(hole_out[16..], [0; 8]);
    assert_eq!(bc1::assemble(&[b0, b1, b2].concat(), &holes).as_deref(), Ok(&hole_out[..]));
}

/// The 8-bit model 11 as US 11,234,023 prints it (its `base_table[8][12]`, counted from 1).
#[test]
fn model_matches_the_patent_excerpt() {
    let base = [0u32, 7575, 14276, 25440, 41008, 56352, 64256, 65344, 65408];
    let freq = [7575u16, 6701, 5582, 3892, 1918, 494, 34, 1, 1];
    assert_eq!(FREQ[7][11], freq);
    let mut at = 0;
    for k in 0..9 {
        assert_eq!(at, base[k]);
        at += freq[k] as u32 * if k < 2 { 1 } else { 1 << (k - 1) };
    }
    assert_eq!(at, 65536);
    for (bits, per_q) in FREQ.iter().enumerate() {
        for f in per_q {
            let total: u32 = (0..9).map(|k| f[k] as u32 * if k < 2 { 1 } else { 1 << (k - 1) }).sum();
            assert_eq!(total, 65536);
            assert!(f[..bits + 2].iter().all(|&v| v >= 1) && f[bits + 2..].iter().all(|&v| v == 0));
        }
    }
}

// ---------------------------------------------------------------------------------------
// a test encoder: any model, any fragment sizes
// ---------------------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

/// (frequency, first slot) of `sym` in model (`bits`, `q`).
fn slot(bits: usize, q: usize, sym: u32) -> (u32, u32) {
    let seg = (32 - sym.leading_zeros()) as usize;
    let base: u32 = (0..seg).map(|k| FREQ[bits - 1][q][k] as u32 * if k < 2 { 1 } else { 1 << (k - 1) }).sum();
    let f = FREQ[bits - 1][q][seg] as u32;
    let cf = base + f * (sym - ((1u32 << seg) >> 1));
    if cf == 0 {
        (f - 1, 1)
    } else {
        (f, cf)
    }
}

/// One flushed fragment: header, state, payload. `trim`: drop the zero bytes an encoder
/// that starts at `L` may drop (those its first symbol pushed out).
fn fragment(symbols: &[u8], lane_bits: usize, z: usize, q: usize, trim: bool) -> Vec<u8> {
    let mut x = L;
    let mut emitted = Vec::new();
    let mut droppable = 0;
    for (k, &s) in symbols.iter().rev().enumerate() {
        let (f, cf) = slot(lane_bits - z, q, s as u32);
        while x >= f << 16 {
            emitted.push(x as u8);
            x >>= 8;
        }
        if k == 0 {
            droppable = emitted.len();
        }
        x = (x / f) << 16 | (x % f + cf);
    }
    let mut body = x.to_be_bytes().to_vec();
    body.extend(emitted.iter().rev());
    if trim {
        body.truncate(body.len() - droppable);
    }
    let mode = (z << 6 | 0x10 | q) as u8;
    let mut out = if body.len() <= 256 {
        vec![mode, (body.len() - 1) as u8]
    } else {
        let n = body.len() - 257;
        vec![mode | 0x20, n as u8, (n >> 8) as u8]
    };
    out.extend(body);
    out
}

/// A stream of random symbols and fragments, with its symbols (the decoder's order). The
/// lanes' bytes are put in the order the decoder asks for them, so this checks the decoder
/// against the encoder's arithmetic and itself, not the byte order against anything.
fn random_stream(widths: &[u8], units: usize, rng: &mut Rng) -> (Vec<u8>, Vec<u8>) {
    let n = widths.len();
    let mut symbols = vec![0u8; units * n];
    let mut queues: Vec<std::collections::VecDeque<u8>> = Vec::new();
    for (l, &w) in widths.iter().enumerate() {
        let mut bytes = Vec::new();
        let mut at = 0;
        while at < units {
            let longest = if rng.below(4) == 0 { 700 } else { 40 };
            let len = (1 + rng.below(longest) as usize).min(units - at);
            let z = rng.below(4).min(w as u32 - 1) as usize;
            let q = rng.below(16) as usize;
            let bits = w as usize - z;
            // the steeper models leave one slot to most symbols: keep those symbols rare
            let frag: Vec<u8> = (0..len)
                .map(|_| {
                    let top = if q >= 8 && rng.below(8) != 0 { 1u32 << bits.min(2) } else { 1 << bits };
                    rng.below(top) as u8
                })
                .collect();
            for (k, &s) in frag.iter().enumerate() {
                symbols[(at + k) * n + l] = s;
            }
            bytes.extend(fragment(&frag, w as usize, z, q, rng.below(2) == 0));
            at += len;
        }
        queues.push(bytes.into());
    }
    let mut stream = Vec::new();
    let got = lanes::run(widths, units, |lane| {
        let b = queues[lane].pop_front();
        stream.extend(b);
        b
    })
    .expect("the encoder's own fragments decode");
    assert_eq!(got, symbols);
    assert!(queues.iter().all(|q| q.is_empty()), "every fragment was read to its end");
    (stream, symbols)
}

#[test]
fn random_streams_round_trip_in_every_model() {
    let mut rng = Rng(0x5EED);
    let dump = std::env::var_os("XCL_BCPACK_DUMP").map(std::path::PathBuf::from);
    for round in 0..60 {
        for mode in [1u8, 3, 4, 5, 6, 7] {
            let widths = lane_widths(mode).unwrap();
            let units = [1, 2, 7, 64, 300, 2048][round % 6] * if mode == 1 || mode == 4 { 1 } else { 2 };
            let (stream, symbols) = random_stream(widths, units, &mut rng);
            assert_eq!(decode_lanes(mode, &stream, units * 8).as_deref(), Ok(&symbols[..]), "mode {mode} round {round}");
            // every byte of the stream is needed
            assert!(decode_lanes(mode, &stream[..stream.len() - 1], units * 8).is_err());
            if let Some(dir) = &dump {
                // for a comparison with other decoders: the stream and its lanes, lane by lane
                let stem = dir.join(format!("r{round}-bc{mode}-{units}"));
                let n = widths.len();
                let lane_major: Vec<u8> = (0..units * n).map(|i| symbols[(i % units) * n + i / units]).collect();
                std::fs::write(stem.with_extension("bin"), &stream).unwrap();
                std::fs::write(stem.with_extension("lanes"), lane_major).unwrap();
            }
        }
    }
}

/// Damaged input ends in an answer, never a panic or a hang. (The answer can be other
/// symbols: a stream has no checksum, and under a uniform model any bytes are symbols.)
#[test]
fn damaged_streams_end_cleanly() {
    let mut rng = Rng(7);
    for stem in ["bc1-2-0", "bc1-16-1", "bc1-16-9", "bc7-16-1", "bc1-700-2", "bc7-700-2"] {
        let (mode, units) = fixture(stem);
        let bin = data(&format!("{stem}.bin")).unwrap();
        let check = |bytes: &[u8], units: usize| {
            let lanes = decode_lanes(mode, bytes, units * 8);
            if mode == 1 {
                assert_eq!(super::decode(1, bytes, units * 8).is_ok(), lanes.is_ok());
            }
            lanes
        };
        // every byte of a stream is some lane's: a shorter one is never whole
        for cut in (0..bin.len()).step_by((bin.len() / 300).max(1)) {
            assert!(check(&bin[..cut], units).is_err(), "{stem} cut at {cut}");
        }
        for _ in 0..400 {
            let mut b = bin.clone();
            let at = rng.below(b.len() as u32) as usize;
            match rng.below(3) {
                0 => b[at] ^= 1 << rng.below(8),
                1 => b[at] = rng.next() as u8,
                _ => {
                    b.remove(at);
                }
            }
            let _ = check(&b, units);
        }
        // the wrong number of blocks, and a size that is no number of blocks
        assert!(check(&bin, units + 1).is_err());
        assert!(check(&bin, units - 1).is_err());
        assert_eq!(decode_lanes(mode, &bin, units * 8 + 4), Err(Error::OutputSize));
    }
    // noise, and headers that claim more than there is
    for len in [0usize, 1, 2, 3, 47, 48, 64, 500, 5000] {
        for _ in 0..40 {
            let noise: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            for mode in [1u8, 7] {
                let _ = decode_lanes(mode, &noise, 64 * 8);
                let _ = super::decode(mode, &noise, 64 * 8);
            }
        }
    }
    assert!(decode_lanes(1, &[0xFF; 4096], 1 << 20).is_err());
    assert!(decode_lanes(7, &[0x00; 4096], 1 << 20).is_err());
    assert_eq!(decode_lanes(2, &[], 16), Err(Error::Unsupported));
    assert_eq!(decode_lanes(1, &[], 0), Ok(Vec::new()));
}

/// The tables XCL ships (`bc1_fitted.rs`) are whole and in the layout [`bc1::assemble`] reads.
#[test]
fn fitted_bc1_tables_are_well_formed() {
    let t = &super::bc1_fitted::TABLES;
    // every quadrant once, the four uniform ones first
    let mut seen = [false; 256];
    for &q in &t.quadrants {
        assert!(!std::mem::replace(&mut seen[q as usize], true), "quadrant {q:#04x} twice");
    }
    let uniform = |s: usize| t.quadrants[s] & 3;
    for s in 0..4 {
        assert_eq!(t.quadrants[s], uniform(s) * 0x55);
    }
    for h in t.history.iter().flatten() {
        assert!(h.iter().zip([32, 64, 32, 32, 64, 32]).all(|(&c, range)| c < range));
    }

    // a first block that takes the endpoints of the block before the stream as they are
    let mut s = [0u8; 12];
    s[0] = 2;
    s[8..].copy_from_slice(&[0, 1, 2, 3]);
    let b = bc1::assemble(&s, t).unwrap();
    let e = t.history[255].unwrap().map(u16::from);
    assert_eq!(b[..2], (e[0] << 11 | e[1] << 5 | e[2]).to_le_bytes());
    assert_eq!(b[2..4], (e[3] << 11 | e[4] << 5 | e[5]).to_le_bytes());
    let row = |left: usize, right: usize| uniform(left) * 5 | uniform(right) * 5 << 4;
    assert_eq!(b[4..], [row(0, 1), row(0, 1), row(2, 3), row(2, 3)]);

    // every readable entry has a value: bcpack-rs never fitted slot 4, which real chunks do
    // read, so it is ported from ivall/xcl 8cad517 (VIctory's dictionary)
    assert!(t.history[0].is_none());
    assert_eq!(t.history.iter().flatten().count(), 255);
    assert_eq!(t.history[4], Some([12, 27, 13, 6, 12, 6]));

    // a first block that takes the endpoints of the block 252 before the stream (slot 4),
    // plus its residuals
    let mut s = [0u8; 12];
    s[0] = 7;
    s[2] = 2;
    s[8..].copy_from_slice(&[0, 1, 2, 3]);
    let (b, unfitted) = bc1::assemble_counted(&s, t).unwrap();
    assert_eq!(unfitted, 0);
    assert_eq!(b[..2], (13u16 << 11 | 27 << 5 | 13).to_le_bytes());
    assert_eq!(&b[2..4], &(6u16 << 11 | 12 << 5 | 6).to_le_bytes());
    assert_eq!(b[4..], [row(0, 1), row(0, 1), row(2, 3), row(2, 3)]);

    // and through the entropy stage: a two-block stream whose block 0 refers 252 blocks back
    // (slot 4) and whose block 1 takes block 0's endpoints
    let symbols = [7u8, 0, 2, 4, 6, 0, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let stream = stream_from_symbols(lane_widths(1).unwrap(), &symbols);
    let (blocks, unfitted) = bc1::decode_counted(&stream, 16, t).expect("the format allows it");
    assert_eq!(unfitted, 0);
    // block 0 on slot 4 + (1, 2, 3), (0, 1, 1); quadrant symbol 0 is a uniform quadrant
    assert_eq!(blocks[..2], (13u16 << 11 | 29 << 5 | 16).to_le_bytes());
    assert_eq!(blocks[2..4], (6u16 << 11 | 13 << 5 | 7).to_le_bytes());
    assert_eq!(blocks[4..8], [uniform(0) * 0x55; 4]);
    // block 1 one block back, no change
    assert_eq!(blocks[8..10], blocks[..2]);
    assert_eq!(blocks[10..12], blocks[2..4]);
}

// ---------------------------------------------------------------------------------------
// BC5
// ---------------------------------------------------------------------------------------

/// A stream of one flush per lane, carrying `symbols` (unit-major) through the fragment
/// encoder of this file. The lanes' bytes are put in the order the decoder asks for them.
fn stream_from_symbols(widths: &[u8], symbols: &[u8]) -> Vec<u8> {
    let n = widths.len();
    let units = symbols.len() / n;
    let mut queues: Vec<std::collections::VecDeque<u8>> = Vec::new();
    for (l, &w) in widths.iter().enumerate() {
        let lane: Vec<u8> = (0..units).map(|u| symbols[u * n + l]).collect();
        queues.push(fragment(&lane, w as usize, 0, 0, true).into());
    }
    let mut stream = Vec::new();
    lanes::run(widths, units, |lane| {
        let b = queues[lane].pop_front();
        stream.extend(b);
        b
    })
    .expect("the encoder's own fragments decode");
    assert!(queues.iter().all(|q| q.is_empty()), "every fragment was read to its end");
    stream
}

/// Lane 0's (E, O) symbols for the distance `d`: the inverse of [`bc5::distance`].
fn bc5_distance_symbols(d: usize) -> (u8, u8) {
    let (x, low) = if d <= 127 { (d, 0u8) } else { (255 - d, 1u8) };
    let mut e = low;
    let mut o = 0u8;
    for k in 0..4 {
        o |= (((x >> (2 * k)) & 1) as u8) << k;
    }
    for k in 0..3 {
        e |= (((x >> (2 * k + 1)) & 1) as u8) << (k + 1);
    }
    (e, o)
}

/// The residue symbol for `want` from its prediction `base`, modulo 256.
fn bc5_residue(want: u8, base: u8) -> u8 {
    let delta = (want as i32 - base as i32).rem_euclid(256);
    let delta = if delta > 127 { delta - 256 } else { delta };
    if delta >= 0 {
        (2 * delta) as u8
    } else {
        (-2 * delta - 1) as u8
    }
}

/// One BC5 block by hand: both channels, a reference inside the stream and one before it.
#[test]
fn bc5_block_rebuild() {
    let t = bc5_tables();
    let mut s = vec![[0u8; 24]; 3];
    // block 0: no reference, so both channels predict 128; channel 0's endpoints grow by 1
    // and 2, channel 1's by 0 and 3; every index pair is symbol 0, which the test table maps
    // to (0, 0)
    s[0][2] = 2;
    s[0][14] = 4;
    s[0][15] = 6;
    // block 1: one block back, the same endpoints; lane 4's E symbol 58 names (2, 7)
    let (e1, o1) = bc5_distance_symbols(1);
    assert_eq!((e1, o1), (0, 1));
    s[1][0] = e1;
    s[1][12] = o1;
    s[1][4] = 58;
    // block 2: 255 back, before the stream: history slot 256 + 2 - 255 = 3
    let (e2, o2) = bc5_distance_symbols(255);
    assert_eq!((e2, o2), (1, 0));
    s[2][0] = e2;
    s[2][12] = o2;
    let out = bc5::assemble(&s.concat(), t).unwrap();
    assert_eq!(out.len(), 48);
    assert_eq!(out[..8], [129, 130, 0, 0, 0, 0, 0, 0]);
    assert_eq!(out[8..16], [128, 131, 0, 0, 0, 0, 0, 0]);
    assert_eq!(out[16..24], [129, 130, 58, 0, 0, 0, 0, 0]);
    assert_eq!(out[24..32], [128, 131, 0, 0, 0, 0, 0, 0]);
    let h = t.preset[3].unwrap();
    assert_eq!(out[32..34], h[..2]);
    assert_eq!(out[40..42], h[2..]);
    assert_eq!(out[34..40], [0; 6]);
    assert_eq!(out[42..48], [0; 6]);

    // the distance mapping of the format notes, and its inverse over every value
    assert_eq!(bc5::distance(0, 0), 0);
    assert_eq!(bc5::distance(0, 1), 1);
    assert_eq!(bc5::distance(2, 0), 2);
    assert_eq!(bc5::distance(1, 0), 255);
    for d in 0..256 {
        let (e, o) = bc5_distance_symbols(d);
        assert_eq!(bc5::distance(e, o), d);
    }

    // an unknown history entry stops the stream instead of guessing
    let mut holes = bc5::Tables { pairs: t.pairs, preset: t.preset };
    holes.preset[3] = None;
    assert_eq!(bc5::assemble(&s.concat(), &holes), Err(Error::History { block: 2, slot: 3 }));
}

/// Blocks built here, coded into lane symbols, encoded into a stream, decoded and compared.
#[test]
fn bc5_blocks_round_trip_through_a_stream() {
    let t = bc5_tables();
    let widths = lane_widths(5).unwrap();
    let inverse: Vec<Option<u8>> = {
        let mut inv = vec![None; 64];
        for (sym, &(a, b)) in t.pairs.iter().enumerate() {
            inv[(a * 8 + b) as usize] = Some(sym as u8);
        }
        inv
    };
    let mut rng = Rng(0xB0C5);
    for &blocks in &[1usize, 2, 9, 64, 300] {
        let mut want = vec![[0u8; 16]; blocks];
        let mut idx = vec![[0u64; 2]; blocks];
        for i in 0..blocks {
            want[i][0] = rng.next() as u8;
            want[i][1] = rng.next() as u8;
            want[i][8] = rng.next() as u8;
            want[i][9] = rng.next() as u8;
            for ch in 0..2 {
                let mut w = 0u64;
                for p in 0..16 {
                    w |= (rng.below(8) as u64) << (3 * p);
                }
                idx[i][ch] = w;
                want[i][8 * ch + 2..8 * ch + 8].copy_from_slice(&w.to_le_bytes()[..6]);
            }
        }
        let mut symbols = vec![0u8; blocks * 24];
        for i in 0..blocks {
            // early blocks read the history, the others a block behind them
            let d = if i == 0 {
                0
            } else if i % 11 == 0 && i < 250 {
                251 + i % 4
            } else {
                1 + (i % 7).min(i - 1)
            };
            let base = if d == 0 {
                [128u8; 4]
            } else if d <= i {
                let p = &want[i - d];
                [p[0], p[1], p[8], p[9]]
            } else {
                t.preset[256 + i - d].expect("the test tables fit every slot")
            };
            let (e, o) = bc5_distance_symbols(d);
            symbols[i * 24] = e;
            symbols[i * 24 + 12] = o;
            for ch in 0..2 {
                symbols[i * 24 + 2 + ch] = bc5_residue(want[i][8 * ch], base[2 * ch]);
                symbols[i * 24 + 12 + 2 + ch] = bc5_residue(want[i][8 * ch + 1], base[2 * ch + 1]);
                let w = idx[i][ch];
                let px = |p: usize| ((w >> (3 * p)) & 7) as usize;
                for r in 0..4 {
                    symbols[i * 24 + 4 + 4 * ch + r] = inverse[px(4 * r) * 8 + px(4 * r + 1)].unwrap();
                    symbols[i * 24 + 12 + 4 + 4 * ch + r] = inverse[px(4 * r + 2) * 8 + px(4 * r + 3)].unwrap();
                }
            }
        }
        assert_eq!(bc5::assemble(&symbols, t).unwrap(), want.concat(), "{blocks} blocks");
        let stream = stream_from_symbols(widths, &symbols);
        assert_eq!(super::decode(5, &stream, blocks * 16).as_deref(), Ok(&want.concat()[..]), "{blocks} blocks");
        // every byte of the stream is needed, and the size must be a whole number of blocks
        assert!(super::decode(5, &stream[..stream.len() - 1], blocks * 16).is_err(), "{blocks} blocks");
        assert_eq!(super::decode(5, &stream, blocks * 16 + 1), Err(Error::OutputSize));
    }
}

/// The BC5 fixture of `tests/data/bcpack` (an independent encoder) goes through the entropy
/// stage and the rebuild. Its lane 1 holds noise, which the rebuild ignores.
#[test]
fn bc5_fixture_decodes_to_blocks() {
    let bin = data("bc5-16-1.bin").unwrap();
    let blocks = super::decode(5, &bin, 8 * 16).expect("the fixture decodes");
    assert_eq!(blocks.len(), 8 * 16);
}

/// The BC5 tables XCL ships (`bc5_tables.rs`) are whole and in the layout [`bc5::assemble`]
/// reads.
#[test]
fn fitted_bc5_tables_are_well_formed() {
    let t = &super::bc5_tables::TABLES;
    let mut seen = [false; 64];
    for &(a, b) in &t.pairs {
        assert!(a < 8 && b < 8);
        assert!(!std::mem::replace(&mut seen[(a * 8 + b) as usize], true), "pair ({a}, {b}) twice");
    }
    // slot 0 is never read; 214 of the 255 readable slots were fitted, the three the source
    // reports as confirmed among them
    assert!(t.preset[0].is_none());
    assert_eq!(t.preset.iter().flatten().count(), 214);
    for slot in [182, 183, 222] {
        assert!(t.preset[slot].is_some());
    }

    // a first block that takes the endpoints of the block 253 before the stream (slot 3) as
    // they are
    let mut s = [0u8; 24];
    let (e, o) = bc5_distance_symbols(253);
    s[0] = e;
    s[12] = o;
    let b = bc5::assemble(&s, t).unwrap();
    let h = t.preset[3].unwrap();
    assert_eq!(b[..2], h[..2]);
    assert_eq!(b[8..10], h[2..]);
    assert_eq!(b[2..8], [0; 6]);
    assert_eq!(b[10..16], [0; 6]);

    // an entry the fit does not have ends the rebuild
    let mut unknown = [0u8; 24];
    let (e, o) = bc5_distance_symbols(254); // slot 2
    unknown[0] = e;
    unknown[12] = o;
    assert_eq!(bc5::assemble(&unknown, t), Err(Error::History { block: 0, slot: 2 }));
}

// ---------------------------------------------------------------------------------------
// BC4 and BC3 (tables made up for the fixtures, as generate.py built them)
// ---------------------------------------------------------------------------------------

/// The made-up constants the BC3/BC4/BC5 fixtures were built with (`generate.py`): pair symbol
/// `s` is the selectors `(5s + 3) mod 64`, history entry `slot` holds `(7 slot + 13 k) mod 256`.
fn made_up_pairs() -> [u8; 64] {
    std::array::from_fn(|s| ((5 * s + 3) % 64) as u8)
}

fn made_up_history<const N: usize>(slot: usize) -> Option<[u8; N]> {
    Some(std::array::from_fn(|k| ((7 * slot + 13 * k) % 256) as u8))
}

fn bc4_tables() -> bc4::Tables {
    bc4::Tables { history: std::array::from_fn(made_up_history), pair_codes: made_up_pairs() }
}

#[test]
fn bc4_fixture_decodes_to_its_blocks() {
    let tables = bc4_tables();
    let bin = data("bc4-300-4.bin").unwrap();
    let want = data("bc4-300-4.bc4").unwrap();
    assert_eq!(bc4::decode(&bin, 300 * 8, &tables).as_deref(), Ok(&want[..]));
    // a block that predicts from an entry nobody knows is refused, not guessed
    let lanes = data("bc4-300-4.lanes").unwrap();
    let i = (0..300).find(|&i| bc4::back_reference(lanes[i]) > i).unwrap();
    let slot = 256 + i - bc4::back_reference(lanes[i]);
    let mut gap = bc4_tables();
    gap.history[slot] = None;
    assert_eq!(bc4::decode(&bin, 300 * 8, &gap), Err(Error::History { block: i, slot }));
    // a cut stream and the lanes' own errors come through
    assert!(bc4::decode(&bin[..bin.len() - 1], 300 * 8, &tables).is_err());
    assert_eq!(bc4::decode(&bin, 300 * 8 + 4, &tables), Err(Error::OutputSize));
}

/// Three blocks by hand: every field, both kinds of prediction, wrapping differences.
#[test]
fn bc4_block_rebuild() {
    let mut tables = bc4_tables();
    tables.pair_codes = std::array::from_fn(|s| s as u8);
    tables.history[255] = Some([250, 3]);
    let halves = [0o01u8, 0o23, 0o45, 0o67, 0o70, 0o52, 0o34, 0o16];
    #[rustfmt::skip]
    let symbols = [
        // 1 block back: the history's last entry; +10 and -4
        2, 0, 20, 7, halves[0], halves[1], halves[2], halves[3], halves[4], halves[5], halves[6], halves[7],
        // 1 block back: block 0; -1 and +0
        2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // no prediction: 128 for both; +1 and -128
        0, 0, 2, 255, 63, 63, 63, 63, 63, 63, 63, 63,
    ];
    let out = bc4::assemble(&symbols, &tables).unwrap();
    // texel (x, y) has bits 3 (4y + x): the left texel of a pair is its low three bits
    let rows = 0o16_34_52_70_67_45_23_01u64.to_le_bytes();
    assert_eq!(out[..8], [4, 255, rows[0], rows[1], rows[2], rows[3], rows[4], rows[5]]);
    assert_eq!(out[8..16], [3, 255, 0, 0, 0, 0, 0, 0]);
    assert_eq!(out[16..24], [129, 0, 255, 255, 255, 255, 255, 255]);
    let mut lane1 = symbols;
    lane1[12 + 1] = 1;
    assert_eq!(bc4::assemble(&lane1, &tables), Err(Error::Lane1 { block: 1 }));
}


fn bc3_tables() -> bc3::Tables {
    bc3::Tables {
        history: std::array::from_fn(made_up_history),
        pair_codes: made_up_pairs(),
        quadrants: std::array::from_fn(|q| ((37 * q + 11) % 256) as u8),
    }
}

#[test]
fn bc3_fixture_decodes_to_its_blocks() {
    let tables = bc3_tables();
    let bin = data("bc3-600-6.bin").unwrap();
    let want = data("bc3-600-6.bc3").unwrap();
    assert_eq!(bc3::decode(&bin, 300 * 16, &tables).as_deref(), Ok(&want[..]));
    let lanes = data("bc3-600-6.lanes").unwrap();
    let d = |i: usize| bc4::back_reference(bc5::interleave(lanes[2 * i], lanes[2 * i + 1]));
    let i = (0..300).find(|&i| d(i) > i).unwrap();
    let mut gap = bc3_tables();
    gap.history[256 + i - d(i)] = None;
    assert_eq!(bc3::decode(&bin, 300 * 16, &gap), Err(Error::History { block: i, slot: 256 + i - d(i) }));
    assert_eq!(bc3::decode(&bin, 300 * 16 - 8, &tables), Err(Error::OutputSize));
}

/// Two blocks by hand: every field, colour differences wrapping in their own range, and a
/// prediction from a rebuilt block's colours.
#[test]
fn bc3_block_rebuild() {
    let mut tables = bc3_tables();
    tables.pair_codes = std::array::from_fn(|s| s as u8);
    tables.quadrants = std::array::from_fn(|q| q as u8);
    tables.history[255] = Some([200, 100, 31, 0, 5, 1, 63, 16]);
    #[rustfmt::skip]
    let symbols = [
        // block 0, u0: one block back (the history's last entry); A0 +1; alpha rows' left
        // halves; R0 +1 (wraps to 0), G0 -1 (wraps to 63), B0 +0; top-left, bottom-left quadrants
        0, 0, 2, 0o01, 0o02, 0o03, 0o04, 2, 1, 0, 0x1B, 0xE4,
        // u1: A1 -1; the rows' right halves; R1 -2 (wraps to 31), G1 +0, B1 +15; top-right,
        // bottom-right quadrants
        1, 0, 1, 0o05, 0o06, 0o07, 0o10, 3, 0, 30, 0x55, 0xAA,
        // block 1: one block back, nothing else
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ];
    let out = bc3::assemble(&symbols, &tables).unwrap();
    let alpha = 0o10_04_07_03_06_02_05_01u64.to_le_bytes();
    let c0 = (0u16 << 11 | 63 << 5 | 5).to_le_bytes();
    let c1 = (31u16 << 11 | 63 << 5 | 31).to_le_bytes();
    #[rustfmt::skip]
    let block = [
        201, 99, alpha[0], alpha[1], alpha[2], alpha[3], alpha[4], alpha[5], c0[0], c0[1], c1[0], c1[1],
        // rows 0-1 from the top quadrants 0x1B | 0x55, rows 2-3 from 0xE4 | 0xAA
        0x5B, 0x51, 0xA4, 0xAE,
    ];
    assert_eq!(out[..16], block);
    let mut copy = block;
    copy[2..8].fill(0);
    copy[12..16].fill(0);
    assert_eq!(out[16..32], copy);
    let mut lane1 = symbols;
    lane1[12 + 1] = 1;
    assert_eq!(bc3::assemble(&lane1, &tables), Err(Error::Lane1 { block: 0 }));
}
