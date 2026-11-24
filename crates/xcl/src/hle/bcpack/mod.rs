//! BCPack, the console's lossless packing of BCn texture blocks (`DSTORAGE_BCPACK_MODE`, the
//! DirectStorage stage between zlib and swizzle): a stream unpacks to plain BCn blocks.
//!
//! A stream has no header. The fields of the blocks are split over 12 or 16 rANS lanes, one
//! symbol per lane for every 8 bytes of output, and the lanes share the byte stream
//! (`lanes.rs`). The blocks are then rebuilt from the symbols; that part differs per format.
//!
//! What is here:
//! * the entropy stage for BC1, BC3, BC4, BC5, BC6H and BC7 ([`lane_widths`]); no lane layout
//!   is published for BC2,
//! * the rebuild of BC1 blocks ([`bc1`]), which needs two tables of constants that are not in
//!   the stream ([`bc1::Tables`]). XCL has the estimates bcpack-rs made of them
//!   (`bc1_fitted.rs`), so its BC1 blocks are an approximation wherever an estimate is off;
//!   the one history entry real chunks read and bcpack-rs never fitted (slot 4) is ported from
//!   ivall/xcl's dictionary, and a block that reads an entry the table still lacks is built
//!   from zeros (as bcpack-rs does) instead of losing its whole chunk; the first such read of
//!   a process is logged,
//! * the rebuild of BC5 blocks ([`bc5`]), which also needs constants that are not in the
//!   stream ([`bc5::Tables`]). XCL has the index-pair table and the estimates
//!   leonida-asset-pipeline made of the history (`bc5_tables.rs`), so its BC5 blocks are an
//!   approximation wherever an estimate is off,
//! * the rebuild of BC3, BC4, BC6H and BC7 blocks ([`victory`], with the format's constant
//!   tables in `tables.rs`), ported from VIctory by ivall/xcl (PR #905). Its tables are the
//!   format's own, so those modes need no fitting. This is the only source XCL can take them
//!   from: the projects that fitted them are GPL-3.0, and VIctory's author allowed its use in
//!   this GPL-2.0-only crate (see `THIRD_PARTY.md`).
//!
//! Nothing here was compared with what a console writes for the same input: the entropy stage
//! follows Microsoft's published decoder sample and the block rules of VIctory, bcpack-rs
//! (Ahmet Dedeler), the clean-room decoder of leonida-asset-pipeline (wxrrry, MIT) and the
//! format notes of SixNG, whose authors report them as matching retail streams. The tests
//! check against streams built independently from those notes (`tests/data/bcpack`), not
//! against hardware output.

pub mod bc1;
mod bc1_fitted;
pub mod bc3;
pub mod bc4;
pub mod bc5;
mod bc5_tables;
mod lanes;
mod rans_tables;
mod tables;
mod victory;

/// Why a stream was not unpacked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// no block rebuild for this mode (or no lane layout, or no tables)
    Unsupported,
    /// the output is not a whole number of blocks, or too large
    OutputSize,
    /// the stream ends before the lane has its bytes
    Truncated { lane: usize },
    /// a fragment narrows the lane's symbols to nothing
    Width { lane: usize },
    /// the state points at the slot no symbol has
    Symbol { lane: usize },
    /// the lane's last symbol leaves a state no encoder starts from
    EndState { lane: usize },
    /// the lanes wait for each other or for a byte their fragments do not have
    Stuck,
    /// bytes other than zero padding follow the stream
    Trailing { consumed: usize },
    /// BC1 lane 1 is not zero; what it then means is not known
    Lane1 { block: usize },
    /// a block refers to an entry of the built-in history that is not known
    History { block: usize, slot: usize },
    /// a block rebuild from `victory` refused the stream, with its reason
    Decode(&'static str),
}

impl Error {
    /// The kind of failure, without its numbers.
    pub fn kind(&self) -> &'static str {
        match self {
            Error::Unsupported => "unsupported",
            Error::OutputSize => "output size",
            Error::Truncated { .. } => "truncated",
            Error::Width { .. } => "width",
            Error::Symbol { .. } => "reserved slot",
            Error::EndState { .. } => "end state",
            Error::Stuck => "stuck",
            Error::Trailing { .. } => "trailing data",
            Error::Lane1 { .. } => "lane 1",
            Error::History { .. } => "unknown history",
            Error::Decode(_) => "decode",
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Error::Unsupported => write!(f, "no decoder for this mode"),
            Error::OutputSize => write!(f, "the output size is not a whole number of blocks"),
            Error::Truncated { lane } => write!(f, "the stream ends while lane {lane} waits for a byte"),
            Error::Width { lane } => write!(f, "a fragment of lane {lane} narrows its symbols to nothing"),
            Error::Symbol { lane } => write!(f, "lane {lane} reads the reserved slot"),
            Error::EndState { lane } => write!(f, "lane {lane} ends in a state no encoder starts from"),
            Error::Stuck => write!(f, "the lanes cannot go on"),
            Error::Trailing { consumed } => write!(f, "data follows the stream's {consumed} bytes"),
            Error::Lane1 { block } => write!(f, "BC1 block {block} has a non-zero lane 1 symbol"),
            Error::History { block, slot } => write!(f, "block {block} refers to the unknown history entry {slot}"),
            Error::Decode(why) => write!(f, "BCPack: {why}"),
        }
    }
}

/// The symbol width of each lane of a `DSTORAGE_BCPACK_MODE` (1 BC1 … 7 BC7), in stream order.
/// Every lane holds one symbol per 8 bytes of output: one per BC1/BC4 block, two per 16-byte
/// block.
pub fn lane_widths(mode: u8) -> Option<&'static [u8]> {
    Some(match mode {
        1 => &[8, 8, 5, 6, 5, 5, 6, 5, 8, 8, 8, 8],
        3 => &[4, 4, 8, 6, 6, 6, 6, 5, 6, 5, 8, 8],
        4 => &[8, 8, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6],
        5 => &[4, 4, 8, 8, 6, 6, 6, 6, 6, 6, 6, 6],
        6 => &[4, 4, 8, 8, 8, 5, 5, 5, 6, 6, 6, 8, 8, 8, 8, 5],
        7 => &[4, 4, 8, 8, 8, 8, 8, 7, 7, 5, 6, 8, 8, 8, 8, 5],
        _ => return None,
    })
}

/// The symbols of a stream that unpacks to `out_len` bytes, unit by unit
/// (`symbols[unit * lanes + lane]`). Zero padding may follow the stream.
pub fn decode_lanes(mode: u8, src: &[u8], out_len: usize) -> Result<Vec<u8>, Error> {
    let widths = lane_widths(mode).ok_or(Error::Unsupported)?;
    if out_len % 8 != 0 {
        return Err(Error::OutputSize);
    }
    let (symbols, consumed) = lanes::decode(src, widths, out_len / 8)?;
    if src[consumed..].iter().any(|&b| b != 0) {
        return Err(Error::Trailing { consumed });
    }
    Ok(symbols)
}

/// The BC1 constants, when XCL has them: the estimates of bcpack-rs, and in the tests the
/// made-up set their fixtures were built with.
fn bc1_tables() -> Option<&'static bc1::Tables> {
    #[cfg(test)]
    {
        Some(tests::tables())
    }
    #[cfg(not(test))]
    {
        Some(&bc1_fitted::TABLES)
    }
}

/// The BC5 tables, when XCL has them: the estimates of leonida-asset-pipeline, and in the
/// tests the made-up set their fixtures were built with.
fn bc5_tables() -> &'static bc5::Tables {
    #[cfg(test)]
    {
        tests::bc5_tables()
    }
    #[cfg(not(test))]
    {
        &bc5_tables::TABLES
    }
}

/// The bytes of one block of `mode`.
pub fn block_size(mode: u8) -> Option<usize> {
    Some(match mode {
        1 | 4 => 8,
        3 | 5 | 6 | 7 => 16,
        _ => return None,
    })
}

/// Whether [`decode`] has a block rebuild for `mode`.
pub fn can_decode(mode: u8) -> bool {
    match mode {
        1 => bc1_tables().is_some(),
        5 => true,
        // the format's own tables (VIctory) rebuild these
        3 | 4 | 6 | 7 => true,
        _ => false,
    }
}

/// Unpack a stream of `mode` to its `out_len` bytes of BCn blocks.
pub fn decode(mode: u8, src: &[u8], out_len: usize) -> Result<Vec<u8>, Error> {
    match (mode, bc1_tables()) {
        (1, Some(t)) => {
            let (blocks, unfitted) = bc1::decode_counted(src, out_len, t)?;
            report_unfitted(unfitted);
            Ok(blocks)
        }
        (5, _) => bc5::decode(src, out_len, bc5_tables()),
        // BC3, BC4, BC6H and BC7: the port of VIctory's decoder and its tables
        (3 | 4 | 6 | 7, _) => {
            let block = block_size(mode).ok_or(Error::Unsupported)?;
            if out_len == 0 || out_len % block != 0 {
                return Err(Error::OutputSize);
            }
            let mut out = vec![0u8; out_len];
            victory::decode(mode, src, &mut out, victory::Tables::builtin())
                .map_err(|e| Error::Decode(e.0))?;
            Ok(out)
        }
        _ => Err(Error::Unsupported),
    }
}

/// One line the first time a BC1 block reads a history entry the fitted table lacks: such a
/// block is rebuilt from zeros and is probably wrong, the rest of its chunk is not.
fn report_unfitted(blocks: usize) {
    if blocks > 0 && !UNFITTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        crate::xlog_info!(
            "DirectStorage: BCPack BC1: {blocks} blocks of this chunk read a history entry the fitted table lacks; they are rebuilt from zeros"
        );
    }
}

static UNFITTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
pub(crate) mod tests;
