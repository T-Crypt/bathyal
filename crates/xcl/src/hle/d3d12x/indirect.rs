//! Indirect argument records: their layout in a command signature, the count buffer that
//! bounds how many run, and the maxima and per-list work budget every record is held to
//! before the GPU executes it.
//!
//! A console command signature is a byte stride and a list of argument descriptions, the
//! same shape the PC uses (the console's own DispatchX maps onto the PC's Dispatch). The
//! metadata is recorded when the signature is created (`device_x::create_command_signature_x`)
//! so a later ExecuteIndirect can read its records without asking the driver.
//!
//! Nothing here touches a device: a record is plain bytes, so the parser and the clamps are
//! unit-testable on their own.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// PC `D3D12_INDIRECT_ARGUMENT_TYPE` values (the console uses these too, bar DispatchX).
pub const ARG_DRAW: u32 = 0;
pub const ARG_DRAW_INDEXED: u32 = 1;
pub const ARG_DISPATCH: u32 = 2;
pub const ARG_VERTEX_BUFFER: u32 = 3;
pub const ARG_INDEX_BUFFER: u32 = 4;
pub const ARG_CONSTANT: u32 = 5;
pub const ARG_CBV: u32 = 6;
pub const ARG_SRV: u32 = 7;
pub const ARG_UAV: u32 = 8;
pub const ARG_DISPATCH_MESH: u32 = 10;
pub const ARG_INCREMENTING_CONSTANT: u32 = 11;

/// documented maxima a record's counts are clamped to (D3D12's own bounds, tightened)
pub const MAX_VERTICES: u32 = 16 << 20;
pub const MAX_INDICES: u32 = 16 << 20;
pub const MAX_INSTANCES: u32 = 1 << 16;
/// D3D12's limit on one dimension of a dispatch
pub const MAX_GROUPS_PER_DIM: u32 = 65_535;

/// Work one command list may charge to indirect records in one recording. A record past it
/// runs empty. `XCL_INDIRECT_BUDGET` overrides it (records, not bytes).
pub fn work_budget() -> u64 {
    static B: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_INDIRECT_BUDGET").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(256 << 20))
}

/// One argument description: where its bytes sit in a record, and (for the arguments that
/// change a root parameter) which parameter it writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arg {
    pub kind: u32,
    /// byte offset of the argument in the record
    pub offset: u32,
    /// bytes the argument takes
    pub size: u32,
    /// root parameter index for Constant / CBV / SRV / UAV arguments
    pub root: u32,
    /// destination 32-bit word of a Constant argument
    pub dest: u32,
    /// 32-bit words of a Constant argument
    pub count: u32,
}

/// A command signature's record layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    /// bytes between records
    pub stride: u32,
    pub args: Vec<Arg>,
}

impl Signature {
    /// From a console command-signature desc (`ByteStride`, `NumArgumentDescs`,
    /// `pArgumentDescs`), with the console argument types already mapped to PC ones.
    ///
    /// # Safety
    /// `desc` points at a committed `D3D12_COMMAND_SIGNATURE_DESC` for 24 bytes.
    pub unsafe fn from_console_desc(desc: *const u8) -> Option<Signature> {
        if desc.is_null() {
            return None;
        }
        let head = std::slice::from_raw_parts(desc, 24);
        let stride = u32::from_le_bytes(head[0..4].try_into().ok()?);
        let n = u32::from_le_bytes(head[4..8].try_into().ok()?) as usize;
        let args = u64::from_le_bytes(head[8..16].try_into().ok()?) as *const [u32; 4];
        if args.is_null() || n == 0 || n > 64 {
            return None;
        }
        let raws: Vec<[u32; 4]> = (0..n).map(|i| args.add(i).read_unaligned()).collect();
        Signature::from_args(stride, &raws)
    }

    /// From a byte stride and the raw argument descriptions. An argument type whose size is
    /// not known makes the whole signature unknown (the caller then runs nothing rather
    /// than pass an unbounded record).
    pub fn from_args(stride: u32, raws: &[[u32; 4]]) -> Option<Signature> {
        if stride == 0 || raws.is_empty() {
            return None;
        }
        let mut offset = 0u32;
        let mut args = Vec::with_capacity(raws.len());
        for a in raws {
            let (kind, size, root, dest, count) = match a[0] {
                ARG_DRAW => (ARG_DRAW, 16, 0, 0, 0),
                ARG_DRAW_INDEXED => (ARG_DRAW_INDEXED, 20, 0, 0, 0),
                ARG_DISPATCH | ARG_DISPATCH_MESH => (a[0], 12, 0, 0, 0),
                ARG_VERTEX_BUFFER | ARG_INDEX_BUFFER => (a[0], 16, 0, 0, 0),
                ARG_CONSTANT => (ARG_CONSTANT, 4u32.checked_mul(a[3])?, a[1], a[2], a[3]),
                ARG_CBV | ARG_SRV | ARG_UAV => (a[0], 8, a[1], 0, 1),
                ARG_INCREMENTING_CONSTANT => (ARG_INCREMENTING_CONSTANT, 4, a[1], 0, 1),
                _ => return None,
            };
            args.push(Arg { kind, offset, size, root, dest, count });
            offset = offset.checked_add(size)?;
        }
        // a stride larger than the arguments is padding; a smaller one would run into the
        // next record
        if offset > stride {
            return None;
        }
        Some(Signature { stride, args })
    }

    /// The one Draw / DrawIndexed / Dispatch argument, if the record carries one.
    pub fn count_arg(&self) -> Option<&Arg> {
        self.args.iter().find(|a| matches!(a.kind, ARG_DRAW | ARG_DRAW_INDEXED | ARG_DISPATCH | ARG_DISPATCH_MESH))
    }

    /// The arguments that change a root parameter, in record order.
    pub fn root_args(&self) -> impl Iterator<Item = &Arg> {
        self.args.iter().filter(|a| (ARG_CONSTANT..=ARG_UAV).contains(&a.kind))
    }
}

/// The kind of count argument a record carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Draw,
    DrawIndexed,
    Dispatch,
    DispatchMesh,
}

/// One indirect argument record, as read from title memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record {
    pub kind: Kind,
    /// Draw: vertex, instance, start vertex, start instance.
    /// DrawIndexed: index, instance, start index, base vertex (u32 bits), start instance.
    /// Dispatch / DispatchMesh: x, y, z.
    pub counts: [u32; 5],
}

impl Record {
    pub fn empty(kind: Kind) -> Record {
        Record { kind, counts: [0; 5] }
    }

    /// Thread groups a two-pass pipeline's compute pass runs for this record, and whether it
    /// is within budget after clamping.
    pub fn groups(&self, max_vertices: u32) -> u32 {
        match self.kind {
            Kind::Dispatch | Kind::DispatchMesh => {
                let (x, y, z) = (self.counts[0] as u64, self.counts[1] as u64, self.counts[2] as u64);
                u32::try_from(x * y * z).unwrap_or(u32::MAX)
            }
            Kind::Draw | Kind::DrawIndexed => {
                let count = self.counts[0] as u64;
                u32::try_from(count.div_ceil(max_vertices.max(1) as u64)).unwrap_or(u32::MAX)
            }
        }
    }
}

/// Read one record's count argument from `bytes` (the record starts at 0).
pub fn record_of(sig: &Signature, bytes: &[u8]) -> Option<Record> {
    let a = sig.count_arg()?;
    let w = |i: usize| -> Option<u32> {
        let at = a.offset as usize + 4 * i;
        Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
    };
    let kind = match a.kind {
        ARG_DRAW => Kind::Draw,
        ARG_DRAW_INDEXED => Kind::DrawIndexed,
        ARG_DISPATCH => Kind::Dispatch,
        ARG_DISPATCH_MESH => Kind::DispatchMesh,
        _ => return None,
    };
    let n = match kind {
        Kind::Draw | Kind::Dispatch | Kind::DispatchMesh => 4,
        Kind::DrawIndexed => 5,
    };
    let mut counts = [0u32; 5];
    for (i, c) in counts.iter_mut().enumerate().take(n) {
        *c = w(i)?;
    }
    Some(Record { kind, counts })
}

/// Parse up to `n` records, each `sig.stride` bytes into `bytes`.
pub fn parse_records(sig: &Signature, bytes: &[u8], n: u32) -> Vec<Record> {
    let stride = sig.stride as usize;
    let mut out = Vec::with_capacity(n.min(1 << 20) as usize);
    for i in 0..n as usize {
        let at = match i.checked_mul(stride) {
            Some(a) => a,
            None => break,
        };
        match bytes.get(at..) {
            Some(b) if b.len() >= stride => match record_of(sig, b) {
                Some(r) => out.push(r),
                None => break,
            },
            _ => break,
        }
    }
    out
}

/// Clamp one record to the documented maxima and charge the list's budget. Returns whether
/// the record still runs anything: a record past the budget is zeroed and returns false.
pub fn clamp_record(r: &mut Record, budget: &mut u64) -> bool {
    match r.kind {
        Kind::Draw => {
            r.counts[0] = r.counts[0].min(MAX_VERTICES);
            r.counts[1] = r.counts[1].min(MAX_INSTANCES);
        }
        Kind::DrawIndexed => {
            r.counts[0] = r.counts[0].min(MAX_INDICES);
            r.counts[1] = r.counts[1].min(MAX_INSTANCES);
            // counts[3] is a signed base vertex: left alone
        }
        Kind::Dispatch | Kind::DispatchMesh => {
            for c in r.counts.iter_mut().take(3) {
                *c = (*c).min(MAX_GROUPS_PER_DIM);
            }
        }
    }
    let cost = match r.kind {
        Kind::Draw | Kind::DrawIndexed => r.counts[0] as u64 * (r.counts[1].max(1) as u64),
        Kind::Dispatch | Kind::DispatchMesh => {
            r.counts[0] as u64 * r.counts[1].max(1) as u64 * r.counts[2].max(1) as u64
        }
    };
    if cost > *budget {
        // past the budget: run empty, keeping the kind
        r.counts = [0; 5];
        return false;
    }
    *budget -= cost;
    // an empty record still passes through as empty (zero vertices / zero groups)
    match r.kind {
        Kind::Draw | Kind::DrawIndexed => r.counts[0] != 0 && r.counts[1] != 0,
        Kind::Dispatch | Kind::DispatchMesh => r.counts[0] != 0 && r.counts[1] != 0 && r.counts[2] != 0,
    }
}

/// Write a record's count argument back into `bytes` at `offset` (the clamped values the GPU
/// runs with).
pub fn write_record(sig: &Signature, bytes: &mut [u8], offset: usize, r: &Record) -> bool {
    let Some(a) = sig.count_arg() else { return false };
    let at = offset + a.offset as usize;
    let n = match r.kind {
        Kind::Draw | Kind::Dispatch | Kind::DispatchMesh => 4,
        Kind::DrawIndexed => 5,
    };
    match bytes.get_mut(at..at + 4 * n) {
        Some(dst) => {
            for (i, c) in r.counts.iter().take(n).enumerate() {
                dst[4 * i..4 * i + 4].copy_from_slice(&c.to_le_bytes());
            }
            true
        }
        None => false,
    }
}

static SIGNATURES: Mutex<Option<HashMap<usize, Arc<Signature>>>> = Mutex::new(None);

/// The argument layout of command signature `sig` (an object address), if XCL created it.
pub fn signature_of(sig: usize) -> Option<Arc<Signature>> {
    SIGNATURES.lock().unwrap().as_ref()?.get(&sig).cloned()
}

/// Record the layout of command signature `sig`.
pub fn note_signature(sig: usize, s: Signature) {
    SIGNATURES.lock().unwrap().get_or_insert_with(HashMap::new).insert(sig, Arc::new(s));
}

#[cfg(test)]
pub(crate) fn forget_signature_for_test(sig: usize) {
    if let Some(m) = SIGNATURES.lock().unwrap().as_mut() {
        m.remove(&sig);
    }
}

#[cfg(test)]
static LAST_CLAMPED: Mutex<Option<Vec<Record>>> = Mutex::new(None);

/// Remember the last records an indirect execution clamped, for a test to read back.
#[cfg(test)]
pub(crate) fn remember_for_test(records: &[Record]) {
    *LAST_CLAMPED.lock().unwrap() = Some(records.to_vec());
}

/// The records of the last indirect execution, as they were run (clamped).
#[cfg(test)]
pub(crate) fn last_clamped_for_test() -> Option<Vec<Record>> {
    LAST_CLAMPED.lock().unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draw_signature() -> [u8; 24] {
        // leaked: the desc must keep pointing at it for the parser's read
        let raw: &'static [[u32; 4]] = Box::leak(vec![[ARG_DRAW, 0, 0, 0]].into_boxed_slice());
        let mut head = [0u8; 24];
        head[0..4].copy_from_slice(&16u32.to_le_bytes()); // stride
        head[4..8].copy_from_slice(&1u32.to_le_bytes()); // one argument
        head[8..16].copy_from_slice(&(raw.as_ptr() as u64).to_le_bytes());
        head
    }

    /// A record at `at` with the given words.
    fn put(buf: &mut [u8], at: usize, words: &[u32]) {
        for (i, w) in words.iter().enumerate() {
            buf[at + 4 * i..at + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
        }
    }

    #[test]
    fn parser_honours_count_offset_count_buffer_and_maximum() {
        let head = draw_signature();
        let sig = unsafe { Signature::from_console_desc(head.as_ptr()) }.expect("signature");
        // records: (vertex, instance, start vertex, start instance)
        let mut buf = vec![0u8; 64];
        put(&mut buf, 0, &[4, 2, 0, 0]);
        put(&mut buf, 16, &[6, 1, 0, 0]);
        put(&mut buf, 32, &[8, 3, 0, 0]);
        put(&mut buf, 48, &[10, 1, 0, 0]);
        // a count buffer says two records from its second word
        let count_words = [7u32, 2];
        let count_bytes: Vec<u8> = count_words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let count = u32::from_le_bytes(count_bytes[4..8].try_into().unwrap());
        assert_eq!(count, 2);

        let rs = parse_records(&sig, &buf, count.min(5));
        assert_eq!(rs.len(), 2);
        assert_eq!(rs[0].counts[..4], [4, 2, 0, 0]);
        assert_eq!(rs[1].counts[..4], [6, 1, 0, 0]);

        // the caller's maximum bounds a larger count
        let rs = parse_records(&sig, &buf, 4u32.min(3));
        assert_eq!(rs.len(), 3);
        assert_eq!(rs[2].counts[0], 8);
    }

    #[test]
    fn clamp_applies_maxima_and_the_list_budget() {
        // maxima: a record of all ones clamps, and still runs if the budget allows
        let mut budget = u64::MAX;
        let mut draw = Record { kind: Kind::Draw, counts: [u32::MAX, u32::MAX, 0, 0, 0] };
        assert!(clamp_record(&mut draw, &mut budget));
        assert_eq!(draw.counts[0], MAX_VERTICES);
        assert_eq!(draw.counts[1], MAX_INSTANCES);

        let mut dispatch = Record { kind: Kind::Dispatch, counts: [u32::MAX; 5] };
        let mut budget = u64::MAX;
        assert!(clamp_record(&mut dispatch, &mut budget));
        assert_eq!(&dispatch.counts[..3], &[MAX_GROUPS_PER_DIM; 3]);

        // budget: the first record is charged, the second (past it) is zeroed
        let mut budget = 10u64;
        let a = Record { kind: Kind::Draw, counts: [4, 1, 0, 0, 0] };
        let b = Record { kind: Kind::Draw, counts: [8, 1, 0, 0, 0] };
        let mut a = a;
        let mut b = b;
        assert!(clamp_record(&mut a, &mut budget));
        assert_eq!(budget, 6);
        assert!(!clamp_record(&mut b, &mut budget));
        assert_eq!(b.counts, [0; 5]);

        // zero and the exact budget edge
        let mut budget = 4u64;
        let mut z = Record { kind: Kind::Draw, counts: [0, 0, 0, 0, 0] };
        assert!(!clamp_record(&mut z, &mut budget)); // zero vertices: runs empty
        assert_eq!(budget, 4);
        let mut e = Record { kind: Kind::Draw, counts: [4, 1, 0, 0, 0] };
        assert!(clamp_record(&mut e, &mut budget));
        assert_eq!(budget, 0);
    }

    #[test]
    fn root_arguments_are_located_in_record_order() {
        // stride 32: a draw (16) then a constant (root 2, dest 1, 2 words)
        let raws = [[ARG_DRAW, 0, 0, 0], [ARG_CONSTANT, 2, 1, 2]];
        let sig = Signature::from_args(32, &raws).expect("signature");
        let c = sig.root_args().next().expect("one root argument");
        assert_eq!((c.kind, c.offset, c.size, c.root, c.dest, c.count), (ARG_CONSTANT, 16, 8, 2, 1, 2));
        // an argument type with no known size makes the layout unknown
        assert!(Signature::from_args(16, &[[ARG_DISPATCH_MESH + 100, 0, 0, 0]]).is_none());
        // a stride too small for the arguments is rejected
        assert!(Signature::from_args(8, &[[ARG_DRAW, 0, 0, 0]]).is_none());
    }

    #[test]
    fn rewrite_writes_the_clamped_words_back() {
        let raws = [[ARG_DRAW, 0, 0, 0]];
        let sig = Signature::from_args(16, &raws).unwrap();
        let mut bytes = vec![0u8; 16];
        let mut r = Record { kind: Kind::Draw, counts: [u32::MAX, u32::MAX, 0, 0, 0] };
        let mut budget = u64::MAX;
        clamp_record(&mut r, &mut budget);
        assert!(write_record(&sig, &mut bytes, 0, &r));
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), MAX_VERTICES);
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), MAX_INSTANCES);
    }
}
