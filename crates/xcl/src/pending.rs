//! What the title is still waiting on: one line per category of outstanding work, every
//! `XCL_PENDING_EVERY` seconds (`XCL_PENDING=1`: 30 s, both off by default).
//!
//! A run that sits on a loading screen gives no sign of what it waits for: a request that
//! never completes, an async block nobody signals, a pipeline that is still compiling. Each
//! such site records a *ticket* when it starts and drops it when the work finishes; the
//! reporter reads the ages out. Every line is built from small numbers only (a size, a queue
//! or slot id, a BCPack mode, a timeout) – never file names, paths or game data.
//!
//! The registry costs one short mutex per start and finish, and nothing at all when
//! `enabled()` is false: call sites test it before they make a ticket.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// A category of outstanding work (the order is the report order).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// DirectStorage requests enqueued but not executed yet.
    DStorageRequest = 0,
    /// Status-array slots set to `E_PENDING` whose entry has not run yet.
    DStorageStatus,
    /// Fence signals enqueued on a queue but not yet made.
    DStorageSignal,
    /// XAsync blocks begun but not completed.
    XAsync,
    /// Overlapped file reads still inside `ReadFile`.
    FileRead,
    /// Title threads blocked on a fence event, a frame event, or GPU memory commit.
    TitleWait,
    /// Pipelines whose build (translation and driver compile) is running.
    PsoBuild,
    /// Shader compiles queued for the DXC workers.
    ShaderCompile,
}

const KINDS: usize = 8;

impl Kind {
    pub const ALL: [Kind; KINDS] = [
        Kind::DStorageRequest,
        Kind::DStorageStatus,
        Kind::DStorageSignal,
        Kind::XAsync,
        Kind::FileRead,
        Kind::TitleWait,
        Kind::PsoBuild,
        Kind::ShaderCompile,
    ];

    fn label(self) -> &'static str {
        match self {
            Kind::DStorageRequest => "DirectStorage requests",
            Kind::DStorageStatus => "DirectStorage status slots",
            Kind::DStorageSignal => "DirectStorage fence signals",
            Kind::XAsync => "XAsync blocks",
            Kind::FileRead => "overlapped file reads",
            Kind::TitleWait => "title waits",
            Kind::PsoBuild => "pipeline builds",
            Kind::ShaderCompile => "shader compiles",
        }
    }

    /// A short description of one item: kind, size, queue or slot id, and age is added by the
    /// reporter. Holds no game content.
    fn describe(self, m: &Meta) -> String {
        match self {
            Kind::DStorageRequest => {
                let source = if m.extra & 1 != 0 { "from memory" } else { "from a file" };
                let mode = if m.mode == 0 { "no BCPack".to_string() } else { format!("BCPack mode {}", m.mode) };
                format!("queue {}, {} bytes {source}, {mode}", m.index, m.size)
            }
            Kind::DStorageStatus => format!("status array {}, slot {} still E_PENDING", m.index, m.size),
            Kind::DStorageSignal => format!("queue {}, fence value {:#x} not signalled", m.index, m.size),
            Kind::XAsync => format!("async block {:#x}", m.size),
            Kind::FileRead => format!("{} bytes{}", m.size, if m.extra & 1 != 0 { " (overlapped)" } else { "" }),
            Kind::TitleWait => match m.index {
                0 => format!("frame event wait, type {}, timeout {} ms", m.size, m.extra),
                1 => format!("fence wait for value {:#x}", m.size),
                _ => format!("waiting for GPU memory commit, {} bytes needed free", m.size),
            },
            Kind::PsoBuild => format!("{} pipeline {:#x}", if m.extra & 1 != 0 { "compute" } else { "graphics" }, m.size),
            Kind::ShaderCompile => format!("DXC compile, {} bytes of source", m.size),
        }
    }
}

/// The small numbers a ticket carries; every call site fills only what it has.
#[derive(Clone, Copy, Default, Debug)]
pub struct Meta {
    /// a size in bytes, a fence value, or an identity/address
    pub size: u64,
    /// a queue id, a status-array id, a kind tag, or a slot index
    pub index: u32,
    /// a BCPack mode
    pub mode: u32,
    /// flags: bit 0 memory source / overlapped / compute; or a timeout in ms
    pub extra: u32,
}

struct Item {
    start: Instant,
    meta: Meta,
    /// the "still outstanding after N s" line was already logged
    logged: bool,
}

/// One outstanding item: it is forgotten when the ticket is dropped (or `finish` is called
/// early). `Ticket::NONE` does nothing.
pub struct Ticket {
    kind: u8,
    seq: u64,
}

impl Ticket {
    pub const NONE: Ticket = Ticket { kind: 0, seq: 0 };

    /// Forget the item now.
    pub fn finish(mut self) {
        let (kind, seq) = (self.kind, self.seq);
        self.seq = 0;
        if seq != 0 && (kind as usize) < KINDS {
            registry(kind as usize).lock().unwrap_or_else(|e| e.into_inner()).remove(&seq);
        }
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        if self.seq == 0 {
            return;
        }
        let ki = self.kind as usize;
        if ki < KINDS {
            registry(ki).lock().unwrap_or_else(|e| e.into_inner()).remove(&self.seq);
        }
    }
}

static NEXT: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
static REGS: [OnceLock<Mutex<HashMap<u64, Item>>>; KINDS] = [const { OnceLock::new() }; KINDS];
static ENABLED: AtomicBool = AtomicBool::new(false);

/// An item older than this is logged once on its own line.
const SLOW_ITEM: Duration = Duration::from_secs(10);

fn registry(ki: usize) -> &'static Mutex<HashMap<u64, Item>> {
    REGS[ki].get_or_init(|| Mutex::new(HashMap::new()))
}

/// Whether the reporter runs. Call sites test this first so the off case costs one atomic load.
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// A test that checks the registry turns it on, and off again when it ends.
#[cfg(test)]
pub(crate) fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::Release);
}

/// How many items of `kind` were begun for `size` (`Meta::size`, the identity the site
/// recorded). The registry is one per process and the suite runs its tests in parallel, so a
/// test counts its own items rather than a category.
#[cfg(test)]
pub(crate) fn count_matching(kind: Kind, size: u64) -> usize {
    registry(kind as usize).lock().unwrap_or_else(|e| e.into_inner()).values().filter(|i| i.meta.size == size).count()
}

/// Record an outstanding item (cheap: one short mutex).
pub fn begin(kind: Kind, meta: Meta) -> Ticket {
    let ki = kind as usize;
    let seq = NEXT[ki].fetch_add(1, Ordering::Relaxed) + 1;
    registry(ki).lock().unwrap_or_else(|e| e.into_inner()).insert(seq, Item { start: Instant::now(), meta, logged: false });
    Ticket { kind: ki as u8, seq }
}

/// One category's state for a report or a test.
pub struct Summary {
    pub kind: Kind,
    pub count: usize,
    pub oldest_age: Option<f64>,
    pub oldest_description: Option<String>,
}

/// Counts and the oldest item of every category.
pub fn snapshot() -> Vec<Summary> {
    let now = Instant::now();
    Kind::ALL
        .iter()
        .map(|&kind| {
            let g = registry(kind as usize).lock().unwrap_or_else(|e| e.into_inner());
            let oldest = g.values().min_by_key(|i| i.start);
            Summary {
                kind,
                count: g.len(),
                oldest_age: oldest.map(|i| now.saturating_duration_since(i.start).as_secs_f64()),
                oldest_description: oldest.map(|i| kind.describe(&i.meta)),
            }
        })
        .collect()
}

/// The lines a report would log now, without logging them. Items older than `SLOW_ITEM` that
/// were not logged yet are marked and get their own line.
fn report_lines(now: Instant) -> Vec<String> {
    let mut lines = Vec::new();
    for kind in Kind::ALL {
        let mut g = registry(kind as usize).lock().unwrap_or_else(|e| e.into_inner());
        if g.is_empty() {
            continue;
        }
        if let Some(item) = g.values().min_by_key(|i| i.start) {
            lines.push(format!(
                "pending: {}: {} outstanding, oldest {:.1} s: {}",
                kind.label(),
                g.len(),
                now.saturating_duration_since(item.start).as_secs_f64(),
                kind.describe(&item.meta)
            ));
        }
        for item in g.values_mut() {
            let age = now.saturating_duration_since(item.start);
            if !item.logged && age >= SLOW_ITEM {
                item.logged = true;
                lines.push(format!("pending: {} still outstanding {:.1} s: {}", kind.label(), age.as_secs_f64(), kind.describe(&item.meta)));
            }
        }
    }
    lines
}

/// Log one line per non-empty category, plus each item that just passed `SLOW_ITEM`.
pub fn report() {
    for line in report_lines(Instant::now()) {
        crate::xlog_info!("{line}");
    }
}

/// Read `XCL_PENDING` / `XCL_PENDING_EVERY` and start the reporter thread.
pub fn init() {
    let every = std::env::var("XCL_PENDING_EVERY").ok().and_then(|v| v.trim().parse::<f64>().ok()).filter(|s| *s > 0.0);
    let on = every.is_some() || std::env::var("XCL_PENDING").is_ok_and(|v| !v.trim().is_empty() && v.trim() != "0");
    if !on {
        return;
    }
    let every = every.unwrap_or(30.0);
    ENABLED.store(true, Ordering::Release);
    let started = std::thread::Builder::new().name("xcl-pending".into()).spawn(move || {
        let every = Duration::from_secs_f64(every);
        loop {
            std::thread::sleep(every);
            report();
        }
    });
    if let Err(e) = started {
        crate::xlog_warn!("pending: no reporter thread ({e})");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(size: u64) -> Meta {
        Meta { size, ..Default::default() }
    }

    /// Every registry reports a pending item with its age and description, and forgets it
    /// when the ticket is finished or dropped.
    #[test]
    fn every_registry_reports_a_pending_item_and_drops_it() {
        for kind in Kind::ALL {
            let t = begin(kind, meta(1 << 20));
            let s = snapshot();
            let row = s.iter().find(|s| s.kind == kind).expect("a row per category");
            assert_eq!(row.count, 1, "{kind:?} shows its item");
            assert!(row.oldest_age.is_some_and(|a| a >= 0.0));
            assert!(row.oldest_description.as_ref().is_some_and(|d| !d.is_empty()));
            t.finish();
            let s = snapshot();
            assert_eq!(s.iter().find(|s| s.kind == kind).unwrap().count, 0, "{kind:?} drops it on finish");
        }
    }

    /// Dropping a ticket (an early return on a hot path) also forgets the item.
    #[test]
    fn a_dropped_ticket_is_forgotten() {
        {
            let _t = begin(Kind::FileRead, meta(4096));
            assert_eq!(snapshot().iter().find(|s| s.kind == Kind::FileRead).unwrap().count, 1);
        }
        assert_eq!(snapshot().iter().find(|s| s.kind == Kind::FileRead).unwrap().count, 0);
    }

    /// The description carries the numbers the site recorded and nothing else.
    #[test]
    fn descriptions_hold_the_numbers_of_the_site() {
        assert_eq!(
            Kind::DStorageRequest.describe(&Meta { size: 1048576, index: 2, mode: 7, extra: 0 }),
            "queue 2, 1048576 bytes from a file, BCPack mode 7"
        );
        assert_eq!(Kind::DStorageStatus.describe(&Meta { size: 3, index: 5, ..Default::default() }), "status array 5, slot 3 still E_PENDING");
        assert_eq!(Kind::DStorageSignal.describe(&Meta { size: 0x42, index: 1, ..Default::default() }), "queue 1, fence value 0x42 not signalled");
        assert_eq!(Kind::TitleWait.describe(&Meta { size: 0, index: 0, extra: 1000, ..Default::default() }), "frame event wait, type 0, timeout 1000 ms");
        assert_eq!(Kind::PsoBuild.describe(&Meta { size: 0x1234, extra: 1, ..Default::default() }), "compute pipeline 0x1234");
        assert_eq!(Kind::ShaderCompile.describe(&meta(8192)), "DXC compile, 8192 bytes of source");
        assert_eq!(Kind::XAsync.describe(&meta(0xabcd)), "async block 0xabcd");
        assert_eq!(Kind::FileRead.describe(&Meta { size: 10, extra: 1, ..Default::default() }), "10 bytes (overlapped)");
    }

    /// An item past 10 s gets one line of its own; a second report does not repeat it.
    #[test]
    fn a_slow_item_is_logged_once() {
        let t = begin(Kind::XAsync, meta(0x40));
        {
            let mut g = registry(Kind::XAsync as usize).lock().unwrap();
            let item = g.values_mut().next().unwrap();
            item.start = Instant::now() - Duration::from_secs(11);
        }
        let now = Instant::now();
        let first = report_lines(now);
        assert!(first.iter().any(|l| l.contains("XAsync blocks still outstanding") && l.contains("async block 0x40")), "{first:?}");
        let second = report_lines(now);
        assert!(!second.iter().any(|l| l.contains("still outstanding")), "{second:?}");
        // the category line keeps reporting the item while it is pending
        assert!(second.iter().any(|l| l.contains("XAsync blocks: 1 outstanding, oldest 11.")), "{second:?}");
        t.finish();
    }
}
