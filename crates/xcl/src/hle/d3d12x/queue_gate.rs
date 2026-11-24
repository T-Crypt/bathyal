//! Per-queue submission gate: a console title's `Wait32/64BitValueX` is a GPU stall, not a CPU
//! one. The value it waits for is often written by a later submission – on another queue, or by
//! a `cpu_write` that runs only after earlier GPU work. Blocking the thread that submits the
//! waiting list stops that later submission: the title stood on a black screen after its waits
//! timed out.
//!
//! Here every command queue keeps a FIFO of submissions it cannot run yet. The moment a
//! submission carries an unsatisfied wait, it and every later submission to that queue (lists,
//! Signal, Wait) go into the FIFO and the call returns at once. A thread of XCL's re-checks the
//! head of each FIFO whenever something is submitted or completes, and every 0.5 ms otherwise;
//! when the head's waits hold, it submits the head in order and takes the next. Other queues
//! keep running. A wait that holds at submit time takes the fast path: it goes straight to the
//! queue, as the title submitted it.
//!
//! Satisfaction has two sources:
//! * the memory the CPU sees (a UMA window holds the GPU's value once the writing submission
//!   completed; a sparse window is read back after its queue's submission);
//! * the values `Write32/64BitValue*` recorded for the address. Each record is tagged with the
//!   submission that carries it, and a wait on a value written on ANOTHER queue opens only when
//!   that submission's fence has passed, so the rest of the writer's work is done too. On the
//!   SAME queue the value alone is enough: the earlier work is already ahead in the queue, and a
//!   later write cannot be waited for without a deadlock.
//!
//! A held submission is submitted anyway after `XCL_WAIT_TIMEOUT_MS` (default 10000) with one
//! log line, so the title is never stuck for good.

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, Once, OnceLock};
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Foundation::HRESULT;
use windows::Win32::Graphics::Direct3D12::*;

use super::list_x::PendingWait;

/// PC vtable slots of the command queue used here (the listing's own numbering)
const PC_SIGNAL: usize = 14;
const PC_WAIT: usize = 15;
const PC_QUEUE_EXECUTE_COMMAND_LISTS: usize = 10;

/// Where a submission goes when it cannot run yet.
pub(super) enum Op {
    /// Console/PC command lists, as raw pointers.
    Lists(Vec<usize>),
    /// `Signal(fence, value)` on the queue.
    Signal(usize, u64),
    /// `Wait(fence, value)` on the queue.
    Wait(usize, u64),
}

struct Item {
    op: Op,
    waits: Vec<PendingWait>,
    /// addresses this submission's `Write32/64BitValue*` recorded (tagged with its sequence)
    immediates: Vec<u64>,
    queued: Instant,
}

#[derive(Default)]
struct QueueGate {
    fifo: VecDeque<Item>,
    /// the gate thread is submitting the item it took; new work must queue behind it
    busy: bool,
}

static GATES: Mutex<Option<HashMap<usize, QueueGate>>> = Mutex::new(None);
static WAKE: Condvar = Condvar::new();
static THREAD: Once = Once::new();

/// A `Write32/64BitValue*` value, and the submission that will store it (`None` while the list
/// that recorded it has not been submitted).
#[derive(Clone, Copy)]
struct Immediate {
    value: u32,
    writer: Option<(usize, u64)>,
}

static IMMEDIATES: Mutex<Option<HashMap<u64, Immediate>>> = Mutex::new(None);

/// XCL's fences, one per queue: they say which submissions have completed (`Signal` after
/// each submission, never waited on by the gate thread).
struct QueueFence {
    fence: ID3D12Fence,
    next: u64,
}

unsafe impl Send for QueueFence {}

static FENCES: Mutex<Option<HashMap<usize, QueueFence>>> = Mutex::new(None);

/// 0: not set for a test; the environment is read once instead.
static TEST_TIMEOUT_MS: AtomicU64 = AtomicU64::new(0);

fn timeout_ms() -> u64 {
    let set = TEST_TIMEOUT_MS.load(Ordering::Relaxed);
    if set != 0 {
        return set;
    }
    static DEFAULT: OnceLock<u64> = OnceLock::new();
    *DEFAULT.get_or_init(|| {
        std::env::var("XCL_WAIT_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|&ms| ms > 0)
            .unwrap_or(10_000)
    })
}

fn timeout() -> Duration {
    Duration::from_millis(timeout_ms())
}

// ---- counters (a line every 300 frames) -------------------------------------------------

/// submissions that had to be held since the last report
static GATED: AtomicU64 = AtomicU64::new(0);
/// longest hold (enqueue → submission) since the last report, in microseconds
static LONGEST_US: AtomicU64 = AtomicU64::new(0);
/// holds that ended by the timeout since the last report
static TIMEOUTS: AtomicU64 = AtomicU64::new(0);

/// The gate's counters since the last call: (held submissions, longest hold in ms, timeouts).
pub fn report() -> (u64, u64, u64) {
    (
        GATED.swap(0, Ordering::Relaxed),
        LONGEST_US.swap(0, Ordering::Relaxed) / 1000,
        TIMEOUTS.swap(0, Ordering::Relaxed),
    )
}

// ---- recorded immediate values ----------------------------------------------------------

/// The recorded value for `va` and whether the submission that will store it has completed on
/// its queue. On `reader` itself the writer is accepted as done (GPU order, or the same-queue
/// write that no CPU-side gate can wait for without deadlocking).
fn recorded_for(va: u64, width: u8, reader: usize) -> Option<(u64, bool)> {
    let (value, writer) = {
        let g = IMMEDIATES.lock().unwrap_or_else(|e| e.into_inner());
        let m = g.as_ref()?;
        let value = match width {
            4 => m.get(&va).map(|e| e.value as u64)?,
            8 => m.get(&va)?.value as u64 | ((m.get(&(va + 4))?.value as u64) << 32),
            _ => return None,
        };
        (value, m.get(&va).and_then(|e| e.writer))
    };
    let done = match writer {
        None => false,
        Some((queue, _)) if queue == reader => true,
        Some((queue, seq)) => completed(queue) >= seq,
    };
    Some((value, done))
}

/// A `Write32/64BitValue*` records the values it will store, so a wait on the address sees
/// them before the GPU (or the CPU after the queue's fence) gets there.
pub(super) fn note_immediate(va: u64, words: &[u32]) {
    {
        let mut g = IMMEDIATES.lock().unwrap_or_else(|e| e.into_inner());
        let m = g.get_or_insert_with(HashMap::new);
        if m.len() > 8192 {
            m.clear();
        }
        for (i, &w) in words.iter().enumerate() {
            m.insert(va + 4 * i as u64, Immediate { value: w, writer: None });
        }
    }
    WAKE.notify_all();
}

/// The submission `seq` on `queue` will store `addresses`: its completion gates waits on them.
fn tag(addresses: &[u64], queue: usize, seq: u64) {
    let mut g = IMMEDIATES.lock().unwrap_or_else(|e| e.into_inner());
    let Some(m) = g.as_mut() else { return };
    for &va in addresses {
        if let Some(e) = m.get_mut(&va) {
            e.writer = Some((queue, seq));
        }
    }
}

// ---- per-queue completion fences --------------------------------------------------------

fn next_seq(queue: usize) -> Option<u64> {
    let mut g = FENCES.lock().unwrap_or_else(|e| e.into_inner());
    let m = g.get_or_insert_with(HashMap::new);
    if let Some(f) = m.get_mut(&queue) {
        f.next += 1;
        return Some(f.next);
    }
    let q_raw = queue as *mut c_void;
    let q = unsafe { ID3D12CommandQueue::from_raw_borrowed(&q_raw) }?;
    let mut dev: Option<ID3D12Device> = None;
    unsafe { q.GetDevice(&mut dev) }.ok()?;
    let dev = dev?;
    let fence: ID3D12Fence = unsafe { dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) }.ok()?;
    m.insert(queue, QueueFence { fence, next: 1 });
    Some(1)
}

unsafe fn signal(queue: usize, seq: u64) {
    let fence = {
        let g = FENCES.lock().unwrap_or_else(|e| e.into_inner());
        g.as_ref().and_then(|m| m.get(&queue)).map(|f| f.fence.clone())
    };
    let Some(fence) = fence else { return };
    let q_raw = queue as *mut c_void;
    if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&q_raw) {
        let _ = q.Signal(&fence, seq);
    }
}

fn completed(queue: usize) -> u64 {
    let g = FENCES.lock().unwrap_or_else(|e| e.into_inner());
    g.as_ref().and_then(|m| m.get(&queue)).map_or(0, |f| unsafe { f.fence.GetCompletedValue() })
}

// ---- the gate ---------------------------------------------------------------------------

/// A wait holds when the CPU's memory holds the value, or a `Write*Value*` recorded it and the
/// submission that stores it has completed (`reader` exempt, see `recorded_for`).
unsafe fn holds(w: &PendingWait, reader: usize) -> bool {
    if w.is_satisfied() {
        return true;
    }
    w.satisfied_by_recorded(reader)
}

fn ready(item: &Item, queue: usize) -> bool {
    unsafe { item.waits.iter().all(|w| holds(w, queue)) }
}

fn start_thread() {
    THREAD.call_once(|| {
        let spawned = std::thread::Builder::new().name("xcl-queue-gate".into()).spawn(|| loop {
            let next = {
                let mut g = GATES.lock().unwrap_or_else(|e| e.into_inner());
                let gates = g.get_or_insert_with(HashMap::new);
                let chosen = gates
                    .iter()
                    .find(|(&q, s)| !s.busy && s.fifo.front().is_some_and(|item| ready(item, q) || item.queued.elapsed() >= timeout()))
                    .map(|(&q, _)| q);
                match chosen {
                    Some(q) => {
                        let s = gates.get_mut(&q).expect("chosen queue");
                        let item = s.fifo.pop_front().expect("a head");
                        s.busy = true;
                        Some((q, item))
                    }
                    None => {
                        let _ = WAKE.wait_timeout(g, Duration::from_micros(500));
                        None
                    }
                }
            };
            let Some((queue, item)) = next else { continue };
            let held_us = item.queued.elapsed().as_micros() as u64;
            LONGEST_US.fetch_max(held_us, Ordering::Relaxed);
            if held_us / 1000 >= timeout_ms() && !ready(&item, queue) {
                TIMEOUTS.fetch_add(1, Ordering::Relaxed);
                let value = unsafe { item.waits.first().and_then(|w| w.read_value()) };
                crate::xlog_warn!(
                    "D3D12.X: queue {queue:#x}: a held submission waited {} ms for {} wait(s) (at {:#x}, want {:#x}, memory holds {:x?}); submitted anyway",
                    held_us / 1000,
                    item.waits.len(),
                    item.waits.first().map_or(0, |w| w.va),
                    item.waits.first().map_or(0, |w| w.value),
                    value
                );
            }
            unsafe { run(queue, &item.op, &item.immediates) };
            let mut g = GATES.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(s) = g.get_or_insert_with(HashMap::new).get_mut(&queue) {
                s.busy = false;
            }
            WAKE.notify_all();
        });
        if let Err(e) = spawned {
            crate::xlog_error!("D3D12.X: no thread for the queue gates: {e}");
        }
    });
}

/// Whether `queue` has a submission it has not run yet (a present must follow them).
pub(super) fn has_pending(queue: usize) -> bool {
    let g = GATES.lock().unwrap_or_else(|e| e.into_inner());
    g.as_ref().and_then(|m| m.get(&queue)).is_some_and(|s| s.busy || !s.fifo.is_empty())
}

/// Forget `queue`'s held work (it is being released; nothing will run it).
pub(super) fn forget(queue: usize) {
    if let Some(m) = GATES.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        m.remove(&queue);
    }
    if let Some(m) = FENCES.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        m.remove(&queue);
    }
}

/// Submit `lists` (one submission, in order) to `queue`: at once when they can run and nothing
/// is held, otherwise into the queue's FIFO. `waits[i]` are the waits of `lists[i]`, and
/// `immediates[i]` the addresses `lists[i]` recorded with `Write32/64BitValue*`.
pub(super) unsafe fn gate_lists(queue: usize, lists: &[*mut c_void], waits: Vec<Vec<PendingWait>>, immediates: Vec<Vec<u64>>) {
    start_thread();
    if lists.is_empty() {
        return;
    }
    let base: Vec<usize> = lists.iter().map(|&l| l as usize).collect();
    let (run_lead, lead) = {
        let mut g = GATES.lock().unwrap_or_else(|e| e.into_inner());
        let state = g.get_or_insert_with(HashMap::new).entry(queue).or_default();
        let backlog = state.busy || !state.fifo.is_empty();
        let first = if backlog { 0 } else { waits.iter().position(|w| !w.iter().all(|x| holds(x, queue))).unwrap_or(lists.len()) };
        if !backlog && first == lists.len() {
            // fast path: every wait holds and nothing is held – straight to the queue
            state.busy = true;
            (true, lists.len())
        } else {
            for i in first..lists.len() {
                state.fifo.push_back(Item {
                    op: Op::Lists(vec![base[i]]),
                    waits: waits.get(i).cloned().unwrap_or_default(),
                    immediates: immediates.get(i).cloned().unwrap_or_default(),
                    queued: Instant::now(),
                });
            }
            if !backlog {
                // the lists before the first held one go now; the held ones are in the FIFO
                state.busy = true;
                (true, first)
            } else {
                (false, 0)
            }
        }
    };
    WAKE.notify_all();
    if !run_lead {
        return;
    }
    if lead > 0 {
        let lead_immediates: Vec<u64> = immediates.iter().take(lead).flatten().copied().collect();
        run(queue, &Op::Lists(base[..lead].to_vec()), &lead_immediates);
    }
    {
        let mut g = GATES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = g.get_or_insert_with(HashMap::new).get_mut(&queue) {
            s.busy = false;
        }
    }
    WAKE.notify_all();
}

/// `Signal(fence, value)` on `queue`: at once unless the queue holds work, then in its order.
pub(super) unsafe fn gate_signal(queue: usize, fence: usize, value: u64) -> HRESULT {
    start_thread();
    let run_now = {
        let mut g = GATES.lock().unwrap_or_else(|e| e.into_inner());
        let state = g.get_or_insert_with(HashMap::new).entry(queue).or_default();
        if state.busy || !state.fifo.is_empty() {
            state.fifo.push_back(Item { op: Op::Signal(fence, value), waits: Vec::new(), immediates: Vec::new(), queued: Instant::now() });
            false
        } else {
            state.busy = true;
            true
        }
    };
    WAKE.notify_all();
    if !run_now {
        return HRESULT(0);
    }
    let hr = run(queue, &Op::Signal(fence, value), &[]);
    {
        let mut g = GATES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = g.get_or_insert_with(HashMap::new).get_mut(&queue) {
            s.busy = false;
        }
    }
    WAKE.notify_all();
    HRESULT(hr)
}

/// `Wait(fence, value)` on `queue`: at once unless the queue holds work, then in its order.
pub(super) unsafe fn gate_wait(queue: usize, fence: usize, value: u64) -> HRESULT {
    start_thread();
    let run_now = {
        let mut g = GATES.lock().unwrap_or_else(|e| e.into_inner());
        let state = g.get_or_insert_with(HashMap::new).entry(queue).or_default();
        if state.busy || !state.fifo.is_empty() {
            state.fifo.push_back(Item { op: Op::Wait(fence, value), waits: Vec::new(), immediates: Vec::new(), queued: Instant::now() });
            false
        } else {
            state.busy = true;
            true
        }
    };
    WAKE.notify_all();
    if !run_now {
        return HRESULT(0);
    }
    let hr = run(queue, &Op::Wait(fence, value), &[]);
    {
        let mut g = GATES.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = g.get_or_insert_with(HashMap::new).get_mut(&queue) {
            s.busy = false;
        }
    }
    WAKE.notify_all();
    HRESULT(hr)
}

/// Submit `lists` now, in order, with what follows a submission: the CPU-visible readback of
/// sparse writes, the bottom-of-pipe work, the textures' states, the present fallback.
pub(super) unsafe fn submit_now(queue: usize, lists: &[*mut c_void]) -> i32 {
    let q_raw = queue as *mut c_void;
    super::title_textures::refresh(q_raw);
    type ExecFn = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
    let exec: ExecFn = std::mem::transmute_copy(&super::real::orig_fn(q_raw, PC_QUEUE_EXECUTE_COMMAND_LISTS));
    exec(q_raw, lists.len() as u32, lists.as_ptr());
    super::list_x::readback_sparse_writes(q_raw, lists);
    super::list_x::cpu_writes_after(q_raw, lists);
    super::title_textures::submitted(lists);
    super::present::note_execute(q_raw, lists);
    0
}

/// Submit `lists` now and number the submission, so a waiter on another queue can tell when it
/// is done: the recorded immediate values of the batch carry that number, and the queue's gate
/// fence is signalled with it after the submission.
unsafe fn submit_lists_now(queue: usize, lists: &[usize], immediates: &[u64]) -> i32 {
    let seq = next_seq(queue);
    if let Some(seq) = seq {
        if !immediates.is_empty() {
            tag(immediates, queue, seq);
        }
    }
    let lists: Vec<*mut c_void> = lists.iter().map(|&l| l as *mut c_void).collect();
    let hr = submit_now(queue, &lists);
    if let Some(seq) = seq {
        signal(queue, seq);
    }
    hr
}

/// Submit one list now (`KickoffX`'s path, which must reset the list for continued recording).
pub(super) unsafe fn submit_one_now(queue: usize, list: *mut c_void, immediates: &[u64]) -> i32 {
    submit_lists_now(queue, &[list as usize], immediates)
}

/// Run one item on its queue and answer its `HRESULT` (`0` where the call has none).
unsafe fn run(queue: usize, op: &Op, immediates: &[u64]) -> i32 {
    #[cfg(test)]
    {
        if let Some(f) = *TEST_RUNNER.lock().unwrap_or_else(|e| e.into_inner()) {
            f(queue, op);
            return 0;
        }
    }
    let q_raw = queue as *mut c_void;
    match op {
        Op::Lists(_) => submit_lists_now(queue, lists, immediates),
        Op::Signal(fence, value) => {
            let f: unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> i32 =
                std::mem::transmute_copy(&super::real::orig_fn(q_raw, PC_SIGNAL));
            let hr = f(q_raw, *fence as *mut c_void, *value);
            if hr >= 0 {
                super::fence_x::note_queue_signal(*fence, *value);
            }
            hr
        }
        Op::Wait(fence, value) => {
            let f: unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> i32 =
                std::mem::transmute_copy(&super::real::orig_fn(q_raw, PC_WAIT));
            f(q_raw, *fence as *mut c_void, *value)
        }
    }
}

/// Before `KickoffX` submits its closed list: wait (bounded) for `waits`, then for the queue's
/// held work to drain. `KickoffX` resets the list on the same allocator for the title's
/// continued recording, so its submission cannot be held; this is the one place a wait can
/// still occupy the title's thread, and it is bounded by `XCL_WAIT_TIMEOUT_MS`.
pub(super) unsafe fn wait_for_kickoff(queue: usize, waits: &[PendingWait]) {
    let start = Instant::now();
    let limit = timeout();
    while start.elapsed() < limit {
        if waits.iter().all(|w| holds(w, queue)) && !has_pending(queue) {
            return;
        }
        std::thread::sleep(Duration::from_micros(500));
    }
    crate::xlog_warn!(
        "D3D12.X: KickoffX on queue {queue:#x}: waits still pending after {} ms; submitting anyway",
        start.elapsed().as_millis()
    );
}

#[cfg(test)]
type TestRunner = fn(usize, &Op);
#[cfg(test)]
static TEST_RUNNER: Mutex<Option<TestRunner>> = Mutex::new(None);

/// The gate's submissions without a GPU, for tests: the runner sees each one in order.
#[cfg(test)]
pub(super) fn set_runner_for_test(f: Option<TestRunner>) {
    *TEST_RUNNER.lock().unwrap_or_else(|e| e.into_inner()) = f;
}

/// Empty every FIFO and forget the recorded immediate values.
#[cfg(test)]
pub(super) fn reset_for_test() {
    if let Some(m) = GATES.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        m.clear();
    }
    if let Some(m) = IMMEDIATES.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        m.clear();
    }
    TEST_TIMEOUT_MS.store(0, Ordering::Relaxed);
}

#[cfg(test)]
pub(super) fn set_timeout_ms_for_test(ms: u64) {
    TEST_TIMEOUT_MS.store(ms, Ordering::Relaxed);
}
