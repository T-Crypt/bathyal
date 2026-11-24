//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A draw waits for its graphics pipeline only so long between presents
//! (`resolve_for_draw`), then draws nothing until it is built; a compute pipeline is waited
//! for at its first bind (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.
//! `XCL_PSO_NEVER_SKIP=1` waits for a compute build with no time limit instead of ever
//! dispatching its stand-in; a build that failed is still skipped (see [`decide`]).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

struct Pending {
    /// the stand-in address the title knows the pipeline by
    standin: usize,
    /// the pending-work report forgets the build when this drops
    ticket: Mutex<Option<crate::pending::Ticket>>,
    /// the build, until a worker – or a bind that cannot wait for the queue – takes it
    job: Mutex<Option<Job>>,
    /// the build went to the workers (a lazy one goes on its first `resolve`)
    sent: std::sync::atomic::AtomicBool,
    /// the real pipeline state (raw pointer), 0 while building or when building failed
    real: AtomicUsize,
    done: Mutex<bool>,
    cv: Condvar,
    compute: bool,
    /// a bounded compute wait already timed out; subsequent binds skip waiting until `real` lands
    timed_out: std::sync::atomic::AtomicBool,
    /// the stand-in was released before the real pipeline was built
    orphan: std::sync::atomic::AtomicBool,
    /// its shaders are in the prewarm record (marked once)
    noted: std::sync::atomic::AtomicBool,
    /// it was seen still building and marked in the record (once)
    stall_noted: std::sync::atomic::AtomicBool,
    /// the unconditional wait (`XCL_PSO_NEVER_SKIP=1`) was logged for it
    wait_logged: std::sync::atomic::AtomicBool,
    /// where it goes in the worker queue: 0 before 1 before 2 before 3
    rank: u8,
}

impl Pending {
    fn is_active(&self) -> bool {
        self.sent.load(Ordering::Acquire) && !*self.done.lock().unwrap()
    }
}

/// stand-in handed to the title → its real pipeline
static PENDING: Mutex<Option<HashMap<usize, Arc<Pending>>>> = Mutex::new(None);

/// The build went to a worker (or is being built in a bind): show it in the pending report.
fn note_building(p: &Pending) {
    if !crate::pending::enabled() {
        return;
    }
    let mut t = p.ticket.lock().unwrap();
    if t.is_none() {
        *t = Some(crate::pending::begin(
            crate::pending::Kind::PsoBuild,
            crate::pending::Meta { size: p.standin as u64, extra: if p.compute { 1 } else { 0 }, ..Default::default() },
        ));
    }
}

/// The build is done (`real`: its pipeline, 0 = failed).
fn complete(p: &Pending, real: usize) {
    // the build is over: the pending-work report forgets it
    p.ticket.lock().unwrap().take();
    if real == 0 {
        FAILED.fetch_add(1, Ordering::Relaxed);
    } else {
        BUILT.fetch_add(1, Ordering::Relaxed);
    }
    let orphan = {
        let mut done = p.done.lock().unwrap();
        let orphan = p.orphan.load(Ordering::Acquire);
        if !orphan {
            p.real.store(real, Ordering::Release);
        }
        *done = true;
        p.cv.notify_all();
        orphan
    };
    // outside `done`: the Release override takes PENDING (`counts` takes PENDING, then `done`)
    if orphan {
        release(real);
    }
}
static BUILT: AtomicUsize = AtomicUsize::new(0);
static FAILED: AtomicUsize = AtomicUsize::new(0);

pub fn enabled() -> bool {
    std::env::var("XCL_ASYNC_PSO").map_or(true, |v| v != "0")
}

/// Builds the real pipeline: its raw pointer, 0 = failed.
pub type Job = Box<dyn FnOnce() -> usize + Send>;

/// A worker queue that pops the lowest rank first and keeps arrival order within a rank. The
/// ranks put compute and pipelines that were seen still building ahead of the rest, so a warm
/// run does not reach the world with the builds that hung it still waiting.
struct WorkQueue {
    items: Mutex<VecDeque<(u8, u64, Arc<Pending>)>>,
    next: std::sync::atomic::AtomicU64,
    cv: Condvar,
}

impl WorkQueue {
    fn new() -> Self {
        Self { items: Mutex::new(VecDeque::new()), next: std::sync::atomic::AtomicU64::new(0), cv: Condvar::new() }
    }

    fn push(&self, rank: u8, p: Arc<Pending>) {
        let seq = self.next.fetch_add(1, Ordering::Relaxed);
        let mut q = self.items.lock().unwrap();
        // after every entry of a lower or equal rank: stable within a rank
        let at = q.iter().position(|(r, s, _)| (*r, *s) > (rank, seq)).unwrap_or(q.len());
        q.insert(at, (rank, seq, p));
        self.cv.notify_one();
    }

    fn pop(&self) -> Arc<Pending> {
        let mut q = self.items.lock().unwrap();
        while q.is_empty() {
            q = self.cv.wait(q).unwrap();
        }
        q.pop_front().unwrap().2
    }

    fn spawn(self: &'static Self, prefix: &str, threads: usize) {
        for k in 0..threads {
            let _ = std::thread::Builder::new().name(format!("{prefix}-{k}")).stack_size(64 << 20).spawn(move || loop {
                let p = self.pop();
                // (None: a bind took it and builds it itself)
                let job = p.job.lock().unwrap().take();
                if let Some(job) = job {
                    complete(&p, job());
                }
            });
        }
    }
}

fn queue() -> &'static WorkQueue {
    static Q: OnceLock<&'static WorkQueue> = OnceLock::new();
    *Q.get_or_init(|| {
        let q: &'static WorkQueue = Box::leak(Box::new(WorkQueue::new()));
        q.spawn("xcl-pso", pipeline_threads());
        q
    })
}

/// Dedicated worker pool for compute pipelines so a backlog (or driver stall, #156) of graphics
/// builds cannot starve compute dispatches indefinitely.
fn compute_queue() -> &'static WorkQueue {
    static Q: OnceLock<&'static WorkQueue> = OnceLock::new();
    *Q.get_or_init(|| {
        let q: &'static WorkQueue = Box::leak(Box::new(WorkQueue::new()));
        q.spawn("xcl-pso-compute", (crate::gpu::recomp::dxc::compile_threads() / 2).clamp(1, 4));
        q
    })
}

/// Where a pipeline goes in its worker queue. Compute pipelines go first: the hang evidence is
/// compute (a skipped dispatch leaves a GPU-side wait unsatisfied). Among them, and among
/// graphics ones, a pipeline seen still building in an earlier run goes first.
fn early_rank(compute: bool, stalled: bool) -> u8 {
    match (compute, stalled) {
        (true, true) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (false, false) => 3,
    }
}

/// A pipeline built at its first use, not during load (nothing says it was used before).
const RANK_NORMAL: u8 = 3;

/// Send `p` to its worker pool if not already sent; clears `sent` on failure so a later bind
/// can retry instead of stalling forever.
fn enqueue(p: &Arc<Pending>) -> bool {
    if !p.sent.swap(true, Ordering::AcqRel) {
        let q = if p.compute { compute_queue() } else { queue() };
        q.push(p.rank, p.clone());
        return true;
    }
    true
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        super::pso_disk::forget_pso(p);
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

/// `standin` went to the title; `job` builds the real pipeline on a worker. `stalled`: the
/// record says an earlier run saw it still building, so it goes before the other builds.
pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_as(standin, compute, job, true, RANK_NORMAL);
}

/// `submit` for a pipeline the record knows (`flags` from `pso_disk::wanted_flags`): it goes
/// ahead of the recorded pipelines that were fine before, compute ones first.
pub fn submit_recorded(standin: usize, compute: bool, flags: u8, job: Job) {
    let stalled = flags & super::pso_disk::FLAG_STALLED != 0;
    submit_as(standin, compute, job, true, early_rank(compute, stalled));
}

/// Lazy pipeline creation (`XCL_LAZY_PSO=0`: off): a title's pipelines with console shaders get
/// their stand-in when made and are built when first used – the first bind of a graphics one
/// sends it to the workers, a compute one is built at its first bind. A retail game makes
/// ~51 000 pipelines in its first 30 s and uses few; each one built stays in the driver
/// (~400 KB), and the process ran out of commit.
pub fn lazy() -> bool {
    static L: OnceLock<bool> = OnceLock::new();
    *L.get_or_init(|| enabled() && std::env::var("XCL_LAZY_PSO").map_or(true, |v| v != "0"))
}

/// `submit`, sent to the workers only on its first `resolve` (`lazy`).
pub fn submit_lazy(standin: usize, compute: bool, job: Job) {
    submit_as(standin, compute, job, false, RANK_NORMAL);
}

fn submit_as(standin: usize, compute: bool, job: Job, send: bool, rank: u8) {
    let p = Arc::new(Pending {
        standin,
        ticket: Mutex::new(None),
        job: Mutex::new(Some(job)),
        sent: std::sync::atomic::AtomicBool::new(false),
        real: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
        timed_out: Default::default(),
        orphan: Default::default(),
        noted: Default::default(),
        stall_noted: Default::default(),
        wait_logged: Default::default(),
        rank,
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        note_building(&p);
        let _ = enqueue(&p);
    }
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.lock().unwrap().as_ref()?.get(&pso).cloned()
}

/// The pipeline to use for `pso` now: the real one once built, else `pso` itself.
pub fn resolve(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    match p.real.load(Ordering::Acquire) {
        0 => {
            // a lazy one is wanted now: to the workers
            note_building(&p);
            let _ = enqueue(&p);
            pso
        }
        r => r,
    }
}

/// Whether a pipeline whose build was put off is built now, while the title loads, or on its
/// first use. One whose shaders were drawn with in an earlier run is prewarmed: the library
/// has its driver blob, so the build is a quick `Create*PipelineState`, and the first draw
/// does not wait for it.
pub fn build_during_load(lazy: bool, drawn_before: bool) -> bool {
    !lazy || drawn_before
}

/// A pipeline drawn with in an earlier run went to the workers at its creation (logged once).
pub fn note_prewarmed() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines drawn with in an earlier run are built while the title loads");
    });
}

/// `resolve`, waiting for a pipeline still being built – built here if no worker has started
/// it yet (a queue of a title's hundreds of pipelines must not hold up the one it binds now).
pub fn resolve_wait(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    let start = Instant::now();
    p.sent.store(true, Ordering::Release);
    note_building(&p);
    let job = p.job.lock().unwrap().take();
    let here = job.is_some();
    if let Some(job) = job {
        complete(&p, job());
    }
    let mut done = p.done.lock().unwrap();
    while !*done {
        done = p.cv.wait(done).unwrap();
    }
    let real = p.real.load(Ordering::Acquire);
    drop(done);
    note_long_wait(pso, p.compute, here, start.elapsed());
    if real != 0 && !p.noted.swap(true, Ordering::Relaxed) {
        super::pso_disk::note_drawn(real);
    }
    match real {
        0 => pso,
        r => r,
    }
}

/// A title's thread held this long in one pipeline wait is logged: a retail game's hang
/// watchdog ends the process (`mov dword [0], 0` at title+0x2215c35, #156 / #232) when its
/// threads stop for long, and an NVIDIA driver takes 10 - 17 s for one big translated pixel
/// shader, minutes for a big compute program (#156).
const LONG_WAIT: Duration = Duration::from_secs(2);

fn note_long_wait(pso: usize, compute: bool, here: bool, took: Duration) {
    static N: AtomicUsize = AtomicUsize::new(0);
    if took >= LONG_WAIT && N.fetch_add(1, Ordering::Relaxed) < 32 {
        crate::xlog_warn!(
            "D3D12.X: thread '{}' waited {:.1} s for {} pipeline {pso:#x} ({})",
            std::thread::current().name().unwrap_or("title"),
            took.as_secs_f32(),
            if compute { "compute" } else { "graphics" },
            if here { "built in this thread" } else { "built on a worker" }
        );
    }
}

/// The longest a draw holds the title's threads, between two presents, for graphics pipelines
/// still being built (`XCL_PSO_WAIT_MS`, default 4000; 0: no limit, as before). Past it a draw
/// goes on with its stand-in (draws nothing) and the pipeline is built on a worker.
fn draw_wait_limit() -> Option<Duration> {
    static L: OnceLock<Option<Duration>> = OnceLock::new();
    *L.get_or_init(|| match std::env::var("XCL_PSO_WAIT_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()) {
        Some(0) => None,
        Some(ms) => Some(Duration::from_millis(ms)),
        None => Some(Duration::from_secs(4)),
    })
}

/// Time spent in `resolve_for_draw` waits since present number `epoch`.
struct DrawWait {
    epoch: u64,
    spent: Duration,
}

static DRAW_WAIT: Mutex<DrawWait> = Mutex::new(DrawWait { epoch: 0, spent: Duration::ZERO });

/// For a draw (or ExecuteIndirect) of console code: a compute pipeline is waited for
/// (`resolve_wait`); a graphics one only within what is left of `draw_wait_limit` until the
/// next present, and never built in the title's thread, where a driver compile cannot be cut
/// short. A draw past the limit goes on with the stand-in, logged.
pub fn resolve_for_draw(pso: usize) -> usize {
    resolve_within(pso, draw_wait_limit(), super::present::activity().0, &DRAW_WAIT)
}

fn resolve_within(pso: usize, limit: Option<Duration>, epoch: u64, budget: &Mutex<DrawWait>) -> usize {
    let Some(p) = entry(pso) else { return pso };
    let Some(limit) = limit.filter(|_| !p.compute) else { return resolve_wait(pso) };
    let left = {
        let mut b = budget.lock().unwrap();
        if b.epoch != epoch {
            *b = DrawWait { epoch, spent: Duration::ZERO };
        }
        limit.saturating_sub(b.spent)
    };
    // to the workers (a lazy one has not gone yet)
    note_building(&p);
    let _ = enqueue(&p);
    let start = Instant::now();
    let deadline = start + left;
    let mut done = p.done.lock().unwrap();
    while !*done {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        done = p.cv.wait_timeout(done, deadline - now).unwrap().0;
    }
    let (finished, real) = (*done, p.real.load(Ordering::Acquire));
    drop(done);
    if real != 0 && !p.noted.swap(true, Ordering::Relaxed) {
        // drawn with: the next run builds it while the title loads
        super::pso_disk::note_drawn(real);
    }
    let waited = start.elapsed();
    {
        let mut b = budget.lock().unwrap();
        if b.epoch == epoch {
            b.spent += waited;
        }
    }
    if !finished {
        // drawn with its stand-in: the next run builds it before the other recorded pipelines
        if !p.stall_noted.swap(true, Ordering::Relaxed) {
            super::pso_disk::note_stalled(pso);
        }
        static N: AtomicUsize = AtomicUsize::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        if n < 16 || n.is_power_of_two() {
            crate::xlog_warn!(
                "D3D12.X: graphics pipeline {pso:#x} not built after {:.1} s of waiting since present {epoch} (limit {} ms, XCL_PSO_WAIT_MS) – drawn with its stand-in, still building on a worker ({} such draws)",
                waited.as_secs_f32(),
                limit.as_millis(),
                n + 1
            );
        }
    } else {
        note_long_wait(pso, false, false, waited);
    }
    match real {
        0 => pso,
        r => r,
    }
}

/// Maximum per-frame time title threads wait for uncached compute pipelines on first
/// bind/dispatch (`XCL_COMPUTE_PSO_WAIT_MS`, default 1500 ms; `0` = unbounded). Huge compute
/// shaders can take 20+ s to compile in DXC / the driver, and a loading frame can bind many
/// uncached compute pipelines in a row; sharing one budget per frame keeps the total bounded.
fn compute_wait_timeout() -> Option<std::time::Duration> {
    static LIMIT: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        let ms = std::env::var("XCL_COMPUTE_PSO_WAIT_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(1500);
        if ms == 0 { None } else { Some(std::time::Duration::from_millis(ms)) }
    })
}

/// Shared per-frame compute wait budget, reset at each `PresentX` boundary (`reset_frame_budget`).
struct FrameBudget {
    spent_ms: std::sync::atomic::AtomicU64,
}

impl FrameBudget {
    const fn new() -> Self {
        Self { spent_ms: std::sync::atomic::AtomicU64::new(0) }
    }

    fn remaining(&self, limit: std::time::Duration) -> std::time::Duration {
        let limit_ms = limit.as_millis().min(u64::MAX as u128) as u64;
        let spent = self.spent_ms.load(Ordering::Relaxed);
        std::time::Duration::from_millis(limit_ms.saturating_sub(spent))
    }

    fn consume(&self, elapsed: std::time::Duration) {
        let ms = elapsed.as_millis().min(u64::MAX as u128) as u64;
        if ms > 0 {
            self.spent_ms.fetch_add(ms, Ordering::Relaxed);
        }
    }

    fn reset(&self) {
        self.spent_ms.store(0, Ordering::Relaxed);
    }
}

static FRAME_BUDGET: FrameBudget = FrameBudget::new();

/// Reset the per-frame compute pipeline wait budget (called at `PresentX`).
pub fn reset_frame_budget() {
    FRAME_BUDGET.reset();
}

/// `XCL_PSO_NEVER_SKIP=1`: a compute pipeline whose build is still going is waited for with no
/// time limit, so no dispatch or ExecuteIndirect runs its stand-in. Off by default.
pub fn never_skip() -> bool {
    static N: OnceLock<bool> = OnceLock::new();
    *N.get_or_init(|| std::env::var("XCL_PSO_NEVER_SKIP").is_ok_and(|v| v.trim() == "1"))
}

/// What a bind, dispatch or ExecuteIndirect does about a pipeline that is not ready.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PipelineWait {
    /// the real pipeline is there: use it
    Ready,
    /// still building and `XCL_PSO_NEVER_SKIP=1`: wait for the build with no time limit
    WaitForBuild,
    /// still building, flag off: the bounded wait as before
    BoundedWait,
    /// the build is over and produced nothing: use the stand-in, as today
    Failed,
}

/// The decision for a pipeline whose state is `(real, done)`. `real != 0` means built; a build
/// that failed sets `done` without a `real` pipeline, so a failed build is told apart from one
/// still building by `done`, not by `real`: waiting only ever ends when the build ends, and a
/// failure ends it too.
pub(crate) fn decide(real: usize, done: bool, never_skip: bool) -> PipelineWait {
    if real != 0 {
        PipelineWait::Ready
    } else if done {
        PipelineWait::Failed
    } else if never_skip {
        PipelineWait::WaitForBuild
    } else {
        PipelineWait::BoundedWait
    }
}

/// Bounded wait for a compute pipeline: always builds on `xcl-pso-compute-*` workers (never
/// stalling the title's thread inside DXC/driver), waits at most the remaining per-frame `budget`
/// up to `limit`, and returns the stand-in immediately once timed out or once the frame budget is
/// exhausted until the worker finishes.
fn resolve_compute_within(pso: usize, limit: Option<std::time::Duration>, budget: &FrameBudget) -> usize {
    resolve_compute_decided(pso, limit, budget, never_skip())
}

fn resolve_compute_decided(pso: usize, limit: Option<std::time::Duration>, budget: &FrameBudget, never_skip: bool) -> usize {
    let Some(p) = entry(pso) else { return pso };
    // bound: drawn with this run, so the next one builds it while the title loads (a compute
    // pipeline's stand-in carries the shader identity it was created with). Before the decision,
    // so a wait that never skips still enters the record.
    if !p.noted.swap(true, Ordering::Relaxed) {
        super::pso_disk::note_drawn(pso);
    }
    // `complete` stores `real` and sets `done` under this lock: read both together, so a
    // build finishing now is seen as built (or as failed), never as neither
    let (real_now, done_now) = {
        let done = p.done.lock().unwrap();
        (p.real.load(Ordering::Acquire), *done)
    };
    match decide(real_now, done_now, never_skip) {
        PipelineWait::Ready => return real_now,
        PipelineWait::Failed => return pso,
        PipelineWait::WaitForBuild => return wait_for_build(&p, pso),
        PipelineWait::BoundedWait => {}
    }
    let Some(limit) = limit else {
        return resolve_wait(pso);
    };
    if let r @ 1.. = real_now {
        return r;
    }
    if !enqueue(&p) {
        return pso;
    }
    if p.timed_out.load(Ordering::Acquire) {
        return match p.real.load(Ordering::Acquire) {
            0 => pso,
            r => r,
        };
    }
    let rem_budget = budget.remaining(limit);
    if rem_budget.is_zero() {
        return match p.real.load(Ordering::Acquire) {
            0 => pso,
            r => r,
        };
    }
    let started = std::time::Instant::now();
    let deadline = started + rem_budget;
    let mut done = p.done.lock().unwrap();
    while !*done {
        let rem = deadline.saturating_duration_since(std::time::Instant::now());
        if rem.is_zero() {
            break;
        }
        let (g, res) = p.cv.wait_timeout(done, rem).unwrap();
        done = g;
        if res.timed_out() {
            break;
        }
    }
    let elapsed = started.elapsed();
    if !*done {
        budget.consume(elapsed.max(rem_budget));
        if rem_budget >= limit / 2 && !p.timed_out.swap(true, Ordering::AcqRel) {
            // skipped this run: the next one builds it before the other recorded pipelines
            if !p.stall_noted.swap(true, Ordering::Relaxed) {
                super::pso_disk::note_stalled(pso);
            }
            static LOGGED: AtomicUsize = AtomicUsize::new(0);
            if LOGGED.fetch_add(1, Ordering::Relaxed) < 16 {
                crate::xlog_warn!(
                    "D3D12.X: compute pipeline {pso:#x} still building after {} ms – proceeding with stand-in (continues in background)",
                    elapsed.as_millis()
                );
            }
        }
        return pso;
    }
    budget.consume(elapsed);
    match p.real.load(Ordering::Acquire) {
        0 => pso,
        r => r,
    }
}

/// Wait for a compute build to end, with no time limit (`XCL_PSO_NEVER_SKIP=1`). The wait ends
/// on success or failure alike; a failed build gives the stand-in back, as today. Logged once
/// per pipeline, with the wait at the end.
fn wait_for_build(p: &Arc<Pending>, pso: usize) -> usize {
    if let r @ 1.. = p.real.load(Ordering::Acquire) {
        return r;
    }
    if !enqueue(p) {
        // no worker pool to build it: an unbounded wait could never end
        return pso;
    }
    // the title waits for it: the next run builds it before the other recorded pipelines
    if !p.stall_noted.swap(true, Ordering::Relaxed) {
        super::pso_disk::note_stalled(pso);
    }
    let started = Instant::now();
    let mut done = p.done.lock().unwrap();
    while !*done {
        done = p.cv.wait(done).unwrap();
    }
    let real = p.real.load(Ordering::Acquire);
    drop(done);
    if !p.wait_logged.swap(true, Ordering::Relaxed) {
        let ms = started.elapsed().as_millis();
        if real == 0 {
            crate::xlog_warn!("D3D12.X: compute pipeline {pso:#x} build failed after {ms} ms (XCL_PSO_NEVER_SKIP=1) – stand-in used, as before");
        } else {
            crate::xlog_info!("D3D12.X: compute pipeline {pso:#x} waited {ms} ms for its build (XCL_PSO_NEVER_SKIP=1)");
        }
    }
    match real {
        0 => pso,
        r => r,
    }
}

/// Resolve a compute pipeline with `XCL_COMPUTE_PSO_WAIT_MS` per-frame bounding.
pub fn resolve_compute(pso: usize) -> usize {
    resolve_compute_within(pso, compute_wait_timeout(), &FRAME_BUDGET)
}

/// For binding: compute pipelines wait up to `XCL_COMPUTE_PSO_WAIT_MS` on background workers;
/// graphics pipelines are queued without blocking.
pub fn resolve_for_bind(pso: usize) -> usize {
    match entry(pso) {
        Some(p) if p.compute => resolve_compute(pso),
        Some(_) => resolve(pso),
        None => {
            // a pipeline built in the call (no stand-in): drawn with, so prewarmed next run
            super::pso_disk::note_drawn(pso);
            pso
        }
    }
}

/// For `ExecuteIndirect` (and `Dispatch`): compute pipelines use bounded waits, or an unbounded
/// one under `XCL_PSO_NEVER_SKIP=1`; graphics pipelines keep `resolve_wait`.
pub fn resolve_for_indirect(pso: usize) -> usize {
    match entry(pso) {
        Some(p) if p.compute => resolve_compute(pso),
        Some(_) => resolve_wait(pso),
        None => pso,
    }
}

/// The title released stand-in `pso` for good: forget it (its address may come back for
/// another object) and return its real pipeline to release (0: none, or still being built –
/// the worker's result is then released when it arrives).
pub fn released(pso: usize) -> usize {
    let Some(p) = PENDING.lock().unwrap().as_mut().and_then(|m| m.remove(&pso)) else { return 0 };
    // the stand-in address is not the pipeline: forget what it was registered as
    super::pso_disk::forget_pso(pso);
    let done = p.done.lock().unwrap();
    if *done {
        p.real.swap(0, Ordering::AcqRel)
    } else {
        // the worker stores its result into a Pending nobody will read: mark it orphaned
        p.orphan.store(true, Ordering::Release);
        0
    }
}

/// `pso` is a stand-in whose real pipeline is still being built.
pub fn is_pending(pso: usize) -> bool {
    entry(pso).is_some_and(|p| !*p.done.lock().unwrap())
}

/// Whether `pso` is a real pipeline now: not a stand-in at all, or its build finished with
/// one (a failed build leaves the stand-in).
pub fn is_built(pso: usize) -> bool {
    match entry(pso) {
        Some(p) => *p.done.lock().unwrap() && p.real.load(Ordering::Acquire) != 0,
        None => true,
    }
}

/// How many `xcl-pso-N` workers build pipelines. A worker's cost is mostly the driver's
/// compile (`Create*PipelineState`), so this is how many driver compiles run at once.
///
/// On NVIDIA the driver compiler is what a worker spends its time in and its time grows
/// faster than the program (#156): one big retail compute program takes minutes. #156's
/// measurements disagree about how many creators help: 4.0 on a 4-core 7700K (8 finished the
/// same queue later than 4), and 3.1 / 5.3 here at 4 / 8. So NVIDIA gets a small fixed count
/// and everything else keeps a worker per core minus one, as before. `XCL_PSO_THREADS`
/// overrides both.
fn pipeline_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        let over = std::env::var("XCL_PSO_THREADS").ok().and_then(|v| v.trim().parse::<usize>().ok());
        let n = workers_for(super::real::adapter_vendor() == 0x10DE, over, crate::gpu::recomp::dxc::compile_threads());
        crate::xlog_info!("D3D12.X: {n} pipeline build worker(s) (XCL_PSO_THREADS sets the count)");
        n
    })
}

/// [`pipeline_threads`] from the adapter and the environment, so its test needs no GPU.
fn workers_for(nvidia: bool, over: Option<usize>, dxc: usize) -> usize {
    match over {
        Some(n) => n.clamp(1, 64),
        None if nvidia => NVIDIA_WORKERS,
        None => dxc,
    }
}

/// Measured best on an RTX 4090 (driver 621.01, i9-14900K) with the large synthetic programs
/// of `nv_parallel_driver_compile`: 1 / 2 / 4 / 8 creator threads finished the same queue in
/// 50.0 / 29.7 / 16.5 / 10.1 s, with each program at 8.4 / 9.6 / 9.2 / 10.0 s. So the driver's
/// compiler is not serialised here and more threads do speed the queue up, but each program
/// gets slower. 4 is what the slowest machine in #156 (a 4-core 7700K / RTX 5080) measured
/// best, and it keeps a frame's own pipeline from waiting behind seven other compiles.
const NVIDIA_WORKERS: usize = 4;

/// How many uncached driver compiles may run at once. Each one fans out across the GPU
/// driver's own compiler, so a few of them already fill the machine; more of them fight
/// over the same cores and the load takes longer. `XCL_PSO_COMPILES` overrides it; else it
/// is the pool size on NVIDIA, and the pool capped as before elsewhere.
fn driver_compile_limit() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        if let Some(n) = std::env::var("XCL_PSO_COMPILES").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
            return n.clamp(1, 64);
        }
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        let pool = pipeline_threads();
        if super::real::adapter_vendor() == 0x10DE {
            pool
        } else {
            (cores / 4).clamp(2, 4).min(pool)
        }
    })
}

struct DriverGate {
    /// compiles in the driver, and whether a resource create wants the driver to itself
    running: Mutex<(usize, bool)>,
    cv: Condvar,
}

fn driver_gate() -> &'static DriverGate {
    static G: OnceLock<DriverGate> = OnceLock::new();
    G.get_or_init(|| DriverGate { running: Mutex::new((0, false)), cv: Condvar::new() })
}

/// Hold the driver until the in-flight pipeline compiles finish, and keep new ones from
/// starting. A resource create overlapping a pipeline compile removes the device on this GPU.
pub struct DriverQuiet;

impl Drop for DriverQuiet {
    fn drop(&mut self) {
        let g = driver_gate();
        let mut n = g.running.lock().unwrap();
        n.1 = false;
        g.cv.notify_all();
    }
}

pub fn quiet_driver() -> DriverQuiet {
    let g = driver_gate();
    let mut n = g.running.lock().unwrap();
    n.1 = true;
    if n.0 > 0 {
        crate::xlog_info!("D3D12.X: waiting for {} pipeline compile(s) before creating a resource", n.0);
    }
    while n.0 > 0 {
        n = g.cv.wait(n).unwrap();
    }
    DriverQuiet
}

/// Held across one driver compile of a pipeline the cache did not have.
pub struct DriverGuard;

impl Drop for DriverGuard {
    fn drop(&mut self) {
        let g = driver_gate();
        let mut n = g.running.lock().unwrap();
        n.0 = n.0.saturating_sub(1);
        g.cv.notify_all();
    }
}

/// Block until fewer than `driver_compile_limit` uncached driver compiles are running.
pub fn uncached_compile() -> DriverGuard {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: up to {} new pipelines compile in the driver at once", driver_compile_limit());
    });
    let g = driver_gate();
    let limit = driver_compile_limit();
    let mut n = g.running.lock().unwrap();
    while n.1 || n.0 >= limit {
        n = g.cv.wait(n).unwrap();
    }
    n.0 += 1;
    DriverGuard
}

/// `pso` is a compute stand-in whose real pipeline is still being built.
pub fn is_pending_compute(pso: usize) -> bool {
    entry(pso).is_some_and(|p| p.compute && !*p.done.lock().unwrap())
}

#[cfg(test)]
pub(crate) fn resolve_compute_with_limit_for_test(pso: usize, limit: std::time::Duration) -> usize {
    resolve_compute_within(pso, Some(limit), &FrameBudget::new())
}


/// (built in the background, failed, still building)
pub fn counts() -> (usize, usize, usize) {
    let pending = PENDING.lock().unwrap().as_ref().map_or(0, |m| m.values().filter(|p| !*p.done.lock().unwrap()).count());
    (BUILT.load(Ordering::Relaxed), FAILED.load(Ordering::Relaxed), pending)
}

/// Failed builds and builds actually started or sent to workers, excluding dormant lazy pipelines.
pub fn activity() -> (usize, usize) {
    let active = PENDING.lock().unwrap().as_ref().map_or(0, |m| {
        m.values().filter(|p| p.is_active()).count()
    });
    (FAILED.load(Ordering::Relaxed), active)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn budget() -> Mutex<DrawWait> {
        Mutex::new(DrawWait { epoch: 0, spent: Duration::ZERO })
    }

    fn test_pending(standin: usize, rank: u8) -> Arc<Pending> {
        Arc::new(Pending {
            standin,
            ticket: Mutex::new(None),
            job: Mutex::new(None),
            sent: std::sync::atomic::AtomicBool::new(true),
            real: AtomicUsize::new(0),
            done: Mutex::new(true),
            cv: Condvar::new(),
            compute: false,
            timed_out: Default::default(),
            orphan: Default::default(),
            noted: Default::default(),
            stall_noted: Default::default(),
            wait_logged: Default::default(),
            rank,
        })
    }

    /// The early queue builds compute before graphics, and among each a pipeline seen still
    /// building in an earlier run before the rest; arrival order holds within a rank.
    #[test]
    fn the_early_queue_builds_compute_first_then_ones_seen_stalled() {
        assert_eq!(early_rank(true, true), 0);
        assert_eq!(early_rank(true, false), 1);
        assert_eq!(early_rank(false, true), 2);
        assert_eq!(early_rank(false, false), 3);
        assert!(early_rank(true, false) < early_rank(false, true), "compute goes before a stalled graphics one");

        let q = WorkQueue::new();
        for (i, rank) in [3u8, 2, 1, 0, 3, 1].into_iter().enumerate() {
            q.push(rank, test_pending(0x100 + i, rank));
        }
        let order: Vec<(u8, usize)> = q.items.lock().unwrap().iter().map(|(r, _, p)| (*r, p.standin)).collect();
        assert_eq!(order, vec![(0, 0x103), (1, 0x102), (1, 0x105), (2, 0x101), (3, 0x100), (3, 0x104)]);
    }

    /// The decision behind `XCL_PSO_NEVER_SKIP`: ready, still building with and without the
    /// flag, and a failed build (`done`, no pipeline), which is still skipped under the flag.
    #[test]
    fn the_never_skip_decision_separates_building_from_failed() {
        // ready
        assert_eq!(decide(0x900, true, false), PipelineWait::Ready);
        assert_eq!(decide(0x900, true, true), PipelineWait::Ready);
        // still building, flag off: the bounded wait as before
        assert_eq!(decide(0, false, false), PipelineWait::BoundedWait);
        // still building, flag on: wait with no time limit
        assert_eq!(decide(0, false, true), PipelineWait::WaitForBuild);
        // failed (the build ended with no pipeline): skipped either way
        assert_eq!(decide(0, true, false), PipelineWait::Failed);
        assert_eq!(decide(0, true, true), PipelineWait::Failed);
    }

    /// With the flag, the unbounded wait outlives a short `limit` and returns the real pipeline
    /// once the worker finishes. A failed build ends the wait at once and still gives the
    /// stand-in, so a broken program cannot block the title.
    #[test]
    fn never_skip_waits_through_the_limit_and_still_gives_up_on_a_failure() {
        let (standin, real) = (0x7600_0001usize, 0x7600_0002usize);
        submit_lazy(
            standin,
            true,
            Box::new(move || {
                std::thread::sleep(Duration::from_millis(120));
                real
            }),
        );
        let t = Instant::now();
        assert_eq!(resolve_compute_decided(standin, Some(Duration::from_millis(1)), &FrameBudget::new(), true), real);
        assert!(t.elapsed() >= Duration::from_millis(100), "waited {:?} past a 1 ms limit", t.elapsed());
        assert_eq!(released(standin), real);

        let failed = 0x7600_0003usize;
        submit_lazy(failed, true, Box::new(|| 0));
        let t = Instant::now();
        assert_eq!(resolve_compute_decided(failed, Some(Duration::from_millis(1)), &FrameBudget::new(), true), failed);
        assert!(t.elapsed() < Duration::from_secs(5), "a failed build must not block the title");
        assert!(!is_pending_compute(failed));
        let _ = released(failed);
    }

    /// A graphics build that does not finish (an NVIDIA driver compile of minutes, #156) holds a
    /// draw only up to the limit, once until the next present, and runs on a worker. With
    /// `resolve_wait` the draw waited for it without limit (building it itself when no worker
    /// had taken it), which held the title past its hang watchdog.
    #[test]
    fn a_draw_waits_for_a_graphics_build_only_within_the_limit_and_never_builds_it_itself() {
        let (standin, real) = (0x7410_0001usize, 0x7410_0002usize);
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let (who_tx, who_rx) = mpsc::channel::<Option<String>>();
        submit_lazy(
            standin,
            false,
            Box::new(move || {
                let _ = who_tx.send(std::thread::current().name().map(str::to_string));
                let _ = go_rx.recv();
                real
            }),
        );
        let b = budget();
        let limit = Some(Duration::from_millis(300));
        let t = Instant::now();
        assert_eq!(resolve_within(standin, limit, 1, &b), standin, "drawn with the stand-in");
        let took = t.elapsed();
        assert!(took >= Duration::from_millis(290) && took < Duration::from_secs(5), "waited {took:?}");
        let who = who_rx.recv_timeout(Duration::from_secs(5)).expect("the build went to a worker");
        assert!(who.as_deref().is_some_and(|n| n.starts_with("xcl-pso")), "built on {who:?}");
        // the limit is spent until the next present: the next draw does not wait
        let t = Instant::now();
        assert_eq!(resolve_within(standin, limit, 1, &b), standin);
        assert!(t.elapsed() < Duration::from_millis(200), "waited {:?} with the limit spent", t.elapsed());
        assert!(is_pending(standin));
        // after the next present, once built: the real pipeline
        go_tx.send(()).unwrap();
        assert_eq!(resolve_within(standin, Some(Duration::from_secs(10)), 2, &b), real);
        assert!(!is_pending(standin));
        let _ = released(standin);
    }

    /// Compute pipelines are still waited for (a skipped dispatch corrupts its results), and
    /// `XCL_PSO_WAIT_MS=0` (no limit) waits for graphics ones as before.
    #[test]
    fn compute_pipelines_and_no_limit_wait_as_before() {
        let b = budget();
        let (c, c_real) = (0x7410_0011usize, 0x7410_0012usize);
        submit_lazy(
            c,
            true,
            Box::new(move || {
                std::thread::sleep(Duration::from_millis(100));
                c_real
            }),
        );
        assert_eq!(resolve_within(c, Some(Duration::ZERO), 1, &b), c_real);
        let (g, g_real) = (0x7410_0021usize, 0x7410_0022usize);
        submit_lazy(g, false, Box::new(move || g_real));
        assert_eq!(resolve_within(g, None, 1, &b), g_real);
        assert_eq!(resolve_within(0x7410_0031, Some(Duration::ZERO), 1, &b), 0x7410_0031, "not a stand-in");
        let _ = released(c);
        let _ = released(g);
    }

    #[test]
    fn lazy_pipeline_is_active_only_after_its_build_starts() {
        let standin = usize::MAX - 1000;
        submit_lazy(standin, true, Box::new(move || {
            let p = entry(standin).unwrap();
            assert!(p.is_active());
            0
        }));
        let p = entry(standin).unwrap();
        assert!(!p.is_active());
        assert_eq!(resolve_wait(standin), standin);
        assert!(!p.is_active());
        assert_eq!(released(standin), 0);
    }

    #[test]
    fn bounded_compute_wait_times_out_then_resolves_once_worker_finishes() {
        // not `0x7300_0001`, which `bind`'s test uses: both run in one process, and its
        // release would drop this test's pending entry
        let standin = 0x7300_0101usize;
        let real_pso = 0x7300_0102usize;
        let budget = FrameBudget::new();
        let (tx, rx) = mpsc::channel::<()>();
        submit_lazy(
            standin,
            true,
            Box::new(move || {
                let _ = rx.recv_timeout(Duration::from_secs(5));
                real_pso
            }),
        );

        // First call sends the lazy build to the compute worker pool and returns the stand-in after ~limit.
        let t0 = Instant::now();
        assert_eq!(resolve_compute_within(standin, Some(Duration::from_millis(60)), &budget), standin);
        assert!(t0.elapsed() < Duration::from_secs(1), "bounded wait must not block indefinitely");
        assert!(is_pending_compute(standin));

        // Subsequent calls while still building return immediately (even on a fresh frame budget).
        budget.reset();
        let t1 = Instant::now();
        assert_eq!(resolve_compute_within(standin, Some(Duration::from_secs(2)), &budget), standin);
        assert!(t1.elapsed() < Duration::from_millis(200), "post-timeout calls must return immediately");

        // Unblock the background worker; once complete, resolution immediately returns the real PSO.
        tx.send(()).unwrap();
        let wait = Instant::now();
        while resolve_compute_within(standin, Some(Duration::from_millis(50)), &budget) != real_pso {
            assert!(wait.elapsed() < Duration::from_secs(5), "background compute PSO never completed");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!is_pending_compute(standin));
        assert_eq!(released(standin), real_pso);
    }

    #[test]
    fn bounded_compute_wait_returns_real_when_build_finishes_in_time() {
        let standin = 0x7300_0003usize;
        let real_pso = 0x7300_0004usize;
        let budget = FrameBudget::new();
        let (name_tx, name_rx) = mpsc::channel::<String>();
        submit_lazy(
            standin,
            true,
            Box::new(move || {
                let _ = name_tx.send(std::thread::current().name().unwrap_or("").to_string());
                real_pso
            }),
        );
        assert_eq!(resolve_compute_within(standin, Some(Duration::from_secs(4)), &budget), real_pso);
        let worker_name = name_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(
            worker_name.starts_with("xcl-pso-compute-"),
            "compute pipelines must run on dedicated compute workers, got {worker_name:?}"
        );
        assert_eq!(released(standin), real_pso);
    }

    #[test]
    fn bounded_compute_wait_shares_per_frame_budget_across_pipelines() {
        let standin_a = 0x7300_0005usize;
        let real_a = 0x7300_0006usize;
        let standin_b = 0x7300_0007usize;
        let real_b = 0x7300_0008usize;
        let budget = FrameBudget::new();
        let (tx_a, rx_a) = mpsc::channel::<()>();
        let (tx_b, rx_b) = mpsc::channel::<()>();
        submit_lazy(
            standin_a,
            true,
            Box::new(move || {
                let _ = rx_a.recv_timeout(Duration::from_secs(5));
                real_a
            }),
        );
        submit_lazy(
            standin_b,
            true,
            Box::new(move || {
                let _ = rx_b.recv_timeout(Duration::from_secs(5));
                real_b
            }),
        );

        // First slow pipeline consumes the entire 60 ms frame budget.
        assert_eq!(resolve_compute_within(standin_a, Some(Duration::from_millis(60)), &budget), standin_a);

        // Second uncached pipeline in the same frame is queued but does NOT stall for another 60 ms.
        let t_b = Instant::now();
        assert_eq!(resolve_compute_within(standin_b, Some(Duration::from_millis(60)), &budget), standin_b);
        assert!(
            t_b.elapsed() < Duration::from_millis(20),
            "second pipeline in exhausted frame must not multiply wait time"
        );

        // Next frame refreshes the budget; once `b` finishes, waiting on it resolves `real_b`.
        budget.reset();
        tx_b.send(()).unwrap();
        assert_eq!(resolve_compute_within(standin_b, Some(Duration::from_secs(4)), &budget), real_b);

        tx_a.send(()).unwrap();
        let wait = Instant::now();
        while resolve_compute_within(standin_a, Some(Duration::from_millis(50)), &budget) != real_a {
            assert!(wait.elapsed() < Duration::from_secs(5), "background compute PSO A never completed");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(released(standin_a), real_a);
        assert_eq!(released(standin_b), real_b);
    }

    /// A pipeline whose shaders were drawn with in an earlier run is built while the title
    /// loads, so its first draw finds it ready within the per-frame bound; one the title never
    /// drew waits for its first use (the lazy stand-in).
    #[test]
    fn a_pipeline_drawn_before_is_ready_at_its_first_draw() {
        assert!(build_during_load(false, false), "no lazy creation: the build is in the call");
        assert!(build_during_load(true, true), "drawn before: built during load");
        assert!(!build_during_load(true, false), "never drawn: built at its first use");
        let (standin, real) = (0x7410_0041usize, 0x7410_0042usize);
        assert!(build_during_load(true, true));
        submit(standin, false, Box::new(move || real));
        // the creation is over; the loading screen runs, then the first draw asks for it
        let b = budget();
        let t = Instant::now();
        assert_eq!(resolve_within(standin, Some(Duration::from_millis(400)), 1, &b), real);
        assert!(t.elapsed() < Duration::from_millis(400), "first draw waited {:?}", t.elapsed());
        let _ = released(standin);
    }

    /// The pool: NVIDIA gets the measured best count, everything else keeps a worker per core
    /// minus one, and `XCL_PSO_THREADS` wins over both (clamped to 1..=64).
    #[test]
    fn the_pipeline_pool_size_follows_the_adapter_and_xcl_pso_threads() {
        assert_eq!(workers_for(true, None, 15), NVIDIA_WORKERS);
        assert_eq!(workers_for(true, Some(1), 15), 1);
        assert_eq!(workers_for(true, Some(8), 15), 8);
        assert_eq!(workers_for(false, None, 15), 15);
        assert_eq!(workers_for(false, Some(3), 15), 3);
        assert_eq!(workers_for(false, Some(0), 15), 1, "a pool needs a worker");
        assert_eq!(workers_for(false, Some(1000), 15), 64);
        assert!(NVIDIA_WORKERS >= 1 && NVIDIA_WORKERS <= 8, "measured best on NVIDIA is 4");
    }
}
