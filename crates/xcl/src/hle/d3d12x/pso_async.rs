//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A draw waits for its graphics pipeline only so long between presents
//! (`resolve_for_draw`), then draws nothing until it is built; a compute pipeline is waited
//! for at its first bind (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

struct Pending {
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
}

impl Pending {
    fn is_active(&self) -> bool {
        self.sent.load(Ordering::Acquire) && !*self.done.lock().unwrap()
    }
}

/// stand-in handed to the title → its real pipeline
static PENDING: Mutex<Option<HashMap<usize, Arc<Pending>>>> = Mutex::new(None);

/// The build is done (`real`: its pipeline, 0 = failed).
fn complete(p: &Pending, real: usize) {
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

struct WorkQueue<T> {
    gfx: VecDeque<T>,
    compute: VecDeque<T>,
}

impl<T> WorkQueue<T> {
    fn new() -> Self {
        Self { gfx: VecDeque::new(), compute: VecDeque::new() }
    }

    fn push(&mut self, compute: bool, item: T) {
        if compute {
            self.compute.push_back(item);
        } else {
            self.gfx.push_back(item);
        }
    }

    fn pop_next(&mut self) -> Option<T> {
        self.gfx.pop_front().or_else(|| self.compute.pop_front())
    }
}

struct Pool {
    q: Mutex<WorkQueue<Arc<Pending>>>,
    cv: Condvar,
}

fn pool() -> &'static Arc<Pool> {
    static Q: OnceLock<Arc<Pool>> = OnceLock::new();
    Q.get_or_init(|| {
        let p = Arc::new(Pool { q: Mutex::new(WorkQueue::new()), cv: Condvar::new() });
        let n = crate::gpu::recomp::dxc::compile_threads();
        for k in 0..n {
            let p = p.clone();
            let _ = std::thread::Builder::new().name(format!("xcl-pso-{k}")).stack_size(64 << 20).spawn(move || loop {
                let next = {
                    let mut q = p.q.lock().unwrap();
                    loop {
                        if let Some(item) = q.pop_next() {
                            break item;
                        }
                        q = p.cv.wait(q).unwrap();
                    }
                };
                let job = next.job.lock().unwrap().take();
                if let Some(job) = job {
                    complete(&next, job());
                }
            });
        }
        p
    })
}

fn enqueue(p: Arc<Pending>) {
    let pool = pool();
    let mut q = pool.q.lock().unwrap();
    q.push(p.compute, p);
    pool.cv.notify_one();
}

/// Release a pipeline state XCL owns (raw pointer, 0 = none).
pub fn release(p: usize) {
    if p != 0 {
        unsafe { drop(<windows::Win32::Graphics::Direct3D12::ID3D12PipelineState as windows::core::Interface>::from_raw(p as *mut std::ffi::c_void)) };
    }
}

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_as(standin, compute, job, true);
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
    submit_as(standin, compute, job, false);
}

fn submit_as(standin: usize, compute: bool, job: Job, send: bool) {
    let p = Arc::new(Pending {
        job: Mutex::new(Some(job)),
        sent: std::sync::atomic::AtomicBool::new(false),
        real: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
        timed_out: Default::default(),
        orphan: Default::default(),
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        enqueue(p);
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
            if !p.sent.swap(true, Ordering::AcqRel) {
                enqueue(p);
            }
            pso
        }
        r => r,
    }
}

/// `resolve`, waiting for a pipeline still being built – built here if no worker has started
/// it yet (a queue of a title's hundreds of pipelines must not hold up the one it binds now).
pub fn resolve_wait(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    let start = Instant::now();
    p.sent.store(true, Ordering::Release);
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
    if !p.sent.swap(true, Ordering::AcqRel) {
        let _ = queue().lock().unwrap().send(p.clone());
    }
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
    let waited = start.elapsed();
    {
        let mut b = budget.lock().unwrap();
        if b.epoch == epoch {
            b.spent += waited;
        }
    }
    if !finished {
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
/// bind/dispatch (`XCL_COMPUTE_PSO_WAIT_MS`, default 4000 ms; `0` = unbounded). Huge compute
/// shaders can take 20+ s to compile in DXC / the driver, and a loading frame can bind many
/// uncached compute pipelines in a row; sharing one budget per frame keeps the total bounded.
fn compute_wait_timeout() -> Option<std::time::Duration> {
    static LIMIT: OnceLock<Option<std::time::Duration>> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        let ms = std::env::var("XCL_COMPUTE_PSO_WAIT_MS").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(4000);
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

/// Bounded wait for a compute pipeline: always builds on `xcl-pso-compute-*` workers (never
/// stalling the title's thread inside DXC/driver), waits at most the remaining per-frame `budget`
/// up to `limit`, and returns the stand-in immediately once timed out or once the frame budget is
/// exhausted until the worker finishes.
fn resolve_compute_within(pso: usize, limit: Option<std::time::Duration>, budget: &FrameBudget) -> usize {
    let Some(limit) = limit else {
        return resolve_wait(pso);
    };
    let Some(p) = entry(pso) else { return pso };
    if let r @ 1.. = p.real.load(Ordering::Acquire) {
        return r;
    }
    if !p.sent.swap(true, Ordering::AcqRel) {
        enqueue(p.clone());
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
        None => pso,
    }
}

/// For `ExecuteIndirect`: compute pipelines use bounded waits, graphics pipelines keep `resolve_wait`.
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

/// How many uncached driver compiles may run at once. Each one fans out across the GPU
/// driver's own compiler, so a few of them already fill the machine; more of them fight
/// over the same cores and the load takes longer. `XCL_PSO_COMPILES` overrides it.
fn driver_compile_limit() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        std::env::var("XCL_PSO_COMPILES").ok().and_then(|v| v.parse().ok()).unwrap_or((cores / 4).clamp(2, 4))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn graphics_jobs_dequeue_ahead_of_queued_compute_jobs() {
        let mut q = WorkQueue::new();
        q.push(true, "c1");
        q.push(true, "c2");
        q.push(false, "g1");
        q.push(true, "c3");
        q.push(false, "g2");
        assert_eq!(q.pop_next(), Some("g1"));
        assert_eq!(q.pop_next(), Some("g2"));
        assert_eq!(q.pop_next(), Some("c1"));
        assert_eq!(q.pop_next(), Some("c2"));
        assert_eq!(q.pop_next(), Some("c3"));
        assert_eq!(q.pop_next(), None);
    }

    #[test]
    fn worker_pool_prioritizes_graphics_pipeline_over_queued_compute() {
        let n = crate::gpu::recomp::dxc::compile_threads();
        let (started_tx, started_rx) = mpsc::channel::<usize>();
        let mut gates = Vec::with_capacity(n);
        let mut blockers = Vec::with_capacity(n);
        for i in 0..n {
            let id = 0x7600_0001usize + i * 2;
            let (tx, rx) = mpsc::channel::<()>();
            let stx = started_tx.clone();
            submit(
                id,
                true,
                Box::new(move || {
                    let _ = stx.send(id);
                    let _ = rx.recv_timeout(Duration::from_secs(10));
                    id + 1
                }),
            );
            gates.push(tx);
            blockers.push(id);
        }
        for _ in 0..n {
            started_rx.recv_timeout(Duration::from_secs(5)).expect("worker started blocker");
        }
        let (order_tx, order_rx) = mpsc::channel::<&'static str>();
        let compute_id = 0x7601_0001usize;
        let graphics_id = 0x7601_0003usize;
        let ctx = order_tx.clone();
        let (c_release_tx, c_release_rx) = mpsc::channel::<()>();
        submit(
            compute_id,
            true,
            Box::new(move || {
                let _ = ctx.send("compute");
                let _ = c_release_rx.recv_timeout(Duration::from_secs(5));
                compute_id + 1
            }),
        );
        let gtx = order_tx.clone();
        submit(
            graphics_id,
            false,
            Box::new(move || {
                let _ = gtx.send("graphics");
                graphics_id + 1
            }),
        );
        gates[0].send(()).unwrap();
        assert_eq!(order_rx.recv_timeout(Duration::from_secs(5)).unwrap(), "graphics");
        for g in &gates[1..] {
            let _ = g.send(());
        }
        let _ = c_release_tx.send(());
        assert_eq!(order_rx.recv_timeout(Duration::from_secs(5)).unwrap(), "compute");
        assert_eq!(resolve_wait(graphics_id), graphics_id + 1);
        assert_eq!(resolve_wait(compute_id), compute_id + 1);
        for id in blockers {
            assert_eq!(resolve_wait(id), id + 1);
            let _ = released(id);
        }
        let _ = released(compute_id);
        let _ = released(graphics_id);
    }
}
