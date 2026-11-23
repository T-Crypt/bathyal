//! Non-blocking pipeline creation. Translating a console shader takes milliseconds; compiling
//! the HLSL with DXC takes 0.2 s to seconds, and a title creates thousands of pipelines while
//! it loads. When a pipeline's DXIL is not in the disk cache yet, the title gets a stand-in
//! pipeline (shaders that draw nothing) at once and the real one is built on a worker; binding
//! (`SetPipelineState`, console list replay, the two-pass lookups) uses the real one as soon
//! as it is there. A draw waits for its graphics pipeline only so long between presents
//! (`resolve_for_draw`), then draws nothing until it is built; a compute pipeline is waited
//! for at its first bind (a skipped dispatch would corrupt what it computes).
//! `XCL_ASYNC_PSO=0` builds every pipeline in the call, as before.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

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

fn queue() -> &'static Mutex<mpsc::Sender<Arc<Pending>>> {
    static Q: OnceLock<Mutex<mpsc::Sender<Arc<Pending>>>> = OnceLock::new();
    Q.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Arc<Pending>>();
        let rx = Arc::new(Mutex::new(rx));
        // translation recurses deeply on big programs; DXC itself runs on its own pool
        let n = crate::gpu::recomp::dxc::compile_threads();
        for k in 0..n {
            let rx = rx.clone();
            let _ = std::thread::Builder::new().name(format!("xcl-pso-{k}")).stack_size(64 << 20).spawn(move || loop {
                let next = rx.lock().unwrap().recv();
                let Ok(p) = next else { return };
                // (None: a bind took it and builds it itself)
                let job = p.job.lock().unwrap().take();
                if let Some(job) = job {
                    complete(&p, job());
                }
            });
        }
        Mutex::new(tx)
    })
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
        sent: std::sync::atomic::AtomicBool::new(send),
        real: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
        orphan: Default::default(),
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        let _ = queue().lock().unwrap().send(p);
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
                let _ = queue().lock().unwrap().send(p);
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

/// For binding: compute pipelines are waited for, graphics ones are not.
pub fn resolve_for_bind(pso: usize) -> usize {
    match entry(pso) {
        Some(p) if p.compute => resolve_wait(pso),
        Some(_) => resolve(pso),
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

    fn budget() -> Mutex<DrawWait> {
        Mutex::new(DrawWait { epoch: 0, spent: Duration::ZERO })
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
}
