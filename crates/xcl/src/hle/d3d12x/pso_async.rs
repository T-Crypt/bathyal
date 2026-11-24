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
    /// of its shaders (`used`); 0 = none
    key: u64,
    /// it was used (`used::note` is told once)
    used: std::sync::atomic::AtomicBool,
    /// the stand-in was released before the real pipeline was built
    orphan: std::sync::atomic::AtomicBool,
    /// its shaders are in the prewarm record (marked once)
    noted: std::sync::atomic::AtomicBool,
}

impl Pending {
    fn is_active(&self) -> bool {
        self.sent.load(Ordering::Acquire) && !*self.done.lock().unwrap()
    }
}

impl Pending {
    fn note_use(&self) {
        if self.key != 0 && !self.used.swap(true, Ordering::Relaxed) {
            used::note(self.key);
        }
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
        let start = std::time::Instant::now();
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

fn spawn_pool(prefix: &str, threads: usize) -> Mutex<mpsc::Sender<Arc<Pending>>> {
    let (tx, rx) = mpsc::channel::<Arc<Pending>>();
    let rx = Arc::new(Mutex::new(rx));
    // translation recurses deeply on big programs; DXC itself runs on its own pool
    for k in 0..threads {
        let rx = rx.clone();
        let _ = std::thread::Builder::new().name(format!("{prefix}-{k}")).stack_size(64 << 20).spawn(move || loop {
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
}

fn queue() -> &'static Mutex<mpsc::Sender<Arc<Pending>>> {
    static Q: OnceLock<Mutex<mpsc::Sender<Arc<Pending>>>> = OnceLock::new();
    Q.get_or_init(|| spawn_pool("xcl-pso", pipeline_threads()))
}

/// Dedicated worker pool for compute pipelines so a backlog (or driver stall, #156) of graphics
/// builds cannot starve compute dispatches indefinitely.
fn compute_queue() -> &'static Mutex<mpsc::Sender<Arc<Pending>>> {
    static Q: OnceLock<Mutex<mpsc::Sender<Arc<Pending>>>> = OnceLock::new();
    Q.get_or_init(|| spawn_pool("xcl-pso-compute", (crate::gpu::recomp::dxc::compile_threads() / 2).clamp(1, 4)))
}

/// Send `p` to its worker pool if not already sent; clears `sent` on failure so a later bind
/// can retry instead of stalling forever.
fn enqueue(p: &Arc<Pending>) -> bool {
    if !p.sent.swap(true, Ordering::AcqRel) {
        let q = if p.compute { compute_queue() } else { queue() };
        if q.lock().unwrap().send(p.clone()).is_err() {
            p.sent.store(false, Ordering::Release);
            return false;
        }
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

/// `standin` went to the title; `job` builds the real pipeline on a worker.
pub fn submit(standin: usize, compute: bool, job: Job) {
    submit_as(standin, compute, job, true, 0);
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

/// `submit`, sent to the workers only on its first `resolve` (`lazy`) – or now, at a low
/// priority, when an earlier run used a pipeline of these shaders (`key`, 0 = none; `used`).
pub fn submit_lazy(standin: usize, compute: bool, job: Job, key: u64) {
    submit_as(standin, compute, job, false, key);
}

fn submit_as(standin: usize, compute: bool, job: Job, send: bool, key: u64) {
    let p = Arc::new(Pending {
        standin,
        ticket: Mutex::new(None),
        job: Mutex::new(Some(job)),
        sent: std::sync::atomic::AtomicBool::new(false),
        real: AtomicUsize::new(0),
        done: Mutex::new(false),
        cv: Condvar::new(),
        compute,
        key,
        used: Default::default(),
        orphan: Default::default(),
        noted: Default::default(),
    });
    PENDING.lock().unwrap().get_or_insert_with(HashMap::new).insert(standin, p.clone());
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_info!("D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background");
    });
    if send {
        let _ = queue().lock().unwrap().send(p);
    } else if used::known(key) {
        if let Some(q) = preload() {
            q.list.lock().unwrap().push_back(p);
            q.cv.notify_one();
        }
    }
}

/// The pipelines a title used, by a hash of their shaders, kept between runs
/// (`used-pipelines.txt` beside the DXIL cache, a key a line).
///
/// A lazy pipeline is built at its first use: until then a graphics one draws nothing, and a
/// compute one is waited for on the title's thread. A retail title's first frames of its world
/// use hundreds of pipelines for the first time at once. The pipelines of an earlier run are
/// built while the title loads instead (`preload`): measured on a build that skips a dispatch
/// instead of waiting for its pipeline, ~2 170 dispatches were skipped in those frames without
/// this and 94 – 150 with it.
mod used {
    use std::collections::HashSet;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock, RwLock};

    struct Used {
        earlier: RwLock<HashSet<u64>>,
        now: Mutex<(HashSet<u64>, Option<std::fs::File>)>,
    }

    /// The keys in a `used-pipelines.txt` (lines that are not a hexadecimal key are passed over).
    pub fn parse(text: &str) -> HashSet<u64> {
        text.lines().filter_map(|l| u64::from_str_radix(l.trim(), 16).ok()).filter(|&k| k != 0).collect()
    }

    fn get() -> &'static Used {
        static U: OnceLock<Used> = OnceLock::new();
        U.get_or_init(|| {
            // (a test neither reads nor writes the user's file)
            if cfg!(test) {
                return Used { earlier: RwLock::new(HashSet::new()), now: Mutex::new((HashSet::new(), None)) };
            }
            let path = crate::gpu::recomp::dxc::cache_dir().join("used-pipelines.txt");
            let earlier = parse(&std::fs::read_to_string(&path).unwrap_or_default());
            if !earlier.is_empty() {
                crate::xlog_info!("D3D12.X: the pipelines of {} shader sets an earlier run used are built ahead of their first use", earlier.len());
            }
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok();
            Used { now: Mutex::new((earlier.clone(), file)), earlier: RwLock::new(earlier) }
        })
    }

    /// An earlier run used a pipeline of these shaders.
    pub fn known(key: u64) -> bool {
        key != 0 && get().earlier.read().unwrap().contains(&key)
    }

    /// A pipeline of these shaders is used now: kept for the next run.
    pub fn note(key: u64) {
        if key == 0 {
            return;
        }
        let mut g = get().now.lock().unwrap();
        if g.0.insert(key) {
            if let Some(f) = g.1.as_mut() {
                let _ = writeln!(f, "{key:016x}");
            }
        }
    }

    /// (tests) This run used a pipeline of these shaders.
    #[cfg(test)]
    pub fn noted(key: u64) -> bool {
        get().now.lock().unwrap().0.contains(&key)
    }

    #[cfg(test)]
    pub fn earlier_run_used(key: u64) {
        get().earlier.write().unwrap().insert(key);
    }
}

/// The workers that build the pipelines of an earlier run ahead of their first use
/// (`XCL_PSO_PRELOAD=0`: off). Apart from the ones a bind sends to (`queue`) and at the lowest
/// priority: a pipeline that is bound is not held up by the ones made ahead (whoever takes
/// `job` first builds it), and the title's own threads are not either.
struct Preload {
    list: Mutex<std::collections::VecDeque<Arc<Pending>>>,
    cv: Condvar,
}

fn preload() -> Option<&'static Preload> {
    static Q: OnceLock<Option<Preload>> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let q = Q
        .get_or_init(|| {
            if std::env::var("XCL_PSO_PRELOAD").is_ok_and(|v| v.trim() == "0") {
                return None;
            }
            Some(Preload { list: Mutex::new(Default::default()), cv: Condvar::new() })
        })
        .as_ref()?;
    STARTED.call_once(|| {
        let n = (crate::gpu::recomp::dxc::compile_threads() / 4).clamp(2, 4);
        for k in 0..n {
            let _ = std::thread::Builder::new().name(format!("xcl-pso-pre-{k}")).stack_size(64 << 20).spawn(move || {
                unsafe {
                    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
                    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                }
                static DONE: AtomicUsize = AtomicUsize::new(0);
                loop {
                    let p = {
                        let mut g = q.list.lock().unwrap();
                        loop {
                            if let Some(p) = g.pop_front() {
                                break p;
                            }
                            g = q.cv.wait(g).unwrap();
                        }
                    };
                    // (released by the title meanwhile)
                    if p.orphan.load(Ordering::Acquire) {
                        continue;
                    }
                    // (None: a bind took it and builds it itself)
                    let job = p.job.lock().unwrap().take();
                    if let Some(job) = job {
                        let real = job();
                        complete(&p, real);
                        let n = DONE.fetch_add(1, Ordering::Relaxed) + 1;
                        if n.is_power_of_two() {
                            crate::xlog_info!("D3D12.X: {n} pipelines built ahead of their first use");
                        }
                    }
                }
            });
        }
    });
    Some(q)
}

#[cfg(test)]
mod used_tests {
    use super::*;

    #[test]
    fn the_keys_of_a_file_are_read_and_other_lines_passed_over() {
        let keys = used::parse("00000000000000ff\n\nnot a key\n  1a2b \n0\n");
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&0xff) && keys.contains(&0x1a2b));
    }

    /// A lazy pipeline of shaders an earlier run used is built without a bind; another one is
    /// built only at its first use, and that use is kept for the next run.
    #[test]
    fn a_pipeline_an_earlier_run_used_is_built_ahead_of_its_first_use() {
        let (known, unknown) = (0x7E57_7001usize, 0x7E57_7002usize);
        let (known_key, unknown_key) = (0x5EED_0001u64, 0x5EED_0002u64);
        used::earlier_run_used(known_key);
        submit_lazy(known, false, Box::new(|| 0x4321), known_key);
        submit_lazy(unknown, false, Box::new(|| 0x4322), unknown_key);
        let start = std::time::Instant::now();
        while is_pending(known) && start.elapsed() < std::time::Duration::from_secs(10) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(!is_pending(known), "built by the workers that build ahead, with no bind");
        assert_eq!(resolve(known), 0x4321);
        assert!(is_pending(unknown), "nobody used it and no earlier run did");
        assert!(!used::noted(unknown_key));
        assert_eq!(resolve_wait(unknown), 0x4322);
        assert!(used::noted(unknown_key) && used::noted(known_key), "a first use is kept for the next run");
        released(known);
        released(unknown);
    }
}

fn entry(pso: usize) -> Option<Arc<Pending>> {
    PENDING.lock().unwrap().as_ref()?.get(&pso).cloned()
}

/// The pipeline to use for `pso` now: the real one once built, else `pso` itself.
pub fn resolve(pso: usize) -> usize {
    let Some(p) = entry(pso) else { return pso };
    p.note_use();
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
    p.note_use();
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
        None => {
            // a pipeline built in the call (no stand-in): drawn with, so prewarmed next run
            super::pso_disk::note_drawn(pso);
            pso
        }
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

    #[test]
    fn bounded_compute_wait_times_out_then_resolves_once_worker_finishes() {
        let standin = 0x7300_0001usize;
        let real_pso = 0x7300_0002usize;
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
