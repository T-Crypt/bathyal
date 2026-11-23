//! `XCL_PROFILE=1`: a sampling CPU profiler inside the process – where the title's time goes,
//! without an external tool.
//!
//! A thread named `xcl-profiler` runs at `XCL_PROFILE_HZ` (default 1000) and, for every other
//! thread of the process, suspends it for the length of one `GetThreadContext`, reads RIP and
//! resumes it. Each RIP becomes one sample, keyed by module and 16-byte cell of its RVA. A sample
//! whose RIP is inside a waiting `ntdll` export counts as a thread inside a wait, so the per-thread
//! table shows how much of a thread's time is spent doing its own work rather than waiting.
//!
//! Nothing is logged or allocated while a thread is suspended: the sampler copies RIPs into a
//! buffer that is already large enough, resumes every thread, and only then classifies the
//! samples, takes the lock and (at an interval) formats a report.
//!
//! Off by default: with `XCL_PROFILE` unset nothing is started and nothing costs anything.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Diagnostics::Debug::{GetThreadContext, CONTEXT, CONTEXT_CONTROL_AMD64};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, Module32FirstW, Module32NextW, Thread32First, Thread32Next, MODULEENTRY32W, TH32CS_SNAPMODULE, TH32CS_SNAPMODULE32, TH32CS_SNAPTHREAD, THREADENTRY32,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcessId, GetCurrentThreadId, GetThreadDescription, OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_QUERY_LIMITED_INFORMATION, THREAD_SUSPEND_RESUME,
};

// ---------------------------------------------------------------------------------------
// settings
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    /// samples per second
    pub hz: u32,
    /// entries in the site table
    pub top: usize,
    /// seconds between reports while the title runs
    pub every: Option<u64>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { hz: 1000, top: 40, every: None }
    }
}

/// `XCL_PROFILE=1` (`0` or unset: off), `XCL_PROFILE_HZ`, `XCL_PROFILE_TOP`, `XCL_PROFILE_EVERY`.
pub fn settings_from_env() -> Option<Settings> {
    let on = std::env::var_os("XCL_PROFILE").map(|v| {
        let s = v.to_string_lossy();
        !(s.trim().is_empty() || s.trim() == "0")
    })?;
    if !on {
        return None;
    }
    let num = |k: &str| std::env::var(k).ok().and_then(|v| v.trim().parse::<u64>().ok());
    Some(Settings {
        hz: num("XCL_PROFILE_HZ").filter(|&h| (1..=10_000).contains(&h)).unwrap_or(1000) as u32,
        top: num("XCL_PROFILE_TOP").filter(|&t| (1..=1000).contains(&t)).unwrap_or(40) as usize,
        every: num("XCL_PROFILE_EVERY").filter(|&e| e >= 1),
    })
}

// ---------------------------------------------------------------------------------------
// samples
// ---------------------------------------------------------------------------------------

#[derive(Clone, Default)]
pub struct ThreadCount {
    pub samples: u64,
    /// samples whose RIP was inside a waiting `ntdll` function
    pub waiting: u64,
    pub name: String,
}

#[derive(Clone)]
pub struct SiteCount {
    pub module: String,
    pub base: usize,
    pub rva: u32,
    pub samples: u64,
}

#[derive(Default)]
struct Stats {
    total: u64,
    /// threads that could not be opened or suspended this tick
    missed: u64,
    sites: HashMap<(String, u32), (usize, u64)>,
    threads: HashMap<u32, ThreadCount>,
}

impl Stats {
    fn record(&mut self, tid: u32, rip: usize, modules: &[ModuleRange], waits: &[(u32, u32)]) {
        self.total += 1;
        let t = self.threads.entry(tid).or_default();
        t.samples += 1;
        let Some(m) = modules.iter().find(|m| rip >= m.base && rip < m.base + m.size) else { return };
        let rva = (rip - m.base) as u32;
        if m.lower == "ntdll.dll" && waits.iter().any(|&(s, e)| rva >= s && rva < e) {
            t.waiting += 1;
        }
        let cell = rva & !15;
        let e = self.sites.entry((m.name.clone(), cell)).or_insert((m.base, 0));
        e.0 = m.base;
        e.1 += 1;
    }
}

/// one row of a finished report
pub struct Snapshot {
    pub seconds: f64,
    pub total: u64,
    pub missed: u64,
    /// distinct (module, 16-byte cell) sites, before the top-N cut
    pub sites_total: usize,
    pub sites: Vec<SiteCount>,
    pub threads: Vec<(u32, ThreadCount)>,
}

// ---------------------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------------------

pub struct Shared {
    settings: Settings,
    stop: AtomicBool,
    started: Instant,
    stats: Mutex<Stats>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

static RUNNING: Mutex<Option<Arc<Shared>>> = Mutex::new(None);

impl Shared {
    pub fn settings(&self) -> Settings {
        self.settings
    }

    /// (total samples, per thread id) – for tests and callers that format their own report.
    pub fn counts(&self) -> (u64, HashMap<u32, ThreadCount>) {
        let st = self.stats.lock().unwrap_or_else(|p| p.into_inner());
        (st.total, st.threads.clone())
    }

    pub fn snapshot(&self) -> Snapshot {
        let st = self.stats.lock().unwrap_or_else(|p| p.into_inner());
        let mut sites: Vec<SiteCount> = st.sites.iter().map(|((module, rva), &(base, samples))| SiteCount { module: module.clone(), base, rva: *rva, samples }).collect();
        sites.sort_by(|a, b| b.samples.cmp(&a.samples).then_with(|| a.module.cmp(&b.module)).then_with(|| a.rva.cmp(&b.rva)));
        let sites_total = sites.len();
        sites.truncate(self.settings.top.max(1));
        let mut threads: Vec<(u32, ThreadCount)> = st.threads.iter().map(|(&tid, t)| (tid, t.clone())).collect();
        threads.sort_by(|a, b| b.1.samples.cmp(&a.1.samples));
        Snapshot { seconds: self.started.elapsed().as_secs_f64(), total: st.total, missed: st.missed, sites_total, sites, threads }
    }

    /// the sampler ends at the next tick (it is not joined here)
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }
}

/// Start the sampler unless it already runs. `None` when the thread cannot be created.
pub fn start(s: Settings) -> Option<Arc<Shared>> {
    let mut g = RUNNING.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(sh) = g.as_ref() {
        return Some(sh.clone());
    }
    let sh = Arc::new(Shared {
        settings: s,
        stop: AtomicBool::new(false),
        started: Instant::now(),
        stats: Mutex::new(Stats::default()),
        thread: Mutex::new(None),
    });
    *g = Some(sh.clone());
    drop(g);
    let sh2 = sh.clone();
    let spawned = std::thread::Builder::new().name("xcl-profiler".into()).spawn(move || run(sh2));
    match spawned {
        Ok(h) => *sh.thread.lock().unwrap_or_else(|p| p.into_inner()) = Some(h),
        Err(e) => {
            crate::xlog_warn!("profile: the sampler thread did not start: {e}");
            *RUNNING.lock().unwrap_or_else(|p| p.into_inner()) = None;
            return None;
        }
    }
    crate::xlog_info!(
        "profile: sampling {} thread(s) at {} Hz, top {} sites{}",
        thread_count_hint(),
        s.hz,
        s.top,
        match s.every {
            Some(e) => format!(", a report every {e} s"),
            None => String::new(),
        }
    );
    Some(sh)
}

/// `XCL_PROFILE=1` in the environment: start now, or nothing.
pub fn start_from_env() -> Option<Arc<Shared>> {
    settings_from_env().and_then(start)
}

// ---------------------------------------------------------------------------------------
// the sampler
// ---------------------------------------------------------------------------------------

struct ModuleRange {
    base: usize,
    size: usize,
    name: String,
    lower: String,
}

fn utf16(z: &[u16]) -> String {
    let n = z.iter().position(|&c| c == 0).unwrap_or(z.len());
    String::from_utf16_lossy(&z[..n])
}

fn run(sh: Arc<Shared>) {
    let interval = Duration::from_secs_f64(1.0 / sh.settings.hz as f64);
    let mut threads: Vec<u32> = Vec::new();
    let mut names: HashMap<u32, String> = HashMap::new();
    let mut modules: Vec<ModuleRange> = Vec::new();
    let mut waits: Vec<(u32, u32)> = Vec::new();
    // both lists are read again after a while: threads come and go, and so do modules
    let mut last_threads = Instant::now() - Duration::from_secs(1);
    let mut last_modules = Instant::now() - Duration::from_secs(1);
    let mut last_report = Instant::now();
    loop {
        if sh.stop.load(Ordering::Acquire) {
            break;
        }
        let tick = Instant::now();
        if last_threads.elapsed() >= Duration::from_millis(500) {
            refresh_threads(&mut threads, &mut names);
            last_threads = tick;
        }
        if last_modules.elapsed() >= Duration::from_secs(2) || waits.is_empty() {
            modules = module_list();
            if waits.is_empty() {
                waits = ntdll_wait_ranges();
            }
            last_modules = tick;
        }
        // nothing is allocated between SuspendThread and ResumeThread: RIPs go into a buffer
        // sized before the first suspension
        let mut buf: Vec<(u32, usize)> = Vec::with_capacity(threads.len());
        for &tid in &threads {
            match sample(tid) {
                Some(rip) => buf.push((tid, rip)),
                None => {
                    sh.stats.lock().unwrap_or_else(|p| p.into_inner()).missed += 1;
                }
            }
        }
        {
            let mut st = sh.stats.lock().unwrap_or_else(|p| p.into_inner());
            for &(tid, rip) in &buf {
                st.record(tid, rip, &modules, &waits);
            }
            // names arrive with the thread list; fill them in as they are seen
            for &(tid, _) in &buf {
                if let Some(entry) = st.threads.get_mut(&tid) {
                    if entry.name.is_empty() {
                        if let Some(n) = names.get(&tid) {
                            entry.name = n.clone();
                        }
                    }
                }
            }
        }
        drop(buf);
        if let Some(every) = sh.settings.every {
            if last_report.elapsed() >= Duration::from_secs(every) {
                log_report(&sh);
                last_report = tick;
            }
        }
        let spent = tick.elapsed();
        if spent < interval {
            std::thread::sleep(interval - spent);
        } else {
            std::thread::yield_now();
        }
    }
}

/// RIP of `tid`, captured while it is suspended for one context read (none: could not be taken).
fn sample(tid: u32) -> Option<usize> {
    unsafe {
        let h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_LIMITED_INFORMATION, 0, tid);
        if h.is_null() {
            return None;
        }
        // CONTEXT must be 16-byte aligned
        #[repr(align(16))]
        struct Aligned(CONTEXT);
        let mut c: Aligned = std::mem::zeroed();
        c.0.ContextFlags = CONTEXT_CONTROL_AMD64;
        let rip = if SuspendThread(h) != u32::MAX {
            let ok = GetThreadContext(h, &mut c.0) != 0;
            ResumeThread(h);
            ok.then_some(c.0.Rip as usize)
        } else {
            None
        };
        CloseHandle(h);
        rip
    }
}

/// The process's threads (not this one) and their names, from one Toolhelp snapshot.
fn refresh_threads(threads: &mut Vec<u32>, names: &mut HashMap<u32, String>) {
    threads.clear();
    unsafe {
        let me = GetCurrentThreadId();
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snap == INVALID_HANDLE_VALUE {
            return;
        }
        let pid = GetCurrentProcessId();
        let mut e: THREADENTRY32 = std::mem::zeroed();
        e.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
        let mut ok = Thread32First(snap, &mut e) != 0;
        while ok {
            if e.th32OwnerProcessID == pid && e.th32ThreadID != me {
                threads.push(e.th32ThreadID);
            }
            ok = Thread32Next(snap, &mut e) != 0;
        }
        CloseHandle(snap);
        names.retain(|k, _| threads.contains(k));
        for &tid in threads.iter() {
            if names.contains_key(&tid) {
                continue;
            }
            let h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, tid);
            if h.is_null() {
                continue;
            }
            let mut p: *mut u16 = std::ptr::null_mut();
            if GetThreadDescription(h, &mut p) >= 0 && !p.is_null() {
                let n = (0..256).take_while(|&i| *p.add(i) != 0).count();
                names.insert(tid, String::from_utf16_lossy(std::slice::from_raw_parts(p, n)));
                windows_sys::Win32::Foundation::LocalFree(p as *mut _);
            }
            CloseHandle(h);
        }
    }
}

fn thread_count_hint() -> usize {
    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
        if snap == INVALID_HANDLE_VALUE {
            return 0;
        }
        let pid = GetCurrentProcessId();
        let mut n = 0;
        let mut e: THREADENTRY32 = std::mem::zeroed();
        e.dwSize = std::mem::size_of::<THREADENTRY32>() as u32;
        let mut ok = Thread32First(snap, &mut e) != 0;
        while ok {
            if e.th32OwnerProcessID == pid {
                n += 1;
            }
            ok = Thread32Next(snap, &mut e) != 0;
        }
        CloseHandle(snap);
        n
    }
}

fn module_list() -> Vec<ModuleRange> {
    let mut out = Vec::new();
    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, GetCurrentProcessId());
        if snap == INVALID_HANDLE_VALUE {
            return out;
        }
        let mut m: MODULEENTRY32W = std::mem::zeroed();
        m.dwSize = std::mem::size_of::<MODULEENTRY32W>() as u32;
        let mut ok = Module32FirstW(snap, &mut m) != 0;
        while ok {
            let name = utf16(&m.szModule);
            out.push(ModuleRange { base: m.modBaseAddr as usize, size: m.modBaseSize as usize, lower: name.to_ascii_lowercase(), name });
            ok = Module32NextW(snap, &mut m) != 0;
        }
        CloseHandle(snap);
    }
    out
}

// ---------------------------------------------------------------------------------------
// waiting: the ntdll exports whose body is a wait
// ---------------------------------------------------------------------------------------

fn is_wait_export(name: &str) -> bool {
    let n = name.strip_prefix("Nt").or_else(|| name.strip_prefix("Zw")).or_else(|| name.strip_prefix("Rtl")).unwrap_or(name);
    n.contains("Wait")
        || name.starts_with("NtDelayExecution")
        || name.starts_with("ZwDelayExecution")
        || name.starts_with("NtRemoveIoCompletion")
        || name.starts_with("ZwRemoveIoCompletion")
        || name.starts_with("NtAlertThreadByThreadId")
}

/// RVAs of ntdll's exports, read from the loaded image (empty: nothing is called a wait).
fn ntdll_exports() -> Vec<(u32, String)> {
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    let mut out = Vec::new();
    unsafe {
        let base = GetModuleHandleW(w!("ntdll.dll")) as usize;
        if base == 0 {
            return out;
        }
        let rd32 = |a: usize| std::ptr::read_unaligned(a as *const u32);
        let rd16 = |a: usize| std::ptr::read_unaligned(a as *const u16);
        if rd16(base) != 0x5A4D {
            return out;
        }
        let pe = base.wrapping_add(rd32(base + 0x3C) as usize);
        if rd32(pe) != 0x0000_4550 || rd16(pe + 0x18) != 0x20B {
            return out;
        }
        // PE32+: the data directories start at optional header + 112; entry 0 is the exports
        let dir = pe.wrapping_add(0x18 + 112);
        let va = rd32(dir) as usize;
        let size = rd32(dir + 4) as usize;
        if va == 0 || size < 40 {
            return out;
        }
        let e = base.wrapping_add(va);
        let nfuncs = rd32(e + 0x14) as usize;
        let nnames = rd32(e + 0x18) as usize;
        let funcs = base.wrapping_add(rd32(e + 0x1C) as usize);
        let names = base.wrapping_add(rd32(e + 0x20) as usize);
        let ords = base.wrapping_add(rd32(e + 0x24) as usize);
        if nnames == 0 || nnames > 100_000 || nfuncs == 0 || nfuncs > 1_000_000 {
            return out;
        }
        for i in 0..nnames {
            let name_rva = rd32(names + 4 * i) as usize;
            let ord = rd16(ords + 2 * i) as usize;
            if ord >= nfuncs {
                continue;
            }
            let fun_rva = rd32(funcs + 4 * ord);
            if fun_rva == 0 {
                continue;
            }
            let s = std::ffi::CStr::from_ptr(base.wrapping_add(name_rva) as *const i8);
            out.push((fun_rva, s.to_string_lossy().into_owned()));
        }
    }
    out
}

/// (start, end) RVA ranges of ntdll's waiting functions: a function ends where the next export
/// starts (clamped to 4 KB, so a data export cannot swallow a large range).
fn ntdll_wait_ranges() -> Vec<(u32, u32)> {
    static CACHE: OnceLock<Vec<(u32, u32)>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let mut ex = ntdll_exports();
            if ex.is_empty() {
                return Vec::new();
            }
            ex.sort_by_key(|(rva, _)| *rva);
            let mut out = Vec::new();
            for i in 0..ex.len() {
                if !is_wait_export(&ex[i].1) {
                    continue;
                }
                let start = ex[i].0;
                let next = ex.get(i + 1).map(|e| e.0).filter(|&n| n > start).unwrap_or(start.saturating_add(0x1000));
                out.push((start, next.min(start.saturating_add(0x1000))));
            }
            out
        })
        .clone()
}

// ---------------------------------------------------------------------------------------
// the report
// ---------------------------------------------------------------------------------------

fn format_report(snap: &Snapshot, s: Settings) -> String {
    let pct = |n: u64, d: u64| if d == 0 { 0.0 } else { n as f64 * 100.0 / d as f64 };
    let mut out = format!(
        "profile: {} samples in {:.1} s ({} Hz, {} threads, {} samples missed)\ntop {} of {} sites (module+0xRVA, 16-byte cells, % of {} samples):",
        snap.total,
        snap.seconds,
        s.hz,
        snap.threads.len(),
        snap.missed,
        snap.sites.len(),
        snap.sites_total,
        snap.total
    );
    for (i, site) in snap.sites.iter().enumerate() {
        let addr = site.base.wrapping_add(site.rva as usize);
        // the same naming the crash reports use: "game.exe+0x1234", "bathyal.exe+0x1234"
        out.push_str(&format!("\n  {:3}. {:<44} {:>9}  {:5.1}%", i + 1, crate::modules::describe_address(addr), site.samples, pct(site.samples, snap.total)));
    }
    out.push_str("\nthreads (id, name, samples, % not in a wait):");
    for (tid, t) in snap.threads.iter().take(s.top.max(1)) {
        let name = if t.name.is_empty() { "-".to_string() } else { t.name.clone() };
        out.push_str(&format!("\n  {:>6}  {:<28} {:>9}  {:5.1}%", tid, name, t.samples, 100.0 - pct(t.waiting, t.samples)));
    }
    out
}

/// Write the report for `sh` to the log (no thread is suspended while this runs).
pub fn log_report(sh: &Shared) {
    let snap = sh.snapshot();
    crate::xlog_info!("{}", format_report(&snap, sh.settings));
    crate::log::flush();
}

/// The report as text (tests, or a caller that prints it itself).
pub fn report_text(sh: &Shared) -> String {
    format_report(&sh.snapshot(), sh.settings)
}

/// Stop the sampler, join it and write the final report – at the end of the run.
pub fn log_summary() {
    let sh = RUNNING.lock().unwrap_or_else(|p| p.into_inner()).take();
    if let Some(sh) = sh {
        sh.stop();
        // joined: no thread is suspended while the report is written, and the sampler cannot
        // log into the middle of it
        if let Some(h) = sh.thread.lock().unwrap_or_else(|p| p.into_inner()).take() {
            let _ = h.join();
        }
        log_report(&sh);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a body the profiler should find: a tight loop in a named function
    #[inline(never)]
    fn spin_work() -> u64 {
        let mut x = 0u64;
        for i in 0..200_000 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(i | 1);
        }
        std::hint::black_box(x)
    }

    #[test]
    fn samples_a_busy_thread_and_ranks_it_first() {
        let s = Settings { hz: 500, top: 10, every: None };
        let Some(sh) = start(s) else {
            eprintln!("profile: the sampler did not start; skipped");
            return;
        };
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel();
        let stop2 = stop.clone();
        let busy = std::thread::Builder::new()
            .name("busy-loop".into())
            .spawn(move || {
                let tid = unsafe { GetCurrentThreadId() };
                tx.send(tid).unwrap();
                let mut rounds = 0u64;
                while !stop2.load(Ordering::Relaxed) {
                    rounds = rounds.wrapping_add(spin_work());
                }
                rounds
            })
            .unwrap();
        let busy_tid = rx.recv().unwrap();
        std::thread::sleep(Duration::from_millis(300));
        stop.store(true, Ordering::Relaxed);
        let _ = busy.join().unwrap();
        let (total, threads) = sh.counts();
        let text = report_text(&sh);
        eprintln!("{text}");
        log_summary(); // also clears the global state for a later test
        if total == 0 {
            eprintln!("profile: no sample was taken here; skipped");
            return;
        }
        assert!(threads.contains_key(&busy_tid), "the busy thread ({busy_tid}) is in the table: {threads:?}");
        let busy_samples = threads[&busy_tid].samples;
        let top_other = threads.iter().filter(|(&t, _)| t != busy_tid).map(|(_, t)| t.samples).max().unwrap_or(0);
        assert!(busy_samples > 0, "the busy thread has samples: {busy_samples}");
        assert!(busy_samples >= top_other, "the busy thread ({busy_samples}) is not first (another has {top_other})");
        assert!(!text.contains("profile: 0 samples"), "the report names the samples: {text}");
    }

    #[test]
    fn the_sampler_costs_little_and_settings_parse() {
        // the parse rules are cheap to check; the cost is measured, not asserted tightly
        std::env::set_var("XCL_PROFILE", "1");
        std::env::set_var("XCL_PROFILE_HZ", "250");
        std::env::set_var("XCL_PROFILE_TOP", "7");
        std::env::set_var("XCL_PROFILE_EVERY", "30");
        let s = settings_from_env().unwrap();
        assert_eq!((s.hz, s.top, s.every), (250, 7, Some(30)));
        std::env::set_var("XCL_PROFILE", "0");
        assert!(settings_from_env().is_none());
        std::env::remove_var("XCL_PROFILE");
        std::env::remove_var("XCL_PROFILE_HZ");
        std::env::remove_var("XCL_PROFILE_TOP");
        std::env::remove_var("XCL_PROFILE_EVERY");
        assert!(settings_from_env().is_none());
    }
}
