//! `report.json`: what the title imported, which stubs it actually called, which console
//! paths were not mapped, which game files were redirected – the to-do list for the
//! next iteration.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::sync::{Mutex, MutexGuard, TryLockError};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::resolver::{self, Source};

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct StubCall {
    pub dll: String,
    pub function: String,
    pub calls: u64,
    pub first_caller: String,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct MissingImport {
    pub importer: String,
    pub dll: String,
    pub function: String,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct ModuleInfo {
    pub name: String,
    pub base: String,
    pub size: String,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct VfsInfo {
    pub unmapped_paths: Vec<String>,
    pub redirected_writes: Vec<String>,
    pub whiteouts: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Report {
    pub xcl_version: String,
    pub title: String,
    pub exe: String,
    pub crash: Option<String>,
    pub exit_code: Option<i64>,
    /// what the PC has of what a run needs (DXC, GPU features), checked by `xcl run` first
    #[serde(default)]
    pub environment: Vec<crate::preflight::Check>,
    /// counts of static imports by how they were resolved
    pub imports_by_source: BTreeMap<String, usize>,
    /// imports that ended up as auto-stubs (static analysis)
    pub stubbed_imports: Vec<MissingImport>,
    /// stubs the title actually called at run time (sorted by call count)
    pub called_stubs: Vec<StubCall>,
    pub modules: Vec<ModuleInfo>,
    pub vfs: VfsInfo,
    /// shader formats seen at pipeline creation (dxil / dxbc / container-unknown / rdna2-raw …)
    #[serde(default)]
    pub shaders: crate::gpu::shader::ShaderStats,
    /// the DXC that compiled them: folder and versions of dxcompiler.dll and dxil.dll (empty:
    /// none was loaded)
    #[serde(default)]
    pub dxc: crate::gpu::recomp::dxc::Toolchain,
    /// D3D12.X backend: adapter, interface layouts, unknown IIDs, fallbacks, presented frames
    #[serde(default)]
    pub d3d12: crate::hle::d3d12x::real::D3d12Stats,
    /// XAudio2: engines, mastering/source voices, XMA voices played as silence, dropped console flags
    #[serde(default)]
    pub audio: crate::hle::xaudio2::AudioStats,
    /// XGameRuntime through the GDK thunks: which classes the title asked for and who answered,
    /// XCL slots it called that XCL does not implement
    #[serde(default)]
    pub xgameruntime: crate::hle::xgameruntime::api::ApiStats,
    /// DirectStorage: BCPack requests unpacked and refused (no decoder) by mode, and bytes filled under
    /// `XCL_DSTORAGE_BCPACK_FILL`
    #[serde(default)]
    pub dstorage: crate::hle::dstorage::DStorageStats,
    pub notes: Vec<String>,
}

static CRASH: Mutex<Option<String>> = Mutex::new(None);
static WRITE_LOCK: Mutex<()> = Mutex::new(());

fn source_label(s: &Source) -> String {
    match s {
        Source::Hle(m) => format!("hle:{m}"),
        Source::HleOther(m) => format!("hle-cross:{m}"),
        Source::System(d) => format!("system:{d}"),
        Source::GameDll(d) => format!("game-dll:{d}"),
        Source::Stub => "stub".into(),
    }
}

pub fn build() -> Report {
    let boot = crate::state::boot();
    let records = resolver::records();
    let mut by_source: BTreeMap<String, usize> = BTreeMap::new();
    let mut stubbed = Vec::new();
    for r in &records {
        let label = match &r.source {
            Source::System(_) => "system".to_string(),
            other => source_label(other),
        };
        *by_source.entry(label).or_default() += 1;
        if r.source == Source::Stub {
            stubbed.push(MissingImport { importer: r.importer.clone(), dll: r.dll.clone(), function: r.function.clone() });
        }
    }
    let mut called: Vec<StubCall> = crate::stubs::all_stubs()
        .into_iter()
        .filter(|s| s.calls.load(Ordering::Relaxed) > 0)
        .map(|s| StubCall {
            dll: s.dll.clone(),
            function: s.name.clone(),
            calls: s.calls.load(Ordering::Relaxed),
            first_caller: crate::modules::describe_address(s.first_caller.load(Ordering::Relaxed)),
        })
        .collect();
    called.sort_by(|a, b| b.calls.cmp(&a.calls));
    let modules = crate::modules::mapped_modules()
        .iter()
        .map(|m| ModuleInfo { name: m.name.clone(), base: format!("{:#x}", m.base), size: format!("{:#x}", m.size) })
        .collect();
    let vfs = crate::vfs::get()
        .map(|v| VfsInfo { unmapped_paths: v.unmapped_paths(), redirected_writes: v.redirected_writes(), whiteouts: v.whiteouts() })
        .unwrap_or_default();
    Report {
        xcl_version: env!("CARGO_PKG_VERSION").to_string(),
        title: boot.map(|b| b.title.clone()).unwrap_or_default(),
        exe: boot.map(|b| b.exe_path.to_string_lossy().into_owned()).unwrap_or_default(),
        crash: CRASH.lock().unwrap_or_else(|p| p.into_inner()).clone(),
        exit_code: None,
        environment: Vec::new(),
        imports_by_source: by_source,
        stubbed_imports: stubbed,
        called_stubs: called,
        modules,
        vfs,
        shaders: crate::gpu::shader::stats(),
        dxc: crate::gpu::recomp::dxc::toolchain(),
        d3d12: crate::hle::d3d12x::real::stats(),
        audio: crate::hle::xaudio2::stats(),
        xgameruntime: crate::hle::xgameruntime::api::stats(),
        dstorage: crate::hle::dstorage::stats(),
        notes: vec![
            "D3D12.X methods missing on PC show up in called_stubs as '<Interface>::#slot (not on PC)'.".into(),
            "Console-format shaders are replaced (fallback_pipelines) until the RDNA2 recompiler is finished.".into(),
            "Direct Nt* calls on files do not go through the VFS.".into(),
        ],
    }
}

thread_local! {
    static IN_WRITE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct InWriteGuard;
impl InWriteGuard {
    fn enter() -> Option<Self> {
        IN_WRITE.with(|c| {
            if c.get() {
                None
            } else {
                c.set(true);
                Some(InWriteGuard)
            }
        })
    }
}
impl Drop for InWriteGuard {
    fn drop(&mut self) {
        IN_WRITE.with(|c| c.set(false));
    }
}

/// Fallback report when a full `build()` cannot be completed (e.g. re-entrant crash or lock contention).
fn build_fallback(crash: Option<String>) -> Report {
    let boot = crate::state::boot();
    Report {
        xcl_version: env!("CARGO_PKG_VERSION").to_string(),
        title: boot.map(|b| b.title.clone()).unwrap_or_default(),
        exe: boot.map(|b| b.exe_path.to_string_lossy().into_owned()).unwrap_or_default(),
        crash,
        notes: vec![
            "Fallback crash report written due to lock contention or re-entrant crash during exit.".into(),
        ],
        ..Default::default()
    }
}

fn write_report_file(path: &std::path::Path, r: &Report) -> bool {
    if let Ok(s) = serde_json::to_string_pretty(r) {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(path, s).is_ok()
    } else {
        false
    }
}

/// `m.try_lock()`, retried for `wait`: never an endless wait – the holder may be this very
/// thread, crashed inside `build()`. Recovers from mutex poisoning.
fn lock_within<T>(m: &Mutex<T>, wait: Duration) -> Option<MutexGuard<'_, T>> {
    let end = Instant::now() + wait;
    loop {
        match m.try_lock() {
            Ok(g) => return Some(g),
            Err(TryLockError::Poisoned(p)) => return Some(p.into_inner()),
            Err(TryLockError::WouldBlock) if Instant::now() < end => {
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(_) => return None,
        }
    }
}

/// Write `report.json` (host side). `crash` is remembered for later writes.
pub fn write_final(crash: Option<String>) {
    // the process ends: a cursor held in the XCL window is let go
    crate::hle::kbm::release();

    let wait = if crash.is_some() { Duration::from_secs(2) } else { Duration::ZERO };

    // Record crash in CRASH static first so it is never lost regardless of lock state
    if let Some(c) = crash.as_ref() {
        if let Some(mut g) = lock_within(&CRASH, wait) {
            g.get_or_insert_with(|| c.clone());
        }
    }

    // Check re-entrancy on the same thread: if this thread already holds WRITE_LOCK and crashed
    // inside build() or write_final(), avoid deadlocking on WRITE_LOCK or re-entering build()
    let Some(_in_write_guard) = InWriteGuard::enter() else {
        let recorded = CRASH.lock().unwrap_or_else(|p| p.into_inner()).clone();
        if let Some(boot) = crate::state::boot() {
            let path = boot.report_path();
            let mut r = load(&path).unwrap_or_else(|| build_fallback(None));
            if recorded.is_some() {
                r.crash = recorded;
            }
            write_report_file(&path, &r);
        }
        return;
    };

    // A crash is kept even while another thread writes the report, and is written once that
    // thread is done – the process ends right after (#106); other calls give way at once
    let lock_guard = lock_within(&WRITE_LOCK, wait);
    let Some(_g) = lock_guard else {
        // WRITE_LOCK had contention or timed out. If there is a crash recorded,
        // ensure it is persisted even without the lock so crash info is never dropped.
        let recorded = CRASH.lock().unwrap_or_else(|p| p.into_inner()).clone();
        if recorded.is_some() {
            if let Some(boot) = crate::state::boot() {
                let path = boot.report_path();
                let mut r = load(&path).unwrap_or_else(|| build_fallback(None));
                r.crash = recorded;
                write_report_file(&path, &r);
            }
        }
        return;
    };

    let Some(boot) = crate::state::boot() else { return };
    let mut r = build();

    // Re-check CRASH in case another thread recorded a crash while build() was running
    if r.crash.is_none() {
        let current = CRASH.lock().unwrap_or_else(|p| p.into_inner()).clone();
        if current.is_some() {
            r.crash = current;
        }
    }

    let path = boot.report_path();
    write_report_file(&path, &r);

    // If a crash arrived while writing to disk, update the file before releasing WRITE_LOCK
    if r.crash.is_none() {
        let current = CRASH.lock().unwrap_or_else(|p| p.into_inner()).clone();
        if let Some(c) = current {
            r.crash = Some(c);
            write_report_file(&path, &r);
        }
    }
}

pub fn load(path: &std::path::Path) -> Option<Report> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_SERIAL: Mutex<()> = Mutex::new(());

    #[test]
    fn a_crash_is_kept_while_another_thread_writes_the_report() {
        let _lock = TEST_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        *CRASH.lock().unwrap_or_else(|p| p.into_inner()) = None;
        let (held, wait_held) = std::sync::mpsc::channel();
        let writer = std::thread::spawn(move || {
            let _g = WRITE_LOCK.lock().unwrap();
            held.send(()).unwrap();
            std::thread::sleep(Duration::from_millis(100));
        });
        wait_held.recv().unwrap();
        // no boot record in unit tests: nothing is written to disk
        write_final(Some("boom".into()));
        writer.join().unwrap();
        assert_eq!(CRASH.lock().unwrap_or_else(|p| p.into_inner()).as_deref(), Some("boom"));
        *CRASH.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }

    #[test]
    fn lock_within_recovers_from_poisoned_mutex() {
        let m = Mutex::new(42);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = m.lock().unwrap();
            panic!("poisoning mutex");
        }));
        assert!(m.is_poisoned());
        let guard = lock_within(&m, Duration::from_millis(50));
        assert!(guard.is_some());
        assert_eq!(*guard.unwrap(), 42);
    }

    #[test]
    fn write_final_recovers_when_crash_mutex_is_poisoned() {
        let _lock = TEST_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        *CRASH.lock().unwrap_or_else(|p| p.into_inner()) = None;
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = CRASH.lock().unwrap_or_else(|p| p.into_inner());
            panic!("poisoning CRASH");
        }));
        write_final(Some("recovered".into()));
        assert_eq!(CRASH.lock().unwrap_or_else(|p| p.into_inner()).as_deref(), Some("recovered"));
        *CRASH.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }

    #[test]
    fn first_crash_payload_is_preserved_under_contention() {
        let _lock = TEST_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        *CRASH.lock().unwrap_or_else(|p| p.into_inner()) = None;
        write_final(Some("first-crash".into()));
        write_final(Some("second-crash".into()));
        assert_eq!(CRASH.lock().unwrap_or_else(|p| p.into_inner()).as_deref(), Some("first-crash"));
        *CRASH.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }

    #[test]
    fn reentrant_write_final_does_not_deadlock() {
        let _lock = TEST_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        *CRASH.lock().unwrap_or_else(|p| p.into_inner()) = None;
        let _guard = InWriteGuard::enter().unwrap();
        // Inside write_final simulated: a crash occurs and write_final is called re-entrantly
        write_final(Some("reentrant-crash".into()));
        assert_eq!(CRASH.lock().unwrap_or_else(|p| p.into_inner()).as_deref(), Some("reentrant-crash"));
        drop(_guard);
        *CRASH.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }

    #[test]
    fn concurrent_exit_and_crash_persists_without_deadlock() {
        let _lock = TEST_SERIAL.lock().unwrap_or_else(|p| p.into_inner());
        *CRASH.lock().unwrap_or_else(|p| p.into_inner()) = None;
        let mut handles = Vec::new();
        for i in 0..10 {
            handles.push(std::thread::spawn(move || {
                if i % 2 == 0 {
                    write_final(Some(format!("concurrent-crash-{i}")));
                } else {
                    write_final(None);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let recorded = CRASH.lock().unwrap_or_else(|p| p.into_inner()).clone();
        assert!(recorded.is_some());
        assert!(recorded.unwrap().starts_with("concurrent-crash-"));
        *CRASH.lock().unwrap_or_else(|p| p.into_inner()) = None;
    }

    #[test]
    fn fallback_report_structure_is_valid() {
        let r = build_fallback(Some("test-fallback-crash".into()));
        assert_eq!(r.crash.as_deref(), Some("test-fallback-crash"));
        let json = serde_json::to_string_pretty(&r).expect("fallback report must serialize to valid JSON");
        let parsed: Report = serde_json::from_str(&json).expect("fallback report must deserialize");
        assert_eq!(parsed.crash.as_deref(), Some("test-fallback-crash"));
    }
}
