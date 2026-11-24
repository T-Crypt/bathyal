//! Work on memory the GPU does not see, done by the CPU once the GPU has finished the work
//! before it: Write32/64BitValueBottomOfPipeX, and CopyMemoryToMemoryX with a side outside
//! unified memory.
//!
//! A console title marks its GPU progress with bottom-of-pipe writes into its own memory and
//! waits on them (a retail game: plain `VirtualAlloc` memory, outside XCL's unified memory).
//! The PC GPU cannot write there; dropped, the title waited for ever after its second frame.
//! It also copies through its memory the GPU has no address for (command list #60 with a side
//! outside the windows). Here each queue gets a fence and completion worker of XCL's: after a
//! submission whose lists carry such work the queue signals its fence, and its worker thread
//! does the work in submission order when the GPU is there – what bottom of pipe means, without
//! one stalled queue blocking another queue's work.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{mpsc, Mutex};

use windows::core::Interface;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;
use windows_sys::Win32::System::Threading as th;

/// A bottom-of-pipe write into memory outside unified memory, applied by the CPU once the
/// queue's fence reaches the submission that recorded it.
#[derive(Clone)]
pub(super) enum Write {
    /// Immediate dwords to store starting at `va`.
    Immediate(usize, Vec<u32>),
    /// A 64-bit value resolved by the GPU into a persistently mapped READBACK buffer slot
    /// (`src`), copied atomically to `va` once the queue's fence completes. Holding `_buf` keeps
    /// the mapping alive until the worker finishes.
    ReadbackU64 { va: usize, src: usize, _buf: ID3D12Resource },
}

/// A copy the CPU does after the queue's earlier work: (destination, source, bytes).
/// Its bytes are read from the source when it runs, so they are what the GPU has left there.
pub(super) type Copy = (usize, usize, usize);

/// What a worker does for one submission, in this order.
pub(super) enum Work {
    Words(Write),
    Copy(Copy),
}

struct Job {
    fence: ID3D12Fence,
    value: u64,
    work: Vec<Work>,
}

// SAFETY: the fence is a free-threaded COM object; the writes are plain data
unsafe impl Send for Job {}

/// queue → its fence, the last value signalled, and its dedicated completion worker
struct QueueFence {
    fence: ID3D12Fence,
    value: u64,
    tx: mpsc::Sender<Job>,
}

unsafe impl Send for QueueFence {}

static FENCES: Mutex<Option<HashMap<usize, QueueFence>>> = Mutex::new(None);
static NEXT_WORKER_ID: AtomicU32 = AtomicU32::new(0);

/// how long a queue's worker waits for one submission before writing anyway (a removed device,
/// a queue that never gets there)
const WAIT_MS: u32 = 10_000;

fn spawn_queue_worker() -> mpsc::Sender<Job> {
    let (tx, rx) = mpsc::channel::<Job>();
    let id = NEXT_WORKER_ID.fetch_add(1, Ordering::Relaxed);
    let spawned = std::thread::Builder::new().name(format!("xcl-cpu-writes-{id}")).spawn(move || unsafe {
        let event = th::CreateEventW(std::ptr::null(), 0, 0, std::ptr::null());
        for job in rx {
            if job.fence.GetCompletedValue() < job.value && job.fence.SetEventOnCompletion(job.value, HANDLE(event as *mut c_void)).is_ok() {
                th::WaitForSingleObject(event, WAIT_MS);
            }
            for w in &job.work {
                match w {
                    Work::Words(Write::Immediate(va, words)) => write_now(*va, words),
                    Work::Words(Write::ReadbackU64 { va, src, .. }) => {
                        let value = (*(*src as *const AtomicU64)).load(Ordering::Acquire);
                        write_u64_now(*va, value);
                    }
                    Work::Copy((dst, src, bytes)) => copy_now(*dst, *src, *bytes),
                }
            }
        }
        if !event.is_null() {
            windows_sys::Win32::Foundation::CloseHandle(event);
        }
    });
    if let Err(e) = spawned {
        crate::xlog_error!("D3D12.X: no thread for the writes after the GPU: {e}");
    }
    tx
}

/// Drop the fence and worker channel for a released command queue so its worker exits once
/// any queued work drains, and a reused pointer starts fresh.
pub(super) fn forget_queue(queue: usize) {
    if let Some(m) = FENCES.lock().unwrap().as_mut() {
        m.remove(&queue);
    }
}

/// The dwords at `va`, if the memory is still there and writable.
unsafe fn write_now(va: usize, words: &[u32]) {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_READWRITE};
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    if VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.State != MEM_COMMIT {
        return;
    }
    if m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) == 0 || (m.BaseAddress as usize + m.RegionSize) < va + 4 * words.len() {
        return;
    }
    for (i, &w) in words.iter().enumerate() {
        (*((va + 4 * i) as *const AtomicU32)).store(w, Ordering::Release);
    }
}

/// An 8-byte aligned 64-bit value at `va`, if the memory is still there and writable.
unsafe fn write_u64_now(va: usize, value: u64) {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_GUARD, PAGE_READWRITE};
    let Some(end) = va.checked_add(8) else { return };
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    if VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 || m.State != MEM_COMMIT {
        return;
    }
    if m.Protect & PAGE_GUARD != 0 || m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) == 0 || (m.BaseAddress as usize).saturating_add(m.RegionSize) < end {
        return;
    }
    (*(va as *const AtomicU64)).store(value, Ordering::Release);
}

/// `[va, va + bytes)` as the CPU sees it: committed, and writable when `write`.
unsafe fn commit_check(va: usize, bytes: usize, write: bool) -> bool {
    use windows_sys::Win32::System::Memory::{
        VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, PAGE_EXECUTE_READWRITE, PAGE_READONLY, PAGE_READWRITE,
    };
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(va, bytes)) {
        // the coherent windows are committed read-write as a whole (no VirtualQuery walk)
        return true;
    }
    let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
    let found = VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) != 0;
    let ok = if write {
        m.Protect & (PAGE_READWRITE | PAGE_EXECUTE_READWRITE) != 0
    } else {
        m.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READWRITE) != 0
    };
    found && m.State == MEM_COMMIT && ok && m.BaseAddress as usize + m.RegionSize >= va + bytes
}

/// Whether `[va, va + bytes)` is memory the CPU can write for the GPU later.
pub(super) fn writable(va: usize, bytes: usize) -> bool {
    crate::hle::kernelx::lazy::ensure(va, bytes);
    unsafe { commit_check(va, bytes, true) }
}

/// Whether `[va, va + bytes)` is memory the CPU can read for a copy.
pub(super) fn readable(va: usize, bytes: usize) -> bool {
    crate::hle::kernelx::lazy::ensure(va, bytes);
    unsafe { commit_check(va, bytes, false) }
}

/// `bytes` from `src` to `dst` on the CPU, when both are memory it can reach; a destination in
/// a sparse window then goes to that window's video memory for the GPU's later work.
pub(super) unsafe fn copy_now(dst: usize, src: usize, bytes: usize) {
    if bytes == 0 || !writable(dst, bytes) || !readable(src, bytes) {
        return;
    }
    std::ptr::copy(src as *const u8, dst as *mut u8, bytes);
    super::sparse::flush_written(dst, bytes);
}

/// `work` of lists just submitted to `queue`: done when the GPU has finished them.
pub(super) unsafe fn after(queue: *mut c_void, work: Vec<Work>) {
    if work.is_empty() || queue.is_null() {
        return;
    }
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return };
    let (tx, job) = {
        let mut g = FENCES.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new);
        let entry = match m.entry(queue as usize) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let mut dev: Option<ID3D12Device> = None;
                if q.GetDevice(&mut dev).is_err() {
                    return;
                }
                let Some(dev) = dev else { return };
                let Ok(fence) = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) else { return };
                v.insert(QueueFence { fence, value: 0, tx: spawn_queue_worker() })
            }
        };
        entry.value += 1;
        if q.Signal(&entry.fence, entry.value).is_err() {
            return;
        }
        (entry.tx.clone(), Job { fence: entry.fence.clone(), value: entry.value, work })
    };
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| crate::xlog_info!("D3D12.X: bottom-of-pipe writes outside unified memory are done by the CPU after the GPU ({} in the first submission)", job.work.len()));
    let _ = tx.send(job);
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Dxgi::*;

    /// The value lands after the queue has done what came before it.
    #[test]
    fn writes_land_after_the_queue() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let cell = Box::new([0u32; 2]);
            let va = cell.as_ptr() as usize;
            assert!(writable(va, 8));
            after(q.as_raw(), vec![Work::Words(Write::Immediate(va, vec![7, 9]))]);
            let t = std::time::Instant::now();
            while (*(va as *const AtomicU32)).load(Ordering::Acquire) != 7 {
                assert!(t.elapsed().as_secs() < 5, "never written");
                std::thread::yield_now();
            }
            assert_eq!(*cell, [7, 9]);
        }
    }

    /// A copy of the title's memory outside unified memory lands too, after the queue has done
    /// what came before it.
    #[test]
    fn copies_land_after_the_queue() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let src = Box::new([0xA5u8; 0x400]);
            let dst = Box::new([0u8; 0x400]);
            let (s, d) = (src.as_ptr() as usize, dst.as_ptr() as usize);
            assert!(writable(d, 0x400) && readable(s, 0x400));
            after(q.as_raw(), vec![Work::Copy((d, s, 0x400))]);
            let t = std::time::Instant::now();
            while *dst.first().unwrap() != 0xA5 {
                assert!(t.elapsed().as_secs() < 5, "never copied");
                std::thread::yield_now();
            }
            assert_eq!(*dst, *src, "the whole range");
        }
    }

    /// A copy the CPU cannot make (a source that is no memory of the title's) changes nothing.
    #[test]
    fn a_copy_the_cpu_cannot_make_changes_nothing() {
        unsafe {
            let dst = Box::new([0u8; 0x100]);
            let d = dst.as_ptr() as usize;
            copy_now(d, 0x10, 0x100);
            assert!(!readable(0x10, 0x100));
            assert_eq!(*dst, [0u8; 0x100]);
            copy_now(d, d, 0);
            assert_eq!(*dst, [0u8; 0x100]);
        }
    }

    /// A `ReadbackU64` write copies the 64-bit value from the mapped readback buffer after the
    /// queue finishes.
    #[test]
    fn readback_u64_lands_after_the_queue() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                };
                if D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                    crate::gpu::recomp::runner::skip_test("no GPU");
                    return;
                }
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let hp = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 8,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut rb: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).unwrap();
            let rb = rb.unwrap();
            let mut ptr: *mut c_void = std::ptr::null_mut();
            rb.Map(0, None, Some(&mut ptr)).unwrap();
            let expected = 0x1234_5678_9ABC_DEF0u64;
            (*(ptr as *const AtomicU64)).store(expected, Ordering::Release);

            let dst = Box::new(0u64);
            let va = dst.as_ref() as *const u64 as usize;
            assert!(writable(va, 8));
            after(q.as_raw(), vec![Work::Words(Write::ReadbackU64 { va, src: ptr as usize, _buf: rb })]);
            let t = std::time::Instant::now();
            while (*(va as *const AtomicU64)).load(Ordering::Acquire) != expected {
                assert!(t.elapsed().as_secs() < 5, "readback value never written");
                std::thread::yield_now();
            }
            assert_eq!(*dst, expected);
        }
    }

    /// A copy of the title's memory outside unified memory lands too, after the queue has done
    /// what came before it.
    #[test]
    fn copies_land_after_the_queue() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.unwrap();
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let src = Box::new([0xA5u8; 0x400]);
            let dst = Box::new([0u8; 0x400]);
            let (s, d) = (src.as_ptr() as usize, dst.as_ptr() as usize);
            assert!(writable(d, 0x400) && readable(s, 0x400));
            after(q.as_raw(), vec![Work::Copy((d, s, 0x400))]);
            let t = std::time::Instant::now();
            while *dst.first().unwrap() != 0xA5 {
                assert!(t.elapsed().as_secs() < 5, "never copied");
                std::thread::yield_now();
            }
            assert_eq!(*dst, *src, "the whole range");
        }
    }

    /// A copy the CPU cannot make (a source that is no memory of the title's) changes nothing.
    #[test]
    fn a_copy_the_cpu_cannot_make_changes_nothing() {
        unsafe {
            let dst = Box::new([0u8; 0x100]);
            let d = dst.as_ptr() as usize;
            copy_now(d, 0x10, 0x100);
            assert!(!readable(0x10, 0x100));
            assert_eq!(*dst, [0u8; 0x100]);
            copy_now(d, d, 0);
            assert_eq!(*dst, [0u8; 0x100]);
        }
    }
}
