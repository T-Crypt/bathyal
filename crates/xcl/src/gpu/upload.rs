//! `D3D12_HEAP_TYPE_GPU_UPLOAD`: video memory the CPU maps write-combined over the BAR.
//!
//! The console's memory is unified: title code writes a buffer and the GPU reads it where it
//! wrote it. A PC keeps title memory in system RAM and sends what the CPU wrote to video
//! memory before each submit (`d3d12x::sparse`: GetWriteWatch, then copies). A GPU_UPLOAD heap
//! (Agility SDK 1.613+, `D3D12_FEATURE_D3D12_OPTIONS16::GPUUploadHeapSupported`, ReBAR) is
//! VRAM that the CPU writes directly, so a block there needs no write-watch page, no upload
//! copy and no GPU wait.
//!
//! It cannot back the title's *own* addresses yet. Title memory is addressed in four 4 GiB
//! console windows and the recompiler turns a console address into `window buffer VA + low 32
//! bits`; a GPU_UPLOAD heap has a driver-chosen GPU VA and a driver-chosen CPU mapping, and
//! neither is a window address. A title allocation whose address the CPU dereferences would
//! then have two different addresses (its pointer and its GPU VA), which the window model
//! cannot express. So this first step backs what XCL itself writes and binds by GPU VA: the
//! per-draw user-SGPR block (`io`) and the 256-byte `res` block that translated shaders read
//! through root UAVs, with the argument table left in console memory (shaders dereference it
//! by address). See HANDOFF "GPU_UPLOAD".
//!
//! Off unless `XCL_GPU_UPLOAD=1`. `XCL_GPU_UPLOAD_MB` sets the heap size (default 64).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};

/// `D3D12_HEAP_TYPE_GPU_UPLOAD` (Agility SDK 1.613; the crate's bindings may predate it)
const HEAP_TYPE_GPU_UPLOAD: i32 = 5;
/// granularity of an allocation in the heap (a resource placement alignment)
const GRAIN: usize = 0x1_0000;
/// small enough for `io` / `res` blocks and far from the card's 24 GB
const DEFAULT_MB: usize = 64;
const MIN_MB: usize = 1;
const MAX_MB: usize = 2048;

/// Whether `XCL_GPU_UPLOAD` asks for this path (default off).
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_GPU_UPLOAD").is_ok_and(|v| v != "0" && !v.is_empty()))
}

fn heap_bytes() -> usize {
    let mb = std::env::var("XCL_GPU_UPLOAD_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(DEFAULT_MB);
    mb.clamp(MIN_MB, MAX_MB) << 20
}

/// Bookkeeping of one heap's byte ranges, apart from D3D12 so it is testable without a GPU.
#[derive(Default)]
pub struct Ranges {
    size: usize,
    free: Vec<(usize, usize)>,
    used: BTreeMap<usize, usize>,
}

impl Ranges {
    pub fn new(size: usize) -> Ranges {
        let size = size / GRAIN * GRAIN;
        Ranges { size, free: if size == 0 { Vec::new() } else { vec![(0, size)] }, used: BTreeMap::new() }
    }

    /// A `GRAIN`-aligned offset of `n` bytes, first fit.
    pub fn take(&mut self, n: usize, align: usize) -> Option<usize> {
        let n = n.max(1).checked_next_multiple_of(GRAIN)?;
        let align = align.max(GRAIN).next_power_of_two();
        let found = self.free.iter().position(|&(o, l)| {
            let at = (o + align - 1) / align * align;
            at + n <= o + l
        })?;
        let (o, l) = self.free.remove(found);
        let at = (o + align - 1) / align * align;
        if at > o {
            self.free.push((o, at - o));
        }
        if at + n < o + l {
            self.free.push((at + n, o + l - at - n));
        }
        self.used.insert(at, n);
        Some(at)
    }

    /// Give an offset from `take` back; coalesces with its neighbours.
    pub fn give(&mut self, at: usize) -> bool {
        let Some(n) = self.used.remove(&at) else { return false };
        let (mut start, mut len) = (at, n);
        if let Some(i) = self.free.iter().position(|&(o, l)| o + l == at) {
            let (o, l) = self.free.swap_remove(i);
            start = o;
            len += l;
        }
        if let Some(i) = self.free.iter().position(|&(o, _)| o == start + len) {
            let (_, l) = self.free.swap_remove(i);
            len += l;
        }
        self.free.push((start, len));
        true
    }

    pub fn used_bytes(&self) -> usize {
        self.used.values().sum()
    }
}

struct Heap {
    /// the placed buffer covering the whole heap (one resource: allocations are subranges)
    _buffer: ID3D12Resource,
    _heap: ID3D12Heap,
    /// CPU (write-combined) address of offset 0 and the resource's GPU VA
    cpu: usize,
    gpu: u64,
    ranges: Ranges,
}

// D3D12 resources and heaps are free-threaded
unsafe impl Send for Heap {}

static ARENA: Mutex<Option<Heap>> = Mutex::new(None);
/// whether the heap exists, readable without the lock (the address translation asks often)
static CREATED: AtomicBool = AtomicBool::new(false);
/// the start-up probe's verdict, once
static SUPPORT: Mutex<Option<String>> = Mutex::new(None);

/// `CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16)` as a raw two-word struct: the crate's
/// binding of the console's field names may not include `GPUUploadHeapSupported` yet.
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Options16 {
    dynamic_depth_bias: i32,
    gpu_upload_heap_supported: i32,
}

fn features(dev: &ID3D12Device) -> bool {
    let mut o = Options16::default();
    let ok = unsafe {
        dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &mut o as *mut _ as *mut std::ffi::c_void, std::mem::size_of::<Options16>() as u32).is_ok()
    };
    let supported = ok && o.gpu_upload_heap_supported != 0;
    crate::xlog_info!(
        "GPU_UPLOAD: OPTIONS16 {} (GPUUploadHeapSupported {})",
        if ok { "queried" } else { "not answered by this runtime" },
        if supported { "yes" } else { "no" }
    );
    supported
}

/// Make the heap at device creation, when `XCL_GPU_UPLOAD=1`. Returns whether allocations are
/// served from it. The heap creation is the real test of support (the feature query is only
/// logged): an inbox runtime that does not know OPTIONS16 refuses the heap.
pub fn init(dev: &ID3D12Device) -> bool {
    if !enabled() {
        return false;
    }
    let queried = features(dev);
    let cap = heap_bytes();
    let heap = create(dev, cap);
    let verdict = match &heap {
        Some(h) => format!("{} MB at GPU VA {:#x} (CPU {:#x}), OPTIONS16 {}", cap >> 20, h.gpu, h.cpu, if queried { "yes" } else { "no" }),
        None if queried => format!("{} MB: could not be created", cap >> 20),
        None => format!("{} MB: not supported (OPTIONS16 says no and the heap was refused)", cap >> 20),
    };
    if let Some(h) = heap {
        crate::xlog_info!("GPU_UPLOAD: heap ready, {verdict}");
        *ARENA.lock().unwrap() = Some(h);
        CREATED.store(true, Ordering::Release);
        *SUPPORT.lock().unwrap() = Some(verdict);
        true
    } else {
        crate::xlog_warn!("GPU_UPLOAD: off, {verdict}; CPU-written buffers stay in unified memory");
        *SUPPORT.lock().unwrap() = Some(format!("off: {verdict}"));
        false
    }
}

fn create(dev: &ID3D12Device, cap: usize) -> Option<Heap> {
    let mut heap: Option<ID3D12Heap> = None;
    let made: windows::core::Result<()> = unsafe {
        dev.CreateHeap(&D3D12_HEAP_DESC {
            SizeInBytes: cap as u64,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE(HEAP_TYPE_GPU_UPLOAD),
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
            },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
        }, &mut heap)
    };
    made.map_err(|e| {
        crate::xlog_warn!("GPU_UPLOAD: CreateHeap({} MB) failed: {e}", cap >> 20);
    })
    .ok()?;
    let heap = heap?;
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: cap as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    };
    let mut buffer: Option<ID3D12Resource> = None;
    unsafe { dev.CreatePlacedResource(&heap, 0, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer) }
        .map_err(|e| {
            crate::xlog_warn!("GPU_UPLOAD: CreatePlacedResource over the heap failed: {e}");
        })
        .ok()?;
    let buffer = buffer?;
    let mut cpu: *mut std::ffi::c_void = std::ptr::null_mut();
    unsafe { buffer.Map(0, None, Some(&mut cpu)) }
        .map_err(|e| {
            crate::xlog_warn!("GPU_UPLOAD: Map of the heap buffer failed: {e}");
        })
        .ok()?;
    if cpu.is_null() {
        crate::xlog_warn!("GPU_UPLOAD: Map of the heap buffer gave no pointer");
        return None;
    }
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    if gpu == 0 {
        crate::xlog_warn!("GPU_UPLOAD: the heap buffer has no GPU address");
        return None;
    }
    Some(Heap { _buffer: buffer, _heap: heap, cpu: cpu as usize, gpu, ranges: Ranges::new(cap) })
}

/// Whether allocations are served from the GPU_UPLOAD heap now.
pub fn active() -> bool {
    CREATED.load(Ordering::Acquire)
}

/// A zero-copy block of `n` bytes (`GRAIN`-aligned): its CPU address (write-combined). None
/// when the path is off, full, or could not be made.
pub fn alloc(n: usize, align: usize) -> Option<usize> {
    let mut g = ARENA.lock().unwrap();
    let h = g.as_mut()?;
    let at = h.ranges.take(n, align)?;
    Some(h.cpu + at)
}

/// Give a block from `alloc` back (false: not one of ours).
pub fn free(p: usize) -> bool {
    if !active() {
        return false;
    }
    let mut g = ARENA.lock().unwrap();
    let Some(h) = g.as_mut() else { return false };
    if p < h.cpu || p >= h.cpu + h.ranges.size {
        return false;
    }
    h.ranges.give(p - h.cpu)
}

/// The GPU address of a CPU address in the heap (None outside it).
pub fn va_of(p: usize) -> Option<u64> {
    if !active() {
        return None;
    }
    let g = ARENA.lock().unwrap();
    let h = g.as_ref()?;
    (p >= h.cpu && p < h.cpu + h.ranges.size).then(|| h.gpu + (p - h.cpu) as u64)
}

/// Whether `va` is a GPU address of the heap (a PC resource address, never a console one).
pub fn owns_gpu(va: u64) -> bool {
    if !active() {
        return false;
    }
    let g = ARENA.lock().unwrap();
    g.as_ref().is_some_and(|h| va >= h.gpu && va < h.gpu + h.ranges.size as u64)
}

/// A line about the heap for the statistics log, or None when the path is off.
pub fn stats() -> Option<String> {
    let g = ARENA.lock().unwrap();
    let h = g.as_ref()?;
    Some(format!("GPU_UPLOAD {} KB of {} MB", h.ranges.used_bytes() >> 10, h.ranges.size >> 20))
}

/// What the start-up probe found (for the report / tests).
pub fn support() -> Option<String> {
    SUPPORT.lock().unwrap().clone()
}

#[cfg(test)]
mod tests {
    use super::Ranges;

    #[test]
    fn ranges_hand_out_aligned_blocks_and_never_overlap() {
        let mut r = Ranges::new(4 << 20);
        let a = r.take(1000, 256).unwrap();
        let b = r.take(0x2_0000, 0x1_0000).unwrap();
        let c = r.take(64, 64).unwrap();
        assert_eq!(a % 0x1_0000, 0);
        assert_eq!(b, 0x1_0000);
        assert_eq!(c, 0x3_0000);
        assert_eq!(r.used_bytes(), 0x1_0000 + 0x2_0000 + 0x1_0000, "each block is a whole grain");
        assert!(a < b || b < a);
    }

    #[test]
    fn a_given_block_is_reused_and_neighbours_coalesce() {
        let mut r = Ranges::new(8 << 20);
        let a = r.take(0x1_0000, 0x1_0000).unwrap();
        let b = r.take(0x1_0000, 0x1_0000).unwrap();
        assert_eq!((a, b), (0, 0x1_0000));
        assert!(r.give(b));
        assert!(r.give(a));
        assert_eq!(r.used_bytes(), 0);
        let big = r.take(0x2_0000, 0x1_0000).unwrap();
        assert_eq!(big, 0, "the two grains came back and coalesced");
        assert!(!r.give(big + 1), "an offset that was not handed out is refused");
    }

    #[test]
    fn a_full_heap_refuses_and_zero_size_is_not_a_heap() {
        let mut r = Ranges::new(0x1_0000);
        assert!(r.take(0x1_0000, 0x1_0000).is_some());
        assert!(r.take(0x1_0000, 0x1_0000).is_none());
        assert!(Ranges::new(0).take(0x1_0000, 0x1_0000).is_none());
    }

    #[test]
    fn addresses_outside_the_heap_are_not_ours() {
        // a test process has no heap: nothing is served, and nothing looks like a console
        // address (so `uma_va_of` / `is_title_address` keep their old answers)
        assert!(!super::free(0x10_0000));
        assert_eq!(super::va_of(0x10_0000), None);
        assert!(!super::owns_gpu(1 << 40));
    }
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;

    /// The heap D3D12 offers here: OPTIONS16's word, then a real heap, a CPU write, and a
    /// compute shader reading that write through a root UAV. Skips without a device or DXC.
    #[test]
    fn gpu_upload_heap_cpu_writes_reach_a_compute_shader() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev).is_err() {
                println!("no D3D12 device – skipping");
                return;
            }
            let dev = dev.expect("device");
            let mut o = Options16::default();
            let queried = dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &mut o as *mut _ as *mut std::ffi::c_void, std::mem::size_of::<Options16>() as u32).is_ok();
            println!("OPTIONS16: queried {queried}, GPUUploadHeapSupported {}", o.gpu_upload_heap_supported != 0);
            if std::env::var_os("XCL_GPU_UPLOAD_PROBE").is_some() {
                for cap in [1usize << 20, 64 << 20, 1024 << 20] {
                    if let Some(h) = create(&dev, cap) {
                        println!("PROBE {cap:#x} bytes: CPU {:#x} (window index {}), GPU VA {:#x}", h.cpu, (h.cpu >> 32) & 3, h.gpu);
                    }
                }
            }
            let Some(h) = create(&dev, 0x1_0000) else {
                println!("no GPU_UPLOAD heap on this runtime/GPU – skipping");
                return;
            };
            // the CPU writes two words into the mapped (write-combined) video memory
            let p = h.cpu as *mut u8;
            std::ptr::write_volatile(p.add(4) as *mut u32, 20);
            std::ptr::write_volatile(p.add(8) as *mut u32, 22);
            std::ptr::write_volatile(p as *mut u32, 0);
            let dxil = match crate::gpu::recomp::dxc::compile(
                "RWByteAddressBuffer b : register(u0);\n[numthreads(1,1,1)] void main() { b.Store(0, b.Load(4) + b.Load(8)); }\n",
                "main",
                "cs_6_6",
                &[],
            ) {
                Ok(d) => d,
                Err(e) => {
                    println!("no DXC ({e}) – skipping");
                    return;
                }
            };
            let param = D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            };
            let desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: 1, pParameters: &param, ..Default::default() };
            let mut blob = None;
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).expect("serialize RS");
            let blob = blob.expect("RS blob");
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())).expect("root signature");
            let pso: ID3D12PipelineState = dev
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const _, BytecodeLength: dxil.len() },
                    ..Default::default()
                })
                .expect("compute PSO");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("allocator");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, Some(&pso)).expect("list");
            list.SetComputeRootSignature(&rs);
            list.SetComputeRootUnorderedAccessView(0, h.gpu);
            list.Dispatch(1, 1, 1);
            list.Close().expect("close");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            queue.Signal(&fence, 1).expect("signal");
            let ev = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            fence.SetEventOnCompletion(1, ev).expect("fence event");
            windows::Win32::System::Threading::WaitForSingleObject(ev, 10_000);
            let _ = windows::Win32::Foundation::CloseHandle(ev);
            // the GPU's write is read through the same mapping (unmap/map so the read is not
            // served from a write-combining buffer)
            h._buffer.Unmap(0, None);
            let mut cpu2: *mut std::ffi::c_void = std::ptr::null_mut();
            h._buffer.Map(0, None, Some(&mut cpu2)).expect("second Map");
            let got = std::ptr::read_volatile(cpu2 as *const u32);
            h._buffer.Unmap(0, None);
            assert_eq!(got, 42, "the shader's sum of the CPU's two words in GPU_UPLOAD memory");
        }
    }
}

// ---------------------------------------------------------------------------------------
// The upload window: one of the four console memory windows backed by a GPU_UPLOAD heap
// ---------------------------------------------------------------------------------------

/// `XCL_GPU_UPLOAD_WINDOW=1`: make one 4 GiB console window a GPU_UPLOAD heap (off by default).
pub fn window_mode() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_GPU_UPLOAD_WINDOW").is_ok_and(|v| v != "0" && !v.is_empty()))
}

fn window_want() -> usize {
    let mb = std::env::var("XCL_GPU_UPLOAD_WINDOW_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(1024);
    (mb.clamp(64, 3072)) << 20
}

/// The device the title made (set at device creation, before `open_uma`); the input heap is
/// built from it so that the window layout can follow the heap's CPU mapping.
static DEVICE: Mutex<Option<ID3D12Device>> = Mutex::new(None);

pub fn set_device(dev: &ID3D12Device) {
    *DEVICE.lock().unwrap() = Some(dev.clone());
}

/// The device, while it is alive (None before the title creates it).
pub fn device() -> Option<ID3D12Device> {
    DEVICE.lock().unwrap().clone()
}

/// A GPU_UPLOAD heap that is one console window: the CPU mapping `cpu`, the GPU address `gpu`
/// of the buffer over it, and the window index its mapping lands in.
pub struct Window {
    _heap: ID3D12Heap,
    _buffer: ID3D12Resource,
    pub index: usize,
    /// the driver's CPU mapping (the title's allocation address in the window)
    pub cpu: usize,
    /// the resource's GPU address; the window's shader base is `gpu - low32(cpu)`
    pub gpu: u64,
    /// bytes from `cpu` on (a claim on the window's address space)
    pub len: usize,
}

unsafe impl Send for Window {}

static WINDOW: Mutex<Option<Window>> = Mutex::new(None);

/// The window's shader base (what `uma_window_vas` holds for it and what shaders add the
/// 32-bit offset to): the heap's GPU address minus the mapping's offset in its 4 GiB block.
pub fn window_shader_va(w: &Window) -> u64 {
    w.gpu - (w.cpu as u64 & 0xFFFF_FFFF)
}

/// `window_mode`: make the heap (once) and return it. Shrunk so that the mapping's offset in
/// its 4 GiB block plus the heap stays inside one 32-bit offset (`low32(cpu) + len <= 4 GiB`).
pub fn ensure_window() -> Option<usize> {
    if !window_mode() {
        return None;
    }
    {
        let g = WINDOW.lock().unwrap();
        if let Some(w) = g.as_ref() {
            return Some(w.index);
        }
    }
    let Some(dev) = device() else {
        static ONCE: OnceLock<()> = OnceLock::new();
        ONCE.get_or_init(|| {
            crate::xlog_warn!("GPU_UPLOAD window: off – the title allocated memory before its D3D12 device existed (the windows are already laid out)")
        });
        return None;
    };
    let mut len = window_want();
    let h = create(&dev, len)?;
    let index = (h.cpu >> 32) & 3;
    let low = h.cpu & 0xFFFF_FFFF;
    if low + len > (1usize << 32) {
        len = ((1usize << 32) - low) / GRAIN * GRAIN;
        if len < (64 << 20) {
            crate::xlog_warn!("GPU_UPLOAD window: off – the heap maps at +{low:#x} in its window (low32), leaving only {len} bytes of 32-bit offsets");
            return None;
        }
        crate::xlog_info!("GPU_UPLOAD window: {} MB (the mapping at +{low:#x} leaves {len} bytes before the 32-bit offset wraps)", len >> 20);
    }
    let w = Window { _heap: h._heap, _buffer: h._buffer, index, cpu: h.cpu, gpu: h.gpu, len };
    crate::xlog_info!(
        "GPU_UPLOAD window {} at CPU {:#x} (GPU VA {:#x}, shader base {:#x}), {} MB",
        w.index,
        w.cpu,
        w.gpu,
        window_shader_va(&w),
        w.len >> 20
    );
    let i = w.index;
    *WINDOW.lock().unwrap() = Some(w);
    // the heap's CPU mapping is the window's memory: do not count it in the small arena
    Some(i)
}

/// The window, once made.
pub fn window() -> Option<(usize, usize, u64, usize)> {
    let g = WINDOW.lock().unwrap();
    g.as_ref().map(|w| (w.index, w.cpu, window_shader_va(w), w.len))
}

/// Bytes of the window in use, of its size (for the frame log).
pub fn window_used(used: usize) -> String {
    let g = WINDOW.lock().unwrap();
    match g.as_ref() {
        Some(w) => format!("{} MB of {} MB", used >> 20, w.len >> 20),
        None => "-".into(),
    }
}

#[cfg(test)]
mod window_tests {
    use super::*;
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;

    /// The upload window end to end: the CPU writes at the window's address (the driver's CPU
    /// mapping), and a compute shader reads those words through the window's shader base
    /// (`H - low32(M)`) plus the 32-bit console offset – no upload copy in between.
    #[test]
    fn gpu_upload_window_cpu_writes_reach_a_compute_shader() {
        if std::env::var_os("XCL_GPU_UPLOAD_WINDOW").is_some_and(|v| v == "0") {
            println!("XCL_GPU_UPLOAD_WINDOW=0 – skipping");
            return;
        }
        unsafe {
            std::env::set_var("XCL_GPU_UPLOAD_WINDOW", "1");
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev).is_err() {
                println!("no D3D12 device – skipping");
                return;
            }
            let dev = dev.expect("device");
            set_device(&dev);
            let Some(_k) = ensure_window() else {
                println!("no GPU_UPLOAD window here – skipping");
                return;
            };
            let (index, cpu, base, len) = window().expect("window");
            let low = cpu & 0xFFFF_FFFF;
            println!("upload window {index}: CPU {cpu:#x}, shader base {base:#x}, {len} bytes, low {low:#x}");
            let p = cpu as *mut u8;
            std::ptr::write_volatile(p.add(4) as *mut u32, 20);
            std::ptr::write_volatile(p.add(8) as *mut u32, 22);
            std::ptr::write_volatile(p as *mut u32, 0);
            let hlsl = format!(
                "RWByteAddressBuffer b : register(u0);\n[numthreads(1,1,1)] void main() {{ b.Store({low}, b.Load({a}) + b.Load({b})); }}\n",
                a = low + 4,
                b = low + 8
            );
            let dxil = match crate::gpu::recomp::dxc::compile(&hlsl, "main", "cs_6_6", &[]) {
                Ok(d) => d,
                Err(e) => {
                    println!("no DXC ({e}) – skipping");
                    return;
                }
            };
            let param = D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            };
            let desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: 1, pParameters: &param, ..Default::default() };
            let mut blob = None;
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).expect("serialize RS");
            let blob = blob.expect("RS blob");
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())).expect("root signature");
            let pso: ID3D12PipelineState = dev
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const _, BytecodeLength: dxil.len() },
                    ..Default::default()
                })
                .expect("compute PSO");
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("queue");
            let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("allocator");
            let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, Some(&pso)).expect("list");
            list.SetComputeRootSignature(&rs);
            list.SetComputeRootUnorderedAccessView(0, base);
            list.Dispatch(1, 1, 1);
            list.Close().expect("close");
            let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>().unwrap())]);
            queue.Signal(&fence, 1).expect("signal");
            let ev = windows::Win32::System::Threading::CreateEventW(None, false, false, None).expect("event");
            fence.SetEventOnCompletion(1, ev).expect("fence event");
            windows::Win32::System::Threading::WaitForSingleObject(ev, 10_000);
            let _ = windows::Win32::Foundation::CloseHandle(ev);
            let got = std::ptr::read_volatile(cpu as *const u32);
            assert_eq!(got, 42, "the shader's sum of the CPU's two words in the GPU_UPLOAD window");
        }
    }
}
