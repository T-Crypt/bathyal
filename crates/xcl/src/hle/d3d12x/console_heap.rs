//! Console descriptor heaps (CBV/SRV/UAV and samplers).
//!
//! On the console a descriptor is GPU-readable memory: 32 bytes per CBV/SRV/UAV slot (T# /
//! V#), 16 per sampler (S#), and a descriptor table in a root argument is a 32-bit GPU address
//! (shaders load it with the high half 0 – amd_fidelityfx_upscaler_xbox.dll, HANDOFF "FSR4").
//! XCL gives console code heaps whose slots live in unified memory (so the table addresses
//! console code computes from the heap start work as they are in the recompiler's `mem`
//! model) and keeps PC descriptors beside each: view slot k → an SRV and, next to it, a UAV;
//! sampler slot k → PC sampler k. A slot holds the descriptor in the XCL shader ABI
//! (`desc::xcl_image_words`: T# word 0 = the SRV's index in the shader-visible PC heap;
//! `desc::xcl_sampler_words`: S# word 3 [11:0] = k; V# with its unified-memory address).
//!
//! View descriptors are values on the console: titles create them in a CPU-only heap and copy
//! them into the shader-visible one as memory (the console header's inline
//! CopyDescriptorsSimple; XDefiant copies every texture that way), and a descriptor table may
//! be in any heap. So all console view heaps share one shader-visible PC heap (`Global`):
//! each gets a range there, T# word 0 is an index into it – a T# copied anywhere still names
//! its PC descriptor – and that heap is what lists bind. The title's own PC heap stays as a
//! CPU-only mirror (where views are made, copies come from and UAV clears take their CPU
//! handle). A used slot gets its own two descriptors in the shared heap when it is first
//! touched – a run per slot, never a whole page, so a heap of many slots uses exactly what it
//! fills and no slot can alias another (a page-based reservation ran out of the cap on titles
//! with large heaps, and the slots past it fell back to indices of other heaps). When the cap
//! is reached a slot reads the universal null descriptor instead.
//! when a slot of that page is first used: titles make large heaps and fill few slots
//! (Minecraft: eight view heaps of 73 727 slots – 147 454 PC descriptors each, more in all
//! than the 1 000 000 an NVIDIA shared heap holds – with a few hundred views in them). A heap
//! left out of the shared heap would be lost: lists bind only the shared heap for views.
//! Without the shared heap (it could not be made) a heap keeps its own PC heap: SRV at 2k,
//! UAV at 2k + 1.

use std::collections::HashMap;
use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;

/// bytes of a CBV/SRV/UAV slot (T# / V#) and of a sampler slot (S#)
pub const SLOT: u32 = 32;
pub const SAMPLER_SLOT: u32 = 16;

/// Whose memory a heap's slots are in.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Memory {
    /// a sparse VRAM allocation in window 0
    Sparse,
    /// committed system RAM
    Uma,
    /// the title's (its allocation hook gave it: `kernelx::runtime_alloc`)
    Title,
}

struct Heap {
    obj: usize,
    shadow: usize,
    n: u32,
    pc_cpu: u64,
    /// 0: not shader visible
    pc_gpu: u64,
    inc: u32,
    sampler: bool,
    /// the heap's views are in the shared heap (`placed` has the slots that got a place)
    shared: bool,
    /// view slot k → the shared-heap index of its SRV (the UAV is at + 1)
    placed: HashMap<u32, u32>,
    /// sampler heaps: PC descriptors the heap really has (the title's count, capped), the
    /// slots kept at their own index and the scratch descriptors the slots above them share
    pc_n: u32,
    keep: u32,
    scratch: u32,
    /// shader visible as the title made it
    visible: bool,
    memory: Memory,
}

impl Heap {
    fn slot_bytes(&self) -> u32 {
        if self.sampler {
            SAMPLER_SLOT
        } else {
            SLOT
        }
    }
}

static HEAPS: Mutex<Vec<Heap>> = Mutex::new(Vec::new());

/// The shader-visible PC heap of all console view heaps.
struct Global {
    dev: ID3D12Device,
    heap: ID3D12DescriptorHeap,
    cpu: u64,
    gpu: u64,
    inc: u32,
    cap: u32,
    next: u32,
    /// released ranges (first, count)
    free: Vec<(u32, u32)>,
}

// D3D12 descriptor heaps and devices are free-threaded
unsafe impl Send for Global {}

static GLOBAL: Mutex<Option<Global>> = Mutex::new(None);
static GLOBAL_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// descriptors of the shared heap at most (the GPU may take more: OPTIONS19)
const GLOBAL_MAX: u32 = 1 << 21;

/// Slot ranges are taken in runs; a table may be in any heap, and a run that cannot be had
/// as a whole is rebuilt from single slots (`place`) – see the module comment.

/// Map an environment override (`XCL_HEAP_CAP`) or `d` to the cap it names: at most `d`, and
/// never 0.
pub fn capped(d: u32) -> u32 {
    let over = std::env::var("XCL_HEAP_CAP").ok().and_then(|v| v.trim().parse::<u32>().ok()).filter(|&v| v > 0);
    over.map_or(d, |v| v.min(d))
}

/// PC sampler descriptors of a shader-visible sampler heap at most: D3D12's
/// `D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE` (2 048).
pub const SAMPLER_MAX: u32 = 2048;

/// The null descriptors' pool in the shared heap: 8 image types × 3 format classes and the
/// universal one (2D, 32-bit float), two descriptors (SRV, UAV) each.
const NULL_KINDS: usize = 25;
const UNIVERSAL: usize = 24;
pub const NULL_POOL: u32 = 2 * NULL_KINDS as u32;
/// first descriptor of the null pool in the shared heap: at its start (this tree has no
/// memory-window views in front of it)
const NULL_BASE: u32 = 0;
/// the universal null's SRV: an image slot with no view of its own points here
pub const NULL_INDEX: u32 = NULL_BASE + 2 * UNIVERSAL as u32;
/// a view slot of a heap in the shared heap that found no place there
const NULL_SLOT: u32 = NULL_INDEX;


/// PC descriptors of a view slot: an SRV and, next to it, a UAV.
pub const SLOT_DESCRIPTORS: u32 = 2;

/// A first-fit range of `count` in `free` (released ranges), else at `next` below `cap`.
fn take_range(free: &mut Vec<(u32, u32)>, next: &mut u32, cap: u32, count: u32) -> Option<u32> {
    if let Some(i) = free.iter().position(|&(_, n)| n >= count) {
        let (f, n) = free[i];
        if n == count {
            free.swap_remove(i);
        } else {
            free[i] = (f + count, n - count);
        }
        return Some(f);
    }
    if cap - *next < count {
        return None;
    }
    *next += count;
    Some(*next - count)
}

/// The shared shader-visible heap, made on first use. False: there is none (it could not be
/// made) – view heaps then keep their own.
pub unsafe fn shared_heap(dev: &ID3D12Device) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let mut g = GLOBAL.lock().unwrap();
    let mut made = false;
    if g.is_none() {
        if GLOBAL_FAILED.load(Relaxed) {
            return false;
        }
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS19::default();
        let max = if dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS19, &mut o as *mut _ as *mut std::ffi::c_void, std::mem::size_of_val(&o) as u32).is_ok() && o.MaxViewDescriptorHeapSize > 0 {
            o.MaxViewDescriptorHeapSize
        } else {
            1_000_000
        };
        let cap = capped(max.min(GLOBAL_MAX)).max(NULL_BASE + NULL_POOL);
        let d = D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, NumDescriptors: cap, Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, NodeMask: 0 };
        let heap: ID3D12DescriptorHeap = match dev.CreateDescriptorHeap(&d) {
            Ok(h) => h,
            Err(e) => {
                GLOBAL_FAILED.store(true, Relaxed);
                crate::xlog_warn!("console descriptor heaps: no shared shader-visible heap of {cap} ({e}) – each keeps its own");
                return false;
            }
        };
        crate::xlog_info!("console descriptor heaps: one shader-visible PC heap of {cap} view descriptors for all of them");
        *g = Some(Global {
            dev: dev.clone(),
            cpu: heap.GetCPUDescriptorHandleForHeapStart().ptr as u64,
            gpu: heap.GetGPUDescriptorHandleForHeapStart().ptr,
            inc: dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
            heap,
            cap,
            // the null descriptors every empty slot points at come first
            next: NULL_BASE + NULL_POOL,
            free: Vec::new(),
        });
        made = true;
    }
    drop(g);
    if made {
        // the descriptor every empty view slot reads, made before any list can bind the heap
        make_universal_null(dev);
    }
    true
}

/// The place in the shared heap of slot `k` of `h`: its SRV's index there, taken now if the
/// slot has none yet. One run per used slot, so a slot never needs its neighbours to lie
/// next to it: a heap of many slots uses exactly what it fills, and no slot aliases another.
/// None: the heap is not in the shared heap, or that one is full.
fn place(h: &mut Heap, k: u32) -> Option<u32> {
    debug_assert!(h.shared);
    if let Some(&i) = h.placed.get(&k) {
        return Some(i);
    }
    let i = {
        let mut g = GLOBAL.lock().unwrap();
        let s = g.as_mut()?;
        let (cap, mut next) = (s.cap, s.next);
        let r = take_range(&mut s.free, &mut next, cap, SLOT_DESCRIPTORS);
        s.next = next;
        if r.is_none() {
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
                crate::xlog_warn!("console descriptor heaps: the shared heap ({cap}) is full – a view slot past it reads the universal null");
            }
        }
        r
    };
    let Some(i) = i else {
        // the highest title slot without a place in the shared heap, logged once per new one
        static MAX: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if k > MAX.load(std::sync::atomic::Ordering::Relaxed) {
            MAX.store(k, std::sync::atomic::Ordering::Relaxed);
            static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 32 {
                crate::xlog_info!("console descriptor heaps: the highest title slot outside the shared heap is {k} (heap {:#x})", h.obj);
            }
        }
        return None;
    };
    h.placed.insert(k, i);
    Some(i)
}

/// The shared range starting at `first` (`count` descriptors) is free again.
pub fn release_global(first: u32, count: u32) {
    if let Some(s) = GLOBAL.lock().unwrap().as_mut() {
        s.free.push((first, count));
    }
}

/// The shader-visible PC heap lists bind for view descriptors (None: none shared).
pub fn global_heap() -> Option<ID3D12DescriptorHeap> {
    GLOBAL.lock().unwrap().as_ref().map(|s| s.heap.clone())
}

/// Whether `obj` (a title's view heap) has its descriptors in the shared heap.
pub fn in_global(obj: usize) -> bool {
    HEAPS.lock().unwrap().iter().any(|h| h.obj == obj && h.shared)
}

/// The index of sampler slot `k` in `pc_n` PC descriptors: the slot's own below `keep`, else
/// a scratch descriptor above it. `scratch` 0: every slot has its own.
fn sampler_index(pc_n: u32, keep: u32, scratch: u32, k: u32) -> u32 {
    if k < keep || scratch == 0 {
        k.min(pc_n.saturating_sub(1))
    } else {
        keep + (k - keep) % scratch
    }
}

/// The slots kept at their own index and the scratch descriptors above them for a sampler
/// heap of `n` slots whose PC heap has `pc_n`.
fn sampler_layout(n: u32, pc_n: u32) -> (u32, u32) {
    if n <= pc_n {
        return (n, 0);
    }
    let scratch = (n - pc_n).min(pc_n / 4).max(1).min(pc_n);
    (pc_n - scratch, scratch)
}

/// Memory for a heap of `bytes` from the title's allocation hook – where the title expects the
/// heap then. Only what descriptor tables can address (window 0) and is committed; anything else
/// goes back to the title.
unsafe fn title_memory(u: &crate::gpu::uma::Uma, bytes: usize) -> Option<usize> {
    use crate::hle::kernelx::{runtime_alloc, runtime_free, D3D_DESCRIPTOR_HEAP};
    let p = runtime_alloc(bytes, D3D_DESCRIPTOR_HEAP)?;
    let readable = crate::gpu::shader::readable(p, bytes);
    if readable && u.window_of(p) == Some(0) && u.window_of(p + bytes - 1) == Some(0) {
        return Some(p);
    }
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 8 {
        crate::xlog_warn!("console descriptor heap: the title's memory {p:#x} ({bytes:#x} bytes) not used (window {:?}, committed {readable})", u.window_of(p));
    }
    runtime_free(p, D3D_DESCRIPTOR_HEAP);
    None
}

/// `obj` is a PC heap from `pc_cpu` / `pc_gpu` (increment `inc`; `pc_gpu` 0 when not shader
/// visible) with `pc_n` PC descriptors (`d` samplers, or 2 per view slot): give it `n` console
/// slots in unified memory. `shared`: its views go to the shared heap (`shared_heap`), one
/// run per used slot; `visible`: what the title asked for.
#[allow(clippy::too_many_arguments)]
pub fn register(obj: usize, n: u32, pc_n: u32, pc_cpu: u64, pc_gpu: u64, inc: u32, sampler: bool, shared: bool, visible: bool) -> bool {
    let Some(u) = crate::gpu::uma::get() else { return false };
    let bytes = n.max(1) * if sampler { SAMPLER_SLOT } else { SLOT };
    // Descriptor tables are 32-bit console addresses (high dword 0): must live in window 0.
    // On the sparse-window path, `sparse::alloc_in(0, ...)` commits fresh zeroed pages so we
    // avoid touching unused pages with `write_bytes` (which would mark them dirty for write-watch).
    let (p, memory) = if let Some(p) = unsafe { title_memory(u, bytes as usize) } {
        (p as *mut u8, Memory::Title)
    } else if u.is_sparse_window(0) {
        let Some((addr, _)) = (unsafe { super::sparse::alloc_in(0, bytes as usize, 256) }) else {
            return false;
        };
        (addr as *mut u8, Memory::Sparse)
    } else {
        let Some(p) = u.alloc_in(0, bytes as usize, 256) else { return false };
        unsafe { std::ptr::write_bytes(p, 0, bytes as usize) };
        (p, Memory::Uma)
    };
    {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 48 {
            crate::xlog_info!(
                "console descriptor heap {obj:#x}: {n} {} slots at {} {:#x}, {}",
                if sampler { "sampler" } else { "view" },
                match memory {
                    Memory::Sparse => "sparse VRAM",
                    Memory::Uma => "UMA",
                    Memory::Title => "the title's memory",
                },
                p as usize - u.base(),
                if visible { "shader visible" } else { "CPU only" }
            );
        }
    }
    let shared = shared && !sampler && GLOBAL.lock().unwrap().is_some();
    let (keep, scratch) = if sampler { sampler_layout(n, pc_n.max(1)) } else { (0, 0) };
    HEAPS.lock().unwrap().push(Heap { obj, shadow: p as usize, n, pc_cpu, pc_gpu, inc, sampler, shared, placed: HashMap::new(), pc_n, keep, scratch, visible, memory });
    fill_empty(obj, n);
    true
}

/// A slot nothing was written to decodes as a null resource of the kind a shader may read it
/// as, never as some other live descriptor.
///
/// A view slot gets the universal null: a 1×1 2D float T# whose SRV is the null pool's
/// universal entry, and whose fields are also a V# of no records – a texture read returns
/// zero and so does a buffer or constant read. A sampler slot gets the point/wrap sampler at
/// its own PC index.
fn fill_empty(obj: usize, n: u32) {
    let (at, sampler, keep, scratch, pc_n) = {
        let heaps = HEAPS.lock().unwrap();
        let Some(h) = heaps.iter().find(|h| h.obj == obj) else { return };
        // a heap whose views are not in the shared heap has no null SRV to point at
        if !h.sampler && !h.shared {
            return;
        }
        (h.shadow, h.sampler, h.keep, h.scratch, h.pc_n)
    };
    let null = crate::gpu::desc::null_image_words(NULL_INDEX, crate::gpu::desc::ImageType::D2, 0);
    for k in 0..n as usize {
        unsafe {
            if sampler {
                let w = [0u32, 0, 0, sampler_index(pc_n, keep, scratch, k as u32)];
                std::ptr::copy_nonoverlapping(w.as_ptr(), (at + k * SAMPLER_SLOT as usize) as *mut u32, 4);
            } else {
                std::ptr::copy_nonoverlapping(null.as_ptr(), (at + k * SLOT as usize) as *mut u32, 8);
            }
        }
    }
}

/// The heap object was released for good: its slots go back to unified memory.
pub fn unregister(obj: usize) {
    let gone = {
        let mut heaps = HEAPS.lock().unwrap();
        heaps.iter().position(|h| h.obj == obj).map(|i| heaps.swap_remove(i))
    };
    if let Some(h) = gone {
        let bytes = h.n.max(1) as usize * if h.sampler { SAMPLER_SLOT as usize } else { SLOT as usize };
        if let Some(m) = NAMED.lock().unwrap().as_mut() {
            let end = h.shadow + bytes;
            m.retain(|&a, _| a < h.shadow || a >= end);
        }
        for &f in h.placed.values() {
            release_global(f, SLOT_DESCRIPTORS);
        }
        match h.memory {
            Memory::Sparse => {
                unsafe { super::sparse::free(h.shadow) };
            }
            Memory::Uma => {
                if let Some(u) = crate::gpu::uma::get() {
                    u.free(h.shadow);
                }
            }
            Memory::Title => unsafe { crate::hle::kernelx::runtime_free(h.shadow, crate::hle::kernelx::D3D_DESCRIPTOR_HEAP) },
        }
    }
}

/// Start of the console slots of heap `obj` (CPU handle = GPU handle on the console).
pub fn start(obj: usize) -> Option<u64> {
    HEAPS.lock().unwrap().iter().find(|h| h.obj == obj).map(|h| h.shadow as u64)
}

/// A console slot: where its descriptor goes and its PC descriptors.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub at: usize,
    /// views: PC heap index of the SRV (the UAV is at + 1); samplers: of the sampler
    pub index: u32,
    /// PC CPU handles of the SRV / UAV where views are made (the title's heap)
    pub pc_srv: u64,
    pub pc_uav: u64,
    /// PC GPU handle of the UAV (0: heap not shader visible)
    pub pc_gpu_uav: u64,
    pub sampler: bool,
    /// CPU handle of the SRV in the shared heap (0: the heap is not in it, or the slot has
    /// no place there)
    pub shared_srv: u64,
    /// the slot's number in its heap
    pub k: u32,
    /// the heap's views live in the shared heap (a bound heap of its own otherwise)
    pub shared: bool,
}

/// The slot behind a console CPU handle (None: not one of XCL's console heaps). A view slot
/// of a heap in the shared heap gets its place there now if it has none yet.
pub fn slot(cpu: u64) -> Option<Slot> {
    let mut heaps = HEAPS.lock().unwrap();
    let h = heaps.iter_mut().find(|h| cpu >= h.shadow as u64 && cpu < (h.shadow + (h.n * h.slot_bytes()) as usize) as u64)?;
    let k = ((cpu - h.shadow as u64) / h.slot_bytes() as u64) as u32;
    if h.sampler {
        let index = sampler_index(h.pc_n, h.keep, h.scratch, k);
        return Some(Slot { at: h.shadow + (k * SAMPLER_SLOT) as usize, index, pc_srv: h.pc_cpu + (index * h.inc) as u64, pc_uav: 0, pc_gpu_uav: 0, sampler: true, shared_srv: 0, k, shared: false });
    }
    let uav = ((2 * k + 1) * h.inc) as u64;
    let shared = if h.shared { place(h, k) } else { None }.and_then(|i| {
        let g = GLOBAL.lock().unwrap();
        let g = g.as_ref()?;
        Some((i, g.gpu + (i as u64 + 1) * g.inc as u64, g.cpu + i as u64 * g.inc as u64))
    });
    let (index, pc_gpu_uav, shared_srv) = if h.shared {
        // no place: the slot reads the universal null rather than a descriptor of another heap
        shared.unwrap_or((NULL_SLOT, 0, 0))
    } else {
        (2 * k, if h.pc_gpu == 0 { 0 } else { h.pc_gpu + uav }, 0)
    };
    Some(Slot { at: h.shadow + (k * SLOT) as usize, index, pc_srv: h.pc_cpu + (2 * k * h.inc) as u64, pc_uav: h.pc_cpu + uav, pc_gpu_uav, sampler: false, shared_srv, k, shared: h.shared })
}

/// The view just made in slot `s` (its SRV, or with `uav` its UAV) goes to the shared heap.
pub unsafe fn publish(s: &Slot, uav: bool) {
    if s.shared_srv == 0 {
        return;
    }
    if let Some(g) = GLOBAL.lock().unwrap().as_ref() {
        let (dst, src) = if uav { (s.shared_srv + g.inc as u64, s.pc_uav) } else { (s.shared_srv, s.pc_srv) };
        g.dev.CopyDescriptorsSimple(1, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: dst as usize }, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: src as usize }, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    }
}

/// `n` view slots from `first` were copied (in the title's heap): they go to the shared heap,
/// one run per slot – neighbouring slots need not lie next to each other there.
pub unsafe fn publish_range(first: &Slot, n: u32) {
    for done in 0..n {
        let s = if done == 0 { Some(*first) } else { slot((first.at + (done * SLOT) as usize) as u64) };
        let Some(s) = s else { return };
        publish(&s, false);
        publish(&s, true);
    }
}

/// view slot address → the resource its view names (`XCL_TRACE_FRAME` diagnostics)
static RES: Mutex<Option<std::collections::HashMap<usize, usize>>> = Mutex::new(None);

/// view slot address → the resource its view names, always kept: when a resource is retired
/// its slots are nulled before the next submission reads the heap.
static NAMED: Mutex<Option<std::collections::HashMap<usize, usize>>> = Mutex::new(None);

/// The view in slot `s` names `res` (0: none).
pub fn note_resource(s: &Slot, res: usize) {
    if s.sampler {
        return;
    }
    {
        let mut g = NAMED.lock().unwrap();
        let m = g.get_or_insert_with(Default::default);
        if res == 0 {
            m.remove(&s.at);
        } else {
            m.insert(s.at, res);
        }
    }
    if !super::present::trace_frame() {
        return;
    }
    let mut g = RES.lock().unwrap();
    let m = g.get_or_insert_with(Default::default);
    if res == 0 {
        m.remove(&s.at);
    } else {
        m.insert(s.at, res);
    }
}

=
/// `res` was released for good: every console slot that named it becomes a null slot, and the
/// shared shader-visible PC heap gets null descriptors in the places those slots name. No
/// later submission can read a descriptor of memory that is gone.
pub fn retire_resource(res: usize) {
    if res == 0 {
        return;
    }
    let (slots, dev, inc) = {
        let mut g = NAMED.lock().unwrap();
        let m = g.get_or_insert_with(Default::default);
        let slots: Vec<usize> = m.iter().filter(|&(_, &r)| r == res).map(|(&a, _)| a).collect();
        m.retain(|_, &mut r| r != res);
        let (dev, inc) = GLOBAL
            .lock()
            .unwrap()
            .as_ref()
            .map_or((None, 0), |s| (Some(s.dev.clone()), s.inc));
        (slots, dev, inc)
    };
    for at in slots {
        let Some(s) = slot(at as u64) else { continue };
        write(&s, &[0; 8]);
        let Some(dev) = &dev else { continue };
        unsafe {
            dev.CreateShaderResourceView(None::<&ID3D12Resource>, None::<*const D3D12_SHADER_RESOURCE_VIEW_DESC>, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: s.pc_srv as usize });
            if s.pc_uav != 0 {
                dev.CreateUnorderedAccessView(None::<&ID3D12Resource>, None::<&ID3D12Resource>, None::<*const D3D12_UNORDERED_ACCESS_VIEW_DESC>, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: s.pc_uav as usize });
            }
            if s.shared_srv != 0 {
                dev.CreateShaderResourceView(None::<&ID3D12Resource>, None::<*const D3D12_SHADER_RESOURCE_VIEW_DESC>, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: s.shared_srv as usize });
                dev.CreateUnorderedAccessView(None::<&ID3D12Resource>, None::<&ID3D12Resource>, None::<*const D3D12_UNORDERED_ACCESS_VIEW_DESC>, D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (s.shared_srv + inc as u64) as usize });
            }
        }
    }
}

/// The resource of the view slot at UMA address `at`.
pub fn resource_at(at: usize) -> Option<usize> {
    RES.lock().unwrap().as_ref()?.get(&at).copied()
}

pub fn write(s: &Slot, words: &[u32]) {
    let n = words.len().min(if s.sampler { 4 } else { 8 });
    unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), s.at as *mut u32, n) };
}

// -----------------------------------------------------------------------------------------
// null descriptors
// -----------------------------------------------------------------------------------------

/// The null views of the shared heap: a typed null SRV (and UAV) of every image dimension and
/// format class, two descriptors each. 0 = not made, 1 = failed, 2 = made. An SRV is at
/// `NULL_BASE + 2·kind`, its UAV at + 1.
static NULL_STATE: Mutex<[u8; NULL_KINDS]> = Mutex::new([0; NULL_KINDS]);
static NULL_LOCK: Mutex<()> = Mutex::new(());

fn image_kind(ty: crate::gpu::desc::ImageType) -> usize {
    use crate::gpu::desc::ImageType as T;
    match ty {
        T::D1 => 0,
        T::D2 => 1,
        T::D3 => 2,
        T::Cube => 3,
        T::D1Array => 4,
        T::D2Array => 5,
        T::D2Msaa => 6,
        T::D2MsaaArray => 7,
    }
}

/// The class (0 float, 1 uint, 2 int) of a GFX10 image format, as `img_cls` decodes it.
pub fn format_class(format: u32) -> u32 {
    match (crate::gpu::recomp::emit::buf_format_info(format.min(77) as usize) >> 3) & 7 {
        4 => 1,
        5 => 2,
        _ => 0,
    }
}

/// The typed null SRV of image type `ty` and format class `class` in the shared heap: made
/// now if it does not exist yet. None: no shared heap, or the descriptor could not be made.
///
/// A null descriptor whose description is complete is legal D3D12 and reads as zero; a null
/// description is not – it makes a real null descriptor, and a typed read of that removed the
/// device at a retail title's start-up.
pub unsafe fn null_view_index(dev: &ID3D12Device, ty: crate::gpu::desc::ImageType, class: u32) -> Option<u32> {
    let kind = image_kind(ty) * 3 + class.min(2) as usize;
    let index = NULL_BASE + 2 * kind as u32;
    if NULL_STATE.lock().unwrap()[kind] == 2 {
        return Some(index);
    }
    let _one = NULL_LOCK.lock().unwrap();
    if NULL_STATE.lock().unwrap()[kind] == 2 {
        return Some(index);
    }
    let (cpu, inc) = {
        let g = GLOBAL.lock().unwrap();
        let s = g.as_ref()?;
        (s.cpu, s.inc)
    };
    let class = class.min(2);
    let srv = null_srv_desc(ty, class);
    dev.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&srv), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (cpu + index as u64 * inc as u64) as usize });
    if !matches!(ty, crate::gpu::desc::ImageType::D2Msaa | crate::gpu::desc::ImageType::D2MsaaArray) {
        let uav = null_uav_desc(ty, class);
        dev.CreateUnorderedAccessView(None::<&ID3D12Resource>, None::<&ID3D12Resource>, Some(&uav), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (cpu + (index as u64 + 1) * inc as u64) as usize });
    }
    NULL_STATE.lock().unwrap()[kind] = 2;
    Some(index)
}

/// The universal null at `NULL_INDEX`: a 2D float one, made at the shared heap's creation so
/// no empty slot ever reads an undefined descriptor.
unsafe fn make_universal_null(dev: &ID3D12Device) {
    let (cpu, inc) = {
        let g = GLOBAL.lock().unwrap();
        let Some(s) = g.as_ref() else { return };
        (s.cpu, s.inc)
    };
    let d = null_srv_desc(crate::gpu::desc::ImageType::D2, 0);
    dev.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&d), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (cpu + NULL_INDEX as u64 * inc as u64) as usize });
    let u = null_uav_desc(crate::gpu::desc::ImageType::D2, 0);
    dev.CreateUnorderedAccessView(None::<&ID3D12Resource>, None::<&ID3D12Resource>, Some(&u), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: (cpu + (NULL_INDEX as u64 + 1) * inc as u64) as usize });
    NULL_STATE.lock().unwrap()[image_kind(crate::gpu::desc::ImageType::D2) * 3] = 2;
}

fn null_dxgi_format(class: u32) -> windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT {
    use windows::Win32::Graphics::Dxgi::Common::*;
    match class {
        1 => DXGI_FORMAT_R32G32B32A32_UINT,
        2 => DXGI_FORMAT_R32G32B32A32_SINT,
        _ => DXGI_FORMAT_R32G32B32A32_FLOAT,
    }
}

fn null_view_dimension(ty: crate::gpu::desc::ImageType) -> D3D12_SRV_DIMENSION {
    use crate::gpu::desc::ImageType as T;
    match ty {
        T::D1 => D3D12_SRV_DIMENSION_TEXTURE1D,
        T::D1Array => D3D12_SRV_DIMENSION_TEXTURE1DARRAY,
        T::D2 => D3D12_SRV_DIMENSION_TEXTURE2D,
        T::D2Array => D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
        T::D2Msaa => D3D12_SRV_DIMENSION_TEXTURE2DMS,
        T::D2MsaaArray => D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY,
        T::D3 => D3D12_SRV_DIMENSION_TEXTURE3D,
        T::Cube => D3D12_SRV_DIMENSION_TEXTURECUBEARRAY,
    }
}

fn null_srv_desc(ty: crate::gpu::desc::ImageType, class: u32) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    use crate::gpu::desc::ImageType as T;
    let mut d = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: null_dxgi_format(class),
        ViewDimension: null_view_dimension(ty),
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        ..Default::default()
    };
    let a = &mut d.Anonymous;
    match ty {
        T::D1 => a.Texture1D = D3D12_TEX1D_SRV { MipLevels: 1, ..Default::default() },
        T::D1Array => a.Texture1DArray = D3D12_TEX1D_ARRAY_SRV { ArraySize: 1, MipLevels: 1, ..Default::default() },
        T::D2 | T::D2Msaa => a.Texture2D = D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
        T::D2Array | T::D2MsaaArray => a.Texture2DArray = D3D12_TEX2D_ARRAY_SRV { ArraySize: 1, MipLevels: 1, ..Default::default() },
        T::D3 => a.Texture3D = D3D12_TEX3D_SRV { MipLevels: 1, ..Default::default() },
        T::Cube => a.TextureCubeArray = D3D12_TEXCUBE_ARRAY_SRV { NumCubes: 1, MipLevels: 1, ..Default::default() },
    }
    d
}

fn null_uav_desc(ty: crate::gpu::desc::ImageType, class: u32) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    use crate::gpu::desc::ImageType as T;
    let mut d = D3D12_UNORDERED_ACCESS_VIEW_DESC { Format: null_dxgi_format(class), ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D, ..Default::default() };
    let a = &mut d.Anonymous;
    match ty {
        T::D1 => {
            d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
            a.Texture1D = D3D12_TEX1D_UAV { MipSlice: 0 };
        }
        T::D1Array => {
            d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            a.Texture1DArray = D3D12_TEX1D_ARRAY_UAV { MipSlice: 0, FirstArraySlice: 0, ArraySize: 1 };
        }
        T::D2Array => {
            d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            a.Texture2DArray = D3D12_TEX2D_ARRAY_UAV { MipSlice: 0, FirstArraySlice: 0, ArraySize: 1, PlaneSlice: 0 };
        }
        T::D3 => {
            d.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            a.Texture3D = D3D12_TEX3D_UAV { MipSlice: 0, FirstWSlice: 0, WSize: 1 };
        }
        _ => a.Texture2D = D3D12_TEX2D_UAV { MipSlice: 0, PlaneSlice: 0 },
    }
    d
}

/// What a view of no resource asks for: a buffer (raw or structured) or a texture of a
/// dimension and format class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NullKind {
    Buffer,
    Image(crate::gpu::desc::ImageType, u32),
}

/// The console words of a view slot whose view has no resource: a null of the kind the view
/// asks for, never eight zero words (word 0 of a T# is a heap index, and 0 is a live
/// descriptor of another kind). A texture slot points at the typed null of its dimension; a
/// buffer slot is a V# of no records. Both read as zero and never index past the heap.
pub unsafe fn write_null(dev: &ID3D12Device, s: &Slot, kind: NullKind) {
    if s.sampler {
        write(s, &[0, 0, 0, s.index & 0xFFF]);
        return;
    }
    match kind {
        NullKind::Buffer => write(s, &crate::gpu::desc::null_buffer_words()),
        NullKind::Image(ty, class) => {
            let class = class.min(2);
            if s.shared {
                // a place of the slot's own if it has one, else the universal null
                let index = if s.shared_srv != 0 { null_view_index(dev, ty, class) } else { None }.unwrap_or(NULL_INDEX);
                write(s, &crate::gpu::desc::null_image_words(index, ty, class));
            } else {
                // the title's own CPU-only heap: a typed null at the slot's PC handle
                let d = null_srv_desc(ty, class);
                dev.CreateShaderResourceView(None::<&ID3D12Resource>, Some(&d), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: s.pc_srv as usize });
                write(s, &crate::gpu::desc::null_image_words(s.index, ty, class));
            }
        }
    }
}

/// Copy a console slot's descriptor to another slot of the same kind, the PC heap index
/// in it (T# word 0, S# word 3 [11:0]) renumbered for the destination. Buffer V# carry
/// their address and stay as they are, and so does a null view (its word 0 is the null
/// pool's, not the destination's).
pub fn copy_words(src: &Slot, dst: &Slot) {
    let n = if src.sampler { 4 } else { 8 };
    let mut w = [0u32; 8];
    unsafe { std::ptr::copy_nonoverlapping(src.at as *const u32, w.as_mut_ptr(), n) };
    let null = !src.sampler && w[3] >> 28 >= 8 && w[0] == NULL_INDEX;
    if src.sampler {
        w[3] = (w[3] & !0xFFF) | (dst.index & 0xFFF);
    } else if w[3] >> 28 >= 8 && !null && (!dst.shared || dst.shared_srv != 0) {
        w[0] = dst.index;
    }
    write(dst, &w[..n]);
    if !src.sampler {
        let named = NAMED.lock().unwrap().as_ref().and_then(|m| m.get(&src.at).copied()).unwrap_or(0);
        note_resource(dst, named);
    }
}

/// `XCL_TRACE_FRAME`: what the slots of each shader-visible view heap hold – images whose
/// PC index is the slot's own (made or copied through the device) or another (copied by the
/// title itself, as memory), buffers, empty.
pub fn census() {
    let heaps = HEAPS.lock().unwrap();
    // samplers: S# word 3 [11:0] is the PC index in the sampler heap the slot belongs to
    for h in heaps.iter().filter(|h| h.sampler) {
        let (mut empty, mut own, mut other) = (0u32, 0u32, 0u32);
        let mut examples: Vec<(u32, u32)> = Vec::new();
        for k in 0..h.n {
            let w = unsafe { std::slice::from_raw_parts((h.shadow + (k * SAMPLER_SLOT) as usize) as *const u32, 4) };
            if w.iter().all(|&x| x == 0) {
                empty += 1;
            } else if w[3] & 0xFFF == k {
                own += 1;
            } else {
                other += 1;
                if examples.len() < 8 {
                    examples.push((k, w[3] & 0xFFF));
                }
            }
        }
        crate::xlog_info!(
            "frame trace: sampler heap {:#x} ({} slots, {}): {own} with their own index, {other} with another (slot, index: {examples:?}), {empty} empty",
            h.obj,
            h.n,
            if h.visible { "shader visible" } else { "CPU only" }
        );
    }
    for h in heaps.iter().filter(|h| !h.sampler) {
        let (mut empty, mut own, mut other, mut buffers) = (0u32, 0u32, 0u32, 0u32);
        let mut examples: Vec<(u32, u32)> = Vec::new();
        let own_index = |k: u32| match h.placed.get(&k) {
            Some(&f) => f,
            None if h.shared => NULL_INDEX,
            None => 2 * k,
        };
        for k in 0..h.n {
            let w = unsafe { std::slice::from_raw_parts((h.shadow + (k * SLOT) as usize) as *const u32, 8) };
            if w.iter().all(|&x| x == 0) {
                empty += 1;
            } else if w[3] >> 28 >= 8 {
                if w[0] == own_index(k) {
                    own += 1;
                } else {
                    other += 1;
                    if examples.len() < 8 {
                        examples.push((k, w[0]));
                    }
                }
            } else {
                buffers += 1;
            }
        }
        crate::xlog_info!(
            "frame trace: view heap {:#x} ({} slots, {}): {own} images with their own PC index, {other} with another (slot, index: {examples:?}), {buffers} buffers, {empty} empty",
            h.obj,
            h.n,
            if h.visible { "shader visible" } else { "CPU only" }
        );
    }
}

/// The descriptor at a console GPU address (low 32 bits, as in a descriptor table), if it
/// is in one of the console heaps.
pub fn words_at(addr32: u32) -> Option<[u32; 8]> {
    let u = crate::gpu::uma::get()?;
    let s = slot((u.base() + addr32 as usize) as u64)?;
    let mut w = [0u32; 8];
    unsafe { std::ptr::copy_nonoverlapping(s.at as *const u32, w.as_mut_ptr(), if s.sampler { 4 } else { 8 }) };
    Some(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ranges of the shared heap: released ones first (first fit, split), then new ones, and
    /// none past its size.
    #[test]
    fn shared_heap_ranges() {
        let (mut free, mut next) = (Vec::new(), 0u32);
        assert_eq!(take_range(&mut free, &mut next, 100, 40), Some(0));
        assert_eq!(take_range(&mut free, &mut next, 100, 40), Some(40));
        assert_eq!(take_range(&mut free, &mut next, 100, 40), None, "20 left");
        free.push((0, 40));
        assert_eq!(take_range(&mut free, &mut next, 100, 30), Some(0));
        assert_eq!(free, vec![(30, 10)]);
        assert_eq!(take_range(&mut free, &mut next, 100, 20), Some(80));
        assert_eq!(take_range(&mut free, &mut next, 100, 10), Some(30));
        assert!(free.is_empty());
    }

    #[test]
    fn slots_copy_and_release() {
        let Some(u) = crate::gpu::uma::get() else { return };
        let rel = |p: u64| (p as usize - u.base()) as u32;
        // views: 4 slots over a PC heap at 0x1000 (increment 32, GPU 0x5000); samplers: 2 at 0x9000
        assert!(register(0xA1, 4, 4, 0x1000, 0x5000, 32, false, false, true));
        assert!(register(0xA2, 2, 2, 0x9000, 0, 16, true, false, true));
        let (v, sm) = (start(0xA1).unwrap(), start(0xA2).unwrap());
        assert_eq!(u.window_of(v as usize), Some(0), "view heap must live in window 0 (32-bit table addressing)");
        assert_eq!(u.window_of(sm as usize), Some(0), "sampler heap must live in window 0");
        assert!(u.is_sparse(v as usize) && u.is_sparse(sm as usize), "descriptor heaps backed by sparse VRAM");
        let s1 = slot(v + 32).unwrap();
        assert_eq!((s1.index, s1.pc_srv, s1.pc_uav, s1.pc_gpu_uav), (2, 0x1040, 0x1060, 0x5060));
        // a 2D T# from slot 1 to slot 3: word 0 = the destination's SRV index
        write(&s1, &[2, 0x0470_0000, 0, 9 << 28, 0, 0, 0, 0]);
        let s3 = slot(v + 96).unwrap();
        copy_words(&s1, &s3);
        assert_eq!(words_at(rel(v + 96)).unwrap()[..4], [6, 0x0470_0000, 0, 9 << 28]);
        // a null T# keeps the null pool's index when it is copied
        let nullw = crate::gpu::desc::null_image_words(NULL_INDEX, crate::gpu::desc::ImageType::D2, 0);
        write(&s1, &nullw);
        copy_words(&s1, &s3);
        assert_eq!(words_at(rel(v + 96)).unwrap(), nullw, "a null view stays the null of its kind");
        // a buffer V# keeps its address
        write(&s1, &[0x1234, 0, 64, 0x3101_6FAC, 0, 0, 0, 0]);
        copy_words(&s1, &s3);
        assert_eq!(words_at(rel(v + 96)).unwrap()[0], 0x1234);
        // a raytracing BVH SRSRC keeps its base address and box-sort flag verbatim
        let bvh = crate::gpu::desc::bvh_words(0x12_3456_7800, true);
        write(&s1, &[bvh[0], bvh[1], bvh[2], bvh[3], 0, 0, 0, 0]);
        copy_words(&s1, &s3);
        assert_eq!(words_at(rel(v + 96)).unwrap()[..4], bvh);
        // samplers: 16-byte slots, S# word 3 [11:0] = sampler index
        let (q0, q1) = (slot(sm).unwrap(), slot(sm + 16).unwrap());
        assert!(q1.sampler && q1.index == 1 && q1.pc_srv == 0x9010);
        write(&q0, &[1, 2, 3, 0xABC0_0000]);
        copy_words(&q0, &q1);
        assert_eq!(words_at(rel(sm + 16)).unwrap()[..4], [1, 2, 3, 0xABC0_0001]);
        assert!(slot(sm + 32).is_none(), "past the last sampler slot");
        unregister(0xA1);
        unregister(0xA2);
        assert!(start(0xA1).is_none() && slot(v).is_none() && slot(sm).is_none());
    }

    /// A sampler slot nothing was written to is the point/wrap sampler at its own index: a
    /// table entry the title left empty samples as a defined state, not as slot 0's sampler.
    #[test]
    fn empty_sampler_slots_are_point_wrap_at_their_own_index() {
        let Some(u) = crate::gpu::uma::get() else { return };
        let rel = |p: u64| (p as usize - u.base()) as u32;
        assert!(register(0xA3, 6, 6, 0x9000, 0, 16, true, false, true));
        let sm = start(0xA3).unwrap();
        for k in 0..6u64 {
            let w = words_at(rel(sm + k * SAMPLER_SLOT as u64)).unwrap();
            assert_eq!(&w[..4], &[0, 0, 0, k as u32], "slot {k} is the point/wrap sampler at its own index");
        }
        unregister(0xA3);
    }

    /// A sampler heap above the PC limit keeps its first slots at their own index and maps the
    /// slots above them into the scratch descriptors at the top of the heap – every index
    /// inside the PC heap, and the slots a table straddles do not alias each other.
    #[test]
    fn sampler_heaps_above_the_pc_limit_map_into_scratch() {
        let cap = 16;
        let (keep, scratch) = sampler_layout(24, cap);
        assert_eq!((keep, scratch), (12, 4));
        let idx: Vec<u32> = (0..24).map(|k| sampler_index(cap, keep, scratch, k)).collect();
        assert!(idx.iter().all(|&i| i < cap), "every PC index is inside the capped heap");
        assert_eq!(&idx[..12], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        // the slots around the boundary a table crosses do not share a descriptor
        assert_eq!(&idx[10..16], &[10, 11, 12, 13, 14, 15]);
        // a heap that fits: every slot keeps its own index
        assert_eq!(sampler_layout(8, cap), (8, 0));
        assert_eq!((0..8).map(|k| sampler_index(cap, 8, 0, k)).collect::<Vec<_>>(), (0..8).collect::<Vec<_>>());
        // never an index past the heap
        assert!(idx.iter().all(|&i| i < cap));
    }

    /// A view slot takes its own run of the shared heap; when the cap is reached it reads the
    /// universal null instead of an index that belongs to another heap (an alias: the sampled
    /// pixel then came from a texture the draw never bound).
    #[test]
    fn a_table_that_crosses_the_cap_never_aliases() {
        const CAP: u32 = 32;
        fn place(free: &mut Vec<(u32, u32)>, next: &mut u32, k: u32, out: &mut Vec<(u32, u32)>) -> Option<u32> {
            let i = take_range(free, next, CAP, SLOT_DESCRIPTORS)?;
            out.push((k, i));
            Some(i)
        }
        let (mut free, mut next, mut placed) = (Vec::new(), 0u32, Vec::new());
        // a table of 20 slots starting below the cap: the first 16 get runs, the rest none
        let got: Vec<(u32, Option<u32>)> = (0..20).map(|k| (k, place(&mut free, &mut next, k, &mut placed))).collect();
        assert!(got[..16].iter().all(|(_, i)| i.is_some()), "slots below the cap get a place");
        assert!(got[16..].iter().all(|(_, i)| i.is_none()), "slots past the cap get none");
        let idx: Vec<u32> = got.iter().filter_map(|(_, i)| *i).collect();
        assert!(idx.iter().all(|&i| i < CAP), "every table index lies inside the cap");
        let mut uniq = idx.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq, idx, "no two live slots share a descriptor");
        // a released run is reused (the last one placed), a live one is not
        let last = *idx.last().unwrap();
        free.push((last, SLOT_DESCRIPTORS));
        assert_eq!(place(&mut free, &mut next, 20, &mut placed), Some(last));
        assert_eq!(place(&mut free, &mut next, 21, &mut placed), None, "the cap is full again");
    }

    /// Minecraft's view heaps: eight of 73 727 slots. Each used slot takes its own two
    /// descriptors, so the slots a heap fills never need the whole heap's worth of the shared
    /// heap, and no slot shares an index with another.
    #[test]
    fn large_heaps_take_shared_slots_as_they_are_used() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        if crate::gpu::uma::get().is_none() {
            return;
        }
        let dev = unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("WARP");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            dev.expect("D3D12 device")
        };
        assert!(unsafe { shared_heap(&dev) });
        const N: u32 = 73_727;
        let objs: Vec<usize> = (0..8).map(|i| 0xB000 + i).collect();
        for &o in &objs {
            assert!(register(o, N, N, 0x10_0000, 0, 32, false, true, true));
            assert!(in_global(o));
        }
        let mut seen = Vec::new();
        for &o in &objs {
            let v = start(o).unwrap();
            // the first slot, one in the middle, the heap's last slot
            let s: Vec<Slot> = [0, N / 2, N - 1].iter().map(|&k| slot(v + (k * SLOT) as u64).unwrap()).collect();
            assert!(s.iter().all(|x| x.shared_srv != 0), "heap {o:#x}: a slot without a place in the shared heap");
            assert!(s[1].index > s[0].index && s[2].index > s[1].index, "a slot's run is its own");
            assert_eq!(s[0].pc_srv, 0x10_0000, "views are still made in the title's heap");
            assert_eq!(s[2].pc_srv, 0x10_0000 + (2 * (N - 1) * 32) as u64);
            for x in &s {
                assert!(!seen.contains(&x.index), "two slots share the PC index {}", x.index);
                seen.push(x.index);
            }
            // the same slot again: the same place
            assert_eq!(slot(v).unwrap().index, s[0].index);
            // only the slots used took a place, not the heap
            let placed = HEAPS.lock().unwrap().iter().find(|h| h.obj == o).unwrap().placed.len();
            assert_eq!(placed, 3, "heap {o:#x} placed {placed} slots");
        }
        for &o in &objs {
            unregister(o);
        }
        // two heaps of 4 slots take 8 descriptors each, not a whole page
        assert!(register(0xB100, 4, 4, 0x10_0000, 0, 32, false, true, true) && register(0xB101, 4, 4, 0x20_0000, 0, 32, false, true, true));
        let (a, b) = (slot(start(0xB100).unwrap()).unwrap(), slot(start(0xB101).unwrap()).unwrap());
        assert!(a.shared_srv != 0 && b.shared_srv != 0);
        assert_ne!(a.index, b.index, "two heaps' slots never share a descriptor");
        let a2 = slot(start(0xB100).unwrap() + SLOT as u64).unwrap();
        assert_ne!(a2.index, a.index, "a heap's slots have runs of their own");
        assert_ne!(a2.index, b.index);
        unregister(0xB100);
        unregister(0xB101);
    }

    /// An empty view slot reads a null of every kind: the universal null is a 1×1 2D T# whose
    /// SRV is the null pool's, and its fields are also a V# of no records.
    #[test]
    fn empty_view_slots_read_a_null_of_their_kind() {
        let w = crate::gpu::desc::null_image_words(NULL_INDEX, crate::gpu::desc::ImageType::D2, 0);
        let i = crate::gpu::desc::image(&w).expect("the universal null is a T#");
        assert_eq!((i.ty, i.width, i.height, i.depth), (crate::gpu::desc::ImageType::D2, 1, 1, 1));
        assert_eq!(i.format, crate::gpu::desc::null_class_format(0));
        assert_eq!(w[0], NULL_INDEX, "word 0 is the null pool's SRV, not index 0 of the window UAVs");
        let v = crate::gpu::desc::buffer(&[w[0], w[1], w[2], w[3]]);
        assert_eq!(v.num_records, 0, "the same words are a V# of no records: a buffer read returns zero");
        assert_eq!(w[2], 0);
    }

    /// A typed null SRV for every image type: a complete description (format and dimension),
    /// never a NULL one.
    #[test]
    fn null_srvs_carry_a_complete_description() {
        use crate::gpu::desc::ImageType as T;
        for ty in [T::D1, T::D2, T::D3, T::Cube, T::D1Array, T::D2Array, T::D2Msaa, T::D2MsaaArray] {
            for class in 0..3 {
                let d = null_srv_desc(ty, class);
                assert_eq!(d.Format, null_dxgi_format(class), "{ty:?} class {class}");
                assert_ne!(d.Format.0, 0, "a null SRV names a format");
                assert_eq!(d.ViewDimension, null_view_dimension(ty));
                assert_eq!(d.Shader4ComponentMapping, D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING);
                let u = null_uav_desc(ty, class);
                assert_ne!(u.Format.0, 0);
                if !matches!(ty, T::D2Msaa | T::D2MsaaArray) {
                    assert_ne!(u.ViewDimension.0, 0, "a null UAV names a dimension");
                }
            }
        }
    }
}
