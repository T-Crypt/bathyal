//! Textures a title places in its own memory (CreatePlacedResourceX): on the console the GPU
//! samples that memory, laid out as the console's layout library says (`hle::xg`, checked
//! against the console's own). Here a PC texture holds a detiled copy: uploaded when the
//! texture is created and again before the title's next ExecuteCommandLists once the CPU wrote
//! any of its pages (unified memory and the title's graphics reservations are write-watched,
//! `gpu::uma`, `kernelx::VirtualAlloc`) – textures streamed in after creation (mips loaded
//! later, physical pages mapped in, DirectStorage) included.
//!
//! The first upload is not done when the texture is made (a title makes thousands while it
//! streams – Fortnite's render thread then spent its time in CreatePlacedResourceX and its
//! render command fences ran late): it is pending until the title's next ExecuteCommandLists,
//! where all pending and rewritten textures go in batches on one list each.
//!
//! The copy waits for the GPU on an XCL queue of the device. Memory that is not readable
//! (reserved, not mapped yet) reads as zero. GPU writes into the title's memory (copies to the
//! address) are not seen; the title's own GPU work on the texture goes to the PC texture.
//!
//! A copy starts and ends in the state the title's submitted work left the texture in: its
//! transitions of these textures are noted per PC command list as they are recorded
//! (`note_transitions`) and applied, in submission order, when the list is submitted
//! (`submitted`) – per subresource once a transition names one.
//!
//! The PC texture is a reserved resource on 64 KiB pages of video memory, one per page of the
//! title's address, shared by every such texture placed there (`vram::place_title_texture`) –
//! a title streaming its world places new textures over the old ones' memory; where it cannot
//! be (an address not 64 KiB aligned, a texture the driver does not reserve) it is committed.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use crate::gpu::texture::{self, Subresource};
use crate::hle::xg;

/// GetWriteWatch with atomic reset of the reported writes.
const WRITE_WATCH_FLAG_RESET: u32 = 1;
#[link(name = "kernel32")]
extern "system" {
    fn GetWriteWatch(flags: u32, base: *const c_void, size: usize, addresses: *mut *mut c_void, count: *mut usize, granularity: *mut u32) -> u32;
}

/// (device, address its bytes are read from, layout, states) of an upload
type Upload = (usize, usize, Arc<xg::Texture>, Vec<D3D12_RESOURCE_STATES>);

struct Entry {
    device: usize,
    /// the title's address of the texture
    va: usize,
    /// where its bytes are read and its CPU writes watched: `va`, or – for a texture in a GPU
    /// view of the title's CPU memory with nothing mapped there – the same offset in the CPU
    /// view (`kernelx::mirrors`)
    src: usize,
    /// `va` lies in a GPU view: `src` is checked again when that view's mappings change
    in_gpu_view: bool,
    tex: Arc<xg::Texture>,
    /// the state the title's submitted work leaves the texture in (uploads leave it there):
    /// the creation state, then what its transitions give – one state for every subresource,
    /// or one per subresource (D3D12 order) while they differ
    states: Vec<D3D12_RESOURCE_STATES>,
    /// not uploaded yet (made since the last submit)
    pending: bool,
    /// the title's GPU work wrote it (CopyTextureRegion into it): its CPU view is stale from
    /// then on and no longer uploaded
    gpu_written: bool,
    /// the order it was placed in: textures placed over the same memory share its video memory
    /// (`vram::place_title_texture`), and as on the console the newest one's texels are what it
    /// holds – uploads go oldest first
    seq: u64,
}

/// the next texture's `Entry::seq`
static SEQ: AtomicU64 = AtomicU64::new(0);

/// textures waiting for their first upload
static PENDING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

// raw COM pointers of the title's objects; uploads hold a reference while they run
unsafe impl Send for Entry {}

/// PC texture → its place in the title's memory
static TEXTURES: Mutex<BTreeMap<usize, Entry>> = Mutex::new(BTreeMap::new());
// The dirty-page scan consumes write-watch state for every submission thread.
// Keep the scan, pending-list update and GPU uploads together.
static REFRESH: Mutex<()> = Mutex::new(());
static TEXTURES_GEN: AtomicU64 = AtomicU64::new(0);
static TEXTURES_LEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// cached `watch_ranges` keyed by `(sparse::mapped_gen, TEXTURES_GEN, graphics_reservations_gen, sparse::visible_gen)`
static WATCH_CACHE: Mutex<Option<((u64, u64, u64, u64), Arc<Vec<(usize, usize)>>)>> = Mutex::new(None);
/// the queue uploads go to, per device
static QUEUES: Mutex<BTreeMap<usize, ID3D12CommandQueue>> = Mutex::new(BTreeMap::new());
static UPLOADS: AtomicU64 = AtomicU64::new(0);

/// Number of textures kept in title memory and uploads done so far.
pub fn counts() -> (usize, u64) {
    (TEXTURES.lock().unwrap().len(), UPLOADS.load(Ordering::Relaxed))
}

/// The console layout of a texture the title places (its console desc `desc`) when it can be
/// kept in step with the memory (`register`); Err: why not.
pub(super) fn layout(desc: &[u8; 0x38]) -> Result<xg::Texture, String> {
    let tex = xg::Texture::new(desc).map_err(|hr| format!("layout {hr:#x}"))?;
    // the console's 32-bit 116 / 117 floats are half floats on the PC (`pc_resource_desc`): the
    // title's bytes are not the resource's texels
    if matches!(u32::from_le_bytes(desc[0x20..0x24].try_into().unwrap()), 116 | 117) {
        return Err("a console float format the PC holds as R16G16B16A16_FLOAT".into());
    }
    // Render targets and depth buffers are the GPU's (their metadata planes, empty planes) –
    // except a small single-sample colour target with texels: a retail title makes its glyph
    // caches that way (256x256 R8 with ALLOW_RENDER_TARGET, two planes) and fills them from the
    // CPU. Left to the GPU they stayed as they were made and its menus' body text lost most of
    // its letters. Such a target is uploaded like any texture when the CPU wrote its pages; one
    // the title only renders to is not CPU-written and so not uploaded over.
    // `XCL_SYNC_SMALL_TARGETS=0` turns this off.
    let small_target = {
        let (w, h) = (u64::from_le_bytes(desc[0x10..0x18].try_into().unwrap()), u32::from_le_bytes(desc[0x18..0x1c].try_into().unwrap()));
        let (samples, flags) = (u32::from_le_bytes(desc[0x24..0x28].try_into().unwrap()), u32::from_le_bytes(desc[0x30..0x34].try_into().unwrap()));
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let off = *OFF.get_or_init(|| std::env::var("XCL_SYNC_SMALL_TARGETS").is_ok_and(|v| v.trim() == "0"));
        // (64x64 and up: 1x1 placeholders are not worth an entry; 0x2: a depth buffer)
        !off && w <= 1024 && h <= 1024 && w * h as u64 >= 64 * 64 && samples <= 1 && flags & 0x2 == 0
    };
    if tex.texels().is_none() || (tex.planes.len() != 1 && !small_target) {
        return Err("a render target / depth buffer".into());
    }
    if tex.planes.len() != 1 {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 2 || n.is_power_of_two() {
            crate::xlog_info!("D3D12.X: a small render target ({} planes) is kept in step with the title's memory – a glyph cache the CPU fills ({n} so far)", tex.planes.len());
        }
    }
    Ok(tex)
}

/// A texture the title placed at `va` (its console layout `tex`, from `layout`): kept in step
/// with the memory.
pub(super) unsafe fn register(device: *mut c_void, res: *mut c_void, va: usize, tex: xg::Texture, state: D3D12_RESOURCE_STATES) -> Result<(), String> {
    if ID3D12Resource::from_raw_borrowed(&res).is_none() {
        return Err("no resource".into());
    }
    let src = mirror_source(va);
    if src != va {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed) + 1;
        if n <= 2 || n.is_power_of_two() {
            crate::xlog_info!("D3D12.X: a texture at {va:#x} lies in a GPU view of the title's CPU memory: kept in step with {src:#x} ({n} so far)");
        }
    }
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let in_gpu_view = crate::hle::kernelx::mirrors::source(va).is_some();
    let e = Entry { device: device as usize, va, src, in_gpu_view, tex: Arc::new(tex), states: vec![state], pending: true, gpu_written: false, seq };
    {
        let mut t = TEXTURES.lock().unwrap();
        if let Some(old) = t.insert(res as usize, e) {
            if old.pending {
                PENDING.fetch_sub(1, Ordering::Relaxed);
            }
        }
        TEXTURES_LEN.store(t.len(), Ordering::Relaxed);
        PENDING.fetch_add(1, Ordering::Relaxed);
    }
    TEXTURES_GEN.fetch_add(1, Ordering::Release);
    Ok(())
}

/// Where a texture at `va` is read from: the CPU view behind a GPU view (`kernelx::mirrors`)
/// when nothing is mapped at `va` itself, else `va`. A physical page mapped at `va` counts
/// before its first use too, while it is still reserved memory (`kernelx::physical_page_at`).
fn mirror_source(va: usize) -> usize {
    let Some(src) = crate::hle::kernelx::mirrors::source(va) else { return va };
    if crate::hle::kernelx::physical_page_at(va) {
        return va;
    }
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let mapped = unsafe { VirtualQuery(va as *const c_void, &mut m, std::mem::size_of_val(&m)) } != 0 && m.State == MEM_COMMIT;
    if mapped { va } else { src }
}

/// Physical pages were mapped or unmapped in a GPU view since the last look: textures there
/// whose source changed (a page now mapped under one, or gone from it) read from the new one,
/// uploaded again at this submit.
fn recheck_gpu_view_sources() {
    static SEEN: AtomicU64 = AtomicU64::new(0);
    let gen = crate::hle::kernelx::gpu_view_map_gen();
    if SEEN.swap(gen, Ordering::AcqRel) == gen {
        return;
    }
    let mut changed = false;
    for e in TEXTURES.lock().unwrap().values_mut() {
        changed |= recheck_source(e);
    }
    if changed {
        TEXTURES_GEN.fetch_add(1, Ordering::Release);
    }
}

/// `e`'s source again (a texture in a GPU view): when it changed, `e` reads the new one and is
/// uploaded at the next submit. Whether it changed.
fn recheck_source(e: &mut Entry) -> bool {
    if !e.in_gpu_view {
        return false;
    }
    let src = mirror_source(e.va);
    if src == e.src {
        return false;
    }
    e.src = src;
    if !e.pending {
        e.pending = true;
        PENDING.fetch_add(1, Ordering::Relaxed);
    }
    true
}

/// The title's GPU work writes texture `res` (a copy into it): on the console the memory then
/// holds what the GPU wrote; here the CPU's view of it is stale, so it is not uploaded again
/// (a CPU write elsewhere on its pages – a neighbour, an allocator's header – would upload the
/// stale view over the copy: Fortnite's font atlas is zeros in memory, its glyphs are copied in
/// by the GPU). The first upload, if still pending, stays: it runs before the title's submit
/// that copies.
pub fn note_gpu_write(res: usize) {
    if let Some(e) = TEXTURES.lock().unwrap().get_mut(&res) {
        if !e.gpu_written {
            e.gpu_written = true;
            GPU_WRITTEN.fetch_add(1, Ordering::Relaxed);
        }
    }
}

static GPU_WRITTEN: AtomicU64 = AtomicU64::new(0);

/// The textures kept in step whose memory `[va, va + len)` overlaps (PC texture, its address).
pub fn overlapping(va: usize, len: usize) -> Vec<(usize, usize)> {
    TEXTURES.lock().unwrap().iter().filter(|(_, e)| e.va < va + len && va < e.va + e.tex.size() as usize).map(|(&r, e)| (r, e.va)).take(4).collect()
}

/// The PC texture was released for good.
pub fn forget(res: usize) {
    let removed = {
        let mut t = TEXTURES.lock().unwrap();
        let r = t.remove(&res);
        TEXTURES_LEN.store(t.len(), Ordering::Relaxed);
        r
    };
    if let Some(e) = removed {
        if e.pending {
            PENDING.fetch_sub(1, Ordering::Relaxed);
        }
        TEXTURES_GEN.fetch_add(1, Ordering::Release);
    }
}

/// The title-memory address of a texture kept in step with it.
pub fn address(res: usize) -> Option<usize> {
    TEXTURES.lock().unwrap().get(&res).map(|e| e.va)
}

/// (texture, subresource or ALL_SUBRESOURCES, state after) of a transition the title recorded
type Transition = (usize, u32, D3D12_RESOURCE_STATES);

/// PC command list → the transitions of textures kept in step it recorded since it was made or
/// last reset, in order. They apply when the list is submitted – each time: a closed list may
/// run again.
static LIST_NOTES: Mutex<BTreeMap<usize, Vec<Transition>>> = Mutex::new(BTreeMap::new());

/// Whether any texture is kept in step (callers skip collecting transitions otherwise).
pub fn any() -> bool {
    TEXTURES_LEN.load(Ordering::Relaxed) != 0
}

/// The title recorded `transitions` on PC list `list`: those of textures kept in step are noted
/// for when the list is submitted.
pub fn note_transitions(list: usize, transitions: &[Transition]) {
    if !any() || transitions.is_empty() {
        return;
    }
    let kept: Vec<Transition> = {
        let t = TEXTURES.lock().unwrap();
        transitions.iter().filter(|(r, _, _)| t.contains_key(r)).copied().collect()
    };
    if !kept.is_empty() {
        LIST_NOTES.lock().unwrap().entry(list).or_default().extend(kept);
    }
}

/// PC list `list` starts a new recording (made, reset, released): its noted transitions go.
pub fn list_reset(list: usize) {
    let mut g = LIST_NOTES.lock().unwrap();
    if !g.is_empty() {
        g.remove(&list);
    }
}

/// The PC lists `lists` were submitted, in this order: the textures kept in step are in the
/// states their transitions leave them in.
pub fn submitted(lists: &[*mut c_void]) {
    let notes: Vec<Transition> = {
        let g = LIST_NOTES.lock().unwrap();
        if g.is_empty() {
            return;
        }
        lists.iter().filter_map(|&l| g.get(&(l as usize))).flatten().copied().collect()
    };
    if notes.is_empty() {
        return;
    }
    let mut t = TEXTURES.lock().unwrap();
    for (res, sub, after) in notes {
        if let Some(e) = t.get_mut(&res) {
            let n = subresource_count(&e.tex);
            apply_transition(&mut e.states, n, sub, after);
        }
    }
}

/// Subresources of a texture kept in step (as `subresources` makes them).
fn subresource_count(tex: &xg::Texture) -> usize {
    tex.texels().map_or(1, |s| s.array as usize * s.chain.levels.len())
}

/// `states` (one for all `n` subresources, or one each) after a transition of subresource `sub`
/// (or all of them) to `after`; a subresource the texture does not have is ignored.
fn apply_transition(states: &mut Vec<D3D12_RESOURCE_STATES>, n: usize, sub: u32, after: D3D12_RESOURCE_STATES) {
    if sub == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES || (n <= 1 && sub == 0) {
        *states = vec![after];
        return;
    }
    let k = sub as usize;
    if k >= n {
        return;
    }
    if states.len() != n {
        *states = vec![states[0]; n];
    }
    states[k] = after;
    if states.iter().all(|&s| s == after) {
        states.truncate(1);
    }
}

thread_local! {
    /// XCL's own submits on this thread (texture uploads): they read no title memory on the GPU,
    /// so they do not look for CPU writes first – that look is the title's submits' job
    static BUSY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Reusable `GetWriteWatch` output buffer (avoids 128 KiB alloc+zero+free per submit).
    static WATCH_BUF: std::cell::RefCell<Vec<*mut c_void>> = const { std::cell::RefCell::new(Vec::new()) };
    /// Reusable dirty page list for `refresh_once`.
    static DIRTY_BUF: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The number of watched-range computations so far (the covered-share line is logged now and
/// then).
static WATCH_COMPUTES: AtomicU64 = AtomicU64::new(0);

/// `XCL_WATCH_TILES=all`: watch every mapped tile instead of the tiles a GPU-visible object or a
/// GPU command covers (`sparse::visible_ranges`: XMem graphics allocations, placed buffers,
/// console and view heaps, XCL's own arenas, and the tiles copies name). The narrow set is the
/// default: on a retail title every mapped tile was about 3 GB and 23.7 ms a look.
/// It is not exact for a tile a title commits in a window without an allocation and then hands to
/// the GPU by raw address, which on the console is GPU memory as a whole.
fn watch_all_tiles() -> bool {
    static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *A.get_or_init(super::sparse::watch_all_tiles_env)
}

fn cached_watch_ranges(u: &crate::gpu::uma::Uma) -> Arc<Vec<(usize, usize)>> {
    let key = (
        super::sparse::mapped_gen(),
        TEXTURES_GEN.load(Ordering::Acquire),
        crate::hle::kernelx::graphics_reservations_gen(),
        super::sparse::visible_gen(),
    );
    {
        let g = WATCH_CACHE.lock().unwrap();
        if let Some((k, ref ranges)) = *g {
            if k == key {
                return Arc::clone(ranges);
            }
        }
    }
    // Recompute the watched ranges (textures kept in step + the window tiles to watch, clipped
    // to write-watched regions) only when the set of textures, tiles, allocations, or
    // reservations changed: GetWriteWatch walks every page of every range, so this set is what a
    // look costs.
    let mut regions = u.watch_regions();
    regions.extend(crate::hle::kernelx::graphics_reservations());
    let mut wanted: Vec<(usize, usize)> = TEXTURES.lock().unwrap().values().map(|e| (e.src & !0xFFF, (e.src + e.tex.size() as usize + 0xFFF) & !0xFFF)).collect();
    let mapped = super::sparse::mapped_ranges();
    let all = watch_all_tiles();
    let mut tiles = if all { mapped.clone() } else { super::sparse::visible_ranges() };
    // a title that has made no GPU-visible object yet: nothing would be watched, so watch all of
    // its tiles until one exists
    let fell_back = tiles.is_empty() && !mapped.is_empty();
    if fell_back {
        tiles = mapped.clone();
    }
    let n = WATCH_COMPUTES.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        let sum = |rs: &[(usize, usize)]| rs.iter().map(|r| r.1).sum::<usize>() >> 20;
        crate::xlog_info!(
            "write-watch: watching {} MB of the {} MB of mapped title tiles ({} of {} tile runs; {}); XCL_WATCH_TILES=all watches them all",
            sum(&tiles),
            sum(&mapped),
            tiles.len(),
            mapped.len(),
            if all { "every mapped tile" } else if fell_back { "no GPU-visible object yet, so every mapped tile" } else { "the tiles resources, heaps and copies cover" }
        );
    }
    wanted.extend(tiles);
    let computed = Arc::new(watch_ranges(&regions, wanted));
    *WATCH_CACHE.lock().unwrap() = Some((key, Arc::clone(&computed)));
    computed
}

/// Run `f` as XCL's own GPU work: ExecuteCommandLists inside it does not refresh.
pub(super) fn own_work<R>(f: impl FnOnce() -> R) -> R {
    let was = BUSY.with(|b| b.replace(true));
    let r = f();
    BUSY.with(|b| b.set(was));
    r
}

/// Whether the current thread is inside XCL's own internal GPU setup/upload/readback work.
pub(super) fn is_own_work() -> bool {
    BUSY.with(|b| b.get())
}

/// Before the title's GPU work: textures whose memory the CPU wrote since the last look are
/// uploaded again.
pub(super) unsafe fn refresh(queue: *mut c_void) {
    // the uploads' own ExecuteCommandLists come back here
    if BUSY.with(|b| b.get()) {
        return;
    }
    let _refresh = REFRESH.lock().unwrap();
    own_work(|| refresh_once(queue));
}

/// The write-watch ranges split into at most `WATCH_THREADS` parts of about equal size, or one
/// part when the set is small enough that threads would cost more than they save.
fn watch_parts(ranges: &[(usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    split_watch(ranges, watch_threads())
}

fn split_watch(ranges: &[(usize, usize)], threads: usize) -> Vec<Vec<(usize, usize)>> {
    const MIN_PARALLEL: usize = 256 << 20;
    let total: usize = ranges.iter().map(|r| r.1).sum();
    if ranges.len() < 2 || total < MIN_PARALLEL || threads < 2 {
        return vec![ranges.to_vec()];
    }
    let k = threads.min(ranges.len());
    let share = total.div_ceil(k);
    let mut parts: Vec<Vec<(usize, usize)>> = vec![Vec::new()];
    let mut acc = 0usize;
    for &r in ranges {
        if acc >= share && parts.len() < k {
            parts.push(Vec::new());
            acc = 0;
        }
        parts.last_mut().unwrap().push(r);
        acc += r.1;
    }
    parts
}

/// `XCL_WATCH_THREADS` (default 1 = the serial loop; the kernel serialises the walks, so more threads
/// measured no faster): threads that read the write-watch.
fn watch_threads() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("XCL_WATCH_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1).clamp(1, 32))
}

/// GetWriteWatch with reset over `ranges`: every written page's address goes to `dirty`.
///
/// When the buffer fills, the call reports the pages that fitted and leaves the rest dirty, and
/// the next call walks the whole range again from its base (measured: a 2 GiB range with 100k
/// dirty pages and a 16384-entry buffer costs one walk per bufferful). A look that finds more
/// pages than the buffer holds therefore costs several walks; the buffer grows to hold what the
/// range has, up to `MAX_WATCH_BUF` entries.
unsafe fn watch_into(ranges: &[(usize, usize)], buf: &mut Vec<*mut c_void>, dirty: &mut Vec<usize>, page: &mut u32) {
    /// 4M pages (16 GiB of 4 KiB pages) held as 32 MiB of pointers
    const MAX_WATCH_BUF: usize = 4 << 20;
    if buf.len() < 16384 {
        buf.resize(16384, std::ptr::null_mut());
    }
    for &(base, size) in ranges {
        loop {
            let mut n = buf.len();
            if GetWriteWatch(WRITE_WATCH_FLAG_RESET, base as *const c_void, size, buf.as_mut_ptr(), &mut n, page) != 0 {
                break;
            }
            dirty.extend(buf[..n].iter().map(|&p| p as usize));
            if n < buf.len() {
                break;
            }
            if buf.len() < MAX_WATCH_BUF {
                buf.resize((buf.len() * 4).min(MAX_WATCH_BUF), std::ptr::null_mut());
            }
        }
    }
}

type WatchJob = (Vec<(usize, usize)>, std::sync::mpsc::Sender<(Vec<usize>, u32)>);

/// The write-watch threads: a shared queue of jobs, each answered with (dirty pages, granularity).
struct WatchPool(Mutex<std::sync::mpsc::Sender<WatchJob>>);

impl WatchPool {
    fn send(&self, job: WatchJob) {
        let _ = self.0.lock().unwrap().send(job);
    }
}

fn watch_pool() -> &'static WatchPool {
    static POOL: std::sync::OnceLock<WatchPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<WatchJob>();
        let rx = Arc::new(Mutex::new(rx));
        for i in 0..watch_threads() {
            let rx = Arc::clone(&rx);
            let _ = std::thread::Builder::new().name(format!("xcl-watch-{i}")).spawn(move || {
                let mut buf: Vec<*mut c_void> = Vec::new();
                loop {
                    let job = rx.lock().unwrap().recv();
                    let Ok((ranges, reply)) = job else { return };
                    let mut found = Vec::new();
                    let mut page = 0u32;
                    unsafe { watch_into(&ranges, &mut buf, &mut found, &mut page) };
                    let _ = reply.send((found, page));
                }
            });
        }
        WatchPool(Mutex::new(tx))
    })
}

/// Whether this submit reads the write-watch. By default every submit does: a page the CPU wrote
/// between two submits (descriptors, buffers in sparse windows) must reach video memory before the
/// second one. GetWriteWatch walks every watched page, though (gigabytes of mapped sparse tiles on a
/// newer retail title: about 15 ms a call, several submits a frame), so `XCL_WATCH_PER_FRAME=1`
/// reads it only at the first submit after a present: faster, but a write made between two submits
/// of one frame reaches video memory one frame late.
fn scan_this_submit() -> bool {
    static PER_FRAME: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*PER_FRAME.get_or_init(|| std::env::var("XCL_WATCH_PER_FRAME").is_ok_and(|v| v != "0")) {
        return true;
    }
    static LAST: AtomicU64 = AtomicU64::new(u64::MAX);
    let now = super::present::presents();
    LAST.swap(now, Ordering::Relaxed) != now
}

unsafe fn refresh_once(queue: *mut c_void) {
    let Some(u) = crate::gpu::uma::get() else { return };
    let sparse = u.has_sparse();
    if !sparse && TEXTURES_LEN.load(Ordering::Relaxed) == 0 {
        return;
    }
    recheck_gpu_view_sources();
    let watch_t0 = std::time::Instant::now();
    let mut page = 0x1000u32;
    let ranges = cached_watch_ranges(u);
    let scan = scan_this_submit();
    let dirty_len = DIRTY_BUF.with(|dcell| {
        let mut dirty = dcell.borrow_mut();
        dirty.clear();
        if !scan {
            return 0;
        }
        let parts = watch_parts(&ranges);
        if parts.len() <= 1 {
            WATCH_BUF.with(|wcell| watch_into(&ranges, &mut wcell.borrow_mut(), &mut dirty, &mut page));
        } else {
            // Large watched sets: GetWriteWatch walks every page of a range in the kernel, one
            // range after another, and that walk held the title's submitting thread for most of
            // its frame. The ranges are split over a pool of long-lived threads (created once,
            // so the title's per-thread setup runs only for them); each range is still read and
            // reset by exactly one call, so the dirty set is the same as the serial loop's.
            let (tx, rx) = std::sync::mpsc::channel();
            let n = parts.len();
            for part in parts {
                watch_pool().send((part, tx.clone()));
            }
            drop(tx);
            for _ in 0..n {
                if let Ok((found, g)) = rx.recv() {
                    dirty.extend(found);
                    if g != 0 {
                        page = g;
                    }
                }
            }
        }
        dirty.sort_unstable();
        dirty.len()
    });
    super::sparse::note_watch(
        watch_t0.elapsed(),
        dirty_len,
        if scan { ranges.len() } else { 0 },
        if scan { ranges.iter().map(|r| r.1).sum() } else { 0 },
    );
    if super::present::trace_frame() {
        // every 600th look: what the CPU's view of a few textures holds now (does the title's
        // data reach the address at all, and is the texture still kept in step?)
        static LOOKS: AtomicU64 = AtomicU64::new(0);
        if LOOKS.fetch_add(1, Ordering::Relaxed) % 600 == 599 {
            let t = TEXTURES.lock().unwrap();
            // 16 samples of 4 KiB over the whole texture (GFX10 keeps the small levels first,
            // level 0 last)
            let content = |e: &Entry| -> usize {
                let size = e.tex.size() as usize;
                (0..16usize)
                    .map(|k| e.src + (size.saturating_sub(4096)) * k / 15)
                    .filter(|&a| crate::gpu::shader::readable(a, 4096))
                    .map(|a| std::slice::from_raw_parts(a as *const u8, 4096).iter().filter(|&&b| b != 0).count())
                    .sum()
            };
            let (mut with, mut without) = (0, 0);
            let mut shown = 0;
            for e in t.values() {
                let c = content(e);
                if c > 0 { with += 1 } else { without += 1 }
                if c > 0 && shown < 4 {
                    shown += 1;
                    crate::xlog_info!("frame trace: title texture at {:#x} ({} bytes) has data in memory ({c} of 65536 sampled bytes); GPU-written {}", e.va, e.tex.size(), e.gpu_written);
                }
            }
            crate::xlog_info!("frame trace: title textures: {with} with data in memory, {without} all zero; {dirty_len} dirty pages this look");
        }
    }
    // the title's own memory: what the CPU wrote goes to its video memory first
    if let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) {
        DIRTY_BUF.with(|dcell| {
            let dirty = dcell.borrow();
            super::sparse::before_execute(q, &dirty, page as usize);
            super::vram::before_execute(q);
        });
    }
    if dirty_len == 0 && PENDING.load(Ordering::Relaxed) == 0 {
        return;
    }
    // textures made since the last submit and those with a written page; a reference each
    // while they are uploaded
    let mut todo: Vec<(u64, ID3D12Resource, Upload)> = DIRTY_BUF.with(|dcell| {
        let dirty = dcell.borrow();
        let mut t = TEXTURES.lock().unwrap();
        t.iter_mut()
            .filter(|(_, e)| {
                if e.pending {
                    return true;
                }
                if e.gpu_written {
                    return false;
                }
                let i = dirty.partition_point(|&p| p + page as usize <= e.src);
                dirty.get(i).is_some_and(|&p| p < e.src + e.tex.size() as usize)
            })
            .filter_map(|(&res, e)| {
                if e.pending {
                    e.pending = false;
                    PENDING.fetch_sub(1, Ordering::Relaxed);
                }
                let p = res as *mut c_void;
                ID3D12Resource::from_raw_borrowed(&p).map(|r| (e.seq, r.clone(), (e.device, e.src, Arc::clone(&e.tex), e.states.clone())))
            })
            .collect()
    });
    upload_all(in_placement_order(&mut todo));
}

/// The uploads `todo` (placement order, PC texture, upload) oldest placement first: where
/// textures share video memory the newest one's upload is the last word.
fn in_placement_order<R, U>(todo: &mut Vec<(u64, R, U)>) -> Vec<(R, U)> {
    todo.sort_unstable_by_key(|t| t.0);
    todo.drain(..).map(|(_, r, u)| (r, u)).collect()
}

/// staging bytes of one batch at most
const BATCH_BYTES: usize = 256 << 20;

/// The uploads `todo`, in batches of up to `BATCH_BYTES` (one list and one wait each).
unsafe fn upload_all(todo: Vec<(ID3D12Resource, Upload)>) {
    static N: AtomicU64 = AtomicU64::new(0);
    let warn = |e: &str| {
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            crate::xlog_warn!("title texture: upload failed: {e}");
        }
    };
    let mut todo = todo.into_iter().peekable();
    while todo.peek().is_some() {
        let mut batch = Vec::new();
        let mut bytes = 0usize;
        let mut device = 0usize;
        while let Some((_, (dev, _, tex, _))) = todo.peek() {
            if !batch.is_empty() && (bytes + tex.size() as usize > BATCH_BYTES || *dev != device) {
                break;
            }
            let (r, (dev, src, tex, states)) = todo.next().unwrap();
            device = dev;
            bytes += tex.size() as usize;
            let subs = subresources(src, &tex);
            if super::present::trace_frame() {
                note_content(src, &r, &subs);
            }
            batch.push((r, subs, states));
        }
        let d = device as *mut c_void;
        let Some(dev) = ID3D12Device::from_raw_borrowed(&d) else { continue };
        let queue = match upload_queue(dev, device) {
            Ok(q) => q,
            Err(e) => {
                warn(&e);
                continue;
            }
        };
        // textures on pages: their tiles' mappings come first
        super::vram::before_execute(&queue);
        match own_work(|| texture::upload_batch(dev, &queue, &batch)) {
            Ok(failed) => {
                for (_, e) in failed {
                    warn(&e);
                }
                UPLOADS.fetch_add(batch.len() as u64, Ordering::Relaxed);
            }
            Err(e) => warn(&e),
        }
    }
}

/// `XCL_TRACE_FRAME`: how much of a title texture's top level is not zero, each time it is
/// uploaded (the first 48 uploads of textures of 64x64 and more).
fn note_content(va: usize, r: &ID3D12Resource, subs: &[texture::Subresource]) {
    static N: AtomicU64 = AtomicU64::new(0);
    let Some(top) = subs.first() else { return };
    let d = unsafe { r.GetDesc() };
    if d.Width < 64 || d.Height < 64 || N.fetch_add(1, Ordering::Relaxed) >= 48 {
        return;
    }
    let nonzero = top.data.iter().filter(|&&b| b != 0).count();
    crate::xlog_info!(
        "frame trace: title texture at {va:#x} ({}x{} format {}) uploaded: {nonzero} of {} bytes of level 0 not zero",
        d.Width,
        d.Height,
        d.Format.0,
        top.data.len()
    );
}

/// `XCL_TRACE_FRAME`: what the GPU holds in level 0 of title texture `res` (read back on XCL's
/// queue; the texture is taken to be in a shader-resource state) against the CPU's view of the
/// memory now. Each texture once, four at most.
pub unsafe fn probe_gpu_content(res: usize) {
    static SEEN: Mutex<Vec<usize>> = Mutex::new(Vec::new());
    {
        let mut g = SEEN.lock().unwrap();
        if g.contains(&res) || g.len() >= 4 {
            return;
        }
        g.push(res);
    }
    let Some((device, va, src, tex, gpu_written)) = TEXTURES.lock().unwrap().get(&res).map(|e| (e.device, e.va, e.src, Arc::clone(&e.tex), e.gpu_written)) else { return };
    let d = device as *mut c_void;
    let r = res as *mut c_void;
    let (Some(dev), Some(r)) = (ID3D12Device::from_raw_borrowed(&d), ID3D12Resource::from_raw_borrowed(&r)) else { return };
    let rd = r.GetDesc();
    let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let (mut rows, mut row_bytes, mut total) = (0u32, 0u64, 0u64);
    dev.GetCopyableFootprints(&rd, 0, 1, 0, Some(&mut fp), Some(&mut rows), Some(&mut row_bytes), Some(&mut total));
    let result = own_work(|| -> Result<Vec<u8>, String> {
        let queue = upload_queue(dev, device)?;
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(|e| e.to_string())?;
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).map_err(|e| e.to_string())?;
        let mut rb: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: total,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut rb,
        )
        .map_err(|e| e.to_string())?;
        let rb = rb.ok_or("no readback buffer")?;
        let barrier = |a: D3D12_RESOURCE_STATES, b: D3D12_RESOURCE_STATES| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER { pResource: std::mem::transmute_copy(r), Subresource: 0, StateBefore: a, StateAfter: b }),
            },
        };
        let srv = D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE;
        list.ResourceBarrier(&[barrier(srv, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(r), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        list.ResourceBarrier(&[barrier(D3D12_RESOURCE_STATE_COPY_SOURCE, srv)]);
        list.Close().map_err(|e| e.to_string())?;
        queue.ExecuteCommandLists(&[Some(list.cast().map_err(|e| e.to_string())?)]);
        texture::wait(dev, &queue)?;
        let mut p: *mut c_void = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut p)).map_err(|e| e.to_string())?;
        let mut out = Vec::with_capacity(rows as usize * row_bytes as usize);
        for y in 0..rows as usize {
            out.extend_from_slice(std::slice::from_raw_parts((p as *const u8).add(y * fp.Footprint.RowPitch as usize), row_bytes as usize));
        }
        rb.Unmap(0, None);
        Ok(out)
    });
    let cpu = subresources(src, &tex).into_iter().next().map(|s| s.data).unwrap_or_default();
    match result {
        Ok(gpu) => {
            let nonzero = gpu.iter().filter(|&&b| b != 0).count();
            let same = gpu.iter().zip(cpu.iter()).filter(|(a, b)| a == b).count();
            crate::xlog_info!(
                "frame trace: GPU content of title texture {res:#x} at {va:#x} ({}x{} format {}, {} mips, GPU-written {gpu_written}): {nonzero} of {} bytes of level 0 not zero; {same} equal to the CPU view ({} of its bytes not zero); first bytes {:02x?}",
                rd.Width,
                rd.Height,
                rd.Format.0,
                rd.MipLevels,
                gpu.len(),
                cpu.iter().filter(|&&b| b != 0).count(),
                &gpu[..gpu.len().min(16)]
            );
        }
        Err(e) => crate::xlog_warn!("frame trace: GPU content of title texture {res:#x}: {e}"),
    }
}

/// XCL's queue for uploads on `dev`.
unsafe fn upload_queue(dev: &ID3D12Device, device: usize) -> Result<ID3D12CommandQueue, String> {
    let mut q = QUEUES.lock().unwrap();
    if let Some(q) = q.get(&device) {
        return Ok(q.clone());
    }
    let made: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).map_err(|e| e.to_string())?;
    q.insert(device, made.clone());
    Ok(made)
}

/// Ranges closer than this are scanned as one. A gap costs the walk over it, a call costs its
/// fixed cost: about 6.9 µs a call and 0.35 µs a MiB walked on a 2 GiB synthetic reservation of
/// idle memory (the same bytes in one call are 13× cheaper than in 512 calls), but about 7.8 µs
/// a MiB over the memory a running title has just written, with 13.5 calls a submit – so on a
/// title's own set the break-even gap is near 1 MiB, not 20.
///
/// `XCL_WATCH_GAP_MB` overrides it (1-1024). Any value is exact – a gap is only walked, never
/// read for writes it cannot report – so it is a cost knob: on a fragmented set of watched tiles
/// a smaller gap buys fewer bytes walked, on a sparse one a larger gap buys fewer calls.
/// `cargo test -q --lib watch_bench -- --ignored --nocapture` prints both costs with the memory
/// written through, and the log's "ranges" count in the sparse-window timing line shows what the
/// title's own layout does.
const WATCH_GAP: usize = 1 << 20;

fn watch_gap() -> usize {
    static G: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *G.get_or_init(|| {
        std::env::var("XCL_WATCH_GAP_MB")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map_or(WATCH_GAP, |mb| mb.clamp(1, 1024) << 20)
    })
}

/// The parts of `wanted` (start, end) inside `regions` (base, size), merged (gaps under
/// `WATCH_GAP` too) and sorted, as (base, size) – each within one region (a write-watch call
/// covers one allocation).
fn watch_ranges(regions: &[(usize, usize)], wanted: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    watch_ranges_gap(regions, wanted, watch_gap())
}

fn watch_ranges_gap(regions: &[(usize, usize)], mut wanted: Vec<(usize, usize)>, gap: usize) -> Vec<(usize, usize)> {
    wanted.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, b) in wanted {
        match merged.last_mut() {
            Some(last) if a <= last.1.saturating_add(gap) => last.1 = last.1.max(b),
            _ => merged.push((a, b)),
        }
    }
    let mut out = Vec::new();
    for (a, b) in merged {
        for &(base, size) in regions {
            let (lo, hi) = (a.max(base), b.min(base + size));
            if lo < hi {
                out.push((lo, hi - lo));
            }
        }
    }
    out.sort_unstable();
    out
}

/// `len` bytes at `at`: the committed, readable pages copied, the rest zero (a streaming pool
/// has only some of its pages mapped).
unsafe fn read_committed(at: usize, len: usize) -> Vec<u8> {
    use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT};
    let mut out = vec![0u8; len];
    // (the coherent window is committed as a whole: no `VirtualQuery`, which sizes a region by
    // walking it)
    if crate::gpu::uma::get().is_some_and(|u| u.in_committed_window(at, len)) {
        std::ptr::copy_nonoverlapping(at as *const u8, out.as_mut_ptr(), len);
        return out;
    }
    let mut p = at;
    while p < at + len {
        let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
        if VirtualQuery(p as *const c_void, &mut m, std::mem::size_of_val(&m)) == 0 {
            break;
        }
        let end = (m.BaseAddress as usize + m.RegionSize).min(at + len);
        let readable = m.State == MEM_COMMIT && m.Protect & 0xEE != 0 && m.Protect & 0x100 == 0; // R / RW / WC / X-R…, no guard
        if readable {
            std::ptr::copy_nonoverlapping(p as *const u8, out.as_mut_ptr().add(p - at), end - p);
        }
        p = end;
    }
    out
}

/// Every subresource (D3D12 order) of a texture from the title's memory at `va`.
unsafe fn subresources(va: usize, tex: &xg::Texture) -> Vec<Subresource> {
    let Some(tex) = tex.texels() else { return Vec::new() };
    let c = &tex.chain;
    let bpe = tex.bpe as usize;
    if tex.depth > 0 {
        let mem = read_committed(va, tex.size() as usize);
        return c.levels.iter().enumerate().map(|(i, l)| Subresource { depth: l.depth, rows: l.height, row_bytes: l.width * bpe, data: c.detile_volume(&mem, i) }).collect();
    }
    let mut out = Vec::with_capacity(tex.array as usize * c.levels.len());
    for s in 0..tex.array as usize {
        let slice = read_committed(va + s * c.slice_bytes, c.slice_bytes);
        for (i, l) in c.levels.iter().enumerate() {
            out.push(Subresource { depth: 1, rows: l.height, row_bytes: l.width * bpe, data: c.detile_level(&slice, i, s) });
        }
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;

    /// A physical page mapped into a GPU view but not used yet (reserved until its first
    /// access, `kernelx::lazy`) is that address's own source, not the CPU view behind it.
    /// (Test by Kalmarv, review of PR #283.)
    #[test]
    fn a_lazily_mapped_page_in_a_gpu_view_is_its_own_source() {
        use crate::hle::kernelx;
        use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
        const VIEW: usize = 16 << 20;
        unsafe {
            let Some(_) = crate::gpu::uma::get() else { return };
            let cpu = VirtualAlloc(std::ptr::null(), VIEW, MEM_RESERVE, PAGE_READWRITE) as usize;
            let gpu = VirtualAlloc(std::ptr::null(), VIEW, MEM_RESERVE, PAGE_READWRITE) as usize;
            kernelx::mirrors::reserved(cpu, VIEW, 0x2040_2000, 0x04);
            kernelx::mirrors::reserved(gpu, VIEW, 0x2040_2000, 0x801);
            assert_eq!(mirror_source(gpu + 0x2_0000), cpu + 0x2_0000); // nothing mapped: the CPU view
            let (mut n, mut pages) = (1usize, [0usize; 1]);
            assert_eq!(kernelx::AllocateTitlePhysicalPages(std::ptr::null_mut(), 0, &mut n, pages.as_mut_ptr()), 1);
            let at = gpu + 0x2_0000;
            // a texture placed there before the page is mapped reads the CPU view …
            let mut d = [0u8; 0x38];
            d[0..4].copy_from_slice(&3u32.to_le_bytes());
            d[0x10..0x18].copy_from_slice(&64u64.to_le_bytes());
            d[0x18..0x1C].copy_from_slice(&64u32.to_le_bytes());
            d[0x1C..0x1E].copy_from_slice(&1u16.to_le_bytes());
            d[0x1E..0x20].copy_from_slice(&1u16.to_le_bytes());
            d[0x20..0x24].copy_from_slice(&28u32.to_le_bytes());
            d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
            d[0x2C..0x30].copy_from_slice(&0x105u32.to_le_bytes());
            let tex = Arc::new(xg::Texture::new(&d).unwrap());
            let mut e = Entry { device: 0, va: at, src: mirror_source(at), in_gpu_view: true, tex, states: vec![D3D12_RESOURCE_STATE_COMMON], pending: false, gpu_written: false, seq: 0 };
            assert_eq!(e.src, cpu + 0x2_0000);
            assert!(!recheck_source(&mut e), "nothing changed yet");
            assert_eq!(kernelx::MapTitlePhysicalPages(at as *mut c_void, 1, 0, 4, pages.as_mut_ptr()) as usize, at);
            // … and the page mapped under it later, uploaded again
            assert!(recheck_source(&mut e));
            assert_eq!(e.src, at);
            assert!(e.pending);
            PENDING.fetch_sub(1, Ordering::Relaxed);
            let before = mirror_source(at);
            (at as *mut u32).write_volatile(7); // its first use commits it
            let after = mirror_source(at);
            let mut none = [0usize; 1];
            kernelx::MapTitlePhysicalPages(at as *mut c_void, 1, 0, 4, none.as_mut_ptr());
            kernelx::FreeTitlePhysicalPages(std::ptr::null_mut(), 1, pages.as_mut_ptr());
            kernelx::mirrors::released(cpu);
            VirtualFree(cpu as *mut c_void, 0, MEM_RELEASE);
            VirtualFree(gpu as *mut c_void, 0, MEM_RELEASE);
            assert_eq!(after, at);
            assert_eq!(before, at, "a page mapped at the address but not used yet is its source too");
        }
    }

    const SRV: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0);
    const ALL: u32 = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    /// One state while every subresource has it, one each while they differ.
    #[test]
    fn transitions_per_subresource_and_back_to_one_state() {
        let mut s = vec![D3D12_RESOURCE_STATE_COMMON];
        apply_transition(&mut s, 4, ALL, SRV);
        assert_eq!(s, vec![SRV]);
        apply_transition(&mut s, 4, 2, D3D12_RESOURCE_STATE_COPY_SOURCE);
        assert_eq!(s, vec![SRV, SRV, D3D12_RESOURCE_STATE_COPY_SOURCE, SRV]);
        apply_transition(&mut s, 4, 9, D3D12_RESOURCE_STATE_COPY_DEST); // no subresource 9
        assert_eq!(s, vec![SRV, SRV, D3D12_RESOURCE_STATE_COPY_SOURCE, SRV]);
        apply_transition(&mut s, 4, 2, SRV);
        assert_eq!(s, vec![SRV]);
        apply_transition(&mut s, 4, 1, D3D12_RESOURCE_STATE_COPY_DEST);
        apply_transition(&mut s, 4, ALL, D3D12_RESOURCE_STATE_COMMON);
        assert_eq!(s, vec![D3D12_RESOURCE_STATE_COMMON]);
        // a texture of one subresource: subresource 0 is all of it
        let mut one = vec![SRV];
        apply_transition(&mut one, 1, 0, D3D12_RESOURCE_STATE_COPY_SOURCE);
        assert_eq!(one, vec![D3D12_RESOURCE_STATE_COPY_SOURCE]);
    }

    /// A texture kept in step: the title's transitions count once their list is submitted, in
    /// submission order, each time it is; a Reset forgets them; other resources are not noted.
    #[test]
    fn transitions_apply_when_their_list_is_submitted() {
        // RGBA8 64×64, 3 mips, 2 slices (6 subresources), 4KB_S
        let mut d = [0u8; 0x38];
        d[0..4].copy_from_slice(&3u32.to_le_bytes());
        d[0x10..0x18].copy_from_slice(&64u64.to_le_bytes());
        d[0x18..0x1C].copy_from_slice(&64u32.to_le_bytes());
        d[0x1C..0x1E].copy_from_slice(&2u16.to_le_bytes());
        d[0x1E..0x20].copy_from_slice(&3u16.to_le_bytes());
        d[0x20..0x24].copy_from_slice(&28u32.to_le_bytes());
        d[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
        d[0x2C..0x30].copy_from_slice(&0x105u32.to_le_bytes());
        let tex = Arc::new(xg::Texture::new(&d).unwrap());
        assert_eq!(subresource_count(&tex), 6);
        // fake resource / list keys, never dereferenced: GPU-written, so no refresh uploads it
        let (res, other, l1, l2) = (0x7E57_0000_1000usize, 0x7E57_0000_2000usize, 0x7E57_0000_3000usize, 0x7E57_0000_4000usize);
        TEXTURES.lock().unwrap().insert(res, Entry { device: 0, va: 0x7E57_0000_0000, src: 0x7E57_0000_0000, in_gpu_view: false, tex, states: vec![D3D12_RESOURCE_STATE_COMMON], pending: false, gpu_written: true, seq: 0 });
        TEXTURES_LEN.fetch_add(1, Ordering::Relaxed);
        let states = || TEXTURES.lock().unwrap().get(&res).unwrap().states.clone();
        let (p1, p2) = (l1 as *mut c_void, l2 as *mut c_void);

        note_transitions(l1, &[(res, ALL, SRV), (other, ALL, D3D12_RESOURCE_STATE_RENDER_TARGET), (res, 4, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        assert_eq!(LIST_NOTES.lock().unwrap().get(&l1).map(|v| v.len()), Some(2), "only the texture kept in step is noted");
        assert_eq!(states(), vec![D3D12_RESOURCE_STATE_COMMON], "recorded, not run yet");
        note_transitions(l2, &[(res, 4, SRV), (res, 0, D3D12_RESOURCE_STATE_COPY_DEST)]);
        submitted(&[p1]);
        let mut want = vec![SRV; 6];
        want[4] = D3D12_RESOURCE_STATE_COPY_SOURCE;
        assert_eq!(states(), want);
        submitted(&[p2]);
        want[4] = SRV;
        want[0] = D3D12_RESOURCE_STATE_COPY_DEST;
        assert_eq!(states(), want);
        // l1 runs again (a closed list may): its transitions again, after l2's
        submitted(&[p2, p1]);
        assert_eq!(states(), {
            let mut w = vec![SRV; 6];
            w[4] = D3D12_RESOURCE_STATE_COPY_SOURCE;
            w
        });
        list_reset(l1);
        list_reset(l2);
        submitted(&[p1, p2]);
        assert_eq!(states()[4], D3D12_RESOURCE_STATE_COPY_SOURCE, "a reset list has nothing to apply");
        forget(res);
        assert!(LIST_NOTES.lock().unwrap().get(&l1).is_none());
    }

    /// Only the wanted parts of the watched regions are scanned, merged, one region each.
    #[test]
    fn watch_ranges_clip_and_merge() {
        let regions = [(0x10000, 0x10000), (0x40000, 0x10000), (0x1000_0000, 0x1000_0000)];
        let g = |wanted, gap| watch_ranges_gap(&regions, wanted, gap);
        let wanted = vec![(0x12000, 0x13000), (0x11000, 0x12800), (0x1F000, 0x41000), (0x60000, 0x61000)];
        assert_eq!(g(wanted, WATCH_GAP), vec![(0x11000, 0xF000), (0x40000, 0x10000)]);
        // ranges 16 MiB apart: one scan at a 16 MiB gap, two at the default 1 MiB
        let wanted = vec![(0x1000_0000, 0x1000_1000), (0x1100_0000, 0x1100_1000), (0x1300_0000, 0x1300_1000)];
        assert_eq!(g(wanted.clone(), 16 << 20), vec![(0x1000_0000, 0x100_1000), (0x1300_0000, 0x1000)]);
        assert_eq!(g(wanted, WATCH_GAP), vec![(0x1000_0000, 0x1000), (0x1100_0000, 0x1000), (0x1300_0000, 0x1000)]);
        // a call costs more than the walk over a gap of up to `WATCH_GAP`: 8 MiB merges at 16, 40 MiB not
        let wanted = vec![(0x1000_0000, 0x1000_1000), (0x1080_1000, 0x1080_2000)];
        assert_eq!(g(wanted.clone(), 16 << 20), vec![(0x1000_0000, 0x80_2000)]);
        assert_eq!(g(wanted, 1 << 20), vec![(0x1000_0000, 0x1000), (0x1080_1000, 0x1000)]);
        let wanted = vec![(0x1000_0000, 0x1000_1000), (0x1280_0000, 0x1280_1000)];
        assert_eq!(g(wanted, 16 << 20), vec![(0x1000_0000, 0x1000), (0x1280_0000, 0x1000)]);
    }

    /// Whatever the gap, the same wanted bytes are watched: a gap is only walked, never a place a
    /// write can hide (the knob `XCL_WATCH_GAP_MB` is a cost choice, not a correctness one).
    #[test]
    fn any_gap_covers_every_wanted_byte() {
        let regions = [(0x1000_0000usize, 0x2000_0000usize)];
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let mut wanted: Vec<(usize, usize)> = Vec::new();
        for _ in 0..200 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let a = 0x1000_0000 + (x as usize % (0x1F00_0000 / 0x1000)) * 0x1000;
            let b = a + ((x >> 32) as usize % 300) * 0x1000;
            wanted.push((a, b));
        }
        let covered = |by: &[(usize, usize)]| -> bool {
            wanted.iter().all(|&(a, b)| by.iter().any(|&(lo, n)| lo <= a && b <= lo + n))
        };
        for gap in [0usize, 0x1000, 1 << 20, 4 << 20, 16 << 20, 256 << 20] {
            let out = watch_ranges_gap(&regions, wanted.clone(), gap);
            assert!(covered(&out), "gap {gap:#x}: wanted bytes missing {out:x?}");
            assert!(out.iter().all(|&(lo, n)| lo >= regions[0].0 && lo + n <= regions[0].0 + regions[0].1));
        }
        // and a bigger gap never makes more calls
        let calls = |gap: usize| watch_ranges_gap(&regions, wanted.clone(), gap).len();
        assert!(calls(256 << 20) <= calls(16 << 20) && calls(16 << 20) <= calls(0x1000));
    }
}

#[cfg(test)]
mod watch_parts_tests {
    use super::split_watch;

    #[test]
    fn small_sets_stay_serial_and_large_ones_split_without_losing_a_range() {
        let small = vec![(0x1000usize, 0x10000usize), (0x20000, 0x10000)];
        assert_eq!(split_watch(&small, 8), vec![small.clone()]);
        let big: Vec<(usize, usize)> = (0..40).map(|i| (0x1_0000_0000 + i * (64 << 20), 64 << 20)).collect();
        assert_eq!(split_watch(&big, 1), vec![big.clone()], "one thread: the serial loop");
        let parts = split_watch(&big, 8);
        assert!(parts.len() > 1 && parts.len() <= 8, "{} parts", parts.len());
        let flat: Vec<(usize, usize)> = parts.concat();
        assert_eq!(flat, big, "every range exactly once, in order");
    }
}

/// What the write-watch costs, measured on this machine (Windows only): the kernel walk over a
/// synthetic write-watched reservation, and how that cost splits into a fixed part per call, a
/// part per page and a part per dirty page. `XCL_BENCH_WATCH_MB` (default 512) is the size of the
/// reservation the benches use.
///
/// `cargo test -q --lib watch_bench -- --ignored --nocapture` runs the benches; the first test
/// runs in the normal suite (64 MiB).
#[cfg(all(test, windows))]
mod watch_bench {
    use super::{watch_into, GetWriteWatch, WRITE_WATCH_FLAG_RESET};
    use std::ffi::c_void;
    use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};

    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// `len` bytes of write-watched memory, freed on drop.
    struct Watched(*mut c_void, usize);

    impl Watched {
        fn new(len: usize) -> Self {
            let p = unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) };
            assert!(!p.is_null(), "VirtualAlloc({len}, MEM_WRITE_WATCH) failed");
            Watched(p, len)
        }

        /// write one byte of `pages` 4 KiB pages, spread over the range
        fn touch(&self, pages: usize) {
            let step = (self.1 / pages.max(1) & !0xFFF).max(0x1000);
            for k in 0..pages {
                unsafe { std::ptr::write_volatile((self.0 as *mut u8).add(k * step), 1) };
            }
        }
    }

    impl Drop for Watched {
        fn drop(&mut self) {
            unsafe { VirtualFree(self.0, 0, MEM_RELEASE) };
        }
    }

    fn mb() -> usize {
        std::env::var("XCL_BENCH_WATCH_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(512).clamp(16, 2048)
    }

    /// More dirty pages than the buffer starts with: `watch_into` grows the buffer and returns
    /// every page without a second walk over the range.
    #[test]
    fn more_dirty_pages_than_the_buffer_holds_are_all_returned() {
        let w = Watched::new(128 << 20);
        let pages = 20_000usize;
        w.touch(pages);
        let mut buf: Vec<*mut c_void> = Vec::new();
        let mut dirty: Vec<usize> = Vec::new();
        let mut gran = 0u32;
        unsafe { watch_into(&[(w.0 as usize, w.1)], &mut buf, &mut dirty, &mut gran) };
        dirty.sort_unstable();
        dirty.dedup();
        assert_eq!(dirty.len(), pages, "every written page reported once");
        assert!(buf.len() >= pages, "the buffer grew to hold the range's pages ({} entries)", buf.len());
    }

    #[test]
    #[ignore = "bench: up to 2 GiB, cost by bytes and by dirty pages"]
    fn get_write_watch_cost_by_bytes_and_dirty_pages() {
        let mb = mb();
        let w = Watched::new(mb << 20);
        let mut buf = vec![std::ptr::null_mut::<c_void>(); 1 << 16];
        for &dirty in &[0usize, 16, 1024, 16384] {
            w.touch(dirty);
            let t = std::time::Instant::now();
            let (mut n, mut gran) = (buf.len(), 0u32);
            let r = unsafe { GetWriteWatch(WRITE_WATCH_FLAG_RESET, w.0, w.1, buf.as_mut_ptr(), &mut n, &mut gran) };
            let d = t.elapsed().as_secs_f64();
            assert_eq!(r, 0, "GetWriteWatch failed");
            assert_eq!(n, dirty, "{dirty} pages written, {n} reported");
            println!(
                "watched {mb} MiB ({:.0} pages), {n} dirty: {:.2} ms ({:.2} ns/page, {:.3} µs/dirty page)",
                w.1 as f64 / 4096.0,
                d * 1e3,
                d * 1e9 / (w.1 as f64 / 4096.0),
                d * 1e6 / n.max(1) as f64
            );
        }
    }

    /// The same bytes in one call and in many: the fixed cost of a call.
    #[test]
    #[ignore = "bench: 512 MiB in 1 call vs calls of 1, 4 and 16 MiB"]
    fn cost_of_many_small_ranges() {
        let len = mb() << 20;
        let w = Watched::new(len);
        let mut buf = vec![std::ptr::null_mut::<c_void>(); 1 << 16];
        for chunk in [len, 16 << 20, 4 << 20, 1 << 20] {
            w.touch(64);
            let mut got = 0usize;
            let t = std::time::Instant::now();
            for base in (0..len).step_by(chunk) {
                let (mut n, mut gran) = (buf.len(), 0u32);
                let r = unsafe { GetWriteWatch(WRITE_WATCH_FLAG_RESET, (w.0 as *const u8).add(base) as *const c_void, chunk, buf.as_mut_ptr(), &mut n, &mut gran) };
                assert_eq!(r, 0);
                got += n;
            }
            let d = t.elapsed().as_secs_f64();
            println!(
                "{} MiB in {} calls of {} MiB: {:.2} ms, {got} pages ({:.1} µs a call)",
                len >> 20,
                len / chunk,
                chunk >> 20,
                d * 1e3,
                d * 1e6 / (len / chunk) as f64
            );
            assert_eq!(got, 64);
        }
    }

    /// A buffer that fills: the call reports what fitted and the rest stay dirty, so the next
    /// call walks the whole range again (why `watch_into` grows the buffer).
    #[test]
    #[ignore = "bench: 2 GiB, 100k dirty pages, small vs grown buffer"]
    fn a_full_buffer_costs_one_walk_per_bufferful() {
        let w = Watched::new(2048 << 20);
        let dirty = 100_000usize;
        for entries in [16384usize, 0] {
            w.touch(dirty);
            let mut buf = vec![std::ptr::null_mut::<c_void>(); if entries == 0 { dirty + 1024 } else { entries }];
            let t = std::time::Instant::now();
            let (mut got, mut calls, mut gran) = (0usize, 0usize, 0u32);
            loop {
                let mut n = buf.len();
                let r = unsafe { GetWriteWatch(WRITE_WATCH_FLAG_RESET, w.0, w.1, buf.as_mut_ptr(), &mut n, &mut gran) };
                assert_eq!(r, 0);
                got += n;
                calls += 1;
                if n < buf.len() {
                    break;
                }
            }
            println!("buffer {} entries: {calls} calls, {got} of {dirty} pages, {:.2} ms", buf.len(), t.elapsed().as_secs_f64() * 1e3);
            assert_eq!(got, dirty);
        }
    }

    /// What a look costs with memory in use, and what narrowing the set saves: the same
    /// synthetic window (64 KiB tiles, every other tile mapped, 1 in `XCL_BENCH_WATCH_COVERED` of
    /// them covered by a GPU-visible object) written through before every look, so the kernel's
    /// dirty list is as busy as a running title's, read as every mapped tile against as the
    /// covered ones only, at several gap sizes.
    ///
    /// `XCL_BENCH_WATCH_MB` (default 512) is the window, `XCL_BENCH_WATCH_COVERED` (default 8)
    /// the share of the mapped tiles that stays watched, `XCL_BENCH_WATCH_LOOKS` (default 16) the
    /// looks each configuration is timed over, `XCL_BENCH_WATCH_DIRTY` (default 716, the retail
    /// title's pages a look) the pages written before each look.
    #[test]
    #[ignore = "bench: covered tiles against every mapped tile, written through, at several gaps"]
    fn covered_tiles_cost_less_than_every_mapped_tile() {
        const TILE: usize = 0x1_0000;
        let w = Watched::new(mb() << 20);
        let base = w.0 as usize;
        let num = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let every = num("XCL_BENCH_WATCH_COVERED", 8).clamp(1, 64);
        let looks = num("XCL_BENCH_WATCH_LOOKS", 16).max(1);
        let dirty_per_look = num("XCL_BENCH_WATCH_DIRTY", 716).max(1);
        // a title's streaming pattern: every other 64 KiB tile has video memory
        let runs = |covered_only: bool| -> Vec<(usize, usize)> {
            let mut out: Vec<(usize, usize)> = Vec::new();
            for k in (0..w.1 / TILE).step_by(2) {
                if covered_only && k % (2 * every) != 0 {
                    continue;
                }
                let a = base + k * TILE;
                match out.last_mut() {
                    Some(last) if last.0 + last.1 == a => last.1 += TILE,
                    _ => out.push((a, TILE)),
                }
            }
            out
        };
        // the pages a look finds written: spread over the window, as a title's frame leaves them
        let stride = (w.1 / dirty_per_look & !0xFFF).max(0x1000);
        for (what, ranges) in [("every mapped tile", runs(false)), ("covered only", runs(true))] {
            for gap_mb in [16usize, 4, 1, 0] {
                let spans: Vec<(usize, usize)> = ranges.iter().map(|&(a, n)| (a, a + n)).collect();
                let calls = super::watch_ranges_gap(&[(base, w.1)], spans, gap_mb << 20);
                let walked: usize = calls.iter().map(|c| c.1).sum();
                let mut buf: Vec<*mut c_void> = Vec::new();
                let mut dirty: Vec<usize> = Vec::new();
                let mut page = 0u32;
                let mut spent = 0f64;
                let mut found = 0usize;
                for k in 0..looks {
                    // write this look's pages (the same bytes every look: the dirty list stays busy)
                    for p in 0..dirty_per_look {
                        unsafe { std::ptr::write_volatile((base + ((p * stride + k) % w.1 & !0xFFF)) as *mut u8, 1) };
                    }
                    dirty.clear();
                    let t = std::time::Instant::now();
                    unsafe { super::watch_into(&calls, &mut buf, &mut dirty, &mut page) };
                    spent += t.elapsed().as_secs_f64();
                    found += dirty.len();
                }
                let per = spent / looks as f64 * 1e3;
                let mib = (walked >> 20).max(1) as f64;
                println!(
                    "{what}: {} MiB of tiles in {} runs, gap {gap_mb} MiB -> {} calls, {} MiB walked: {per:.2} ms a look, {:.2} µs a MB, {} dirty pages a look",
                    ranges.iter().map(|r| r.1).sum::<usize>() >> 20,
                    ranges.len(),
                    calls.len(),
                    walked >> 20,
                    per * 1e3 / mib,
                    found / looks
                );
            }
        }
    }
}
