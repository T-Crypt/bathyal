//! Driver pipeline blobs (`ID3D12PipelineState::GetCachedBlob`) and the prewarm record.
//!
//! Translating a console shader is cached as DXIL. The driver then compiles that DXIL again
//! on every launch. A blob from `GetCachedBlob` is that compile: the next
//! `Create*PipelineState` passes it in `CachedPSO` and the driver skips the work when the
//! blob still matches this device. A driver update rejects the blob; the caller drops the
//! file and compiles again, and this module starts the run with an empty library.
//!
//! The console shaders a pipeline was drawn with are written to `drawn.txt` beside the blobs.
//! The next run builds those pipelines while the title loads instead of at their first use
//! (`pso_async::build_during_load`).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;

const MAGIC: &[u8; 4] = b"XPSO";
const VERSION: u32 = 1;

fn dir() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        if let Ok(d) = std::env::var("XCL_PSO_CACHE") {
            return PathBuf::from(d);
        }
        let base = std::env::var("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(|_| std::env::temp_dir());
        base.join("xcl").join("pso")
    })
    .clone()
}

fn memory() -> &'static Mutex<std::collections::HashMap<u64, std::sync::Arc<Vec<u8>>>> {
    static M: OnceLock<Mutex<std::collections::HashMap<u64, std::sync::Arc<Vec<u8>>>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// Blobs a pipeline's build took instead of the driver's compile, and builds that had to
/// compile (every built pipeline is one or the other; the log line reads them every 300
/// frames).
static HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// (pipelines built from a saved blob, pipelines the driver had to compile)
pub fn counts() -> (u64, u64) {
    (HITS.load(std::sync::atomic::Ordering::Relaxed), MISSES.load(std::sync::atomic::Ordering::Relaxed))
}

/// A pipeline build is about to try `key`: its blob was found (`hit`) or not.
pub fn note_lookup(hit: bool) {
    if hit {
        HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    } else {
        MISSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

fn path_for(dir: &Path, key: u64) -> PathBuf {
    dir.join(format!("{key:016x}.blob"))
}

fn drawn_path(dir: &Path) -> PathBuf {
    dir.join("drawn.txt")
}

fn payload(file: &[u8]) -> Option<&[u8]> {
    if file.len() < 8 || &file[..4] != MAGIC || u32::from_le_bytes(file[4..8].try_into().ok()?) != VERSION {
        return None;
    }
    let blob = &file[8..];
    (!blob.is_empty() && blob.len() <= 32 << 20).then_some(blob)
}

/// A blob saved for `key`, from this process or from disk.
pub fn load(key: u64) -> Option<std::sync::Arc<Vec<u8>>> {
    if let Some(hit) = memory().lock().unwrap().get(&key).cloned() {
        return Some(hit);
    }
    let blob = std::sync::Arc::new(load_in(&dir(), key)?);
    memory().lock().unwrap().insert(key, blob.clone());
    Some(blob)
}

/// A blob read from the library in `dir` (no in-memory cache: what a fresh run sees).
fn load_in(dir: &Path, key: u64) -> Option<Vec<u8>> {
    let bytes = std::fs::read(path_for(dir, key)).ok()?;
    payload(&bytes).map(<[u8]>::to_vec)
}

fn write_file(key: u64, blob: &[u8]) {
    save_in(&dir(), key, blob);
}

/// Write a blob into the library in `dir`, unless the same bytes are there already.
fn save_in(dir: &Path, key: u64, blob: &[u8]) {
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let path = path_for(dir, key);
    if std::fs::read(&path).ok().and_then(|b| payload(&b).map(|p| p == blob)).unwrap_or(false) {
        return;
    }
    let mut body = Vec::with_capacity(8 + blob.len());
    body.extend_from_slice(MAGIC);
    body.extend_from_slice(&VERSION.to_le_bytes());
    body.extend_from_slice(blob);
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&tmp, &body).is_ok() && std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

fn writer() -> &'static std::sync::mpsc::SyncSender<(u64, std::sync::Arc<Vec<u8>>)> {
    static W: OnceLock<std::sync::mpsc::SyncSender<(u64, std::sync::Arc<Vec<u8>>)>> = OnceLock::new();
    W.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<(u64, std::sync::Arc<Vec<u8>>)>(64);
        let _ = std::thread::Builder::new().name("xcl-pso-cache".into()).spawn(move || {
            while let Ok((key, blob)) = rx.recv() {
                write_file(key, &blob);
            }
        });
        tx
    })
}

/// Remember `blob` now and write it on a side thread, so a driver compile is not stuck on disk.
pub fn store(key: u64, blob: &[u8]) {
    if blob.is_empty() || blob.len() > 32 << 20 {
        return;
    }
    let owned = std::sync::Arc::new(blob.to_vec());
    memory().lock().unwrap().insert(key, owned.clone());
    if let Err(std::sync::mpsc::TrySendError::Full((_, blob))) = writer().try_send((key, owned)) {
        write_file(key, &blob);
    }
}

/// The driver refused this blob (a new driver, or a different pipeline). Do not try it again.
pub fn forget(key: u64) {
    memory().lock().unwrap().remove(&key);
    let _ = std::fs::remove_file(path_for(&dir(), key));
}

/// The device refused a saved blob. A new driver refuses the others too, so the blobs this
/// run has read go with it and the library starts empty; the run logs it once.
pub fn rejected(key: u64) {
    forget(key);
    memory().lock().unwrap().clear();
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        crate::xlog_warn!(
            "D3D12.X: the device refused a saved pipeline blob – pipelines are compiled again this run; delete the pipeline library folder to start clean (XCL_PSO_CACHE names it)"
        );
    });
}

/// Identity of the console shaders a pipeline is built from: what the next run greets a
/// deferred pipeline by, before it is translated (the driver blob's key needs the DXIL, which
/// does not exist yet then). `blobs` is the pipeline's shaders in description order, empty
/// ones left out, as `real::own_*` copies them.
pub fn shaders_sig(blobs: &[&[u8]]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    mix(&mut h, b"sh1");
    for b in blobs {
        mix_u32(&mut h, b.len() as u32);
        mix(&mut h, b);
    }
    h
}

/// Console shader identity of pipelines built this run, by pipeline state.
static PSO_SHADERS: Mutex<Option<HashMap<usize, u64>>> = Mutex::new(None);
/// how many pipelines are in `PSO_SHADERS` (a bind skips the lock while none are)
static SHADERS_KNOWN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Pipeline state `pso` was built from the shaders of `sig`.
pub fn note_pso_shader(pso: usize, sig: u64) {
    if pso == 0 {
        return;
    }
    PSO_SHADERS.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, sig);
    SHADERS_KNOWN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// `pso` was released: it cannot be drawn with again.
pub fn forget_pso(pso: usize) {
    if SHADERS_KNOWN.load(std::sync::atomic::Ordering::Relaxed) == 0 {
        return;
    }
    if PSO_SHADERS.lock().unwrap().as_mut().and_then(|m| m.remove(&pso)).is_some() {
        SHADERS_KNOWN.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// `pso` is bound: what it was built from is drawn with, and the next run builds it during
/// load.
pub fn note_drawn(pso: usize) {
    if SHADERS_KNOWN.load(std::sync::atomic::Ordering::Relaxed) == 0 {
        return;
    }
    let sig = PSO_SHADERS.lock().unwrap().as_ref().and_then(|m| m.get(&pso).copied());
    if let Some(sig) = sig {
        mark_drawn(&dir(), sig);
    }
}

/// The shaders drawn with in an earlier run, and the ones this run has added not yet written.
struct Drawn {
    dir: PathBuf,
    keys: HashSet<u64>,
    unsaved: usize,
}

static DRAWN: Mutex<Option<Drawn>> = Mutex::new(None);
/// how many entries are written at once by the side thread
const DRAWN_BATCH: usize = 64;

fn read_drawn(dir: &Path) -> HashSet<u64> {
    std::fs::read_to_string(drawn_path(dir))
        .map(|text| text.lines().filter_map(|l| u64::from_str_radix(l.trim(), 16).ok()).collect())
        .unwrap_or_default()
}

fn write_drawn(dir: &Path, keys: &HashSet<u64>) {
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let mut text = String::with_capacity(17 * keys.len());
    for k in keys {
        text.push_str(&format!("{k:016x}\n"));
    }
    let path = drawn_path(dir);
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&tmp, &text).is_ok() && std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

fn flush_async(dir: PathBuf, keys: HashSet<u64>) {
    let _ = std::thread::Builder::new().name("xcl-pso-drawn".into()).spawn(move || write_drawn(&dir, &keys));
}

/// The drawn set of `dir`, read from disk the first time it is asked for.
fn with_drawn<R>(dir: &Path, f: impl FnOnce(&mut Drawn) -> R) -> R {
    let mut g = DRAWN.lock().unwrap();
    let stale = g.as_ref().map_or(true, |d| d.dir != dir);
    if stale {
        // the marks not written yet belong to the old dir
        if let Some(d) = g.take().filter(|d| d.unsaved > 0) {
            write_drawn(&d.dir, &d.keys);
        }
        *g = Some(Drawn { dir: dir.to_path_buf(), keys: read_drawn(dir), unsaved: 0 });
    }
    f(g.as_mut().unwrap())
}

/// A shader identity drawn with in an earlier run (its pipeline is built during load).
pub fn wanted_shader(sig: u64) -> bool {
    wanted_in(&dir(), sig)
}

fn wanted_in(dir: &Path, sig: u64) -> bool {
    let wanted = with_drawn(dir, |d| d.keys.contains(&sig));
    static ONCE: std::sync::Once = std::sync::Once::new();
    if wanted {
        ONCE.call_once(|| {
            let n = with_drawn(dir, |d| d.keys.len());
            crate::xlog_info!("D3D12.X: pipeline library: {n} pipeline(s) drawn with in an earlier run are built while the title loads");
        });
    }
    wanted
}

fn mark_drawn(dir: &Path, sig: u64) {
    with_drawn(dir, |d| {
        if d.keys.insert(sig) {
            d.unsaved += 1;
            if d.unsaved >= DRAWN_BATCH {
                d.unsaved = 0;
                flush_async(dir.to_path_buf(), d.keys.clone());
            }
        }
    });
}

/// Write the shaders drawn with since the last write (the frame counter calls it now and then,
/// so a short run records what it drew).
pub fn flush_drawn() {
    with_drawn(&dir(), |d| {
        if d.unsaved > 0 {
            d.unsaved = 0;
            let (dir, keys) = (d.dir.clone(), d.keys.clone());
            flush_async(dir, keys);
        }
    });
}

/// Forget what was drawn with in memory (the file stays): what a restart of the process sees.
#[cfg(test)]
pub(crate) fn forget_drawn_memory() {
    let mut g = DRAWN.lock().unwrap();
    if let Some(mut d) = g.take() {
        if d.unsaved > 0 {
            d.unsaved = 0;
            write_drawn(&d.dir, &d.keys);
        }
    }
}

fn mix(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= b as u64;
        *h = h.wrapping_mul(0x100_0000_01b3);
    }
}

fn mix_u32(h: &mut u64, v: u32) {
    mix(h, &v.to_le_bytes());
}

fn mix_shader(h: &mut u64, s: &D3D12_SHADER_BYTECODE) {
    let n = s.BytecodeLength;
    let p = s.pShaderBytecode as usize;
    if n == 0 || !crate::gpu::shader::readable(p, n) {
        mix_u32(h, 0);
        return;
    }
    mix_u32(h, n as u32);
    mix(h, unsafe { std::slice::from_raw_parts(p as *const u8, n) });
}

fn mix_input(h: &mut u64, layout: &D3D12_INPUT_LAYOUT_DESC) {
    let n = layout.NumElements as usize;
    mix_u32(h, layout.NumElements);
    if n == 0 || n > 32 || layout.pInputElementDescs.is_null() {
        return;
    }
    let bytes = n * std::mem::size_of::<D3D12_INPUT_ELEMENT_DESC>();
    if !crate::gpu::shader::readable(layout.pInputElementDescs as usize, bytes) {
        return;
    }
    let elems = unsafe { std::slice::from_raw_parts(layout.pInputElementDescs, n) };
    for e in elems {
        if !e.SemanticName.is_null() {
            let mut k = 0usize;
            while k < 64 {
                let c = unsafe { *e.SemanticName.0.add(k) };
                mix(h, &[c]);
                if c == 0 {
                    break;
                }
                k += 1;
            }
        }
        mix_u32(h, e.SemanticIndex);
        mix_u32(h, e.Format.0 as u32);
        mix_u32(h, e.InputSlot);
        mix_u32(h, e.AlignedByteOffset);
        mix_u32(h, e.InputSlotClass.0 as u32);
        mix_u32(h, e.InstanceDataStepRate);
    }
}

fn mix_blend(h: &mut u64, b: &D3D12_BLEND_DESC) {
    mix_u32(h, b.AlphaToCoverageEnable.0 as u32);
    mix_u32(h, b.IndependentBlendEnable.0 as u32);
    for rt in &b.RenderTarget {
        mix_u32(h, rt.BlendEnable.0 as u32);
        mix_u32(h, rt.LogicOpEnable.0 as u32);
        mix_u32(h, rt.SrcBlend.0 as u32);
        mix_u32(h, rt.DestBlend.0 as u32);
        mix_u32(h, rt.BlendOp.0 as u32);
        mix_u32(h, rt.SrcBlendAlpha.0 as u32);
        mix_u32(h, rt.DestBlendAlpha.0 as u32);
        mix_u32(h, rt.BlendOpAlpha.0 as u32);
        mix_u32(h, rt.LogicOp.0 as u32);
        mix_u32(h, rt.RenderTargetWriteMask as u32);
    }
}

fn mix_op(h: &mut u64, o: &D3D12_DEPTH_STENCILOP_DESC) {
    mix_u32(h, o.StencilFailOp.0 as u32);
    mix_u32(h, o.StencilDepthFailOp.0 as u32);
    mix_u32(h, o.StencilPassOp.0 as u32);
    mix_u32(h, o.StencilFunc.0 as u32);
}

fn mix_depth(h: &mut u64, z: &D3D12_DEPTH_STENCIL_DESC) {
    mix_u32(h, z.DepthEnable.0 as u32);
    mix_u32(h, z.DepthWriteMask.0 as u32);
    mix_u32(h, z.DepthFunc.0 as u32);
    mix_u32(h, z.StencilEnable.0 as u32);
    mix(h, &[z.StencilReadMask, z.StencilWriteMask]);
    mix_op(h, &z.FrontFace);
    mix_op(h, &z.BackFace);
}

fn mix_raster(h: &mut u64, r: &D3D12_RASTERIZER_DESC) {
    mix_u32(h, r.FillMode.0 as u32);
    mix_u32(h, r.CullMode.0 as u32);
    mix(h, &r.FrontCounterClockwise.0.to_le_bytes());
    mix(h, &r.DepthBias.to_le_bytes());
    mix(h, &r.DepthBiasClamp.to_le_bytes());
    mix(h, &r.SlopeScaledDepthBias.to_le_bytes());
    mix_u32(h, r.DepthClipEnable.0 as u32);
    mix_u32(h, r.MultisampleEnable.0 as u32);
    mix_u32(h, r.AntialiasedLineEnable.0 as u32);
    mix_u32(h, r.ForcedSampleCount);
    mix_u32(h, r.ConservativeRaster.0 as u32);
}

fn mix_samplers(h: &mut u64, samplers: &[([u32; 4], bool)]) {
    mix_u32(h, samplers.len() as u32);
    for (words, comparison) in samplers {
        for w in words {
            mix_u32(h, *w);
        }
        mix(h, &[*comparison as u8]);
    }
}

/// Identity of a graphics pipeline the driver compiled. The root signature pointer is not
/// part of it; `samplers` is, because that is what the XCL root signature was built from.
pub fn graphics_key(d: &D3D12_GRAPHICS_PIPELINE_STATE_DESC, samplers: &[([u32; 4], bool)]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    mix(&mut h, b"gfx1");
    mix_shader(&mut h, &d.VS);
    mix_shader(&mut h, &d.PS);
    mix_shader(&mut h, &d.DS);
    mix_shader(&mut h, &d.HS);
    mix_shader(&mut h, &d.GS);
    mix_blend(&mut h, &d.BlendState);
    mix_u32(&mut h, d.SampleMask);
    mix_raster(&mut h, &d.RasterizerState);
    mix_depth(&mut h, &d.DepthStencilState);
    mix_input(&mut h, &d.InputLayout);
    mix_u32(&mut h, d.IBStripCutValue.0 as u32);
    mix_u32(&mut h, d.PrimitiveTopologyType.0 as u32);
    mix_u32(&mut h, d.NumRenderTargets);
    for f in &d.RTVFormats {
        mix_u32(&mut h, f.0 as u32);
    }
    mix_u32(&mut h, d.DSVFormat.0 as u32);
    mix_u32(&mut h, d.SampleDesc.Count);
    mix_u32(&mut h, d.SampleDesc.Quality);
    mix_u32(&mut h, d.NodeMask);
    mix_u32(&mut h, d.Flags.0 as u32);
    mix_samplers(&mut h, samplers);
    h
}

pub fn compute_key(d: &D3D12_COMPUTE_PIPELINE_STATE_DESC, samplers: &[([u32; 4], bool)]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    mix(&mut h, b"cs1");
    mix_shader(&mut h, &d.CS);
    mix_u32(&mut h, d.NodeMask);
    mix_u32(&mut h, d.Flags.0 as u32);
    mix_samplers(&mut h, samplers);
    h
}

/// Identity of a pipeline state stream (`CreatePipelineState` with subobjects, the form XCL
/// builds mesh pipelines in). Two things in such a stream are not the pipeline: the root
/// signature (an object pointer, different every run – `samplers` stands in for it, as in
/// [`graphics_key`]) and shader subobjects, which are a pointer and a length (the code they
/// point at is mixed instead). A CACHED_PSO subobject is skipped. `buf` must hold whole
/// subobjects, as XCL's `StreamOut` writes them.
pub fn stream_key(buf: &[u8], samplers: &[([u32; 4], bool)]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    mix(&mut h, b"str1");
    let mut off = 0usize;
    while off + 8 <= buf.len() {
        let ty = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        let Some((psize, palign)) = super::null::subobject_payload(ty) else { break };
        let poff = if palign == 8 { 8 } else { 4 };
        let Some(payload) = buf.get(off + poff..off + poff + psize) else { break };
        match ty {
            // the root signature object: `samplers` identifies it
            0 | 19 => {}
            // shaders (VS PS DS HS GS CS AS MS: 1..=6, 24, 25): the code, not the pointer
            1..=6 | 24 | 25 if psize >= 16 => {
                let p = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                let n = u64::from_le_bytes(payload[8..16].try_into().unwrap()) as usize;
                mix_shader(&mut h, &D3D12_SHADER_BYTECODE { pShaderBytecode: p as *const _, BytecodeLength: n });
            }
            // an input layout's elements are behind a pointer too
            12 if psize >= 16 => {
                let p = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as *const D3D12_INPUT_ELEMENT_DESC;
                mix_input(&mut h, &D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: p, NumElements: u32::from_le_bytes(payload[8..12].try_into().unwrap()) });
            }
            _ => {
                mix_u32(&mut h, ty);
                mix(&mut h, payload);
            }
        }
        off += (poff + psize + 7) & !7;
    }
    mix_samplers(&mut h, samplers);
    h
}

/// A console pipeline description, as the title hands it over before any translation.
pub enum ConsoleDesc<'a> {
    Graphics(&'a D3D12_GRAPHICS_PIPELINE_STATE_DESC),
    Compute(&'a D3D12_COMPUTE_PIPELINE_STATE_DESC),
    /// a `CreatePipelineState` stream: whole subobjects in one buffer
    Stream(&'a [u8]),
}

/// The identity of a console pipeline before it is translated: the shader code and every state
/// byte of the description, with this run's pointer values left out (shader and input layout
/// pointers are read, a root signature object is not part of it – the translated pipeline gets
/// XCL's own root signature, built from the samplers the shaders declare). Two equal
/// descriptions give an equal key, so the pipeline is built once.
pub fn console_desc_key(d: ConsoleDesc<'_>) -> u64 {
    match d {
        ConsoleDesc::Graphics(g) => graphics_key(g, &[]),
        ConsoleDesc::Compute(c) => compute_key(c, &[]),
        ConsoleDesc::Stream(s) => console_stream_key(s),
    }
}

/// [`console_desc_key`] for a pipeline state stream. Unlike [`stream_key`], each shader and
/// input layout subobject mixes its type too: the same code in the VS and the CS slots is not
/// the same pipeline.
fn console_stream_key(buf: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    mix(&mut h, b"cstr1");
    let mut off = 0usize;
    while off + 8 <= buf.len() {
        let ty = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
        let Some((psize, palign)) = super::null::subobject_payload(ty) else { break };
        let poff = if palign == 8 { 8 } else { 4 };
        let Some(payload) = buf.get(off + poff..off + poff + psize) else { break };
        match ty {
            // the root signature object and a cached blob are not the pipeline
            0 | 19 => {}
            // shaders (VS PS DS HS GS CS AS MS: 1..=6, 24, 25): the code, not the pointer
            1..=6 | 24 | 25 if psize >= 16 => {
                let p = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as usize;
                let n = u64::from_le_bytes(payload[8..16].try_into().unwrap()) as usize;
                mix_u32(&mut h, ty);
                mix_shader(&mut h, &D3D12_SHADER_BYTECODE { pShaderBytecode: p as *const _, BytecodeLength: n });
            }
            // an input layout's elements are behind a pointer too
            12 if psize >= 16 => {
                let p = u64::from_le_bytes(payload[0..8].try_into().unwrap()) as *const D3D12_INPUT_ELEMENT_DESC;
                mix_u32(&mut h, ty);
                mix_input(&mut h, &D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: p, NumElements: u32::from_le_bytes(payload[8..12].try_into().unwrap()) });
            }
            _ => {
                mix_u32(&mut h, ty);
                mix(&mut h, payload);
            }
        }
        off += (poff + psize + 7) & !7;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A drawn pipeline's shaders are written beside the blobs, and a restart (the set in
    /// memory gone) finds them: the next run builds that pipeline during load. A blob
    /// corrupted on disk leaves the library with nothing for its key.
    #[test]
    fn drawn_shaders_and_blobs_survive_a_restart_and_a_corrupt_blob_leaves_an_empty_library() {
        let dir = std::env::temp_dir().join(format!("xcl-pso-library-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let (key, sig) = (0x7410_0000_0000_00a1u64, 0x7410_0000_0000_00b2u64);
        assert!(!wanted_in(&dir, sig));
        mark_drawn(&dir, sig);
        let blob = b"driver blob";
        save_in(&dir, key, blob);
        // a restart: only the files remain
        forget_drawn_memory();
        assert!(wanted_in(&dir, sig), "the drawn shaders are read back from disk");
        assert_eq!(load_in(&dir, key).as_deref(), Some(&blob[..]), "and so is the blob");
        // a blob from another version, or one cut short, is not used
        std::fs::write(path_for(&dir, key), b"XPSO\x02\x00\x00\x00driver blob").unwrap();
        assert_eq!(load_in(&dir, key), None);
        std::fs::write(path_for(&dir, key), b"trunc").unwrap();
        assert_eq!(load_in(&dir, key), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A blob the device refuses takes the blobs this run has read with it: the library starts
    /// empty, and the run logs it once.
    #[test]
    fn a_blob_the_device_refuses_starts_an_empty_library() {
        let key = 0x7410_0000_0000_00d4u64;
        memory().lock().unwrap().insert(key, std::sync::Arc::new(b"a blob".to_vec()));
        assert!(load(key).is_some());
        rejected(key);
        assert!(load(key).is_none(), "nothing of the refused library is used again");
    }

    /// What a creation does: the blob from the library, else the compile and its blob saved.
    /// The second creation, in what a restart leaves, compiles nothing.
    #[test]
    fn a_library_hit_compiles_nothing_on_the_next_run() {
        let dir = std::env::temp_dir().join(format!("xcl-pso-recreate-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let key = 0x7410_0000_0000_00e5u64;
        let compiles = std::cell::Cell::new(0u32);
        let create = |key: u64| -> Vec<u8> {
            if let Some(blob) = load_in(&dir, key) {
                return blob;
            }
            // the translation's compile, which a library hit must not reach
            compiles.set(compiles.get() + 1);
            let blob = b"freshly compiled".to_vec();
            save_in(&dir, key, &blob);
            blob
        };
        assert_eq!(create(key), b"freshly compiled".to_vec());
        assert_eq!(compiles.get(), 1);
        assert_eq!(create(key), b"freshly compiled".to_vec(), "the library answers the next run");
        assert_eq!(compiles.get(), 1, "and nothing compiles");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two pipelines are told apart by their shaders, and the same shaders at other addresses
    /// are the same pipeline (a restart moves them).
    #[test]
    fn a_shader_identity_is_the_code_and_the_order_of_the_stages() {
        let (vs, ps) = ([7u8; 16], [8u8; 24]);
        let sig = shaders_sig(&[&vs, &ps]);
        assert_eq!(sig, shaders_sig(&[&vs, &ps]));
        assert_ne!(sig, shaders_sig(&[&ps, &vs]), "stage order counts");
        assert_ne!(sig, shaders_sig(&[&vs]), "a missing stage counts");
        let mut other = ps;
        other[0] = 9;
        assert_ne!(sig, shaders_sig(&[&vs, &other]));
    }

    #[test]
    fn shader_bytes_change_the_key_and_the_blob_round_trips() {
        let vs = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut g = D3D12_GRAPHICS_PIPELINE_STATE_DESC::default();
        g.VS = D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as *const _, BytecodeLength: vs.len() };
        g.NumRenderTargets = 1;
        g.RTVFormats[0] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
        let k1 = graphics_key(&g, &[]);
        let vs2 = [1u8, 2, 3, 4, 5, 6, 7, 9];
        g.VS.pShaderBytecode = vs2.as_ptr() as *const _;
        assert_ne!(k1, graphics_key(&g, &[]));
        g.VS.pShaderBytecode = vs.as_ptr() as *const _;
        assert_eq!(k1, graphics_key(&g, &[]));
        assert_ne!(k1, graphics_key(&g, &[([1, 2, 3, 4], false)]));

        let dir = std::env::temp_dir().join(format!("xcl-pso-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = path_for(&dir, k1);
        let blob = b"driver-blob-bytes";
        let mut body = Vec::new();
        body.extend_from_slice(MAGIC);
        body.extend_from_slice(&VERSION.to_le_bytes());
        body.extend_from_slice(blob);
        std::fs::write(&file, &body).unwrap();
        let read = std::fs::read(&file).unwrap();
        assert_eq!(payload(&read), Some(blob.as_slice()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A stream's key is the pipeline, not this run's pointers: the root signature object may
    /// differ (different address) and the shader code may live at another address, and both
    /// leave the key alone. Other code, or other samplers, change it.
    #[test]
    fn a_stream_key_leaves_out_this_runs_pointers() {
        let code_a = [9u8; 32];
        // the same code at another address
        let code_a_again = [9u8; 32];
        let code_b = [8u8; 32];
        let stream = |rs: usize, code: &[u8]| -> Vec<u8> {
            let mut v = Vec::new();
            // ROOT_SIGNATURE (type 0): 4-byte header, pointer payload at 8
            v.extend_from_slice(&0u32.to_le_bytes());
            v.extend_from_slice(&[0u8; 4]);
            v.extend_from_slice(&(rs as u64).to_le_bytes());
            // CS (type 2): 4-byte type, 4 of padding, then D3D12_SHADER_BYTECODE
            v.extend_from_slice(&2u32.to_le_bytes());
            v.extend_from_slice(&[0u8; 4]);
            v.extend_from_slice(&(code.as_ptr() as u64).to_le_bytes());
            v.extend_from_slice(&(code.len() as u64).to_le_bytes());
            // SAMPLE_MASK (type 9): 4-byte header, 4-byte payload
            v.extend_from_slice(&9u32.to_le_bytes());
            v.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
            v
        };
        let (a, b) = (stream(0x1000, &code_a), stream(0x2000, &code_a_again));
        assert_eq!(stream_key(&a, &[]), stream_key(&b, &[]), "pointers are not the pipeline");
        assert_ne!(stream_key(&a, &[]), stream_key(&stream(0x1000, &code_b), &[]), "the code is");
        assert_ne!(stream_key(&a, &[]), stream_key(&a, &[([1, 2, 3, 4], false)]), "the samplers are");
        assert_ne!(stream_key(&a, &[]), stream_key(&a[..a.len() - 4], &[]), "the state is");
    }

    /// Every finished build is a hit or a miss: the blobs a build takes are counted, and a
    /// stored blob comes back.
    #[test]
    fn a_build_from_a_stored_blob_counts_as_a_hit() {
        let key = 0x7410_0000_0000_0001u64;
        super::forget(key);
        let (h0, m0) = counts();
        assert!(load(key).is_none());
        note_lookup(false);
        store(key, b"a driver blob");
        assert_eq!(load(key).map(|b| b.to_vec()), Some(b"a driver blob".to_vec()));
        note_lookup(true);
        super::forget(key);
        let (h1, m1) = counts();
        assert!(h1 > h0 && m1 > m0, "hit {h0}->{h1}, miss {m0}->{m1}");
    }

    fn rs(v: usize) -> std::mem::ManuallyDrop<Option<ID3D12RootSignature>> {
        unsafe { std::mem::transmute::<usize, std::mem::ManuallyDrop<Option<ID3D12RootSignature>>>(v) }
    }

    fn console_graphics(vs: &[u8], elements: &[D3D12_INPUT_ELEMENT_DESC], root: usize) -> D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr() as _, BytecodeLength: vs.len() },
            InputLayout: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: elements.as_ptr(), NumElements: elements.len() as u32 },
            pRootSignature: rs(root),
            ..Default::default()
        }
    }

    fn element(name: &[u8]) -> D3D12_INPUT_ELEMENT_DESC {
        D3D12_INPUT_ELEMENT_DESC { SemanticName: windows::core::PCSTR(name.as_ptr()), Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32_FLOAT, ..Default::default() }
    }

    /// A console description's key is its bytes, not this run's pointers: the same shader code
    /// and input layout at other addresses, and another root signature object, give the same
    /// pipeline. Any changed state byte or shader byte gives another one.
    #[test]
    fn a_console_desc_key_leaves_out_pointer_values_and_keeps_the_bytes() {
        let (vs_a, vs_again) = ([1u8, 2, 3, 4, 5, 6, 7, 8], [1u8, 2, 3, 4, 5, 6, 7, 8]);
        let mut vs_b = vs_a;
        vs_b[0] = 9;
        let (elems_a, elems_again) = ([element(b"POSITION\0")], [element(b"POSITION\0")]);
        let elems_other = [element(b"POSITION1\0")];
        let key = |g: &D3D12_GRAPHICS_PIPELINE_STATE_DESC| console_desc_key(ConsoleDesc::Graphics(g));
        let a = console_graphics(&vs_a, &elems_a, 0x1000);
        let again = console_graphics(&vs_again, &elems_again, 0x2000);
        assert_eq!(key(&a), key(&again), "pointers are not the pipeline");
        assert_ne!(key(&a), key(&console_graphics(&vs_b, &elems_a, 0x1000)), "the code is");
        let mut state = console_graphics(&vs_a, &elems_a, 0x1000);
        state.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
        assert_ne!(key(&a), key(&state), "a state byte is");
        assert_ne!(key(&a), key(&console_graphics(&vs_a, &elems_other, 0x1000)), "the input layout is");
        let mut targets = console_graphics(&vs_a, &elems_a, 0x1000);
        targets.NumRenderTargets = 1;
        assert_ne!(key(&a), key(&targets), "the render targets are");
    }

    /// A compute description is the same: its CS code and flags count, its root signature
    /// object does not.
    #[test]
    fn a_console_compute_key_is_the_code_not_the_root_signature() {
        let (cs_a, cs_again) = ([3u8; 16], [3u8; 16]);
        let mut cs_b = cs_a;
        cs_b[1] = 4;
        let desc = |cs: &[u8], root: usize| D3D12_COMPUTE_PIPELINE_STATE_DESC {
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr() as _, BytecodeLength: cs.len() },
            pRootSignature: rs(root),
            ..Default::default()
        };
        let key = |c: &D3D12_COMPUTE_PIPELINE_STATE_DESC| console_desc_key(ConsoleDesc::Compute(c));
        assert_eq!(key(&desc(&cs_a, 0x10)), key(&desc(&cs_again, 0x20)), "the root signature object is not the pipeline");
        assert_ne!(key(&desc(&cs_a, 0x10)), key(&desc(&cs_b, 0x10)), "the code is");
    }

    /// A stream's console key reads the subobjects' bytes, not their pointers, and counts the
    /// slot: the same code as a VS and as a CS is two pipelines.
    #[test]
    fn a_console_stream_key_counts_the_slot_and_not_the_pointer() {
        let code_a = [7u8; 24];
        let code_again = [7u8; 24];
        let mut code_b = code_a;
        code_b[3] = 8;
        let stream = |ty: u32, code: &[u8], rs: usize| -> Vec<u8> {
            let mut v = Vec::new();
            v.extend_from_slice(&0u32.to_le_bytes());
            v.extend_from_slice(&[0u8; 4]);
            v.extend_from_slice(&(rs as u64).to_le_bytes());
            v.extend_from_slice(&ty.to_le_bytes());
            v.extend_from_slice(&[0u8; 4]);
            v.extend_from_slice(&(code.as_ptr() as u64).to_le_bytes());
            v.extend_from_slice(&(code.len() as u64).to_le_bytes());
            v.extend_from_slice(&9u32.to_le_bytes());
            v.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
            v
        };
        let key = |b: &[u8]| console_desc_key(ConsoleDesc::Stream(b));
        assert_eq!(key(&stream(1, &code_a, 0x1000)), key(&stream(1, &code_again, 0x2000)), "pointers are not the pipeline");
        assert_ne!(key(&stream(1, &code_a, 0x1000)), key(&stream(1, &code_b, 0x1000)), "the code is");
        assert_ne!(key(&stream(1, &code_a, 0x1000)), key(&stream(2, &code_a, 0x1000)), "the slot is");
        let mut mask = stream(1, &code_a, 0x1000);
        *mask.last_mut().unwrap() = 0;
        assert_ne!(key(&stream(1, &code_a, 0x1000)), key(&mask), "a state byte is");
    }
}
