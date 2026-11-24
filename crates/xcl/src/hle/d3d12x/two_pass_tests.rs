//! Pipelines in two passes drawn end to end without a game: synthetic console containers (an
//! NGG mesh program; the ES variant of a VS + a GS) go through the backend device's pipeline
//! hooks, the draws are recorded through the console command list's methods (`bind`,
//! `list_x`) on a list made as for a title, and the render target is read back.
//!
//! Skipped (passing) without a D3D12 device with mesh shaders, unified memory on the GPU or DXC.

use std::ffi::c_void;
use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{bind, device_x, list_x, real, title_list};
use crate::gpu::rdna2::encode;
use crate::gpu::recomp::dxc;
use crate::gpu::xbox::tests::{container, dxil_part, fssm};

/// The tests share the backend device and its counters.
static SERIAL: Mutex<()> = Mutex::new(());

const SIZE: u32 = 32;
const CLEAR: [f32; 4] = [-1.0, -1.0, -1.0, -1.0];
/// VALU source code of v0
const V: u64 = 256;

fn f(x: f32) -> u32 {
    x.to_bits()
}

/// The backend device, when it can run these tests.
fn ready() -> Option<ID3D12Device> {
    if dxc::location().is_err() {
        println!("no DXC – skipping");
        return None;
    }
    if super::backend() != super::Backend::Real {
        println!("no real D3D12 device – skipping");
        return None;
    }
    if real::software_adapter() {
        // WARP / the Basic Render Driver report mesh shaders but reject the translated pipelines
        println!("software adapter (no GPU) – skipping");
        crate::gpu::recomp::runner::skip_test("no GPU");
        return None;
    }
    if real::uma_window_vas().is_empty() {
        println!("unified memory is not open on the GPU – skipping");
        return None;
    }
    let raw = real::device_raw();
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }?.clone();
    let mut o7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
    let mesh = unsafe { dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &mut o7 as *mut _ as _, std::mem::size_of_val(&o7) as u32) }.is_ok()
        && o7.MeshShaderTier.0 >= D3D12_MESH_SHADER_TIER_1.0;
    if !mesh {
        println!("GPU without mesh shaders – skipping");
        return None;
    }
    Some(dev)
}

/// A v1 signature part (ISG1 / OSG1): (name, index, system value, component type, register).
fn signature(elements: &[(&str, u32, u32, u32, u32)]) -> Vec<u8> {
    let mut d = Vec::new();
    d.extend_from_slice(&(elements.len() as u32).to_le_bytes());
    d.extend_from_slice(&8u32.to_le_bytes());
    let names_at = 8 + 32 * elements.len() as u32;
    let mut names = Vec::new();
    for &(name, index, sv, ct, reg) in elements {
        d.extend_from_slice(&0u32.to_le_bytes()); // stream
        d.extend_from_slice(&(names_at + names.len() as u32).to_le_bytes());
        d.extend_from_slice(&index.to_le_bytes());
        d.extend_from_slice(&sv.to_le_bytes());
        d.extend_from_slice(&ct.to_le_bytes());
        d.extend_from_slice(&reg.to_le_bytes());
        d.extend_from_slice(&[0xF, 0xF, 0, 0]);
        d.extend_from_slice(&0u32.to_le_bytes()); // min precision
        names.extend_from_slice(name.as_bytes());
        names.push(0);
    }
    d.extend_from_slice(&names);
    d
}

/// A PSV0 part: `r` = the runtime info (48 bytes).
fn psv0(r: &[u8; 48]) -> Vec<u8> {
    let mut part = 48u32.to_le_bytes().to_vec();
    part.extend_from_slice(r);
    part.extend_from_slice(&0u32.to_le_bytes()); // resources
    part
}

/// A console shader container of DXIL kind `kind` with one FSSM (hardware stage `hw`).
fn console(kind: u32, hw: u8, code: &[u8], isg1: &[u8], osg1: &[u8], psv: Option<Vec<u8>>) -> Vec<u8> {
    let dxil = dxil_part(kind, &[fssm(code, &[(hw, kind as u8)])]);
    let mut parts: Vec<(&[u8; 4], &[u8])> = vec![(b"SFI0", &[0; 8]), (b"ISG1", isg1), (b"OSG1", osg1)];
    if let Some(p) = &psv {
        parts.push((b"PSV0", p));
    }
    parts.push((b"HASH", &[0; 20]));
    parts.push((b"DXIL", &dxil));
    container(&parts)
}

fn prog(list: &[(&str, &str, Vec<(&str, u64)>, Option<u32>)], end: bool) -> Vec<u8> {
    let mut v = Vec::new();
    for (n, e, fl, l) in list {
        v.extend(encode(n, e, fl, *l).unwrap_or_else(|x| panic!("{n}: {x}")));
    }
    if end {
        v.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
    }
    v
}

/// RTS0 v2 with one parameter of `n` root constants (inline: the user SGPRs from s8).
fn rts0_constants(n: u32) -> Vec<u8> {
    let words: [u32; 12] = [2, 1, 24, 0, 0, 0, 1, 0, 36, 0, 0, n];
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// A list made as for a title (wrapped, adopted by `bind` / `list_x`), with a render target,
/// viewport and scissor set on the PC list; `record` records the draw through the console
/// methods; then the target is read back.
struct Target {
    dev: ID3D12Device,
    alloc: ID3D12CommandAllocator,
    raw_list: *mut c_void,
    backing: *mut c_void,
    rt: ID3D12Resource,
    _heap: ID3D12DescriptorHeap,
}

impl Target {
    unsafe fn new(dev: &ID3D12Device) -> Target {
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
        let mut raw_list: *mut c_void = std::ptr::null_mut();
        let iid = windows_sys::core::GUID::from_u128(0x5b160d0f_ac1b_4185_8ba8_b3ae42a5a455);
        let hr = device_x::create_command_list_pc(dev.as_raw(), 0, D3D12_COMMAND_LIST_TYPE_DIRECT.0 as u32, alloc.as_raw() as usize, 0, &iid, &mut raw_list);
        assert!(hr >= 0 && !raw_list.is_null(), "CreateCommandList: {hr:#x}");
        let backing = title_list::backing(raw_list).expect("a title's list");
        let tex = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: SIZE as u64,
            Height: SIZE,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let cv = D3D12_CLEAR_VALUE { Format: DXGI_FORMAT_R32G32B32A32_FLOAT, Anonymous: D3D12_CLEAR_VALUE_0 { Color: CLEAR } };
        let mut rt: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &tex, D3D12_RESOURCE_STATE_RENDER_TARGET, Some(&cv), &mut rt).unwrap();
        let rt = rt.unwrap();
        let heap: ID3D12DescriptorHeap = dev.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() }).unwrap();
        let rtv = heap.GetCPUDescriptorHandleForHeapStart();
        dev.CreateRenderTargetView(&rt, None, rtv);
        let list = ID3D12GraphicsCommandList::from_raw_borrowed(&backing).unwrap();
        list.OMSetRenderTargets(1, Some(&rtv), false, None);
        list.ClearRenderTargetView(rtv, &CLEAR, None);
        list.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: SIZE as f32, Height: SIZE as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
        list.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: SIZE as i32, bottom: SIZE as i32 }]);
        Target { dev: dev.clone(), alloc, raw_list, backing, rt, _heap: heap }
    }

    /// Close, run and read back the target (one RGBA32F texel per pixel, row by row).
    unsafe fn finish(self) -> Vec<[f32; 4]> {
        let dev = &self.dev;
        let list = ID3D12GraphicsCommandList::from_raw_borrowed(&self.backing).unwrap().clone();
        let tex = self.rt.GetDesc();
        let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0u64;
        dev.GetCopyableFootprints(&tex, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total));
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: total,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut rb: Option<ID3D12Resource> = None;
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut rb).unwrap();
        let rb = rb.unwrap();
        list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::transmute_copy(&self.rt),
                    Subresource: 0,
                    StateBefore: D3D12_RESOURCE_STATE_RENDER_TARGET,
                    StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                }),
            },
        }]);
        let dst = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&rb), Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp } };
        let src = D3D12_TEXTURE_COPY_LOCATION { pResource: std::mem::transmute_copy(&self.rt), Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 } };
        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        list.Close().unwrap();
        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.Signal(&fence, 1).unwrap();
        let t0 = std::time::Instant::now();
        while fence.GetCompletedValue() < 1 {
            assert!(t0.elapsed().as_secs() < 20, "the GPU did not finish (device removed: {:?})", dev.GetDeviceRemovedReason());
            std::thread::yield_now();
        }
        assert!(dev.GetDeviceRemovedReason().is_ok(), "device removed: {:?}", dev.GetDeviceRemovedReason());
        let mut q = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut q)).unwrap();
        let pitch = fp.Footprint.RowPitch as usize;
        let mut px = Vec::with_capacity((SIZE * SIZE) as usize);
        for y in 0..SIZE as usize {
            for x in 0..SIZE as usize {
                let c = (q as *const u8).add(y * pitch + x * 16) as *const f32;
                px.push([*c, *c.add(1), *c.add(2), *c.add(3)]);
            }
        }
        rb.Unmap(0, None);
        // the list goes as a title's does (its unified memory with its allocator)
        let release: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(self.raw_list as *const *const usize)).add(2));
        release(self.raw_list);
        drop(self.alloc);
        px
    }
}

/// Is pixel centre (x, y) (pixels, y down) inside triangle `t` given in clip space (y up)?
fn inside(t: [[f32; 2]; 3], x: usize, y: usize, margin: f32) -> bool {
    let p = |v: [f32; 2]| ((v[0] + 1.0) * 0.5 * SIZE as f32, (1.0 - v[1]) * 0.5 * SIZE as f32);
    let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
    let (qx, qy) = (x as f32 + 0.5, y as f32 + 0.5);
    let area = (b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1);
    let l1 = ((b.0 - qx) * (c.1 - qy) - (c.0 - qx) * (b.1 - qy)) / area;
    let l2 = ((c.0 - qx) * (a.1 - qy) - (a.0 - qx) * (c.1 - qy)) / area;
    let l3 = 1.0 - l1 - l2;
    l1.min(l2).min(l3) > margin
}

/// Console NGG program (the console's style: wave index from s3, group from v5 >> 7,
/// GS_ALLOC_REQ for 3 vertices and 1 triangle, exp pos0 / param0 / prim) in an MS container +
/// a PC PS, through `CreatePipelineState`; drawn by the console list's DispatchMesh with two
/// groups: each group's triangle (shifted right by its group) with param0.x = the group.
#[test]
fn gpu_console_mesh_pipeline_draws_through_dispatch_mesh() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dev) = ready() else { return };
    let exp = |tgt: u64, en: u64, v: [u64; 4], done: u64| ("EXP", "ENC_EXP", vec![("TGT", tgt), ("EN", en), ("VSRC0", v[0]), ("VSRC1", v[1]), ("VSRC2", v[2]), ("VSRC3", v[3]), ("DONE", done)], None);
    let mut list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = vec![
        ("S_BFE_U32", "ENC_SOP2", vec![("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x40018)), // wave index
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 193), ("SRC1", 128)], None),
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 193), ("SRC1", V)], None),
        ("V_LSHL_ADD_U32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 0), ("SRC1", 134), ("SRC2", V)], None), // thread in group
        ("V_READFIRSTLANE_B32", "ENC_VOP1", vec![("VDST", 1), ("SRC0", V + 5)], None),
        ("S_LSHR_B32", "ENC_SOP2", vec![("SDST", 1), ("SSRC0", 1), ("SSRC1", 135)], None), // group = v5 >> 7
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 124), ("SSRC0", 255)], Some(1 << 12 | 3)), // 1 primitive, 3 vertices
        ("S_SENDMSG", "ENC_SOPP", vec![("SIMM16", 9)], None),
        ("S_BARRIER", "ENC_SOPP", vec![], None),
        ("V_CMPX_GT_U32", "ENC_VOPC", vec![("SRC0", 131), ("VSRC1", 0)], None), // threads 0..2
        ("BUFFER_LOAD_DWORDX4", "ENC_MUBUF", vec![("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("V_CVT_F32_U32", "ENC_VOP1", vec![("VDST", 20), ("SRC0", 1)], None),
        ("V_ADD_F32", "ENC_VOP2", vec![("VDST", 10), ("SRC0", V + 20), ("VSRC1", 10)], None), // x += group
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 14), ("SRC0", V + 20)], None),
        ("V_CVT_F32_U32", "ENC_VOP1", vec![("VDST", 15), ("SRC0", V)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 16), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 17), ("SRC0", 242)], None),
    ];
    list.push(exp(12, 0xF, [10, 11, 12, 13], 1));
    list.push(exp(32, 0xF, [14, 15, 16, 17], 0));
    list.push(("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 193)], None));
    list.push(("V_CMPX_EQ_U32", "ENC_VOPC", vec![("SRC0", 128), ("VSRC1", 0)], None)); // thread 0
    list.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 21), ("SRC0", 255)], Some(1 << 10 | 2 << 20)));
    list.push(exp(20, 1, [21, 0, 0, 0], 1));
    let code = prog(&list, true);
    let mut r = [0u8; 48];
    r[12..14].copy_from_slice(&128u16.to_le_bytes()); // max vertices
    r[14..16].copy_from_slice(&128u16.to_le_bytes()); // max primitives
    r[24] = 13; // mesh
    r[27] = 2; // triangles
    r[36..40].copy_from_slice(&128u32.to_le_bytes()); // numthreads
    r[40..44].copy_from_slice(&1u32.to_le_bytes());
    r[44..48].copy_from_slice(&1u32.to_le_bytes());
    let ms = console(13, 7, &code, &signature(&[]), &signature(&[("SV_Position", 0, 1, 3, 0), ("TEXCOORD", 0, 0, 3, 1)]), Some(psv0(&r)));
    assert!(crate::gpu::recomp::pso::console_mesh(&ms, None, true).is_ok(), "{:?}", crate::gpu::recomp::pso::console_mesh(&ms, None, true).err());
    let ps = dxc::compile("struct V { float4 pos : SV_Position; float4 p0 : TEXCOORD0; };\nfloat4 main(V v) : SV_Target0 { return v.p0; }\n", "main", "ps_6_6", &[]).unwrap();

    unsafe {
        // the title's mesh pipeline: its root signature, MS (console), PS, state
        let game_rs: ID3D12RootSignature = {
            let mut ser = None;
            D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).unwrap();
            let ser = ser.unwrap();
            dev.CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize())).unwrap()
        };
        #[repr(C, align(8))]
        struct Sub<T> {
            ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
            v: T,
        }
        #[repr(C)]
        struct Stream {
            rs: Sub<*mut c_void>,
            ms: Sub<D3D12_SHADER_BYTECODE>,
            ps: Sub<D3D12_SHADER_BYTECODE>,
            rast: Sub<D3D12_RASTERIZER_DESC>,
            depth: Sub<D3D12_DEPTH_STENCIL_DESC>,
            rt: Sub<D3D12_RT_FORMAT_ARRAY>,
        }
        let code = |b: &[u8]| D3D12_SHADER_BYTECODE { pShaderBytecode: b.as_ptr() as _, BytecodeLength: b.len() };
        let mut rt = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 1, ..Default::default() };
        rt.RTFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        let st = Stream {
            rs: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, v: game_rs.as_raw() },
            ms: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, v: code(&ms) },
            ps: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, v: code(&ps) },
            rast: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, v: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() } },
            depth: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, v: D3D12_DEPTH_STENCIL_DESC::default() },
            rt: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, v: rt },
        };
        let d2: ID3D12Device2 = dev.cast().unwrap();
        let desc = D3D12_PIPELINE_STATE_STREAM_DESC { SizeInBytes: std::mem::size_of::<Stream>(), pPipelineStateSubobjectStream: &st as *const _ as *mut _ };
        let pso: ID3D12PipelineState = d2.CreatePipelineState(&desc).unwrap_or_else(|e| panic!("{e}: {:?}", real::stats().translation_errors));
        let real_pso = super::pso_async::resolve_wait(pso.as_raw() as usize);
        let tp = real::two_pass_of(real_pso).unwrap_or_else(|| panic!("no two-pass parts: {:?}", real::stats().translation_errors));
        assert_eq!(tp.kind, "MS");

        // the vertex table (3 float4) in title memory, its V# in s[16:19] (root constants 8..11)
        let u = crate::gpu::uma::get().unwrap();
        let table = u.alloc(1 << 16, 1 << 16).unwrap() as usize;
        for (i, (x, y)) in [(-1.0f32, -1.0f32), (-1.0, 1.0), (0.0, -1.0)].iter().enumerate() {
            std::ptr::copy_nonoverlapping([f(*x), f(*y), f(0.0), f(1.0)].as_ptr(), (table as *mut u32).add(4 * i), 4);
        }
        let mut constants = [0u32; 12];
        constants[8..12].copy_from_slice(&[table as u32, ((table >> 32) as u32 & 0xFFFF) | 16 << 16, 3, 0]);
        let rs_key = 0x7400_0010usize;
        super::console_list::note_root_signature(rs_key, &rts0_constants(12));

        let t = Target::new(&dev);
        let before = bind::counts_snapshot();
        bind::set_pipeline_state(t.backing, pso.as_raw());
        bind::set_graphics_root_signature(t.backing, rs_key as *mut c_void);
        bind::set_graphics_root_constants(t.backing, 0, 12, constants.as_ptr(), 0);
        bind::dispatch_mesh(t.backing, 2, 1, 1);
        let after = bind::counts_snapshot();
        // (other tests count too: the pixels below are what tells)
        assert!(after[0] > before[0], "the DispatchMesh ran as a translated draw");
        let px = t.finish();
        u.free(table);

        // group g: (-1 + g, -1), (-1 + g, 1), (g, -1); param0 = (g, vertex, 0, 1)
        let (mut hit, mut clear) = ([0usize; 2], 0usize);
        for y in 0..SIZE as usize {
            for x in 0..SIZE as usize {
                let p = px[y * SIZE as usize + x];
                let mut covered = false;
                for g in 0..2 {
                    let gf = g as f32;
                    if inside([[-1.0 + gf, -1.0], [-1.0 + gf, 1.0], [gf, -1.0]], x, y, 0.02) {
                        assert_eq!((p[0], p[2], p[3]), (gf, 0.0, 1.0), "pixel ({x},{y}) in group {g}'s triangle: {p:?}");
                        hit[g] += 1;
                        covered = true;
                    }
                }
                let near = (0..2).any(|g| inside([[-1.0 + g as f32, -1.0], [-1.0 + g as f32, 1.0], [g as f32, -1.0]], x, y, -0.05));
                if !covered && !near {
                    assert_eq!(p, CLEAR, "pixel ({x},{y}) outside both triangles");
                    clear += 1;
                }
            }
        }
        println!("two-pass MS draw: {hit:?} pixels in the two triangles, {clear} clear");
        assert!(hit[0] > 80 && hit[1] > 80 && clear > 300, "{hit:?} / {clear}");
    }
}

/// A legacy GS pipeline: the ES variant of a console VS (POSITION → LDS, s_setpc into the GS)
/// + a GS (each triangle → a 4-vertex strip, the parallelogram a, b, c, b + c − a; TEXCOORD0 =
/// (primitive, vertex)) + a PC PS, through `CreateGraphicsPipelineState`; drawn by the console
/// list's DrawInstanced from a vertex buffer in title memory (two triangles) and by its
/// DrawIndexedInstanced (the second triangle through 16-bit indices).
#[test]
fn gpu_console_gs_pipeline_draws_through_draw_instanced() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dev) = ready() else { return };
    let e = |n: &'static str, enc: &'static str, f: Vec<(&'static str, u64)>| (n, enc, f, None::<u32>);
    // ES: vertex (v9..v12) → LDS at lane·32, then into the GS
    let es = prog(
        &[
            e("V_MBCNT_LO_U32_B32", "ENC_VOP3", vec![("VDST", 20), ("SRC0", 193), ("SRC1", 128)]),
            e("V_MBCNT_HI_U32_B32", "ENC_VOP3", vec![("VDST", 20), ("SRC0", 193), ("SRC1", V + 20)]),
            e("V_LSHLREV_B32", "ENC_VOP2", vec![("VDST", 21), ("SRC0", 133), ("VSRC1", 20)]),
            e("DS_WRITE2_B32", "ENC_DS", vec![("ADDR", 21), ("DATA0", 9), ("DATA1", 10), ("OFFSET0", 0), ("OFFSET1", 1)]),
            e("DS_WRITE2_B32", "ENC_DS", vec![("ADDR", 21), ("DATA0", 11), ("DATA1", 12), ("OFFSET0", 2), ("OFFSET1", 3)]),
            e("S_BARRIER", "ENC_SOPP", vec![]),
            e("S_SETPC_B64", "ENC_SOP1", vec![("SSRC0", 6)]),
        ],
        false,
    );
    // GS: EXEC = primitives of this wave (s3[15:8]), vertices from LDS, parallelogram strip
    let mut gs: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = vec![
        ("S_BFE_U32", "ENC_SOP2", vec![("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x80008)),
        ("S_BFM_B64", "ENC_SOP2", vec![("SDST", 126), ("SSRC0", 0), ("SSRC1", 128)], None),
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", vec![("VDST", 20), ("SRC0", 193), ("SRC1", 128)], None),
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", vec![("VDST", 20), ("SRC0", 193), ("SRC1", V + 20)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", vec![("VDST", 30), ("SRC0", 130), ("VSRC1", 0)], None),
        ("V_AND_B32", "ENC_VOP2", vec![("VDST", 30), ("SRC0", 255), ("VSRC1", 30)], Some(0x3FFFC)),
        ("V_LSHRREV_B32", "ENC_VOP2", vec![("VDST", 31), ("SRC0", 144), ("VSRC1", 0)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", vec![("VDST", 31), ("SRC0", 130), ("VSRC1", 31)], None),
        ("V_LSHLREV_B32", "ENC_VOP2", vec![("VDST", 32), ("SRC0", 130), ("VSRC1", 1)], None),
        ("DS_READ2_B32", "ENC_DS", vec![("VDST", 40), ("ADDR", 30), ("OFFSET0", 0), ("OFFSET1", 1)], None),
        ("DS_READ2_B32", "ENC_DS", vec![("VDST", 42), ("ADDR", 31), ("OFFSET0", 0), ("OFFSET1", 1)], None),
        ("DS_READ2_B32", "ENC_DS", vec![("VDST", 44), ("ADDR", 32), ("OFFSET0", 0), ("OFFSET1", 1)], None),
    ];
    for (d, a, b) in [(46u64, 42u64, 44u64), (47, 43, 45)] {
        gs.push(("V_ADD_F32", "ENC_VOP2", vec![("VDST", d), ("SRC0", V + a), ("VSRC1", b)], None));
        gs.push(("V_SUB_F32", "ENC_VOP2", vec![("VDST", d), ("SRC0", V + d), ("VSRC1", d - 6)], None)); // − x0 / − y0
    }
    // output base: s2 + lane · (4 vertices · 32 bytes)
    gs.push(("V_MAD_U32_U24", "ENC_VOP3", vec![("VDST", 50), ("SRC0", V + 20), ("SRC1", 255), ("SRC2", 2)], Some(128)));
    gs.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 51), ("SRC0", 240)], None)); // 0.5
    gs.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 52), ("SRC0", 242)], None)); // 1.0
    for (k, (x, y)) in [(40u64, 41u64), (42, 43), (44, 45), (46, 47)].into_iter().enumerate() {
        let o = 8 * k as u64;
        gs.push(("DS_WRITE2_B32", "ENC_DS", vec![("ADDR", 50), ("DATA0", x), ("DATA1", y), ("OFFSET0", o), ("OFFSET1", o + 1)], None));
        gs.push(("DS_WRITE2_B32", "ENC_DS", vec![("ADDR", 50), ("DATA0", 51), ("DATA1", 52), ("OFFSET0", o + 2), ("OFFSET1", o + 3)], None));
        gs.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 53), ("SRC0", 128 + k as u64)], None));
        gs.push(("DS_WRITE2_B32", "ENC_DS", vec![("ADDR", 50), ("DATA0", 2), ("DATA1", 53), ("OFFSET0", o + 4), ("OFFSET1", o + 5)], None));
        gs.push(("S_SENDMSG", "ENC_SOPP", vec![("SIMM16", 0x22)], None)); // emit
    }
    gs.push(("S_SENDMSG", "ENC_SOPP", vec![("SIMM16", 0x12)], None)); // cut
    gs.push(("S_SENDMSG", "ENC_SOPP", vec![("SIMM16", 0x3)], None)); // GS done
    let gs = prog(&gs, true);
    let vs_blob = console(1, 2, &es, &signature(&[("POSITION", 0, 0, 3, 0)]), &signature(&[("SV_Position", 0, 1, 3, 0), ("TEXCOORD", 0, 0, 3, 1)]), None);
    let mut r = [0u8; 48];
    r[0..4].copy_from_slice(&3u32.to_le_bytes()); // triangles in
    r[4..8].copy_from_slice(&5u32.to_le_bytes()); // triangle strip out
    r[24] = 2; // geometry
    r[26..28].copy_from_slice(&4u16.to_le_bytes()); // max vertex count
    let gs_blob = console(2, 3, &gs, &signature(&[]), &signature(&[("SV_Position", 0, 1, 3, 0), ("TEXCOORD", 0, 0, 1, 1)]), Some(psv0(&r)));
    let ps = dxc::compile("struct V { float4 pos : SV_Position; nointerpolation uint4 p0 : TEXCOORD0; };\nfloat4 main(V v) : SV_Target0 { return float4(1.0, (float)v.p0.x, (float)v.p0.y, 1.0); }\n", "main", "ps_6_6", &[]).unwrap();

    unsafe {
        let code = |b: &[u8]| D3D12_SHADER_BYTECODE { pShaderBytecode: b.as_ptr() as _, BytecodeLength: b.len() };
        let elems = [D3D12_INPUT_ELEMENT_DESC {
            SemanticName: windows::core::s!("POSITION"),
            Format: DXGI_FORMAT_R32G32B32_FLOAT,
            AlignedByteOffset: 0,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            ..Default::default()
        }];
        let mut d = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            VS: code(&vs_blob),
            GS: code(&gs_blob),
            PS: code(&ps),
            SampleMask: u32::MAX,
            InputLayout: D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: elems.as_ptr(), NumElements: 1 },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
        d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        d.RasterizerState.DepthClipEnable = true.into();
        d.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
        d.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
        let pso: ID3D12PipelineState = dev.CreateGraphicsPipelineState(&d).unwrap_or_else(|e| panic!("{e}: {:?}", real::stats().translation_errors));
        let real_pso = super::pso_async::resolve_wait(pso.as_raw() as usize);
        let tp = real::two_pass_of(real_pso).unwrap_or_else(|| panic!("no two-pass parts: {:?}", real::stats().translation_errors));
        assert_eq!(tp.kind, "GS");
        assert!(matches!(&tp.fetch, Some(Ok(f)) if f.registers == 1 && f.slots == [0] && f.missing.is_empty()), "vertex fetch: {:?}", tp.fetch.as_ref().map(|r| r.as_ref().map(|f| (f.registers, f.slots.clone())).map_err(|e| e.clone())));

        // two triangles (xyz, 12-byte stride) in title memory, then 16-bit indices 3, 4, 5
        let tris: [[[f32; 2]; 3]; 2] = [[[-0.9, -0.9], [-0.5, -0.9], [-0.9, -0.5]], [[0.1, 0.1], [0.5, 0.1], [0.1, 0.5]]];
        let u = crate::gpu::uma::get().unwrap();
        let vb = u.alloc(1 << 16, 1 << 16).unwrap() as usize;
        for (k, v) in tris.iter().flatten().enumerate() {
            std::ptr::copy_nonoverlapping([v[0], v[1], 0.5f32].as_ptr(), (vb as *mut f32).add(3 * k), 3);
        }
        let ib = vb + 0x1000;
        std::ptr::copy_nonoverlapping([3u16, 4, 5].as_ptr(), (ib + 2) as *mut u16, 3); // at an odd 2-byte offset

        let draw = |indexed: bool| -> Vec<[f32; 4]> {
            let t = Target::new(&dev);
            bind::set_pipeline_state(t.backing, pso.as_raw());
            list_x::ia_set_topology_x(t.backing, windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST.0 as u32);
            let view = D3D12_VERTEX_BUFFER_VIEW { BufferLocation: vb as u64, SizeInBytes: 6 * 12, StrideInBytes: 12 };
            list_x::ia_set_vertex_buffers_x(t.backing, 0, 1, &view);
            let before = bind::counts_snapshot();
            if indexed {
                let iv = D3D12_INDEX_BUFFER_VIEW { BufferLocation: (ib + 2) as u64, SizeInBytes: 6, Format: DXGI_FORMAT_R16_UINT };
                list_x::ia_set_index_buffer_x(t.backing, &iv);
                bind::draw_indexed_instanced(t.backing, 3, 1, 0, 0, 0);
            } else {
                bind::draw_instanced(t.backing, 6, 1, 0, 0);
            }
            let after = bind::counts_snapshot();
            assert!(after[0] > before[0], "a translated draw");
            t.finish()
        };
        // the parallelogram of triangle (a, b, c): a, b, c, b + c − a
        let quad = |t: [[f32; 2]; 3]| -> [[[f32; 2]; 3]; 2] {
            let d = [t[1][0] + t[2][0] - t[0][0], t[1][1] + t[2][1] - t[0][1]];
            [[t[0], t[1], t[2]], [t[2], t[1], d]]
        };
        let check = |px: &[[f32; 4]], drawn: &[(usize, [[f32; 2]; 3])]| {
            let (mut hit, mut clear) = (vec![0usize; drawn.len()], 0usize);
            for y in 0..SIZE as usize {
                for x in 0..SIZE as usize {
                    let p = px[y * SIZE as usize + x];
                    let mut near = false;
                    for (k, (prim, t)) in drawn.iter().enumerate() {
                        let q = quad(*t);
                        if q.iter().any(|h| inside(*h, x, y, 0.02)) {
                            assert_eq!((p[0], p[1], p[3]), (1.0, *prim as f32, 1.0), "pixel ({x},{y}) in primitive {prim}'s parallelogram: {p:?}");
                            hit[k] += 1;
                        }
                        near |= q.iter().any(|h| inside(*h, x, y, -0.05));
                    }
                    if !near {
                        assert_eq!(p, CLEAR, "pixel ({x},{y}) outside every parallelogram");
                        clear += 1;
                    }
                }
            }
            println!("two-pass GS draw: {hit:?} pixels in the parallelograms, {clear} clear");
            assert!(hit.iter().all(|&h| h > 10) && clear > 400, "{hit:?} / {clear}");
        };
        check(&draw(false), &[(0, tris[0]), (1, tris[1])]);
        // indexed: one triangle, primitive 0 of the draw, from vertices 3..5
        check(&draw(true), &[(0, tris[1])]);
        u.free(vb);
    }
}

/// The synthetic console NGG program of the DispatchMesh test as a pipeline, for the indirect
/// tests: one triangle per group, the group index added to x.
unsafe fn mesh_pso(dev: &ID3D12Device) -> ID3D12PipelineState {
    let exp = |tgt: u64, en: u64, v: [u64; 4], done: u64| ("EXP", "ENC_EXP", vec![("TGT", tgt), ("EN", en), ("VSRC0", v[0]), ("VSRC1", v[1]), ("VSRC2", v[2]), ("VSRC3", v[3]), ("DONE", done)], None);
    let mut list: Vec<(&str, &str, Vec<(&str, u64)>, Option<u32>)> = vec![
        ("S_BFE_U32", "ENC_SOP2", vec![("SDST", 0), ("SSRC0", 3), ("SSRC1", 255)], Some(0x40018)),
        ("V_MBCNT_LO_U32_B32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 193), ("SRC1", 128)], None),
        ("V_MBCNT_HI_U32_B32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 193), ("SRC1", V)], None),
        ("V_LSHL_ADD_U32", "ENC_VOP3", vec![("VDST", 0), ("SRC0", 0), ("SRC1", 134), ("SRC2", V)], None),
        ("V_READFIRSTLANE_B32", "ENC_VOP1", vec![("VDST", 1), ("SRC0", V + 5)], None),
        ("S_LSHR_B32", "ENC_SOP2", vec![("SDST", 1), ("SSRC0", 1), ("SSRC1", 135)], None),
        ("S_MOV_B32", "ENC_SOP1", vec![("SDST", 124), ("SSRC0", 255)], Some(1 << 12 | 3)),
        ("S_SENDMSG", "ENC_SOPP", vec![("SIMM16", 9)], None),
        ("S_BARRIER", "ENC_SOPP", vec![], None),
        ("V_CMPX_GT_U32", "ENC_VOPC", vec![("SRC0", 131), ("VSRC1", 0)], None),
        ("BUFFER_LOAD_DWORDX4", "ENC_MUBUF", vec![("VDATA", 10), ("VADDR", 0), ("SRSRC", 4), ("SOFFSET", 128), ("IDXEN", 1)], None),
        ("V_CVT_F32_U32", "ENC_VOP1", vec![("VDST", 20), ("SRC0", 1)], None),
        ("V_ADD_F32", "ENC_VOP2", vec![("VDST", 10), ("SRC0", V + 20), ("VSRC1", 10)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 14), ("SRC0", V + 20)], None),
        ("V_CVT_F32_U32", "ENC_VOP1", vec![("VDST", 15), ("SRC0", V)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 16), ("SRC0", 128)], None),
        ("V_MOV_B32", "ENC_VOP1", vec![("VDST", 17), ("SRC0", 242)], None),
    ];
    list.push(exp(12, 0xF, [10, 11, 12, 13], 1));
    list.push(exp(32, 0xF, [14, 15, 16, 17], 0));
    list.push(("S_MOV_B64", "ENC_SOP1", vec![("SDST", 126), ("SSRC0", 193)], None));
    list.push(("V_CMPX_EQ_U32", "ENC_VOPC", vec![("SRC0", 128), ("VSRC1", 0)], None));
    list.push(("V_MOV_B32", "ENC_VOP1", vec![("VDST", 21), ("SRC0", 255)], Some(1 << 10 | 2 << 20)));
    list.push(exp(20, 1, [21, 0, 0, 0], 1));
    let code = prog(&list, true);
    let mut r = [0u8; 48];
    r[12..14].copy_from_slice(&128u16.to_le_bytes());
    r[14..16].copy_from_slice(&128u16.to_le_bytes());
    r[24] = 13;
    r[27] = 2;
    r[36..40].copy_from_slice(&128u32.to_le_bytes());
    r[40..44].copy_from_slice(&1u32.to_le_bytes());
    r[44..48].copy_from_slice(&1u32.to_le_bytes());
    let ms = console(13, 7, &code, &signature(&[]), &signature(&[("SV_Position", 0, 1, 3, 0), ("TEXCOORD", 0, 0, 3, 1)]), Some(psv0(&r)));
    assert!(crate::gpu::recomp::pso::console_mesh(&ms, None, true).is_ok());
    let ps = dxc::compile("struct V { float4 pos : SV_Position; float4 p0 : TEXCOORD0; };\nfloat4 main(V v) : SV_Target0 { return v.p0; }\n", "main", "ps_6_6", &[]).unwrap();
    let game_rs: ID3D12RootSignature = {
        let mut ser = None;
        D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut ser, None).unwrap();
        let ser = ser.unwrap();
        dev.CreateRootSignature(0, std::slice::from_raw_parts(ser.GetBufferPointer() as *const u8, ser.GetBufferSize())).unwrap()
    };
    #[repr(C, align(8))]
    struct Sub<T> {
        ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
        v: T,
    }
    #[repr(C)]
    struct Stream {
        rs: Sub<*mut c_void>,
        ms: Sub<D3D12_SHADER_BYTECODE>,
        ps: Sub<D3D12_SHADER_BYTECODE>,
        rast: Sub<D3D12_RASTERIZER_DESC>,
        depth: Sub<D3D12_DEPTH_STENCIL_DESC>,
        rt: Sub<D3D12_RT_FORMAT_ARRAY>,
    }
    let bcode = |b: &[u8]| D3D12_SHADER_BYTECODE { pShaderBytecode: b.as_ptr() as _, BytecodeLength: b.len() };
    let mut rt = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 1, ..Default::default() };
    rt.RTFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    let st = Stream {
        rs: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, v: game_rs.as_raw() },
        ms: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, v: bcode(&ms) },
        ps: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, v: bcode(&ps) },
        rast: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, v: D3D12_RASTERIZER_DESC { FillMode: D3D12_FILL_MODE_SOLID, CullMode: D3D12_CULL_MODE_NONE, DepthClipEnable: true.into(), ..Default::default() } },
        depth: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, v: D3D12_DEPTH_STENCIL_DESC::default() },
        rt: Sub { ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, v: rt },
    };
    let d2: ID3D12Device2 = dev.cast().unwrap();
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC { SizeInBytes: std::mem::size_of::<Stream>(), pPipelineStateSubobjectStream: &st as *const _ as *mut _ };
    d2.CreatePipelineState(&desc).unwrap_or_else(|e| panic!("{e}: {:?}", real::stats().translation_errors))
}

/// A dispatch command signature XCL knows the layout of (one Dispatch argument, stride 12).
unsafe fn dispatch_signature(dev: &ID3D12Device) -> ID3D12CommandSignature {
    let mut arg: D3D12_INDIRECT_ARGUMENT_DESC = std::mem::zeroed();
    arg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH;
    let desc = D3D12_COMMAND_SIGNATURE_DESC { ByteStride: 12, NumArgumentDescs: 1, pArgumentDescs: &arg, NodeMask: 0 };
    let mut out: Option<ID3D12CommandSignature> = None;
    dev.CreateCommandSignature(&desc, None, &mut out).unwrap();
    let sig = out.unwrap();
    super::indirect::note_signature(sig.as_raw() as usize, super::indirect::Signature::from_args(12, &[[super::indirect::ARG_DISPATCH, 0, 0, 0]]).unwrap());
    sig
}

/// The vertex table of the mesh program and the root constants that point at it.
unsafe fn mesh_table(u: &crate::gpu::uma::Uma) -> (usize, [u32; 12]) {
    let table = u.alloc(1 << 16, 1 << 16).unwrap() as usize;
    for (i, (x, y)) in [(-1.0f32, -1.0f32), (-1.0, 1.0), (0.0, -1.0)].iter().enumerate() {
        std::ptr::copy_nonoverlapping([f(*x), f(*y), f(0.0), f(1.0)].as_ptr(), (table as *mut u32).add(4 * i), 4);
    }
    let mut c = [0u32; 12];
    c[8..12].copy_from_slice(&[table as u32, ((table >> 32) as u32 & 0xFFFF) | 16 << 16, 3, 0]);
    (table, c)
}

/// A two-pass mesh pipeline drawn through ExecuteIndirect with two dispatch records: record 0
/// one group, record 1 two. Both records produce their triangles and each counts as a draw.
#[test]
fn gpu_indirect_two_pass_mesh_runs_each_record() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dev) = ready() else { return };
    unsafe {
        let u = crate::gpu::uma::get().unwrap();
        let pso = mesh_pso(&dev);
        let real_pso = super::pso_async::resolve_wait(pso.as_raw() as usize);
        assert!(real::two_pass_of(real_pso).is_some(), "a two-pass mesh pipeline");
        let (table, constants) = mesh_table(u);
        let rs_key = 0x7400_0020usize;
        super::console_list::note_root_signature(rs_key, &rts0_constants(12));
        let sig = dispatch_signature(&dev);

        // the argument records in title memory: dispatch (1,1,1) then (2,1,1)
        let arg_bytes = 2 * 12usize;
        let args = u.alloc(arg_bytes, 256).unwrap();
        let words: [u32; 12] = [1, 1, 1, 0, 2, 1, 1, 0, 0, 0, 0, 0];
        std::ptr::copy_nonoverlapping(words.as_ptr(), args as *mut u32, words.len());
        let fake_res = 0x1C0_1000usize;
        super::real::note_test_console_buffer(fake_res, args as usize, arg_bytes);

        let t = Target::new(&dev);
        bind::set_pipeline_state(t.backing, pso.as_raw());
        bind::set_graphics_root_signature(t.backing, rs_key as *mut c_void);
        bind::set_graphics_root_constants(t.backing, 0, 12, constants.as_ptr(), 0);
        let before = bind::counts_snapshot();
        bind::execute_indirect(t.backing, sig.as_raw() as usize, 2, fake_res, 0, 0, 0);
        let after = bind::counts_snapshot();
        assert_eq!(after[0] - before[0], 2, "both records ran as translated draws");
        assert_eq!(super::indirect::last_clamped_for_test(), Some(vec![
            super::indirect::Record { kind: super::indirect::Kind::Dispatch, counts: [1, 1, 1, 0, 0] },
            super::indirect::Record { kind: super::indirect::Kind::Dispatch, counts: [2, 1, 1, 0, 0] },
        ]));
        let px = t.finish();
        super::real::forget_test_console_buffer(fake_res);
        u.free(table);
        u.free(args as usize);

        let (mut hit, mut clear) = ([0usize; 2], 0usize);
        for y in 0..SIZE as usize {
            for x in 0..SIZE as usize {
                let p = px[y * SIZE as usize + x];
                let mut covered = false;
                for g in 0..2 {
                    let gf = g as f32;
                    if inside([[-1.0 + gf, -1.0], [-1.0 + gf, 1.0], [gf, -1.0]], x, y, 0.02) {
                        assert_eq!((p[0], p[2], p[3]), (gf, 0.0, 1.0), "pixel ({x},{y}) in group {g}'s triangle: {p:?}");
                        hit[g] += 1;
                        covered = true;
                    }
                }
                let near = (0..2).any(|g| inside([[-1.0 + g as f32, -1.0], [-1.0 + g as f32, 1.0], [g as f32, -1.0]], x, y, -0.05));
                if !covered && !near {
                    assert_eq!(p, CLEAR, "pixel ({x},{y}) outside both triangles");
                    clear += 1;
                }
            }
        }
        println!("indirect two-pass: {hit:?} pixels, {clear} clear");
        assert!(hit[0] > 80 && hit[1] > 80 && clear > 300, "{hit:?} / {clear}");
    }
}

/// A record of all ones is clamped and charged to the list's budget; past the budget it runs
/// empty, so the frame completes and the device stays up.
#[test]
fn gpu_indirect_record_all_ones_is_bounded() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(dev) = ready() else { return };
    unsafe {
        let u = crate::gpu::uma::get().unwrap();
        let pso = mesh_pso(&dev);
        let (table, constants) = mesh_table(u);
        let rs_key = 0x7400_0021usize;
        super::console_list::note_root_signature(rs_key, &rts0_constants(12));
        let sig = dispatch_signature(&dev);

        let args = u.alloc(12, 256).unwrap();
        std::ptr::copy_nonoverlapping([u32::MAX; 3].as_ptr(), args as *mut u32, 3);
        let fake_res = 0x1C0_2000usize;
        super::real::note_test_console_buffer(fake_res, args as usize, 12);

        let t = Target::new(&dev);
        bind::set_pipeline_state(t.backing, pso.as_raw());
        bind::set_graphics_root_signature(t.backing, rs_key as *mut c_void);
        bind::set_graphics_root_constants(t.backing, 0, 12, constants.as_ptr(), 0);
        let before = bind::counts_snapshot();
        bind::execute_indirect(t.backing, sig.as_raw() as usize, 1, fake_res, 0, 0, 0);
        let after = bind::counts_snapshot();
        // the record was clamped to the maxima and then zeroed by the budget: nothing ran
        assert_eq!(after[0], before[0], "no draw ran for the clamped record");
        assert_eq!(super::indirect::last_clamped_for_test(), Some(vec![
            super::indirect::Record { kind: super::indirect::Kind::Dispatch, counts: [0; 5] },
        ]));
        let _ = t.finish(); // the frame completes and the device is not removed
        super::real::forget_test_console_buffer(fake_res);
        u.free(table);
        u.free(args as usize);
    }
}
