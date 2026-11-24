//! A resource released right after its submission: the GPU's copy still reads it, and the PC
//! object must stay alive until the fence the submission is covered by has signalled.
//!
//! The list and the queue are made from the PC device's own entry points (as a title's submission
//! is), the buffers come through the D3D12.X device, and the association between the list and
//! the fence is the one a recording and a queue Signal make (`lifetime::note`, `submit`,
//! `signal`). Skipped (passing) without a real D3D12 device.

use std::ffi::c_void;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::{lifetime, real};

const BYTES: u64 = 64;

/// The backend device, when it can run this test.
fn ready() -> Option<ID3D12Device> {
    if super::backend() != super::Backend::Real {
        println!("no real D3D12 device – skipping");
        return None;
    }
    if real::software_adapter() {
        println!("software adapter (no GPU) – skipping");
        crate::gpu::recomp::runner::skip_test("no GPU");
        return None;
    }
    let raw = real::device_raw();
    unsafe { ID3D12Device::from_raw_borrowed(&raw) }.cloned()
}

/// A buffer through the D3D12.X device (an object a recorded list can hold).
unsafe fn tracked_buffer(dev: &ID3D12Device, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: BYTES,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut r: Option<ID3D12Resource> = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r)
        .expect("CreateCommittedResource");
    r.unwrap()
}

/// The title's `Release` of an object: slot 2 of its vtable.
unsafe fn release(obj: usize) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(obj as *const *const usize)).add(2));
    f(obj as *mut c_void)
}

/// A title releases a buffer it just submitted a copy from; the copy's bytes are right, the
/// buffer outlives the submission, and the device stays up.
#[test]
fn a_resource_released_after_submit_outlives_its_submission() {
    let Some(dev) = ready() else { return };
    unsafe {
        let d = dev.as_raw() as *mut c_void;
        let src = tracked_buffer(&dev, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        let dst = tracked_buffer(&dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
        let rb = tracked_buffer(&dev, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
                eprintln!("m1 buffers");
let pattern: Vec<u8> = (0..BYTES).map(|k| (k as u8).wrapping_mul(7).wrapping_add(3)).collect();
                eprintln!("m2 before map");
let mut p = std::ptr::null_mut();
        src.Map(0, None, Some(&mut p)).expect("Map of the upload buffer");
        std::ptr::copy_nonoverlapping(pattern.as_ptr(), p as *mut u8, BYTES as usize);
        src.Unmap(0, None);

        eprintln!("m3 filled");
        // the queue and the list from the PC device's own entries
        type QF = unsafe extern "system" fn(*mut c_void, *const D3D12_COMMAND_QUEUE_DESC, *const windows::core::GUID, *mut *mut c_void) -> i32;
        let qf: QF = std::mem::transmute(real::orig_fn(d, 8));
        let qdesc = D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() };
        let mut qraw: *mut c_void = std::ptr::null_mut();
        let qiid = ID3D12CommandQueue::IID;
        assert!(qf(d, &qdesc, &qiid, &mut qraw) >= 0, "CreateCommandQueue");
        let queue = ID3D12CommandQueue::from_raw_borrowed(&qraw).expect("queue");
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("allocator");
        type LF = unsafe extern "system" fn(*mut c_void, u32, i32, *mut c_void, *mut c_void, *const windows::core::GUID, *mut *mut c_void) -> i32;
        let lf: LF = std::mem::transmute(real::orig_fn(d, 12));
        let mut raw_list: *mut c_void = std::ptr::null_mut();
        let liid = ID3D12GraphicsCommandList::IID;
        assert!(lf(d, 0, D3D12_COMMAND_LIST_TYPE_DIRECT.0, alloc.as_raw() as *mut c_void, std::ptr::null_mut(), &liid, &mut raw_list) >= 0, "CreateCommandList");
        let list = ID3D12GraphicsCommandList::from_raw_borrowed(&raw_list).expect("PC list");

        eprintln!("m4 list made");
        // two copies name the source, the destination and the readback buffer
        list.CopyBufferRegion(&dst, 0, &src, 0, BYTES);
        list.CopyBufferRegion(&rb, 0, &dst, 0, BYTES);
        for r in [&src, &dst] {
            lifetime::note(raw_list as usize, r.as_raw() as usize);
        }
        eprintln!("m5 copied");
        list.Close().expect("Close");

        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
        let one = [raw_list];
        let ef: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void) = std::mem::transmute(real::orig_fn(qraw, 10));
        lifetime::submit(qraw as usize, &one);
        ef(qraw, 1, one.as_ptr());

        eprintln!("m6 submitted");
        // the title's references go now, on a GPU that may still be running the copy
        release(src.as_raw() as usize);
        release(dst.as_raw() as usize);
        assert!(lifetime::held(src.as_raw() as usize), "a buffer a submitted list names must still be held");
        queue.Signal(&fence, 1).expect("Signal");
        lifetime::signal(qraw as usize, fence.as_raw() as usize, 1);

        eprintln!("m7 released+signal");
        let t0 = std::time::Instant::now();
        while fence.GetCompletedValue() < 1 {
            assert!(t0.elapsed().as_secs() < 20, "the GPU did not finish (device removed: {:?})", dev.GetDeviceRemovedReason());
            std::thread::yield_now();
        }
        lifetime::poll();
        assert!(dev.GetDeviceRemovedReason().is_ok(), "device removed: {:?}", dev.GetDeviceRemovedReason());
        assert!(!lifetime::held(src.as_raw() as usize), "the buffer was not freed after its fence");

        eprintln!("m8 waited");
        let mut q = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut q)).expect("Map of the readback buffer");
        let got = std::slice::from_raw_parts(q as *const u8, BYTES as usize).to_vec();
        rb.Unmap(0, None);
        assert_eq!(got, pattern, "the copy read memory that was not the source's");

        eprintln!("m9 read back");
        release(rb.as_raw() as usize);
        release(raw_list as usize);
        release(qraw as usize);
        drop(alloc);
    }
}
