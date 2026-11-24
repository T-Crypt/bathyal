//! A resource released right after its submission: the GPU's copy still reads it, and the PC
//! object must stay alive until the fence the submission is covered by has signalled.
//!
//! Skipped (passing) without a real D3D12 device.

use std::ffi::c_void;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::{device_x, fence_x, lifetime, real, title_list};

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

fn buffer(dev: &ID3D12Device, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
    if std::env::var_os("XCL_TEST_ORIG").is_some() {
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
        let mut r: *mut c_void = std::ptr::null_mut();
        let iid = windows::Win32::Graphics::Direct3D12::ID3D12Resource::IID;
        unsafe {
            type F = unsafe extern "system" fn(*mut c_void, *const D3D12_HEAP_PROPERTIES, D3D12_HEAP_FLAGS, *const D3D12_RESOURCE_DESC, D3D12_RESOURCE_STATES, *const D3D12_CLEAR_VALUE, *const windows::core::GUID, *mut *mut c_void) -> i32;
            let f: F = std::mem::transmute(real::orig_fn(dev.as_raw() as *mut c_void, 27));
            let hp = D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() };
            assert!(f(dev.as_raw() as *mut c_void, &hp, D3D12_HEAP_FLAG_NONE, &desc, state, std::ptr::null(), &iid, &mut r) >= 0, "orig CreateCommittedResource");
            ID3D12Resource::from_raw(r)
        }
    } else {
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
    unsafe {
        dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r)
            .expect("CreateCommittedResource");
    }
    r.unwrap()
    }
}

/// The title's `Release` of an adopted object: slot 2 of its vtable.
unsafe fn release(obj: usize) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(obj as *const *const usize)).add(2));
    f(obj as *mut c_void)
}

/// A title releases a buffer it just submitted a copy from; the copy's bytes are right and the
/// device stays up.
#[test]
fn a_resource_released_after_submit_outlives_its_submission() {
    let Some(dev) = ready() else { return };
    unsafe {
        let src = buffer(&dev, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        let dst = buffer(&dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
        let rb = buffer(&dev, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        let pattern: Vec<u8> = (0..BYTES).map(|k| (k as u8).wrapping_mul(7).wrapping_add(3)).collect();
        let mut p = std::ptr::null_mut();
        src.Map(0, None, Some(&mut p)).expect("Map of the upload buffer");
        std::ptr::copy_nonoverlapping(pattern.as_ptr(), p as *mut u8, BYTES as usize);
        src.Unmap(0, None);

        let pure = std::env::var_os("XCL_TEST_PURE").is_some();
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("allocator");
        let mut raw_list: *mut c_void = std::ptr::null_mut();
        let iid = windows_sys::core::GUID::from_u128(0x5b160d0f_ac1b_4185_8ba8_b3ae42a5a455);
        let (hr, backing) = if pure {
            type F = unsafe extern "system" fn(*mut c_void, u32, i32, *mut c_void, *mut c_void, *const windows_sys::core::GUID, *mut *mut c_void) -> i32;
            let f: F = std::mem::transmute(real::orig_fn(dev.as_raw() as *mut c_void, 12));
            let hr = f(dev.as_raw() as *mut c_void, 0, D3D12_COMMAND_LIST_TYPE_DIRECT.0, alloc.as_raw() as *mut c_void, std::ptr::null_mut(), &iid, &mut raw_list);
            (hr, raw_list)
        } else {
            let hr = device_x::create_command_list_pc(dev.as_raw(), 0, D3D12_COMMAND_LIST_TYPE_DIRECT.0 as u32, alloc.as_raw() as usize, 0, &iid, &mut raw_list);
            (hr, title_list::backing(raw_list).expect("a title's list"))
        };
        assert!(hr >= 0 && !backing.is_null(), "CreateCommandList: {hr:#x}");
        let list = ID3D12GraphicsCommandList::from_raw_borrowed(&backing).expect("PC list");

        // two copies name the source, the destination and the readback buffer
        list.CopyBufferRegion(&dst, 0, &src, 0, BYTES);
        list.CopyBufferRegion(&rb, 0, &dst, 0, BYTES);
        for r in [&src, &dst] {
            lifetime::note(backing as usize, r.as_raw() as usize);
        }
        list.Close().expect("Close");

        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).expect("queue");
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
        if pure && std::env::var_os("XCL_TEST_MIX").is_none() {
            let qdesc = D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() };
            type QF = unsafe extern "system" fn(*mut c_void, *const D3D12_COMMAND_QUEUE_DESC, *const windows::core::GUID, *mut *mut c_void) -> i32;
            let qf: QF = std::mem::transmute(real::orig_fn(dev.as_raw() as *mut c_void, 8));
            let mut qraw: *mut c_void = std::ptr::null_mut();
            let qiid = windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue::IID;
            assert!(qf(dev.as_raw() as *mut c_void, &qdesc, &qiid, &mut qraw) >= 0);
            type EF = unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void);
            let ef: EF = std::mem::transmute(real::orig_fn(qraw, 10));
            let one = [backing];
            ef(qraw, 1, one.as_ptr());
        } else if std::env::var_os("XCL_TEST_RAWEXEC").is_some() {
            let ef: unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void) = std::mem::transmute(real::orig_fn(queue.as_raw() as *mut c_void, 10));
            let one = [list.as_raw() as *mut c_void];
            ef(queue.as_raw() as *mut c_void, 1, one.as_ptr());
        } else {
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
        }

        // the title's references go now, on a GPU that may still be running the copy
        release(src.as_raw() as usize);
        release(dst.as_raw() as usize);
        if std::env::var_os("XCL_TEST_ORIG").is_none() {
            assert!(lifetime::held(src.as_raw() as usize), "a buffer a submitted list names must still be held");
        }
        queue.Signal(&fence, 1).expect("Signal");
        fence_x::note_queue_signal(queue.as_raw() as usize, fence.as_raw() as usize, 1);

        let t0 = std::time::Instant::now();
        while fence.GetCompletedValue() < 1 {
            assert!(t0.elapsed().as_secs() < 20, "the GPU did not finish (device removed: {:?})", dev.GetDeviceRemovedReason());
            std::thread::yield_now();
        }
        lifetime::poll();
        assert!(dev.GetDeviceRemovedReason().is_ok(), "device removed: {:?}", dev.GetDeviceRemovedReason());
        if std::env::var_os("XCL_TEST_ORIG").is_none() {
            assert!(!lifetime::held(src.as_raw() as usize), "the buffer was not freed after its fence");
        }

        let mut q = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut q)).expect("Map of the readback buffer");
        let got = std::slice::from_raw_parts(q as *const u8, BYTES as usize).to_vec();
        rb.Unmap(0, None);
        assert_eq!(got, pattern, "the copy read memory that was not the source's");
        release(rb.as_raw() as usize);

        // the title's list goes last (its allocator is what keeps its unified memory)
        release(raw_list as usize);
        drop(alloc);
    }
}
