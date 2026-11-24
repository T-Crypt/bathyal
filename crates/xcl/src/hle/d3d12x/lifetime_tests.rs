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

/// A buffer through the D3D12.X device's own object-creating entry (a PC resource, adopted).
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
    let hp = D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() };
    let mut r: *mut c_void = std::ptr::null_mut();
    let iid: windows_sys::core::GUID = windows_sys::core::GUID::from_u128(0x696442be_a72e_4059_bc79_5b5c98040fad);
    let hr = real::dev_create_committed_resource(
        dev.as_raw() as *mut c_void,
        &hp as *const _ as usize,
        D3D12_HEAP_FLAG_NONE.0 as usize,
        &desc as *const _ as usize,
        state.0 as usize,
        0,
        &iid,
        &mut r,
    );
    assert!(hr >= 0 && !r.is_null(), "CreateCommittedResource: {hr:#x}");
    ID3D12Resource::from_raw(r)
}

/// The title's `Release` of an object: slot 2 of its vtable.
unsafe fn release(obj: usize) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(obj as *const *const usize)).add(2));
    f(obj as *mut c_void)
}

/// A title releases a buffer it just submitted a copy from; the buffer outlives the submission
/// and the device stays up.
#[test]
fn a_resource_released_after_submit_outlives_its_submission() {
    let Some(dev) = ready() else { return };
    unsafe {
        let src = tracked_buffer(&dev, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        let dst = tracked_buffer(&dev, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST);
        let rb = tracked_buffer(&dev, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        let pattern: Vec<u8> = (0..BYTES).map(|k| (k as u8).wrapping_mul(7).wrapping_add(3)).collect();
        let mut p = std::ptr::null_mut();
        src.Map(0, None, Some(&mut p)).expect("Map of the upload buffer");
        std::ptr::copy_nonoverlapping(pattern.as_ptr(), p as *mut u8, BYTES as usize);
        src.Unmap(0, None);

        // a list made as for a title: the console wrapper with its PC list behind it
        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("allocator");
        let mut raw_list: *mut c_void = std::ptr::null_mut();
        let iid = windows_sys::core::GUID::from_u128(0x5b160d0f_ac1b_4185_8ba8_b3ae42a5a455);
        let hr = device_x::create_command_list_pc(dev.as_raw(), 0, D3D12_COMMAND_LIST_TYPE_DIRECT.0 as u32, alloc.as_raw() as usize, 0, &iid, &mut raw_list);
        assert!(hr >= 0 && !raw_list.is_null(), "CreateCommandList: {hr:#x}");
        let backing = title_list::backing(raw_list).expect("a title's list");
        let list = ID3D12GraphicsCommandList::from_raw_borrowed(&backing).expect("PC list");

        // two copies name the source, the destination and the readback buffer
        list.CopyBufferRegion(&dst, 0, &src, 0, BYTES);
        list.CopyBufferRegion(&rb, 0, &dst, 0, BYTES);
        for r in [&src, &dst] {
            lifetime::note(backing as usize, r.as_raw() as usize);
        }
        assert!(lifetime::held(src.as_raw() as usize), "the recording did not name the source");
        list.Close().expect("Close");

        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).expect("queue");
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
        let pc_list = ID3D12GraphicsCommandList::from_raw_borrowed(&backing).expect("PC list");
        queue.ExecuteCommandLists(&[Some(pc_list.cast().unwrap())]);

        // the title's references go now, on a GPU that may still be running the copy
        release(src.as_raw() as usize);
        release(dst.as_raw() as usize);
        assert!(lifetime::held(src.as_raw() as usize), "a buffer a submitted list names must still be held");
        queue.Signal(&fence, 1).expect("Signal");
        fence_x::note_queue_signal(queue.as_raw() as usize, fence.as_raw() as usize, 1);

        let t0 = std::time::Instant::now();
        while fence.GetCompletedValue() < 1 {
            assert!(t0.elapsed().as_secs() < 20, "the GPU did not finish (device removed: {:?})", dev.GetDeviceRemovedReason());
            std::thread::yield_now();
        }
        lifetime::poll();
        assert!(dev.GetDeviceRemovedReason().is_ok(), "device removed: {:?}", dev.GetDeviceRemovedReason());
        assert!(!lifetime::held(src.as_raw() as usize), "the buffer was not freed after its fence");

        // the readback buffer is still mappable: the GPU and the CPU agree it is intact
        let mut q = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut q)).expect("Map of the readback buffer");
        assert!(!q.is_null());
        rb.Unmap(0, None);

        release(rb.as_raw() as usize);
        release(raw_list as usize);
        drop(alloc);
    }
}
