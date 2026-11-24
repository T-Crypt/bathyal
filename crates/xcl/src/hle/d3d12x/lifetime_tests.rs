//! A resource released right after its submission: the GPU's work still reads it, and the
//! PC object must stay alive until the fence the submission is covered by has signalled.
//!
//! Skipped (passing) without a real D3D12 device or without unified memory open on the GPU.

use std::ffi::c_void;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::{fence_x, lifetime, list_x, real};

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
    let dev = unsafe { ID3D12Device::from_raw_borrowed(&raw) }?.clone();
    Some(dev)
}

fn buffer(dev: &ID3D12Device, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
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

/// The title's `Release` of an adopted object: slot 2 of its vtable.
unsafe fn release(obj: usize) -> u32 {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(*(*(obj as *const *const usize)).add(2));
    f(obj as *mut c_void)
}

/// A title releases a buffer it just submitted a copy from; the copy and the released bytes
/// are both still right, and the device stays up.
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

        let alloc: ID3D12CommandAllocator = dev.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).expect("allocator");
        let list: ID3D12GraphicsCommandList = dev.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None::<&ID3D12PipelineState>).expect("CreateCommandList");
        let raw_list = list.as_raw() as *mut c_void;

        // the two copies name the source, the destination and the readback buffer
        list_x::copy_buffer_region_x(raw_list, dst.as_raw() as usize, 0, src.as_raw() as usize, 0, BYTES);
        list_x::copy_buffer_region_x(raw_list, rb.as_raw() as usize, 0, dst.as_raw() as usize, 0, BYTES);
        list.Close().expect("Close");

        let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).expect("queue");
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
        queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);

        // the title's references go now, on a GPU that may still be running the copy
        if std::env::var_os("XCL_TEST_KEEP").is_none() {
            release(src.as_raw() as usize);
            release(dst.as_raw() as usize);
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
        if std::env::var_os("XCL_TEST_KEEP").is_none() {
            assert!(!lifetime::held(src.as_raw() as usize), "the buffer was not freed after its fence");
        }

        let mut q = std::ptr::null_mut();
        rb.Map(0, None, Some(&mut q)).expect("Map of the readback buffer");
        let got = std::slice::from_raw_parts(q as *const u8, BYTES as usize).to_vec();
        rb.Unmap(0, None);
        assert_eq!(got, pattern, "the copy read memory that was not the source's");
        release(rb.as_raw() as usize);

        // the list goes last (its allocator is what keeps its unified memory)
        drop(list);
        drop(alloc);
    }
}
