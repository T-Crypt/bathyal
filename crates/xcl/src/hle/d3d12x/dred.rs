//! DRED (Device Removed Extended Data) log formatting.
//!
//! The driver reports, for each command list, the ops it recorded and the index of the last
//! one that completed. Printing all of them is too much for a log: a window of three ops
//! before and after the last completed one names the operation that hung. This module keeps
//! that formatting pure so it is tested without a device.

/// The name of a D3D12_AUTO_BREADCRUMB_OP value (D3D12 headers, 0..=48); an unknown value is
/// printed as its number.
pub fn op_name(op: i32) -> String {
    let name = match op {
        0 => "SetMarker",
        1 => "BeginEvent",
        2 => "EndEvent",
        3 => "DrawInstanced",
        4 => "DrawIndexedInstanced",
        5 => "ExecuteIndirect",
        6 => "Dispatch",
        7 => "CopyBufferRegion",
        8 => "CopyTextureRegion",
        9 => "CopyResource",
        10 => "CopyTiles",
        11 => "ResolveSubresource",
        12 => "ClearRenderTargetView",
        13 => "ClearUnorderedAccessView",
        14 => "ClearDepthStencilView",
        15 => "ResourceBarrier",
        16 => "ExecuteBundle",
        17 => "Present",
        18 => "ResolveQueryData",
        19 => "BeginSubmission",
        20 => "EndSubmission",
        21 => "DecodeFrame",
        22 => "ProcessFrames",
        23 => "AtomicCopyBufferUINT",
        24 => "AtomicCopyBufferUINT64",
        25 => "ResolveSubresourceRegion",
        26 => "WriteBufferImmediate",
        27 => "DecodeFrame1",
        28 => "SetProtectedResourceSession",
        29 => "DecodeFrame2",
        30 => "ProcessFrames1",
        31 => "BuildRaytracingAccelerationStructure",
        32 => "EmitRaytracingAccelerationStructurePostbuildInfo",
        33 => "CopyRaytracingAccelerationStructure",
        34 => "DispatchRays",
        35 => "InitializeMetaCommand",
        36 => "ExecuteMetaCommand",
        37 => "EstimateMotion",
        38 => "ResolveMotionVectorHeap",
        39 => "SetPipelineState1",
        40 => "InitializeExtensionCommand",
        41 => "ExecuteExtensionCommand",
        42 => "DispatchMesh",
        43 => "EncodeFrame",
        44 => "ResolveEncoderOutputMetadata",
        45 => "Barrier",
        46 => "BeginCommandList",
        47 => "DispatchGraph",
        48 => "SetProgram",
        _ => return op.to_string(),
    };
    name.to_string()
}

/// The ops of a command list around `last` (three before, three after): `index:Name`, with
/// the last completed one marked `>`. `last` past the end of `ops` marks the last op.
pub fn breadcrumb_window(ops: &[i32], last: u32) -> String {
    if ops.is_empty() {
        return String::new();
    }
    let at = (last as usize).min(ops.len() - 1);
    let start = at.saturating_sub(3);
    let end = (at + 4).min(ops.len());
    let mut s = String::new();
    for (i, &op) in ops.iter().enumerate().take(end).skip(start) {
        if !s.is_empty() {
            s.push(' ');
        }
        if i == at {
            s.push('>');
        }
        s.push_str(&format!("{i}:{}", op_name(op)));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breadcrumb_window_shows_three_each_side() {
        let ops: Vec<i32> = (0..=10).collect();
        assert_eq!(
            breadcrumb_window(&ops, 5),
            "2:EndEvent 3:DrawInstanced 4:DrawIndexedInstanced >5:ExecuteIndirect 6:Dispatch 7:CopyBufferRegion 8:CopyTextureRegion"
        );
    }

    #[test]
    fn breadcrumb_window_clamps_at_the_ends() {
        assert_eq!(breadcrumb_window(&[3, 4, 5], 0), ">0:DrawInstanced 1:DrawIndexedInstanced 2:ExecuteIndirect");
        assert_eq!(breadcrumb_window(&[3, 4, 5], 9), "0:DrawInstanced 1:DrawIndexedInstanced >2:ExecuteIndirect");
        assert_eq!(breadcrumb_window(&[], 0), "");
    }

    #[test]
    fn unknown_op_is_its_number() {
        assert_eq!(op_name(6), "Dispatch");
        assert_eq!(op_name(49), "49");
        assert_eq!(op_name(1000), "1000");
    }
}
