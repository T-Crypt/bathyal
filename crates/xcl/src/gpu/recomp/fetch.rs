//! The vertex fetch of a legacy GS pipeline drawn in two passes.
//!
//! The geometry-mode compute pass (`Mode::Geometry`) reads its ES vertex inputs pre-fetched
//! from `mem`: one float4 (raw bits) per ES input register for each vertex of each triangle,
//! at io[`IO_GS_VERTICES`] (`super::IO_GS_VERTICES`). The console's fetch shader would have
//! made them from the title's vertex buffers; here a small compute shader does, from the
//! pipeline's input layout and the list's vertex and index buffers, before the draw's two
//! passes.
//!
//! Its root signature ([`root_signature`]): b0 = [`CONSTANTS`] root constants (see
//! [`Constants`]), t0 = index buffer, t1..t8 = the vertex buffer slots the layout uses
//! ([`MAX_SLOTS`]), u0 = the fetched vertices. Buffers are bound at a 4-byte aligned address
//! with the byte delta to the view's start and the view's size in the constants: nothing is
//! read past a view (the root descriptors have no bounds).

use std::fmt::Write;

/// vertex buffer slots one fetch can read
pub const MAX_SLOTS: usize = 8;
/// root constants of the fetch shader (dwords)
pub const CONSTANTS: u32 = 8 + 4 * MAX_SLOTS as u32;
/// threads per group of the fetch shader (one vertex each)
pub const THREADS: u32 = 64;

/// One ES input register the fetch fills: from vertex buffer `slot` (index into the fetch's
/// slot list) at byte `offset` of each vertex (or instance), in DXGI format `format`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Element {
    pub register: u32,
    pub slot: u32,
    pub offset: u32,
    pub format: u32,
    pub per_instance: bool,
}

/// The fetch shader's root constants.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct Constants {
    /// vertices to fetch (3 per triangle)
    pub count: u32,
    /// first index (indexed) or first vertex
    pub first: u32,
    pub base_vertex: i32,
    /// bit 0: indexed, bit 1: 16-bit indices
    pub flags: u32,
    /// index buffer: bytes from the bound address to the view, bytes readable from the bound address
    pub ib_delta: u32,
    pub ib_bytes: u32,
    pub first_instance: u32,
    pub pad: u32,
    /// per slot: delta, stride, bytes readable, 0
    pub slots: [[u32; 4]; MAX_SLOTS],
}

impl Constants {
    pub fn words(&self) -> Vec<u32> {
        let mut w = vec![self.count, self.first, self.base_vertex as u32, self.flags, self.ib_delta, self.ib_bytes, self.first_instance, self.pad];
        for s in &self.slots {
            w.extend_from_slice(s);
        }
        w
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Num {
    Float,
    Unorm,
    Snorm,
    Uint,
    Sint,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pack {
    /// n components of 32 / 16 / 8 bits
    Bits(u32),
    /// R10G10B10A2
    Rgb10a2,
}

/// (components, packing, number kind, BGRA order) of a DXGI vertex format.
fn format_info(f: u32) -> Option<(u32, Pack, Num, bool)> {
    use Num::*;
    use Pack::*;
    Some(match f {
        2 => (4, Bits(32), Float, false),
        3 => (4, Bits(32), Uint, false),
        4 => (4, Bits(32), Sint, false),
        6 => (3, Bits(32), Float, false),
        7 => (3, Bits(32), Uint, false),
        8 => (3, Bits(32), Sint, false),
        10 => (4, Bits(16), Float, false),
        11 => (4, Bits(16), Unorm, false),
        12 => (4, Bits(16), Uint, false),
        13 => (4, Bits(16), Snorm, false),
        14 => (4, Bits(16), Sint, false),
        16 => (2, Bits(32), Float, false),
        17 => (2, Bits(32), Uint, false),
        18 => (2, Bits(32), Sint, false),
        24 => (4, Rgb10a2, Unorm, false),
        25 => (4, Rgb10a2, Uint, false),
        28 => (4, Bits(8), Unorm, false),
        30 => (4, Bits(8), Uint, false),
        31 => (4, Bits(8), Snorm, false),
        32 => (4, Bits(8), Sint, false),
        34 => (2, Bits(16), Float, false),
        35 => (2, Bits(16), Unorm, false),
        36 => (2, Bits(16), Uint, false),
        37 => (2, Bits(16), Snorm, false),
        38 => (2, Bits(16), Sint, false),
        41 => (1, Bits(32), Float, false),
        42 => (1, Bits(32), Uint, false),
        43 => (1, Bits(32), Sint, false),
        49 => (2, Bits(8), Unorm, false),
        50 => (2, Bits(8), Uint, false),
        51 => (2, Bits(8), Snorm, false),
        52 => (2, Bits(8), Sint, false),
        54 => (1, Bits(16), Float, false),
        56 => (1, Bits(16), Unorm, false),
        57 => (1, Bits(16), Uint, false),
        58 => (1, Bits(16), Snorm, false),
        59 => (1, Bits(16), Sint, false),
        61 => (1, Bits(8), Unorm, false),
        62 => (1, Bits(8), Uint, false),
        63 => (1, Bits(8), Snorm, false),
        64 => (1, Bits(8), Sint, false),
        87 => (4, Bits(8), Unorm, true),
        _ => return None,
    })
}

/// Whether the fetch can read DXGI format `f`.
pub fn supported(f: u32) -> bool {
    format_info(f).is_some()
}

/// Bytes of one element in DXGI format `f` (for D3D12_APPEND_ALIGNED_ELEMENT).
pub fn bytes(f: u32) -> Option<u32> {
    format_info(f).map(|(n, pack, _, _)| match pack {
        Pack::Bits(b) => n * b / 8,
        Pack::Rgb10a2 => 4,
    })
}

/// HLSL that makes component `i` (raw bits, uint) of the element whose words are w0..w3.
fn component(i: u32, pack: Pack, num: Num) -> String {
    let (raw, bits) = match pack {
        Pack::Bits(32) => (format!("w{i}"), 32),
        Pack::Bits(b) => {
            let per_word = 32 / b;
            let (w, sh) = (i / per_word, (i % per_word) * b);
            (format!("((w{w} >> {sh}u) & {:#x}u)", (1u64 << b) - 1), b)
        }
        Pack::Rgb10a2 => {
            let (sh, b) = [(0, 10), (10, 10), (20, 10), (30, 2)][i as usize];
            (format!("((w0 >> {sh}u) & {:#x}u)", (1u32 << b) - 1), b)
        }
    };
    let max = ((1u64 << bits) - 1) as f64;
    let smax = ((1u64 << (bits - 1)) - 1) as f64;
    let sext = |r: &str| if bits == 32 { format!("int({r})") } else { format!("(int({r} << {}u) >> {})", 32 - bits, 32 - bits) };
    match (num, bits) {
        (Num::Float, 32) | (Num::Uint, _) => raw,
        (Num::Sint, _) => format!("uint({})", sext(&raw)),
        (Num::Float, 16) => format!("asuint(f16tof32({raw}))"),
        (Num::Float, _) => raw,
        (Num::Unorm, _) => format!("asuint(float({raw}) / {max:.1})"),
        (Num::Snorm, _) => format!("asuint(max(float({}) / {smax:.1}, -1.0))", sext(&raw)),
    }
}

/// The fetch compute shader for ES input registers `0..registers` (each vertex: `registers`
/// float4s, registers without an element zero).
pub fn hlsl(registers: u32, elements: &[Element]) -> Result<String, String> {
    let mut h = String::new();
    h.push_str("ByteAddressBuffer ib : register(t0);\n");
    for s in 0..MAX_SLOTS {
        let _ = writeln!(h, "ByteAddressBuffer vb{s} : register(t{});", s + 1);
    }
    h.push_str("RWByteAddressBuffer outb : register(u0);\n");
    let _ = writeln!(
        h,
        "cbuffer C : register(b0) {{ uint count; uint first; int base_vertex; uint flags; uint ib_delta; uint ib_bytes; uint first_instance; uint pad0; uint4 slot[{MAX_SLOTS}]; }};"
    );
    // a dword at any byte address, 0 past the readable bytes
    h.push_str(
        "uint ld(ByteAddressBuffer b, uint a, uint size)\n{\n    if (a + 4u > size) return 0u;\n    uint s = (a & 3u) * 8u;\n    uint lo = b.Load(a & ~3u);\n    if (s == 0u) return lo;\n    uint hi = (a & ~3u) + 8u <= size ? b.Load((a & ~3u) + 4u) : 0u;\n    return (lo >> s) | (hi << (32u - s));\n}\n",
    );
    let _ = writeln!(h, "[numthreads({THREADS}, 1, 1)]\nvoid main(uint id : SV_DispatchThreadID)\n{{");
    h.push_str("    if (id >= count) return;\n    uint vtx;\n");
    h.push_str("    if ((flags & 1u) != 0u) {\n        uint a = ib_delta + (first + id) * ((flags & 2u) != 0u ? 2u : 4u);\n        uint w = a + ((flags & 2u) != 0u ? 2u : 4u) <= ib_bytes ? ib.Load(a & ~3u) : 0u;\n        uint i = (flags & 2u) != 0u ? ((a & 2u) != 0u ? (w >> 16) : (w & 0xFFFFu)) : w;\n        vtx = uint(int(i) + base_vertex);\n    } else {\n        vtx = first + id;\n    }\n");
    let _ = writeln!(h, "    uint o = id * {}u;", registers * 16);
    for r in 0..registers {
        let Some(e) = elements.iter().find(|e| e.register == r) else {
            let _ = writeln!(h, "    outb.Store4(o + {}u, uint4(0u, 0u, 0u, 0u));", r * 16);
            continue;
        };
        if e.slot as usize >= MAX_SLOTS {
            return Err(format!("register {r}: vertex buffer slot {} of more than {MAX_SLOTS}", e.slot));
        }
        let (n, pack, num, bgra) = format_info(e.format).ok_or_else(|| format!("register {r}: vertex format {} not fetched yet", e.format))?;
        let words = match pack {
            Pack::Bits(b) => (n * b).div_ceil(32),
            Pack::Rgb10a2 => 1,
        };
        let index = if e.per_instance { "first_instance" } else { "vtx" };
        let s = e.slot;
        let _ = writeln!(h, "    {{\n        uint a = slot[{s}].x + {index} * slot[{s}].y + {}u;", e.offset);
        for w in 0..words {
            let _ = writeln!(h, "        uint w{w} = ld(vb{s}, a + {}u, slot[{s}].z);", 4 * w);
        }
        let one = if matches!(num, Num::Uint | Num::Sint) { "1u" } else { "0x3F800000u" };
        let mut c: Vec<String> = (0..4).map(|i| if i < n { component(i, pack, num) } else if i == 3 { one.into() } else { "0u".into() }).collect();
        if bgra {
            c.swap(0, 2);
        }
        let _ = writeln!(h, "        outb.Store4(o + {}u, uint4({}, {}, {}, {}));\n    }}", r * 16, c[0], c[1], c[2], c[3]);
    }
    h.push_str("}\n");
    Ok(h)
}

/// The fetch shader's root signature (see the module docs).
pub fn root_signature(dev: &windows::Win32::Graphics::Direct3D12::ID3D12Device) -> Result<windows::Win32::Graphics::Direct3D12::ID3D12RootSignature, String> {
    use windows::Win32::Graphics::Direct3D12::*;
    let mut params = vec![D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: CONSTANTS } },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }];
    for t in 0..=MAX_SLOTS as u32 {
        params.push(D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
            Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: t, RegisterSpace: 0 } },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
    }
    params.push(D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
        Anonymous: D3D12_ROOT_PARAMETER_0 { Descriptor: D3D12_ROOT_DESCRIPTOR { ShaderRegister: 0, RegisterSpace: 0 } },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    });
    let desc = D3D12_ROOT_SIGNATURE_DESC { NumParameters: params.len() as u32, pParameters: params.as_ptr(), ..Default::default() };
    unsafe {
        let mut blob = None;
        let mut err = None;
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, Some(&mut err)).map_err(|e| format!("serialize the fetch root signature: {e}"))?;
        let blob = blob.ok_or("no root signature blob")?;
        dev.CreateRootSignature(0, std::slice::from_raw_parts(blob.GetBufferPointer() as *const u8, blob.GetBufferSize())).map_err(|e| format!("fetch root signature: {e}"))
    }
}

/// root parameter of the index buffer, of vertex buffer slot 0, of the output
pub const PARAM_IB: u32 = 1;
pub const PARAM_VB0: u32 = 2;
pub const PARAM_OUT: u32 = 2 + MAX_SLOTS as u32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetch_shader_reads_each_register_once_and_zeroes_the_rest() {
        let e = [
            Element { register: 0, slot: 0, offset: 0, format: 6, per_instance: false },
            Element { register: 2, slot: 1, offset: 4, format: 28, per_instance: true },
        ];
        let h = hlsl(3, &e).unwrap();
        assert!(h.contains("uint o = id * 48u;"));
        assert!(h.contains("outb.Store4(o + 16u, uint4(0u, 0u, 0u, 0u));"), "{h}");
        // R32G32B32_FLOAT: three words, w = 1.0
        assert!(h.contains("outb.Store4(o + 0u, uint4(w0, w1, w2, 0x3F800000u));"), "{h}");
        // R8G8B8A8_UNORM per instance, one word
        assert!(h.contains("slot[1].x + first_instance * slot[1].y + 4u"), "{h}");
        assert!(h.contains("asuint(float(((w0 >> 8u) & 0xffu)) / 255.0)"), "{h}");
        assert!(hlsl(1, &[Element { register: 0, slot: 0, offset: 0, format: 26, per_instance: false }]).is_err(), "R11G11B10 not fetched yet");
        assert!(hlsl(1, &[Element { register: 0, slot: 9, offset: 0, format: 2, per_instance: false }]).is_err());
        assert_eq!(Constants::default().words().len(), CONSTANTS as usize);
    }

    #[test]
    fn signed_and_bgra_components() {
        assert_eq!(component(1, Pack::Bits(16), Num::Sint), "uint((int(((w0 >> 16u) & 0xffffu) << 16u) >> 16))");
        let h = hlsl(1, &[Element { register: 0, slot: 0, offset: 0, format: 87, per_instance: false }]).unwrap();
        assert!(h.contains("uint4(asuint(float(((w0 >> 16u) & 0xffu)) / 255.0), asuint(float(((w0 >> 8u) & 0xffu)) / 255.0), asuint(float(((w0 >> 0u) & 0xffu)) / 255.0)"), "{h}");
    }
}
