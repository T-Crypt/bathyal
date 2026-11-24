# Notice

Zenith-X is a modified version of XCL (https://github.com/ivall/xcl), copyright its authors, released
under the GNU General Public License version 2 (`LICENSE`). The full XCL history is kept in this
repository; each commit names its author. Changes made in Zenith-X are listed in its commit history
from the fork point (`60565c473`) on, and are released under the same license.

Third-party components keep their own notices. The full list, with the files each one covers, is in
`THIRD_PARTY.md`. The sources whose code or data is part of Zenith-X, and their licenses, are:

- AnyPS5 (https://github.com/boykopovar/AnyPS5): GPL-2.0-only, portions of the recompiler logic,
  ported with annotation.
- FFmpeg (https://ffmpeg.org/): LGPL-2.1+, the XMA1/XMA2 decoder and its tables, ported to Rust with
  annotation.
- AMD Machine-Readable GPU ISA (https://gpuopen.com/machine-readable-isa/): MIT, RDNA2 encoding and
  instruction tables.
- AMD addrlib from Mesa (https://gitlab.freedesktop.org/mesa/mesa/): MIT, GFX10 swizzle tables and
  the mip chain and mip tail layout, ported with annotation.
- silicon-rans (https://github.com/microsoft/silicon-rans): MIT, the BCPack rANS lane and its model
  tables, ported with annotation.
- bcpack-decoder (https://github.com/Tim0219800/bcpack-decoder) and bcpack-rs
  (https://github.com/Ahmet-Dedeler/bcpack-rs): MIT, the BCPack stream layout, used as a
  specification; bcpack-rs also supplies BC1 constant tables.
- leonida-asset-pipeline (https://github.com/wxrrry/leonida-asset-pipeline): MIT, the BC5 block rule
  and BC5 constant tables, used as a specification.
- LLVM (https://github.com/llvm/llvm-project): Apache-2.0 WITH LLVM-exception, test data and GFX10
  buffer format values.

Loaded at runtime from the user's own system, and not distributed with Zenith-X: DXC
(`dxcompiler.dll`, `dxil.dll`), optional `libopus` (BSD-3-Clause), and the Windows system libraries
(D3D12, DXGI, XAudio2, MSOpusDecoder and others), used under GPL-2.0's "system library" exception.
The vkd3d-proton patch in `scripts/linux/` applies to the user's own build; vkd3d-proton itself is
not distributed here.

WinDurango, libHttpClient and docs.xbox.com are references only; no code is taken from them.
