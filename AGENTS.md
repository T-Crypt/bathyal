# Working on Bathyal

Bathyal is a fork of XCL (https://github.com/ivall/xcl). Read `README.md` (what works, how to run) and
`HANDOFF.md` (measurements and open work) before changing anything. This file adds the rules for this fork.

## Rules

1. **No game material in the repository, ever:** no executables, archives, shaders in any form (binary,
   disassembly, HLSL, DXIL), textures, audio, carved files, or logs that quote them. Tests synthesize their inputs
   or read local files through the `XCL_SAMPLE_*` variables. Do not name commercial titles in code, docs or
   commit messages: write "a newer retail title".
2. **No keys, decryption or DRM workarounds** in code or docs. The project runs unpacked files the user owns.
3. **Nothing online in the code:** no HTTP clients, no telemetry, no downloads at start. Networking APIs a title
   calls stay offline stand-ins (`hle/netoff.rs`).
4. **Keep XCL's license and credit:** `LICENSE`, `NOTICE.md`, `THIRD_PARTY.md` and the upstream history stay.
   New files are GPL-2.0 too.
5. **Commits under the project identity** (`T-Crypt <T-Crypt@users.noreply.github.com>`), with no personal
   names or addresses and no AI attribution lines.
6. **Agents never push to `main`.** Work on a branch and open a pull request.

## Untrusted executables

A title's executable is untrusted code until it has proven itself. Before its first run:

1. Check its SHA-256 against a source you trust.
2. Keep it out of the read-only package folder; build a game folder with `scripts\setup-game-folder.ps1`.
3. Block its egress: `scripts\block-egress.ps1` (outbound block on `bathyal.exe`; the title runs inside it).
4. Run on a machine with no personal accounts or sensitive data when you can.

## Build, test, run (Windows)

```powershell
cargo build --release --workspace                      # target\release\bathyal.exe
.\scripts\test.ps1                                     # cargo build + cargo test --workspace
.\scripts\run-title.ps1 -Game D:\games\title           # precompiles caches once, then runs; logs in %LOCALAPPDATA%\bathyal\runs
```

## Following XCL

```powershell
git remote add xcl https://github.com/ivall/xcl.git    # once
git fetch xcl
git merge xcl/main                                     # on a branch; keep our README header and NOTICE.md
```

The program is `bathyal.exe`. The crate, module paths, `XCL_*` settings and the `%LOCALAPPDATA%\XCL` data
folder keep XCL's names so these merges stay clean; rename them only together with upstream.

## Notes carried over from another port engine

Another port engine for the same console (C++, D3D12.X on Vulkan) met these issues on a newer retail title.
XCL already covers most of them: AMD-only CPU instructions such as CLZERO, B5G5R5A1 UAVs, A8 targets,
AcpHal's negative-HRESULT asserts, XErrorReport, the TLS index 0 handover and wave64 lane-mask SGPR pairs.
Still worth checking here:

- **R9G9B9E5_SHAREDEXP UAV textures.** PC drivers give that format no typed storage. That engine stores such a
  texture as a mutable R32_UINT image and lets shaders write the packed bits; SRVs and RTVs keep the shared
  exponent format. B5G5R5A1 is handled the same way, as R16_UINT.
- **sRGB UAV textures.** The title asks for UAVs on sRGB textures (2048x2048 seen), which PC D3D12 refuses.
  That engine drops the UAV flag from the console description when the format has no UAV support, and gives
  UAV views the matching UNORM format.
- **GetResourceAllocationInfo with unsupported formats.** The title asks for layouts of formats the PC rejects;
  an error there ended in an int3 in that engine. Answer with the layout of a same-size stand-in format.
- **NVIDIA (32-lane subgroups).** Measured on an RTX 4090: wave64 cross-half operations (readlane, permlane,
  DPP or shuffles across lanes 31/32, wave-wide reductions) are the known fidelity gap in vertex and pixel
  shaders. See `docs/wave64-on-wave32.md`.
