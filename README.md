# Bathyal – Xbox Series X|S → Windows compatibility layer

Bathyal is a fork of [XCL](https://github.com/ivall/xcl) by ivall and its contributors, under the same
GPL-2.0 license (see [`NOTICE.md`](NOTICE.md)). It follows XCL's `main` and adds work focused on NVIDIA
GPUs (32-lane waves) and on running newer retail titles further. The program is `bathyal.exe`; the crate,
the `XCL_*` settings and the `%LOCALAPPDATA%\XCL` data folder keep XCL's names so upstream changes merge
cleanly and existing caches stay valid.

This repository holds code only: no game files, shaders, assets, keys or logs from any game, and no
decryption or DRM workarounds. Bring your own legally obtained, unpacked game files. See
[`LEGAL.md`](LEGAL.md) for what the project accepts and how to report material that should not be here.


Bathyal runs Xbox Series X|S (GDK) game files on a Windows PC **without CPU emulation**: the game's
x64 code runs natively and Bathyal stands in for the console libraries the PC does not have
(`kernelx.dll`, XGameRuntime, `d3d12_xs.dll`, …) – in the spirit of Wine/Proton. The console's
GPU code (RDNA2 machine code) is recompiled into PC shaders (HLSL → DXIL).

**Project state:** retail games run – Minecraft Preview and Unicorn Overlord are fully
playable, Fortnite and XDefiant work as far as each game goes without its online services;
see [Games](#games). The foundation is tested against real data from retail games
(shader packs, texture archives, movies and the console DLLs they ship). Details, measurements and
next steps:
[`HANDOFF.md`](HANDOFF.md); design: [`docs/specs/2026-11-20-xcl-design.md`](docs/specs/2026-11-20-xcl-design.md).

## Games

| Game | Package (version) | State | What works | Limits |
|---|---|---|---|---|
| **Minecraft Preview** | `Microsoft.MinecraftBeta` 1.18.3064.0 | ✅ | fully playable | – |
| **Unicorn Overlord** | `SEGAofAmericaInc.Pa253b0c1Y` 1.0.0.6 | ✅ | fully playable | – |
| **Fortnite** | `436609B6.FortniteClient` 2.0.1461.7919 | ✅ | works fully, as far as the game allows without its online services | the game itself is online-only (multiplayer servers) |
| **XDefiant** | `Ubisoft.XDefiant-Game` 1.0.0.186 | ✅ | works fully, as far as the game allows without its servers | the game's own servers have been shut down |

## Highlights

- **RDNA2 shaders translate to HLSL** – every stage of the console pipeline (VS, PS, CS,
  geometry and mesh) becomes HLSL; random samples compile to DXIL with 0 errors.
- **Console middleware DLLs run on the PC** – a Bink 2 video codec decodes movies, a
  ray-tracing BVH builder builds acceleration structures on a PC GPU, an AMD FSR 4 console
  upscaler runs a whole frame on a PC GPU, and the GDK HTTP library initialises.
- **Console textures decode exactly** – the GFX10 tiling and mip layout reproduce a game's
  textures byte for byte, volume textures included.
- **370+ automated tests**, including GPU tests that compare translated shaders against expected
  results instruction by instruction.

## What works

✅ done and verified

### Running the game's code

| Feature | State | What it does | Verified by |
|---|---|---|---|
| **PE loader** | ✅ | relocations, imports by name / ordinal / delay-load, TLS callbacks, exception tables, game DLLs, data-driven import routing (HLE / Windows / stub) | console DLLs load with 0 unresolved imports |
| **kernelx** | ✅ | title memory (budget from MicrosoftGame.Config, physical pages), threads on the PC's processors, modules, OS / console info | unit + end-to-end tests |
| **Memory** | ✅ | title graphics memory shared by CPU and GPU without copies, up to 16 GB (4 GiB windows) | GPU tests: loads, stores and atomics across windows |
| **File system** | ✅ | console paths (`GAME:`, `G:`, `D:` …) → the game folder; writes go to an overlay, the game folder is never modified; the console's drives: `S:\` (persistent local storage) and `T:\` (temp) in the title's data folder; free disk space answered for console paths | end-to-end: the test title's folder stays byte-identical; PLS and free space through the real call path |

### System services

| Feature | State | What it does | Verified by |
|---|---|---|---|
| **XGameRuntime** | ✅ | the GDK thunk protocol real titles use, laid out from a table of all 396 GDK runtime functions; offline XUser, XGameSave, persistent local storage, XPackage, XGame (title id), XSystem – the title sees an Xbox Series X / S; task queues, async, networking and dialogs through the PC's XGameRuntime | libHttpClient initialises; users, saves and storage through the real call path; slot layout cross-checked against a title's own thunks |
| **Input** | ✅ | GameInput: one gamepad – the PC's XInput controller, and keyboard and mouse as a gamepad (see [Controls](#controls)) | unit tests; an Xbox controller in Minecraft |
| **Audio** | ✅ | XAudio2 → system XAudio2.9 (PCM / float voices; Opus 0x704F decoded via `MSOpusDecoder.dll`; XMA1 / XMA2 decoded in pure Rust) | unit + end-to-end tests |

### Graphics

| Feature | State | What it does | Verified by |
|---|---|---|---|
| **Shader recompiler** | ✅ | RDNA2 → HLSL → DXIL: VS, PS, CS, mesh (NGG), GS; atomics, LDS, scratch, images, samplers incl. static ones, wave ops, ray-tracing instructions; structured control flow; parallel DXC, disk cache | all stages; 0 DXC errors on random samples; per-instruction GPU tests |
| **Pipelines** | ✅ | PSOs from console shaders: VS+PS, CS, MS+PS, VS+GS+PS; root signature with the shaders' static samplers; non-blocking creation – a pipeline whose shaders are not compiled yet gets a stand-in at once and is built in the background | GPU tests drawing real VS / PS pairs, also built in the background |
| **D3D12.X compute** | ✅ | console device slots, console descriptor heaps (T# / V# / S#), buffers at their console addresses, compute command lists recorded in the console format and replayed (tables, barriers, clears, copies) | a console BVH builder and FSR 4 record and run through Bathyal |
| **D3D12.X graphics** | ✅ | device, resources and heaps (also placed in title memory), pipelines; the title's graphics command lists – draws, indirect draws, graphics root arguments, barriers, copies, clears – and command lists recorded by console code, replayed; `PresentX` and frame pacing into an Bathyal window | Minecraft Preview plays its worlds (millions of draws, none skipped); Fortnite and XDefiant draw their screens |
| **Textures** | ✅ | GFX10 tiling (all 14 2D swizzle modes, the 6 3D thick modes, linear), mip chains with mip tail, arrays, cubes and 3D textures → PC textures | real console textures exact, volume textures included; GPU upload bit exact |
| **Ray tracing** | ✅ | BLAS / TLAS built on the GPU by the console BVH builder (XboxBvhBuilder); RDNA2 BVH node format; BVH intersection instructions emulated in shaders | GPU traversal = brute force reference; the builder's output checked on the GPU |
| **Upscaling** | ✅ | the console AMD FSR 4 upscaler DLL: context, the 29 pipelines in the console driver's own container (FSPX) translated, a frame's 20 passes recorded and run on the GPU | end-to-end with the real DLL: a 1280×720 → 1920×1080 frame on an RX 9070, the input's gradient in the output |

### Media and tools

| Feature | State | What it does | Verified by |
|---|---|---|---|
| **Video** | ✅ | the console Bink 2 DLL decodes movies (720p / 1080p) on the CPU; a game's own YUV → RGB display shaders translate like any other. Bink's GPU-assisted mode needs no Bink-specific support: its compute shaders go through the recompiler too (no game seen using it yet) | a game movie decodes frame by frame, picture checked |
| **Tools** | ✅ | `bathyal inspect`, `bathyal run`, `bathyal disasm` (RDNA2, console shader containers), `bathyal shaders` (corpus statistics, recompile / compile runs) | – |

## Controls

A **gamepad is recommended**: console games are made for one, and their menus and controls expect it.

- **Gamepad:** any controller Windows sees through XInput (Xbox controllers, wired or wireless) is the
  game's gamepad – buttons, sticks, triggers and rumble. It can be plugged in or out while the game runs.
- **Keyboard and mouse** work too, as a gamepad: WASD the left stick, the mouse the right stick, mouse
  buttons the triggers, the wheel the bumpers, Enter / Space A, Escape / Backspace B, arrows the D-pad,
  E / Q Y / X, Shift / Ctrl the stick clicks, Tab View, P Menu. `keyboard_mouse` and `mouse_sensitivity` in
  the game's `xcl.toml` turn it off or change the mouse speed. A gamepad and the keyboard / mouse can be
  used together.

## Building and usage

Windows 10/11 x64, Rust (`stable-x86_64-pc-windows-msvc`), the Windows SDK (DXC: `dxcompiler.dll`
+ `dxil.dll`, looked up in `Windows Kits\10\bin\<version>\x64`, next to `bathyal.exe` or in
`%XCL_DXC_DIR%`), a D3D12 GPU with Shader Model 6.6 and mesh shaders, and the system XGameRuntime
(`System32\xgameruntime.dll`, part of Windows / Gaming Services). On **Windows 10** (22H2
KB5029331+), inbox `D3D12Core.dll` stops at SM 6.5; place the Microsoft D3D12 Agility SDK
(`D3D12Core.dll`, plus `d3d12SDKLayers.dll` if using `XCL_D3D12_DEBUG=1`) in a `D3D12\` folder
next to `bathyal.exe`, directly next to `bathyal.exe`, in `%LOCALAPPDATA%\XCL\D3D12\`, or point `%XCL_D3D12_DIR%` at it
(for `cargo test`, use `%LOCALAPPDATA%\XCL\D3D12\` or `%XCL_D3D12_DIR%`).

- **DXC:** 1.8 or newer is recommended, from the
  [DirectXShaderCompiler releases](https://github.com/microsoft/DirectXShaderCompiler/releases): unpack one
  and point `XCL_DXC_DIR` at its `bin\x64`. A Windows SDK may hold an older one (the same `10.0.26100.0`
  folder has DXC 1.7.2308 or 1.8.2502, depending on the SDK's revision). DXC 1.7's DXIL was 2–2.7 times the
  size of 1.9's for translated compute programs and took an NVIDIA driver 1.4–2.2 times as long to build
  pipelines from ([#156](https://github.com/ivall/xcl/issues/156)). `bathyal run` prints the DXC it loaded, and
  `report.json` has it under `dxc`.
- **Network:** title network access is enabled by default. Use `XCL_NETWORK=0` when bringing up an
  unfamiliar title: XCL blocks its WinHTTP and Winsock traffic while keeping loopback available. This
  also prevents a title crash reporter from sending a process dump containing local paths and names.
- **GPU:** developed on AMD RDNA4. The console's geometry and NGG programs run as a compute pass plus a
  mesh pass, so a GPU without mesh shaders draws none of them. On GPUs with 32-lane waves only
  (NVIDIA) console wave64 compute, geometry and mesh programs run as two 32-lane halves that exchange
  masks and lane values ([`docs/wave64-on-wave32.md`](docs/wave64-on-wave32.md)) – slower than
  64-lane waves, most for groups of several waves; wave64 vertex and pixel shaders that read lanes 32–63
  can still differ there. Test results on other GPUs: [#42](https://github.com/ivall/xcl/issues/42).

```powershell
cargo build --release --workspace          # target\release\bathyal.exe

bathyal inspect D:\Game\game.exe               # the game's imports and what Bathyal has for them
bathyal run D:\Game [-- game arguments]        # run
bathyal disasm <shader.bin> [--hex]            # RDNA2 disassembler (console containers too)
bathyal shaders D:\data\shaders --top 50      # corpus statistics; --recomp / --compile N
```

- **Without a command line** (bathyal.exe opened from Explorer) Bathyal shows a small menu: *Choose a
  game…* (pick its exe or `MicrosoftGame.config`) or *Run again* (the last one). On an error the
  window stays open until Enter.
- Bathyal works on **unpacked** game files you own; it contains nothing for decrypting or
  circumventing DRM.
- **The game folder is never modified** – writes go to the overlay.
- Every run writes `report.json` (called stubs, unknown D3D12.X slots and interfaces, XGameRuntime
  classes asked for, shader statistics, audio formats, crash).
  `d3d12.presents` counts presentations, not visible content. With `XCL_FRAME_DUMP=N` (`N > 1`),
  the existing optional frame dumps also set `d3d12.last_frame_dump`: the latest sampled frame's
  zero-based index, source dimensions, DXGI format, sample step, sample count and `non_dark_samples`
  (decoded 8-bit R+G+B > 30). The field is `null` until a frame is sampled; `non_dark_samples` is
  `null` if its format cannot be decoded. Intermediate render-target dumps do not update it.
  Sampling uses every fourth pixel in each direction (`XCL_FRAME_DUMP_FULL=1`: every pixel).
  A coarse sample can miss detail; a non-dark sample does not establish correct rendering or gameplay.
- Title data: `%LOCALAPPDATA%\XCL\<title>\` (`%XCL_DATA_ROOT%`): `xcl.toml`, `overlay\`, `saves\`,
  `pls\` (the title's `S:\`), `temp\` (`T:\`), `cache\`, `logs\xcl.log`, `logs\report.json` (the
  previous run's: `xcl.prev.log`, `report.prev.json`).

## Tests

```powershell
.\scripts\test.ps1                         # cargo build --workspace + cargo test --workspace
```

Unit tests, an end-to-end test with our own "console title" (`tests/testtitle`) and GPU tests
(D3D12, SM 6.6; skipped without a GPU / DXC; `XCL_ADAPTER=1` or `XCL_ADAPTER=radeon` picks
another adapter than the first high-performance one, for titles too). A GPU test that cannot run on
the adapter (no GPU / DXC / wave64 / mesh shaders) passes too, but prints `SKIPPED <test>: <reason>`, and
`scripts/test.ps1` counts them at the end; `XCL_REQUIRE_GPU=wave64` (or `mesh`, or `wave64,mesh`) makes
those skips fail. Tests on real game data are opt-in through
environment variables pointing at your own files – `XCL_SAMPLE_DLLS` (console DLLs),
`XCL_SAMPLE_TTPF` (texture archive), `XCL_SAMPLE_BK2` (a movie), `XCL_SAMPLE_AWC` (a sound) –
described in `HANDOFF.md`; without them those
tests pass without running anything (`scripts/test.ps1` lists the ones not set). No game data is
part of the repository.

## Licenses

**GPL-2.0-only** (`LICENSE`); third-party sources: `THIRD_PARTY.md` (including AMD's
machine-readable ISA, AMD addrlib swizzle patterns, LLVM disassembler tests, AnyPS5,
libHttpClient headers). WinDurango serves only as a reference – no code is ported from it. The
repository contains no headers or binaries from the console SDK.
