# Bathyal work handoff

Bathyal is a fork of XCL that runs Xbox Series X|S (GDK) game files on Windows. This file records
what works, what is open, and the rules for continuing. Design: `docs/specs/2026-11-20-xcl-design.md`.
Day-to-day usage: `README.md`. Change history: `git log` and the pull requests.

## What works

- **Loader and system layer.** PE loader with relocations, imports by name, ordinal and delay-load,
  TLS callbacks and exception tables. `kernelx`, XGameRuntime through the GDK thunk table, the
  console file system with a write overlay, XAudio2 with pure-Rust XMA1/XMA2 decoding.
- **D3D12.X.** Device, resources and heaps (also in title memory), pipelines, console descriptor
  heaps, console-format command lists for compute and graphics, `PresentX` and frame pacing,
  console BVH builder slots, a console FSR 4 upscaler.
- **Shader recompiler.** RDNA2 to HLSL to DXIL for VS, PS, CS, geometry and mesh (NGG) programs:
  atomics, LDS, scratch, images, samplers, wave operations, ray-tracing instructions, structured
  control flow, parallel DXC and a disk cache.
- **Textures and geometry.** GFX10 tiling for 2D, 3D, arrays, cubes and mip chains; a console BVH
  layout with BVH intersection instructions emulated in shaders.

## Notes that apply across titles

- **Wave64 on 32-lane GPUs (NVIDIA).** Console programs run as two 32-lane halves that exchange
  masks and lane values; cross-half operations are approximate. See `docs/wave64-on-wave32.md`.
- **Console memory.** Title memory is shared by CPU and GPU through 4 GiB windows at the console
  address; resource creation, copies and atomics must go through that window.
- **Failure paths.** Some titles end the process with `mov dword [0], 0` in their watchdog after a
  failed D3D12 call. XCL logs the failed call and the matching interface and slot before the game
  stops, so the report shows the cause.
- **Texture formats.** PC drivers give R9G9B9E5_SHAREDEXP and B5G5R5A1 no typed UAV storage; XCL
  stores such textures as mutable integer images and gives UAV views the matching UNORM format.
  A title can ask for UAVs on sRGB textures; XCL drops the UAV flag when the format has no UAV
  support.
- **Console CPU instructions.** AMD-only instructions (CLZERO and others), AcpHal's negative
  HRESULT asserts, XErrorReport, the TLS index 0 handover and wave64 lane-mask SGPR pairs are
  handled in `hle/` and are worth checking first when a new title stalls.

## Tests

`scripts\test.ps1` builds the workspace and runs every test. Unit tests, a synthetic console title
(`tests/testtitle`) and GPU tests need no game data. Tests against real data are opt-in through
environment variables that point at files the user owns: `XCL_SAMPLE_DLLS`, `XCL_SAMPLE_TTPF`,
`XCL_SAMPLE_BK2`, `XCL_SAMPLE_AWC`, `XCL_SAMPLE_FSPX`, `XCL_SAMPLE_XG`, `XCL_SAMPLE_DXC`. Without
them those tests pass without running anything.

## Open work

- Draw-time root argument binding for console graphics command lists recorded after the PSO call.
- Rehydration of prebuilt console BVHs from a game's own geometry descriptions.
- Broader coverage of console command-list slots that games use but current titles have not.

## Out of this repository

Measurements, fixtures and notes for individual titles stay outside this public repository. Keep
the repository free of game files, shaders, assets, keys and logs that quote them.
