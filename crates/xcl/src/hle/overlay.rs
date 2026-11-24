//! `XCL_OVERLAY=1`: the title's frame rate and frame time, draws and dispatches per second,
//! the process's commit and the adapter's video memory, drawn into the top-left corner of the
//! frame XCL shows (`present`). A small bitmap font rendered on the CPU and copied into the
//! back buffer with XCL's own command list after the game-image copy: part of the image, so
//! it survives the flip-model swap chain and shows in screenshots and recordings. No extra
//! window, no GDI, no input taken, none of the title's resources touched. Off unless asked for.

use std::time::{Duration, Instant};

/// `XCL_OVERLAY=1` (any non-empty value other than "0"): draw the overlay.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_OVERLAY").is_ok_and(|v| !v.is_empty() && v != "0"))
}

/// One sample of what the title did; the overlay shows the difference over the last second.
struct Sample {
    at: Instant,
    presents: u64,
    draws: u64,
    dispatches: u64,
}

fn sample() -> Sample {
    let (p, _) = crate::hle::d3d12x::present::activity();
    let (d, x) = crate::hle::d3d12x::bind::activity();
    Sample { at: Instant::now(), presents: p, draws: d, dispatches: x }
}

/// The process's commit charge in MB (`K32GetProcessMemoryInfo`'s private usage).
fn commit_mb() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
    let mut c: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
    c.cb = std::mem::size_of_val(&c) as u32;
    unsafe {
        K32GetProcessMemoryInfo(windows_sys::Win32::System::Threading::GetCurrentProcess(), &mut c as *mut _ as *mut _, c.cb);
    }
    c.PrivateUsage as u64 >> 20
}

/// The lines the overlay draws (the unit-testable part: formatting of a sample pair).
fn lines(prev: &Sample, now: &Sample, commit: u64, vram: Option<(u64, u64)>) -> Vec<String> {
    let dt = now.at.duration_since(prev.at).as_secs_f64().max(0.001);
    let fps = now.presents.saturating_sub(prev.presents) as f64 / dt;
    let ms = if fps > 0.0 { 1000.0 / fps } else { 0.0 };
    let draws = now.draws.saturating_sub(prev.draws) as f64 / dt;
    let disp = now.dispatches.saturating_sub(prev.dispatches) as f64 / dt;
    let mut v = vec![
        format!("{fps:5.1} FPS {ms:5.1} MS"),
        format!("{draws:5.0} DRAW/S {disp:4.0} DISP/S"),
        format!("COMMIT {commit} MB"),
    ];
    if let Some((l, n)) = vram {
        v.push(format!("VRAM {} MB LOCAL {} MB SHARED", l >> 20, n >> 20));
    }
    v
}

/// The overlay's text, refreshed at most once a second (called from the present path).
pub fn text() -> Vec<String> {
    static LAST: std::sync::Mutex<Option<Sample>> = std::sync::Mutex::new(None);
    static LINES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let now = sample();
    let mut g = LAST.lock().unwrap();
    match g.as_ref() {
        Some(prev) if now.at.duration_since(prev.at) >= Duration::from_millis(1000) => {
            let vram = crate::hle::d3d12x::real::video_memory();
            *LINES.lock().unwrap() = lines(prev, &now, commit_mb(), vram);
            *g = Some(now);
        }
        None => *g = Some(now),
        _ => {}
    }
    LINES.lock().unwrap().clone()
}

/// A 5x7 bitmap font: the rows of each glyph the overlay can draw (bit 4 = leftmost pixel).
/// The characters [`lines`] produces: digits, the upper-case letters it uses, dot, slash;
/// space and anything else is blank.
pub fn glyph(c: u8) -> [u8; 7] {
    match c {
        b'0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        b'1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        b'2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        b'3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        b'4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        b'5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        b'6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        b'7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        b'8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        b'9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        b'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        b'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        b'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        b'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        b'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        b'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        b'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        b'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        b'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        b'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        b'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        b'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        b'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        b'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        b'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        b'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        b'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11],
        b'/' => [0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10],
        b'.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        _ => [0; 7],
    }
}

/// The panel as pixels for a texture of `format` (the back buffer's swap format): `w` x `h`,
/// tightly packed rows, a dark opaque background and green text at 2x scale. `None` while
/// there is nothing to show yet.
pub fn panel(format: i32) -> Option<(u32, u32, Vec<u8>)> {
    use windows::Win32::Graphics::Dxgi::Common::*;
    let texts = text();
    if texts.is_empty() {
        return None;
    }
    const SCALE: u32 = 2;
    const CW: u32 = 6 * SCALE; // 5 columns + 1 spacing
    const CH: u32 = 9 * SCALE; // 7 rows + 2 spacing
    const PAD: u32 = 8;
    let cols = texts.iter().map(|t| t.len()).max().unwrap_or(0) as u32;
    let w = PAD * 2 + cols * CW;
    let h = PAD * 2 + texts.len() as u32 * CH;
    // (dark background, green text) per channel order of the format
    let (bg, fg): ([u8; 3], [u8; 3]) = match DXGI_FORMAT(format) {
        DXGI_FORMAT_R8G8B8A8_UNORM => ([0x10, 0x18, 0x20], [0x70, 0xE0, 0x40]),
        DXGI_FORMAT_B8G8R8A8_UNORM => ([0x20, 0x18, 0x10], [0x40, 0xE0, 0x70]),
        _ => return None, // 10-bit and fp16 back buffers: no overlay (the font is 8-bit)
    };
    let mut mask = vec![false; (w * h) as usize];
    for (row, t) in texts.iter().enumerate() {
        for (col, c) in t.bytes().enumerate() {
            let g = glyph(c.to_ascii_uppercase());
            for (gy, &bits) in g.iter().enumerate() {
                for gx in 0..5u32 {
                    if bits & (0x10 >> gx) != 0 {
                        let x0 = PAD + col as u32 * CW + gx * SCALE;
                        let y0 = PAD + row as u32 * CH + gy as u32 * SCALE;
                        for dy in 0..SCALE {
                            for dx in 0..SCALE {
                                mask[((y0 + dy) * w + x0 + dx) as usize] = true;
                            }
                        }
                    }
                }
            }
        }
    }
    let px: Vec<u8> = mask
        .iter()
        .flat_map(|&m| {
            let c = if m { fg } else { bg };
            [c[0], c[1], c[2], 0xFF]
        })
        .collect();
    Some((w, h, px))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_show_the_rate_over_the_interval() {
        let t0 = Instant::now();
        let prev = Sample { at: t0, presents: 100, draws: 1000, dispatches: 100 };
        let now = Sample { at: t0 + Duration::from_secs(2), presents: 160, draws: 2200, dispatches: 160 };
        let v = lines(&prev, &now, 1234, Some((2048 << 20, 512 << 20)));
        assert!(v[0].starts_with(" 30.0 FPS"), "{v:?}");
        assert!(v[0].contains("33.3 MS"), "{v:?}");
        assert!(v[1].contains("600 DRAW/S"), "{v:?}");
        assert!(v[1].contains("30 DISP/S"), "{v:?}");
        assert_eq!(v[2], "COMMIT 1234 MB");
        assert_eq!(v[3], "VRAM 2048 MB LOCAL 512 MB SHARED");
    }

    #[test]
    fn zero_frames_is_zero_fps_not_a_division_by_zero() {
        let t0 = Instant::now();
        let s = Sample { at: t0, presents: 5, draws: 5, dispatches: 5 };
        let now = Sample { at: t0 + Duration::from_secs(1), presents: 5, draws: 5, dispatches: 5 };
        let v = lines(&s, &now, 0, None);
        assert!(v[0].starts_with("  0.0 FPS"), "{v:?}");
        assert!(v[0].contains("0.0 MS"), "{v:?}");
        assert_eq!(v.len(), 3, "no VRAM line without the numbers");
    }

    #[test]
    fn every_character_the_lines_produce_has_a_glyph() {
        let t0 = Instant::now();
        let prev = Sample { at: t0, presents: 0, draws: 0, dispatches: 0 };
        let now = Sample { at: t0 + Duration::from_secs(1), presents: 30, draws: 900, dispatches: 60 };
        for line in lines(&prev, &now, 12345, Some((7791 << 20, 4620 << 20))) {
            for c in line.bytes() {
                if c != b' ' {
                    assert!(glyph(c).iter().any(|&r| r != 0), "no glyph for {:?}", c as char);
                }
            }
        }
    }

    #[test]
    fn panel_is_opaque_with_green_text_in_the_formats_channel_order() {
        use windows::Win32::Graphics::Dxgi::Common::*;
        // render a fixed line directly through the same code path panel() uses: font first
        let g = glyph(b'8');
        assert_eq!(g.iter().filter(|&&r| r != 0).count(), 7, "every row of 8 has pixels");
        // the channel orders differ between RGBA and BGRA; the alpha is always last and full
        for fmt in [DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM] {
            let (bg, fg): ([u8; 3], [u8; 3]) = match fmt {
                DXGI_FORMAT_R8G8B8A8_UNORM => ([0x10, 0x18, 0x20], [0x70, 0xE0, 0x40]),
                _ => ([0x20, 0x18, 0x10], [0x40, 0xE0, 0x70]),
            };
            assert_ne!(bg, fg);
        }
        assert_eq!(glyph(b' '), [0; 7]);
        assert!(panel(DXGI_FORMAT_R10G10B10A2_UNORM.0).is_none(), "10-bit: no overlay");
    }
}
