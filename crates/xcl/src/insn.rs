//! Instructions the console's CPU (Zen 2) has and a PC's may lack – a title raises
//! STATUS_ILLEGAL_INSTRUCTION on them on an Intel CPU before Ice Lake / Alder Lake (CLWB,
//! SHA) or on any Intel CPU (CLZERO, RDPRU, MONITORX / MWAITX and SSE4a's EXTRQ / INSERTQ /
//! MOVNTSS / MOVNTSD, which are AMD's). The exception handler (`eh`) emulates the ones decoded
//! here and steps over them.

/// A memory operand (ModRM with its SIB and displacement): `base + index * scale + disp`, or
/// with `rip` the address of the next instruction + disp; with `a32` (an 0x67 prefix) the
/// address is 32 bits. Registers in CONTEXT order (rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi,
/// r8..r15).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mem {
    pub base: Option<u8>,
    pub index: Option<u8>,
    pub scale: u8,
    pub disp: i32,
    pub rip: bool,
    pub a32: bool,
}

/// A console-only instruction at the faulting address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleOnly {
    /// `clwb m8`: write a cache line back – nothing to do, the PC's memory is coherent
    Clwb,
    /// `clzero`: zero the 64-byte cache line at rAX
    Clzero,
    /// `clzero` loop: `clzero; add rax, 0x40; cmp rax, <reg>; jb -12` – zero `[rax..reg)`
    ClzeroLoop { end_reg: u8 },
    /// `rdpru`: ecx = 0 MPERF / 1 APERF → edx:eax (a counter)
    Rdpru,
    /// `monitorx` / `mwaitx`: a wait the emulation skips
    Monitorx,
    Mwaitx,
    /// `rdpid r64`: TSC_AUX (the processor number) into the register (CONTEXT order: rax, rcx,
    /// rdx, rbx, rsp, rbp, rsi, rdi, r8..r15)
    Rdpid(u8),
    /// SSE4a `extrq xmm, imm8 (length), imm8 (index)` / `extrq xmm1, xmm2` (`src` Some: the
    /// length and index in xmm2's bits 5:0 and 13:8): a bit field of the low quadword, moved
    /// down and zero-extended (the high quadword zeroed). Registers are XMM numbers 0..15.
    Extrq { dst: u8, src: Option<u8>, len: u8, idx: u8 },
    /// SSE4a `insertq xmm1, xmm2, imm8 (length), imm8 (index)` / `insertq xmm1, xmm2` (`reg`:
    /// the length and index in xmm2's bits 69:64 and 77:72): the low bits of xmm2 put into a
    /// bit field of xmm1's low quadword.
    Insertq { dst: u8, src: u8, reg: bool, len: u8, idx: u8 },
    /// SSE4a `movntss m32, xmm` (F3 0F 2B /r, `bytes` 4) / `movntsd m64, xmm` (F2 0F 2B /r,
    /// `bytes` 8): the low dword / quadword of XMM `src` stored without the cache
    Movnts { bytes: u8, src: u8, mem: Mem },
    /// `ud2`: the title's own trap (not emulated)
    Ud2,
}

/// A field length of SSE4a's six bits: 0 is 64.
fn field_len(len: u8) -> u32 {
    match len & 63 {
        0 => 64,
        n => n as u32,
    }
}

/// EXTRQ on a low quadword: `len` bits from bit `idx`, zero-extended (a field past bit 63 is
/// undefined on the console; here it is what a shift gives).
pub fn extrq(v: u64, len: u8, idx: u8) -> u64 {
    let (len, idx) = (field_len(len), (idx & 63) as u32);
    let field = v.checked_shr(idx).unwrap_or(0);
    if len >= 64 {
        field
    } else {
        field & ((1u64 << len) - 1)
    }
}

/// INSERTQ on a low quadword: the low `len` bits of `src` put at bit `idx` of `dst`.
pub fn insertq(dst: u64, src: u64, len: u8, idx: u8) -> u64 {
    let (len, idx) = (field_len(len), (idx & 63) as u32);
    let ones = if len >= 64 { u64::MAX } else { (1u64 << len) - 1 };
    let mask = ones.checked_shl(idx).unwrap_or(0);
    (dst & !mask) | ((src & ones).checked_shl(idx).unwrap_or(0) & mask)
}

/// The address of a memory operand: `regs` in CONTEXT order, `next` the address of the
/// instruction after it (for a rip-relative operand).
pub fn address(m: &Mem, regs: &[u64; 16], next: u64) -> u64 {
    let base = if m.rip { next } else { m.base.map_or(0, |r| regs[r as usize & 15]) };
    let index = m.index.map_or(0, |r| regs[r as usize & 15].wrapping_mul(m.scale as u64));
    let a = base.wrapping_add(index).wrapping_add(m.disp as i64 as u64);
    if m.a32 {
        a & 0xFFFF_FFFF
    } else {
        a
    }
}

/// The memory operand of the ModRM byte at `b[0]` (with its SIB and displacement) and its
/// length; None for a register operand or when the bytes run out. REX.B extends the base, REX.X
/// the index.
fn mem_operand(b: &[u8], rex: u8, a32: bool) -> Option<(Mem, usize)> {
    let len = modrm_len(b)?;
    let (md, rm) = (b[0] >> 6, b[0] & 7);
    if md == 3 {
        return None;
    }
    let (rex_b, rex_x) = ((rex & 1) << 3, (rex & 2) << 2);
    let mut m = Mem { base: Some(rm | rex_b), index: None, scale: 1, disp: 0, rip: false, a32 };
    let mut at = 1;
    if rm == 4 {
        let sib = b[1];
        let (index, base) = (((sib >> 3) & 7) | rex_x, sib & 7);
        m.scale = 1 << (sib >> 6);
        // index 100 is no index (with REX.X it is r12)
        m.index = (index != 4).then_some(index);
        // base 101 with mod 00 is no base and a disp32 (r13 takes mod 01)
        m.base = (md != 0 || base != 5).then_some(base | rex_b);
        at = 2;
    } else if md == 0 && rm == 5 {
        m.base = None;
        m.rip = true;
    }
    m.disp = match len - at {
        0 => 0,
        1 => b[at] as i8 as i32,
        _ => i32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]),
    };
    Some((m, len))
}

/// The length of a ModRM-encoded operand (ModRM, SIB and displacement) at `b[0]`, None when the
/// bytes run out.
fn modrm_len(b: &[u8]) -> Option<usize> {
    let modrm = *b.first()?;
    let (md, rm) = (modrm >> 6, modrm & 7);
    if md == 3 {
        return Some(1);
    }
    let mut len = 1;
    let mut base = rm;
    if rm == 4 {
        base = *b.get(1)? & 7;
        len += 1;
    }
    len += match md {
        0 if base == 5 => 4, // disp32 (rip-relative, or a SIB without a base)
        0 => 0,
        1 => 1,
        _ => 4,
    };
    (b.len() >= len).then_some(len)
}

/// The console-only instruction at the start of `code` and its length.
pub fn decode(code: &[u8]) -> Option<(ConsoleOnly, usize)> {
    let mut i = 0;
    // `rep`: the last of F2 / F3, which decides between them (and over 66) for SSE4a
    let (mut p66, mut pf3, mut rep, mut rex) = (false, false, 0u8, 0u8);
    // for a memory operand: an fs / gs override (its base is not in the context), 0x67
    let (mut fs_gs, mut a32) = (false, false);
    while let Some(&b) = code.get(i) {
        match b {
            0x66 => p66 = true,
            0xF2 => rep = 0xF2,
            0xF3 => {
                pf3 = true;
                rep = 0xF3;
            }
            0x64 | 0x65 => fs_gs = true,
            0x67 => a32 = true,
            0x2E | 0x3E | 0x26 | 0x36 => {}
            // REX counts only right before the opcode: one before a legacy prefix is ignored
            0x40..=0x4F => {
                rex = b;
                i += 1;
                continue;
            }
            _ => break,
        }
        rex = 0;
        i += 1;
    }
    if *code.get(i)? != 0x0F {
        return None;
    }
    match (*code.get(i + 1)?, code.get(i + 2).copied()) {
        (0x01, Some(0xFC)) => {
            // clzero (0F 01 FC); add rax, 0x40 (48 83 C0 40); cmp rax, <reg> ([48|4C] 39 modrm); jb -12 (72 F4)
            // (not after an 0x67: that CLZERO is at EAX, and the jb goes back to the 0F, past the
            // prefix – it is stepped alone, the loop is taken from the next one)
            if code.len() >= i + 12
                && !code[..i].contains(&0x67)
                && code[i + 3..i + 7] == [0x48, 0x83, 0xC0, 0x40]
                && (code[i + 7] == 0x48 || code[i + 7] == 0x4C)
                && code[i + 8] == 0x39
                && (code[i + 9] & 0xC7) == 0xC0
                && code[i + 10..i + 12] == [0x72, 0xF4]
            {
                let rex_r = code[i + 7] == 0x4C;
                let modrm = code[i + 9];
                let reg_idx = ((modrm >> 3) & 7) | if rex_r { 8 } else { 0 };
                if reg_idx != 0 {
                    return Some((ConsoleOnly::ClzeroLoop { end_reg: reg_idx }, i + 12));
                }
            }
            Some((ConsoleOnly::Clzero, i + 3))
        }
        (0x01, Some(0xFD)) => Some((ConsoleOnly::Rdpru, i + 3)),
        (0x01, Some(0xFA)) => Some((ConsoleOnly::Monitorx, i + 3)),
        (0x01, Some(0xFB)) => Some((ConsoleOnly::Mwaitx, i + 3)),
        (0x0B, _) => Some((ConsoleOnly::Ud2, i + 2)),
        // SSE4a: register operands only (ModRM mod 3); REX.R extends reg, REX.B rm
        // (the last of F2 / F3 decides: F2 INSERTQ, F3 #UD; neither with 66: EXTRQ)
        (0x78 | 0x79, Some(m)) if (rep == 0xF2 || (rep == 0 && p66)) && m >> 6 == 3 => {
            let reg = ((m >> 3) & 7) | (rex & 4) << 1;
            let rm = (m & 7) | (rex & 1) << 3;
            let imm = |k: usize| code.get(i + 3 + k).copied();
            match (code[i + 1], rep == 0xF2) {
                // extrq xmm(rm), imm8, imm8 – the ModRM reg field is /0
                (0x78, false) if (m >> 3) & 7 == 0 => Some((ConsoleOnly::Extrq { dst: rm, src: None, len: imm(0)?, idx: imm(1)? }, i + 5)),
                (0x79, false) => Some((ConsoleOnly::Extrq { dst: reg, src: Some(rm), len: 0, idx: 0 }, i + 3)),
                (0x78, true) => Some((ConsoleOnly::Insertq { dst: reg, src: rm, reg: false, len: imm(0)?, idx: imm(1)? }, i + 5)),
                (0x79, true) => Some((ConsoleOnly::Insertq { dst: reg, src: rm, reg: true, len: 0, idx: 0 }, i + 3)),
                _ => None,
            }
        }
        // SSE4a scalar streaming stores, memory operands only: F3 0F 2B movntss, F2 0F 2B
        // movntsd (the last of F2 / F3 decides; without them it is movntps, with 66 alone
        // movntpd). One through fs / gs is left to fault.
        (0x2B, Some(m)) if rep != 0 && m >> 6 != 3 && !fs_gs => {
            let (mem, len) = mem_operand(&code[i + 2..], rex, a32)?;
            let src = ((m >> 3) & 7) | (rex & 4) << 1;
            Some((ConsoleOnly::Movnts { bytes: if rep == 0xF2 { 8 } else { 4 }, src, mem }, i + 2 + len))
        }
        // 66 0F AE /6 with a memory operand (without 66 it is xsaveopt)
        (0xAE, Some(m)) if p66 && (m >> 3) & 7 == 6 && m >> 6 != 3 => Some((ConsoleOnly::Clwb, i + 2 + modrm_len(&code[i + 2..])?)),
        // F3 0F C7 /7 with a register operand
        (0xC7, Some(m)) if pf3 && (m >> 3) & 7 == 7 && m >> 6 == 3 => Some((ConsoleOnly::Rdpid((m & 7) | (rex & 1) << 3), i + 3)),
        _ => None,
    }
}

/// `code` as hex, for a log line.
pub fn hex(code: &[u8]) -> String {
    code.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EXTRQ / INSERTQ as AMD's manual gives them: a length of 0 is 64.
    #[test]
    fn sse4a_bit_fields() {
        assert_eq!(extrq(0x1234_5678_9ABC_DEF0, 8, 16), 0xBC);
        assert_eq!(extrq(0x1234_5678_9ABC_DEF0, 0, 0), 0x1234_5678_9ABC_DEF0);
        assert_eq!(extrq(0x8000_0000_0000_0000, 1, 63), 1);
        assert_eq!(insertq(0xFFFF_FFFF_FFFF_FFFF, 0xAB, 8, 8), 0xFFFF_FFFF_FFFF_ABFF);
        assert_eq!(insertq(0, 0x1FF, 4, 60), 0xF000_0000_0000_0000);
        assert_eq!(insertq(0x1111, 0xDEAD_BEEF_0000_0000, 0, 0), 0xDEAD_BEEF_0000_0000);
        for (v, len, idx) in [(0x0123_4567_89AB_CDEFu64, 12u8, 20u8), (u64::MAX, 63, 1), (0x55, 3, 0)] {
            assert_eq!(extrq(insertq(0, v, len, idx), len, idx), extrq(v, len, 0));
        }
    }

    #[test]
    fn decodes_the_console_only_instructions() {
        assert_eq!(decode(&[0x0F, 0x01, 0xFC]), Some((ConsoleOnly::Clzero, 3)));
        // clzero loop against r10 (4C 39 D0 = cmp rax, r10)
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x4C, 0x39, 0xD0, 0x72, 0xF4]),
            Some((ConsoleOnly::ClzeroLoop { end_reg: 10 }, 12))
        );
        // clzero loop against rdx (48 39 D0 = cmp rax, rdx)
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72, 0xF4]),
            Some((ConsoleOnly::ClzeroLoop { end_reg: 2 }, 12))
        );
        // the same loop after an address-size prefix: that one CLZERO alone
        assert_eq!(decode(&[0x67, 0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72, 0xF4]), Some((ConsoleOnly::Clzero, 4)));
        // cmp with non-rax rm, memory operand, or cmp rax, rax is not a ClzeroLoop
        assert_eq!(decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xCA, 0x72, 0xF4]), Some((ConsoleOnly::Clzero, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0x10, 0x72, 0xF4]), Some((ConsoleOnly::Clzero, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xC0, 0x72, 0xF4]), Some((ConsoleOnly::Clzero, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFD]), Some((ConsoleOnly::Rdpru, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFA]), Some((ConsoleOnly::Monitorx, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFB]), Some((ConsoleOnly::Mwaitx, 3)));
        assert_eq!(decode(&[0x0F, 0x0B]), Some((ConsoleOnly::Ud2, 2)));
        // clwb [rax]; clwb [rcx+0x40]; clwb [rsp+8]; clwb [rip+disp32]; clwb [r8]
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x30]), Some((ConsoleOnly::Clwb, 4)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x71, 0x40]), Some((ConsoleOnly::Clwb, 5)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x74, 0x24, 0x08]), Some((ConsoleOnly::Clwb, 6)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x35, 1, 2, 3, 4]), Some((ConsoleOnly::Clwb, 8)));
        assert_eq!(decode(&[0x66, 0x41, 0x0F, 0xAE, 0x30]), Some((ConsoleOnly::Clwb, 5)));
        // rdpid rax; rdpid r9
        assert_eq!(decode(&[0xF3, 0x0F, 0xC7, 0xF8]), Some((ConsoleOnly::Rdpid(0), 4)));
        assert_eq!(decode(&[0xF3, 0x41, 0x0F, 0xC7, 0xF9]), Some((ConsoleOnly::Rdpid(9), 5)));
        // not console-only: clflushopt (/7), xsaveopt (no 66), a plain mov, truncated bytes
        // SSE4a, as a retail game has it (#216: insertq xmm0, xmm1 with length / index bytes)
        assert_eq!(decode(&[0xF2, 0x0F, 0x78, 0xC1, 0x08, 0x10]), Some((ConsoleOnly::Insertq { dst: 0, src: 1, reg: false, len: 8, idx: 16 }, 6)));
        assert_eq!(decode(&[0xF2, 0x45, 0x0F, 0x79, 0xC1]), Some((ConsoleOnly::Insertq { dst: 8, src: 9, reg: true, len: 0, idx: 0 }, 5)));
        assert_eq!(decode(&[0x66, 0x0F, 0x78, 0xC2, 0x04, 0x08]), Some((ConsoleOnly::Extrq { dst: 2, src: None, len: 4, idx: 8 }, 6)));
        assert_eq!(decode(&[0x66, 0x0F, 0x79, 0xCA]), Some((ConsoleOnly::Extrq { dst: 1, src: Some(2), len: 0, idx: 0 }, 4)));
        assert_eq!(decode(&[0xF2, 0x0F, 0x78, 0xC1, 0x08]), None, "an immediate missing");
        assert_eq!(decode(&[0x0F, 0x78, 0xC1, 0x08, 0x10]), None, "VMREAD without a prefix is not SSE4a");
        // prefixes as a Zen 5 runs them (Kalmarv on #209): the last of F2 / F3 decides, 66 next
        // to them does not; 0x67 and a REX before a legacy prefix are ignored
        let insertq = Some((ConsoleOnly::Insertq { dst: 0, src: 1, reg: false, len: 16, idx: 16 }, 7));
        for code in [[0x66, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0xF2, 0x66, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0xF3, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0x67, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0x41, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10]] {
            assert_eq!(decode(&code), insertq, "{code:02x?}");
        }
        for code in [[0xF2, 0xF3, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0xF3, 0x66, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0x66, 0xF3, 0x0F, 0x78, 0xC0, 0x10, 0x10]] {
            assert_eq!(decode(&code), None, "#UD on the CPU: {code:02x?}");
        }
        assert_eq!(decode(&[0x67, 0x0F, 0x01, 0xFC]), Some((ConsoleOnly::Clzero, 4)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x38]), None);
        assert_eq!(decode(&[0x0F, 0xAE, 0x30]), None);
        assert_eq!(decode(&[0x48, 0x8B, 0x01]), None);
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x74, 0x24]), None);
        assert_eq!(decode(&[0x0F]), None);
    }

    #[test]
    fn decodes_the_sse4a_streaming_stores() {
        let m = |base: Option<u8>, index: Option<u8>, scale: u8, disp: i32| Mem { base, index, scale, disp, rip: false, a32: false };
        // movntsd [rax], xmm0; movntss [rsp+8], xmm1; movntsd [rip+0x10], xmm2
        assert_eq!(decode(&[0xF2, 0x0F, 0x2B, 0x00]), Some((ConsoleOnly::Movnts { bytes: 8, src: 0, mem: m(Some(0), None, 1, 0) }, 4)));
        assert_eq!(decode(&[0xF3, 0x0F, 0x2B, 0x4C, 0x24, 0x08]), Some((ConsoleOnly::Movnts { bytes: 4, src: 1, mem: m(Some(4), None, 1, 8) }, 6)));
        assert_eq!(
            decode(&[0xF2, 0x0F, 0x2B, 0x15, 0x10, 0, 0, 0]),
            Some((ConsoleOnly::Movnts { bytes: 8, src: 2, mem: Mem { rip: true, ..m(None, None, 1, 0x10) } }, 8))
        );
        // movntsd [r8+r12*4-4], xmm9 (REX.RXB); movntsd [r13+0], xmm0; movntsd [0x1000], xmm0 (a SIB
        // without base or index); movntss [rbp+rsi*8+0x12345678], xmm3
        assert_eq!(decode(&[0xF2, 0x47, 0x0F, 0x2B, 0x4C, 0xA0, 0xFC]), Some((ConsoleOnly::Movnts { bytes: 8, src: 9, mem: m(Some(8), Some(12), 4, -4) }, 7)));
        assert_eq!(decode(&[0xF2, 0x41, 0x0F, 0x2B, 0x45, 0x00]), Some((ConsoleOnly::Movnts { bytes: 8, src: 0, mem: m(Some(13), None, 1, 0) }, 6)));
        assert_eq!(decode(&[0xF2, 0x0F, 0x2B, 0x04, 0x25, 0, 0x10, 0, 0]), Some((ConsoleOnly::Movnts { bytes: 8, src: 0, mem: m(None, None, 1, 0x1000) }, 9)));
        assert_eq!(
            decode(&[0xF3, 0x0F, 0x2B, 0x9C, 0xF5, 0x78, 0x56, 0x34, 0x12]),
            Some((ConsoleOnly::Movnts { bytes: 4, src: 3, mem: m(Some(5), Some(6), 8, 0x1234_5678) }, 9))
        );
        // the last of F2 / F3 decides, 66 next to them does not; 0x67 makes the address 32 bits
        assert_eq!(decode(&[0xF2, 0xF3, 0x0F, 0x2B, 0x00]).map(|(c, _)| c), Some(ConsoleOnly::Movnts { bytes: 4, src: 0, mem: m(Some(0), None, 1, 0) }));
        assert_eq!(decode(&[0x66, 0xF2, 0x0F, 0x2B, 0x00]).map(|(c, _)| c), Some(ConsoleOnly::Movnts { bytes: 8, src: 0, mem: m(Some(0), None, 1, 0) }));
        assert_eq!(decode(&[0x67, 0xF2, 0x0F, 0x2B, 0x00]), Some((ConsoleOnly::Movnts { bytes: 8, src: 0, mem: Mem { a32: true, ..m(Some(0), None, 1, 0) } }, 5)));
        // not these: movntps (no prefix), movntpd (66), a register operand, through gs, cut short
        assert_eq!(decode(&[0x0F, 0x2B, 0x00]), None);
        assert_eq!(decode(&[0x66, 0x0F, 0x2B, 0x00]), None);
        assert_eq!(decode(&[0xF2, 0x0F, 0x2B, 0xC0]), None);
        assert_eq!(decode(&[0x65, 0xF2, 0x0F, 0x2B, 0x04, 0x25, 0, 0x10, 0, 0]), None);
        assert_eq!(decode(&[0xF2, 0x0F, 0x2B, 0x44]), None);
        assert_eq!(decode(&[0xF2, 0x0F, 0x2B, 0x80, 1, 2, 3]), None);
    }

    #[test]
    fn memory_operand_addresses() {
        let mut regs = [0u64; 16];
        regs[4] = 0x1000; // rsp
        regs[8] = 0x2000; // r8
        regs[12] = 3; // r12
        regs[0] = 0x1_FFFF_FFF0; // rax
        let m = Mem { base: Some(8), index: Some(12), scale: 4, disp: -4, rip: false, a32: false };
        assert_eq!(address(&m, &regs, 0), 0x2000 + 12 - 4);
        assert_eq!(address(&Mem { base: Some(4), index: None, scale: 1, disp: 8, ..m }, &regs, 0), 0x1008);
        assert_eq!(address(&Mem { base: None, index: None, disp: -0x10, rip: true, ..m }, &regs, 0x5000), 0x4FF0);
        assert_eq!(address(&Mem { base: None, index: None, disp: 0x1000, ..m }, &regs, 0x5000), 0x1000);
        // a 32-bit address wraps at 4 GiB
        assert_eq!(address(&Mem { base: Some(0), index: None, disp: 0x20, a32: true, ..m }, &regs, 0), 0x10);
    }
}
