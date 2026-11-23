//! `XCL_NETWORK=0`: the title runs as on a console without a network. Its WinHTTP requests and
//! its sockets fail as they do there (name not resolved / network unreachable); nothing is
//! answered in a server's place. Loopback connections stay (titles make socket pairs), and so
//! do lookups of the local names (none, this machine's, localhost). Only
//! the title's own imports are concerned (the game's exe and DLLs); off by default.
//!
//! For titles under bring-up: a crash reporter otherwise uploads a memory dump of the process
//! on every crash.

use std::ffi::c_void;

use windows_sys::Win32::Foundation::SetLastError;

pub const NETOFF: &str = "xcl-netoff.dll";

const ERROR_WINHTTP_NAME_NOT_RESOLVED: u32 = 12007;
const WSAENETUNREACH: i32 = 10051;
const WSAHOST_NOT_FOUND: i32 = 11001;

#[link(name = "ws2_32", kind = "raw-dylib")]
extern "system" {
    #[link_name = "connect"]
    fn real_connect(s: usize, name: *const u8, len: i32) -> i32;
    #[link_name = "sendto"]
    fn real_sendto(s: usize, buf: *const u8, len: i32, flags: i32, to: *const u8, tolen: i32) -> i32;
    #[link_name = "getaddrinfo"]
    fn real_getaddrinfo(node: *const u8, service: *const u8, hints: *const c_void, out: *mut *mut c_void) -> i32;
    #[link_name = "GetAddrInfoW"]
    fn real_getaddrinfo_w(node: *const u16, service: *const u16, hints: *const c_void, out: *mut *mut c_void) -> i32;
    #[link_name = "GetAddrInfoExW"]
    fn real_getaddrinfo_ex(n: *const u16, s: *const u16, ns: u32, id: *const c_void, hints: *const c_void, out: *mut *mut c_void, t: *const c_void, ov: *mut c_void, cb: *const c_void, h: *mut c_void) -> i32;
    #[link_name = "freeaddrinfo"]
    fn real_freeaddrinfo(p: *mut c_void);
    fn WSASetLastError(e: i32);
    fn WSAStartup(version: u16, data: *mut u8) -> i32;
}

/// Numeric loopback. Resolving this does not ask DNS.
const LOOPBACK: &[u8] = b"127.0.0.1\0";
const LOOPBACK_W: &[u16] = &[b'1' as u16, b'2' as u16, b'7' as u16, b'.' as u16, b'0' as u16, b'.' as u16, b'0' as u16, b'.' as u16, b'1' as u16, 0];

pub fn enabled() -> bool {
    std::env::var("XCL_NETWORK").is_ok_and(|v| v.trim() == "0")
}

/// (name, address, ws2_32 ordinal – 0: imported by name only)
fn table() -> Vec<(&'static str, usize, u16)> {
    vec![
        ("WinHttpSendRequest", send_request as *const () as usize, 0),
        ("connect", connect as *const () as usize, 4),
        ("sendto", sendto as *const () as usize, 20),
        ("gethostbyname", gethostbyname as *const () as usize, 52),
        ("WSAConnect", wsa_connect as *const () as usize, 0),
        ("WSASendTo", wsa_send_to as *const () as usize, 0),
        ("getaddrinfo", getaddrinfo as *const () as usize, 0),
        ("GetAddrInfoW", getaddrinfo_w as *const () as usize, 0),
        ("GetAddrInfoExW", getaddrinfo_ex as *const () as usize, 0),
    ]
}

pub fn exports() -> Vec<(&'static str, usize)> {
    table().into_iter().map(|(n, a, _)| (n, a)).collect()
}

pub fn ordinals() -> Vec<(u16, &'static str)> {
    table().into_iter().filter(|t| t.2 != 0).map(|(n, _, o)| (o, n)).collect()
}

fn note_msg(msg: &str) {
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 16 {
        crate::xlog_info!("XCL_NETWORK=0: {msg}");
    }
}

fn note(what: &str) {
    note_msg(&format!("{what} fails (no network)"));
}

/// 127.0.0.0/8 or ::1 (sockaddr_in / sockaddr_in6)
unsafe fn loopback(name: *const u8, len: i32) -> bool {
    if name.is_null() || len < 8 {
        return false;
    }
    match (name as *const u16).read_unaligned() {
        2 => *name.add(4) == 127,
        23 if len >= 24 => {
            let a = std::slice::from_raw_parts(name.add(8), 16);
            a[..15].iter().all(|&b| b == 0) && a[15] == 1
        }
        _ => false,
    }
}

unsafe fn unreachable_net(what: &str) -> i32 {
    note(what);
    WSASetLastError(WSAENETUNREACH);
    -1
}

unsafe extern "system" fn send_request(_req: *mut c_void, _headers: *const u16, _hlen: u32, _opt: *const c_void, _olen: u32, _total: u32, _ctx: usize) -> i32 {
    note("WinHttpSendRequest");
    SetLastError(ERROR_WINHTTP_NAME_NOT_RESOLVED);
    0
}

unsafe extern "system" fn connect(s: usize, name: *const u8, len: i32) -> i32 {
    if loopback(name, len) {
        return real_connect(s, name, len);
    }
    unreachable_net("connect")
}

unsafe extern "system" fn sendto(s: usize, buf: *const u8, len: i32, flags: i32, to: *const u8, tolen: i32) -> i32 {
    if to.is_null() || loopback(to, tolen) {
        return real_sendto(s, buf, len, flags, to, tolen);
    }
    unreachable_net("sendto")
}

unsafe extern "system" fn wsa_connect(s: usize, name: *const u8, len: i32, _a: *const c_void, _b: *mut c_void, _c: *const c_void, _d: *const c_void) -> i32 {
    connect(s, name, len)
}

#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn wsa_send_to(_s: usize, _bufs: *const c_void, _n: u32, _sent: *mut u32, _flags: u32, _to: *const u8, _tolen: i32, _ov: *mut c_void, _cb: *const c_void) -> i32 {
    unreachable_net("WSASendTo")
}

unsafe extern "system" fn gethostbyname(_name: *const u8) -> *mut c_void {
    note("gethostbyname");
    WSASetLastError(WSAHOST_NOT_FOUND);
    std::ptr::null_mut()
}

unsafe fn ansi_bytes<'a>(node: *const u8) -> &'a [u8] {
    if node.is_null() {
        return b"";
    }
    let mut n = 0;
    while n < 255 && *node.add(n) != 0 {
        n += 1;
    }
    std::slice::from_raw_parts(node, n)
}

fn local_ansi(node: *const u8) -> bool {
    let s = unsafe { ansi_bytes(node) };
    node.is_null() || s.is_empty() || s.eq_ignore_ascii_case(b"localhost") || s == b"127.0.0.1" || s == b"::1"
}

unsafe fn wide_bytes<'a>(node: *const u16) -> &'a [u16] {
    if node.is_null() {
        return &[];
    }
    let mut n = 0;
    while n < 255 && *node.add(n) != 0 {
        n += 1;
    }
    std::slice::from_raw_parts(node, n)
}

fn wide_is(s: &[u16], ascii: &[u8]) -> bool {
    s.len() == ascii.len() && s.iter().zip(ascii).all(|(c, a)| *c < 128 && (*c as u8).eq_ignore_ascii_case(a))
}

fn local_wide(node: *const u16) -> bool {
    let s = unsafe { wide_bytes(node) };
    node.is_null() || s.is_empty() || wide_is(s, b"localhost") || wide_is(s, b"127.0.0.1") || wide_is(s, b"::1")
}

/// A name that is not local, answered as loopback. The real resolver does this without DNS,
/// so a title that retries a failed lookup does not sit there. `connect` still refuses it
/// unless the title is talking to itself.
unsafe fn loopback_ansi(service: *const u8, hints: *const c_void, out: *mut *mut c_void) -> i32 {
    let hr = real_getaddrinfo(LOOPBACK.as_ptr(), service, hints, out);
    if hr == 0 {
        return 0;
    }
    let hr6 = real_getaddrinfo(b"::1\0".as_ptr(), service, hints, out);
    if hr6 == 0 {
        return 0;
    }
    if !out.is_null() {
        *out = std::ptr::null_mut();
    }
    hr
}

unsafe extern "system" fn getaddrinfo(node: *const u8, service: *const u8, hints: *const c_void, out: *mut *mut c_void) -> i32 {
    if local_ansi(node) {
        return real_getaddrinfo(node, service, hints, out);
    }
    note_msg("getaddrinfo answers 127.0.0.1");
    loopback_ansi(service, hints, out)
}

unsafe extern "system" fn getaddrinfo_w(node: *const u16, service: *const u16, hints: *const c_void, out: *mut *mut c_void) -> i32 {
    if local_wide(node) {
        return real_getaddrinfo_w(node, service, hints, out);
    }
    note_msg("GetAddrInfoW answers 127.0.0.1");
    let hr = real_getaddrinfo_w(LOOPBACK_W.as_ptr(), service, hints, out);
    if hr == 0 {
        return 0;
    }
    const LOOPBACK6_W: &[u16] = &[b':' as u16, b':' as u16, b'1' as u16, 0];
    let hr6 = real_getaddrinfo_w(LOOPBACK6_W.as_ptr(), service, hints, out);
    if hr6 == 0 {
        return 0;
    }
    if !out.is_null() {
        *out = std::ptr::null_mut();
    }
    hr
}

#[allow(clippy::too_many_arguments)]
unsafe extern "system" fn getaddrinfo_ex(n: *const u16, s: *const u16, ns: u32, id: *const c_void, hints: *const c_void, out: *mut *mut c_void, t: *const c_void, ov: *mut c_void, cb: *const c_void, h: *mut c_void) -> i32 {
    let name = if local_wide(n) { n } else { LOOPBACK_W.as_ptr() };
    if name != n {
        note_msg("GetAddrInfoExW answers 127.0.0.1");
    }
    real_getaddrinfo_ex(name, s, ns, id, hints, out, t, ov, cb, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_loopback_goes_through() {
        let v4 = |a: [u8; 4]| [2, 0, 0, 80, a[0], a[1], a[2], a[3], 0, 0, 0, 0, 0, 0, 0, 0];
        let mut v6 = [0u8; 28];
        v6[0] = 23;
        v6[23] = 1;
        unsafe {
            assert!(loopback(v4([127, 0, 0, 1]).as_ptr(), 16));
            assert!(!loopback(v4([8, 8, 8, 8]).as_ptr(), 16));
            assert!(loopback(v6.as_ptr(), 28));
            v6[8] = 0x20;
            assert!(!loopback(v6.as_ptr(), 28));
            assert_eq!(connect(usize::MAX, v4([8, 8, 8, 8]).as_ptr(), 16), -1);
            assert_eq!(send_request(std::ptr::null_mut(), std::ptr::null(), 0, std::ptr::null(), 0, 0, 0), 0);
            let mut wsa = [0u8; 512];
            assert_eq!(WSAStartup(0x0202, wsa.as_mut_ptr()), 0);
            let mut out = std::ptr::null_mut();
            assert_eq!(getaddrinfo(b"example.invalid\0".as_ptr(), std::ptr::null(), std::ptr::null(), &mut out), 0);
            assert!(!out.is_null());
            real_freeaddrinfo(out);
        }
    }
}
