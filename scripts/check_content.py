"""Content gate for pull requests: fails when a change adds what this repository must never hold.

Run by CI on the files a pull request changes (python -I scripts/check_content.py --base origin/main), or by hand
on the whole tree (no --base). Standard library only. Exit 0 = clean, 1 = findings (one line each), 2 = usage.

Checks, on added or modified files only:
  - binaries: Windows PE (MZ ... PE), ELF, and archive/media containers by magic, wherever they are named;
  - file types that are game material here: .exe .dll .rpf .xvd .bk2 .awc .dxbc .dxil .spv .ytd .ydr .pak ...;
  - any file over 1 MiB, outside the allow list below;
  - commercial title names in text (the project writes "a newer retail title");
  - new HTTP/TLS/telemetry crates in Cargo.lock (the code must not go online).
"""
import argparse
import os
import re
import subprocess
import sys

MAX_BYTES = 1 << 20
# Synthetic fixtures that are allowed to be binary (tests build them by hand). Paths relative to the root.
ALLOWED_BINARY_PREFIXES = ("tests/data/testgame/",)
BANNED_EXTENSIONS = {
    ".exe", ".dll", ".sys", ".rpf", ".xvd", ".xvc", ".msixvc", ".appx", ".bk2", ".bik", ".awc", ".xma", ".dxbc",
    ".dxil", ".spv", ".cso", ".ytd", ".ydr", ".ydd", ".yft", ".ybn", ".ymap", ".ytyp", ".pak", ".pdb",
}
MAGIC = [
    (b"MZ", "Windows executable (MZ)"),
    (b"\x7fELF", "ELF executable"),
    (b"RPF", "RPF archive"),
    (b"DXBC", "DXBC shader container"),
    (b"\x03\x02\x23\x07", "SPIR-V module"),
    (b"BIK", "Bink video"),
    (b"KB2", "Bink 2 video"),
    (b"PK\x03\x04", "ZIP archive"),
    (b"7z\xbc\xaf\x27\x1c", "7-Zip archive"),
    (b"Rar!", "RAR archive"),
]
TITLE_NAMES = re.compile(
    r"grand\s+theft\s+auto|\bgta\s*(?:v|vi|5|6|online)\b|rockstar\s+games|\bcyberleek\b|game_scarlett_master",
    re.IGNORECASE)
NETWORK_CRATES = {
    "reqwest", "ureq", "hyper", "curl", "curl-sys", "isahc", "attohttpc", "surf", "native-tls", "rustls",
    "openssl", "openssl-sys", "tungstenite", "tokio-tungstenite", "sentry", "opentelemetry", "h2", "quinn",
}
TEXT_EXTENSIONS = {".rs", ".md", ".toml", ".py", ".ps1", ".yml", ".yaml", ".txt", ".json", ".hlsl", ".cfg", ".bat",
                   ".cmd", ".sh", ".lock"}
SELF = "scripts/check_content.py"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def changed_files(base: str) -> list[str]:
    out = git("diff", "--name-only", "--diff-filter=AMR", "-z", f"{base}...HEAD")
    return [p for p in out.split("\0") if p]


def all_files() -> list[str]:
    return [p for p in git("ls-files", "-z").split("\0") if p]


def crates(lock_text: str) -> set[str]:
    return set(re.findall(r'^name = "([^"]+)"', lock_text, re.MULTILINE))


def check_file(path: str) -> list[str]:
    findings = []
    if not os.path.isfile(path):
        return findings
    rel = path.replace("\\", "/")
    allowed_binary = rel.startswith(ALLOWED_BINARY_PREFIXES)
    ext = os.path.splitext(rel)[1].lower()
    size = os.path.getsize(path)
    if ext in BANNED_EXTENSIONS and not allowed_binary:
        findings.append(f"{rel}: file type {ext} is not allowed in this repository")
    if size > MAX_BYTES and not allowed_binary:
        findings.append(f"{rel}: {size} bytes, over the {MAX_BYTES}-byte limit")
    with open(path, "rb") as f:
        head = f.read(4096)
    if not allowed_binary:
        for magic, what in MAGIC:
            if head.startswith(magic):
                if magic == b"MZ" and b"PE\0\0" not in head:
                    continue
                findings.append(f"{rel}: looks like a {what}")
                break
    if ext in TEXT_EXTENSIONS and rel != SELF and size <= 4 * MAX_BYTES:
        with open(path, encoding="utf-8", errors="replace") as f:
            for number, line in enumerate(f, 1):
                m = TITLE_NAMES.search(line)
                if m:
                    findings.append(f"{rel}:{number}: names a commercial title ({m.group(0)!r}); "
                                    "write \"a newer retail title\"")
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", help="compare against this ref (merge base); without it the whole tree is checked")
    a = ap.parse_args()
    files = changed_files(a.base) if a.base else all_files()
    findings = []
    for path in files:
        findings.extend(check_file(path))
    if a.base and "Cargo.lock" in files:
        try:
            before = crates(git("show", f"{a.base}:Cargo.lock"))
        except subprocess.CalledProcessError:
            before = set()
        with open("Cargo.lock", encoding="utf-8") as f:
            added = crates(f.read()) - before
        for name in sorted(added & NETWORK_CRATES):
            findings.append(f"Cargo.lock: adds network crate {name!r}; the code must not go online")
    for line in findings:
        print(line)
    print(f"check_content: {len(files)} files checked, {len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
