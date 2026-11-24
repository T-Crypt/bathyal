#!/usr/bin/env python3
"""Read a run log and certify what the pipeline path was doing when the GPU stopped.

Answer, from the log alone: which pipelines were not ready at the hang, how far the
translated-shader cache got, and how the skipped work grew. Standard library only; run it as
`python -I tools/pso_report.py LOG [--json]`.

The log is a console capture: UTF-16 with NUL bytes, CRLF, and long lines word-wrapped at the
console width. Decoding strips the NULs, turns CR into newline, and joins wrapped fragments
back into one record before any line is matched. A record that the console cut short is
parsed for what it does hold; a missing tail is reported as missing, not guessed.
"""
import argparse
import json
import re
import sys

# ---------------------------------------------------------------------------------------
# decoding and unwrapping


def decode_log(data: bytes) -> str:
    """Bytes of a console capture to text: UTF-16 by BOM or by its NUL pattern, else UTF-8."""
    if data[:2] == b"\xff\xfe":
        text = data[2:].decode("utf-16-le", "replace")
    elif data[:2] == b"\xfe\xff":
        text = data[2:].decode("utf-16-be", "replace")
    elif data.count(b"\x00") > len(data) // 4:
        text = data.decode("utf-16-le", "replace")
    else:
        text = data.decode("utf-8", "replace")
    return text.replace("\x00", "").replace("\r", "\n")


# A record starts with one of these; anything else after a full-width fragment is a wrap.
_RECORD_START = re.compile(
    r"^(?:"
    r"xcl: |"
    r"[A-Za-z][A-Za-z0-9 ._()/-]{0,40}: |"      # Report: / Log: / GPU: / data: / exe:
    r"Bathyal |Game exited |MicrosoftGame|At |Unimplemented|Traceback|\d+\?\?|"
    r"[+\-*] "
    r")"
)


def unwrap(text: str):
    """Physical lines to records; a wrapped fragment is joined to the record it belongs to.

    The console wraps a long record at its width, so every fragment after the first starts
    without a record prefix (a syscall stack, "background)", "dispatches (0 skipped); ...").
    A line that starts a record ends the one before it.
    """
    records = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        if records and not _RECORD_START.match(line):
            records[-1] = records[-1].rstrip() + " " + line.strip()
        else:
            records.append(line.strip())
    return records


# ---------------------------------------------------------------------------------------
# patterns. Every one matches a prefix only, so a line the console cut short still counts.

RE_FRAMES = re.compile(r"(\d+) frames presented")
RE_STORY = re.compile(r"QueryApiImpl 8836fe87")
RE_CACHE = re.compile(r"pipeline cache: (\d+) hits, (\d+) misses")
RE_DRIVER = re.compile(r"driver pipeline cache: (\d+) pipelines (reused|saved)")
RE_ACTIVITY = re.compile(
    r"console code so far: (\d+) draws with translated pipelines "
    r"\((\d+) skipped, (\d+) with PC pipelines\), (\d+) dispatches(?: \((\d+) skipped\))?")
RE_COMPUTE_STILL = re.compile(r"compute pipeline (0x[0-9a-fA-F]+) still building after (\d+) ms")
RE_GRAPHICS_NOT = re.compile(r"graphics pipeline (0x[0-9a-fA-F]+) not built after")
RE_LONG_WAIT = re.compile(
    r"waited [\d.]+ s for (compute|graphics) pipeline (0x[0-9a-fA-F]+)")
RE_SKIPPED_DISPATCH = re.compile(
    r"Dispatch with a pipeline that has no translated console shader \((0x[0-9a-fA-F]+)\) skipped")
RE_SKIPPED_INDIRECT = re.compile(r"ExecuteIndirect with a pipeline that is still being built skipped")
RE_EPOCH = re.compile(r"since present (\d+)")
RE_REMOVED = re.compile(r"D3D12 device removed: (0x[0-9a-fA-F]+)")
RE_EXIT = re.compile(r"Game exited with code (0x[0-9a-fA-F]+)(?: \((\d+)\))?")
RE_EXIT_KV = re.compile(r"\bexit=(\S+)")

REMOVAL_REASONS = {
    0x887A0005: "DXGI_ERROR_DEVICE_REMOVED",
    0x887A0006: "DXGI_ERROR_DEVICE_HUNG",
    0x887A0007: "DXGI_ERROR_DEVICE_RESET",
    0x887A0020: "DXGI_ERROR_DRIVER_INTERNAL_ERROR",
}

# Kinds a pipeline can be named by, and whether the mention means it was not ready to draw.
NOT_READY = {"still building", "not built", "long wait"}


def analyse(text: str, path: str = "<log>") -> dict:
    records = unwrap(text)

    frame = 0
    story_start = None
    frames_reported = 0
    cache = None
    driver = None
    activity = []
    pipelines = {}          # address -> {count, first_frame, last_frame, kinds, last_index}
    indirect_skipped = []   # (index, frame) for the unaddressed ExecuteIndirect skips
    removal = None
    exit_status = None

    for index, line in enumerate(records):
        m = RE_FRAMES.search(line)
        if m:
            frames_reported = max(frames_reported, int(m.group(1)))
            frame = frames_reported

        if story_start is None and RE_STORY.search(line):
            story_start = {"line": index, "frame": frame}

        m = RE_CACHE.search(line)
        if m:
            cache = {"hits": int(m.group(1)), "misses": int(m.group(2)), "line": index, "frame": frame}

        m = RE_DRIVER.search(line)
        if m:
            what = m.group(2)
            if what == "reused":
                driver = {"reused": int(m.group(1)), "line": index, "frame": frame}
            elif driver is None:
                driver = {"reused": 0, "line": index, "frame": frame}
            driver["saved"] = int(m.group(1)) if what == "saved" else driver.get("saved", 0)

        m = RE_ACTIVITY.search(line)
        if m:
            draws, draws_skipped, draws_pc, dispatches = (int(m.group(i)) for i in (1, 2, 3, 4))
            disp_skipped = int(m.group(5)) if m.group(5) is not None else None
            activity.append({"frame": frame, "line": index, "draws": draws,
                             "draws_skipped": draws_skipped, "draws_pc": draws_pc,
                             "dispatches": dispatches, "dispatches_skipped": disp_skipped})

        # A finer frame number than the 300-frame report, where the log carries one.
        e = RE_EPOCH.search(line)
        if e:
            frame = max(frame, int(e.group(1)))

        kind = addr = None
        m = RE_COMPUTE_STILL.search(line)
        if m:
            kind, addr = "still building", m.group(1)
        else:
            m = RE_GRAPHICS_NOT.search(line)
            if m:
                kind, addr = "not built", m.group(1)
            else:
                m = RE_LONG_WAIT.search(line)
                if m:
                    kind, addr = "long wait", m.group(2)
                else:
                    m = RE_SKIPPED_DISPATCH.search(line)
                    if m:
                        kind, addr = "skipped dispatch", m.group(1)

        if RE_SKIPPED_INDIRECT.search(line):
            indirect_skipped.append({"index": index, "frame": frame})

        if kind:
            p = pipelines.setdefault(addr.lower(), {
                "pso": addr.lower(), "count": 0, "first_frame": frame,
                "last_frame": frame, "kinds": {}, "line": index})
            p["count"] += 1
            p["last_frame"] = frame
            p["kinds"][kind] = p["kinds"].get(kind, 0) + 1
            p["line"] = index

        m = RE_REMOVED.search(line)
        if m and removal is None:
            code = int(m.group(1), 16)
            removal = {"line": index, "frame": frame, "code": m.group(1).lower(),
                       "reason": REMOVAL_REASONS.get(code, "unknown")}

        if exit_status is None:
            m = RE_EXIT.search(line)
            if m:
                exit_status = {"code": m.group(1).lower(),
                               "decimal": int(m.group(2)) if m.group(2) else int(m.group(1), 16),
                               "line": index}
            else:
                m = RE_EXIT_KV.search(line)
                if m:
                    exit_status = {"code": m.group(1), "decimal": None, "line": index}

    ordered = sorted(pipelines.values(), key=lambda p: p["line"])
    named = [{"pso": p["pso"], "count": p["count"], "first_frame": p["first_frame"],
              "last_frame": p["last_frame"], "kinds": p["kinds"]}
             for p in sorted(ordered, key=lambda p: (p["last_frame"], p["line"]))]

    before_removal = [p for p in ordered if removal is None or p["line"] < removal["line"]]
    last_before_removal = [{"pso": p["pso"], "count": p["count"], "first_frame": p["first_frame"],
                            "last_frame": p["last_frame"], "kinds": p["kinds"]}
                           for p in before_removal[-20:]]

    end_frame = removal["frame"] if removal else frames_reported
    window_start = max(0, end_frame - 1000)
    not_ready = [n for n in named
                 if n["last_frame"] >= window_start and set(n["kinds"]) & NOT_READY]
    skipped_dispatch = [n for n in named if "skipped dispatch" in n["kinds"]]

    return {
        "log": path,
        "frames_presented": frames_reported,
        "story_start": story_start,
        "end_status": {"exit": exit_status, "device_removed": removal},
        "cache": cache,
        "driver_pipelines": driver,
        "activity": {"samples": activity, "resolution_note":
                     "frame numbers move in steps of the log's frames-presented report"},
        "slope": slope(activity, end_frame),
        "named_pipelines": named,
        "indirect_skipped": {"count": len(indirect_skipped),
                             "first_frame": indirect_skipped[0]["frame"] if indirect_skipped else None,
                             "last_frame": indirect_skipped[-1]["frame"] if indirect_skipped else None},
        "hang_window": {"from_frame": window_start, "to_frame": end_frame,
                        "not_ready": not_ready,
                        "still_being_built_skipped": sum(
                            1 for s in indirect_skipped if s["frame"] >= window_start),
                        "skipped_dispatch": skipped_dispatch},
        "pipelines_before_removal": last_before_removal,
    }


def slope(samples, end_frame):
    """Skipped dispatches and draws, in the last 1000 frames, as a per-1000-frame rate."""
    window = [s for s in samples if s["frame"] >= end_frame - 1000]
    out = {"from_frame": max(0, end_frame - 1000), "to_frame": end_frame,
           "samples": len(window)}
    for field in ("dispatches_skipped", "draws_skipped"):
        points = [s for s in window if s[field] is not None]
        if len(points) < 2:
            out[field] = None
            continue
        xs = [s["frame"] for s in points]
        ys = [s[field] for s in points]
        if max(xs) == min(xs):
            out[field] = {"per_1000_frames": None, "first": ys[0], "last": ys[-1],
                          "note": "all samples share one frame number; the log reports "
                                  f"frames only every 300 (delta {ys[-1] - ys[0]} seen at one frame)"}
            continue
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        var = sum((x - mx) ** 2 for x in xs)
        per_frame = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
        steps = [(x, y) for i, (x, y) in enumerate(zip(xs, ys))
                 if i == 0 or x != xs[i - 1]]
        out[field] = {"per_1000_frames": round(per_frame * 1000, 1),
                      "first": ys[0], "last": ys[-1],
                      "frames_covered": max(xs) - min(xs),
                      "distinct_frames": len(set(xs)),
                      "last_step": {"from_frame": steps[-2][0], "from_value": steps[-2][1],
                                    "to_frame": steps[-1][0], "to_value": steps[-1][1]}
                      if len(steps) > 1 else None}
    return out


# ---------------------------------------------------------------------------------------
# text report


def _fmt_pipeline(row):
    kinds = ", ".join(f"{k} x{n}" for k, n in sorted(row["kinds"].items()))
    return (f"  {row['pso']}  count {row['count']:>5}  frames {row['first_frame']}-{row['last_frame']}"
            f"  {kinds}")


def render(report) -> str:
    out = []
    out.append(f"log: {report['log']}")
    out.append(f"frames presented: {report['frames_presented']}")
    story = report["story_start"]
    out.append("story start: " + (f"line {story['line']}, frame {story['frame']}"
                                  if story else "not found (no QueryApiImpl 8836fe87 line)"))

    end = report["end_status"]
    removal, exit_status = end["device_removed"], end["exit"]
    if removal:
        out.append(f"end status: device removed {removal['code']} {removal['reason']} "
                   f"at frame {removal['frame']}")
    else:
        out.append("end status: no device removal line")
    if exit_status:
        out.append(f"  exit: {exit_status['code']}"
                   + (f" ({exit_status['decimal']})" if exit_status["decimal"] is not None else ""))
    else:
        out.append("  exit: no exit line")

    cache = report["cache"]
    out.append("pipeline cache (last report): " +
               (f"{cache['hits']} hits, {cache['misses']} misses" if cache else "not found"))
    driver = report["driver_pipelines"]
    if driver:
        out.append(f"driver pipelines reused: {driver['reused']}"
                   + (f" (saved {driver['saved']})" if driver.get("saved") else ""))
    else:
        out.append("driver pipelines reused: not found")

    s = report["slope"]
    def rate(field):
        v = s[field]
        if v is None:
            return "no samples"
        if isinstance(v, dict) and v.get("per_1000_frames") is None:
            return f"{v['first']} -> {v['last']} (no frame span)"
        last = ""
        if v.get("last_step"):
            st = v["last_step"]
            last = (f"; last step at frame {st['to_frame']}: "
                    f"{st['from_value']} -> {st['to_value']}")
        return (f"{v['per_1000_frames']:+.1f} per 1000 frames "
                f"({v['first']} -> {v['last']} over {v['frames_covered']} frames, "
                f"{v['distinct_frames']} distinct frame numbers{last})")
    out.append(f"skipped work, last 1000 frames ({s['from_frame']}-{s['to_frame']}, {s['samples']} samples):")
    out.append(f"  dispatches skipped: {rate('dispatches_skipped')}")
    out.append(f"  draws skipped:      {rate('draws_skipped')}")

    samples = report["activity"]["samples"]
    if samples:
        tail = samples[-11:]
        out.append("  samples (frame, draws, draws skipped, dispatches, dispatches skipped):")
        for row in tail:
            out.append(f"    {row['frame']:>6} {row['draws']:>7} {row['draws_skipped']:>6} "
                       f"{row['dispatches']:>7} "
                       + ("-" if row["dispatches_skipped"] is None else str(row["dispatches_skipped"])))

    named = report["named_pipelines"]
    out.append(f"pipelines named ({len(named)}), by last frame seen:")
    for row in named:
        out.append(_fmt_pipeline(row))
    if not named:
        out.append("  none")

    ind = report["indirect_skipped"]
    out.append(f"ExecuteIndirect skipped on a pipeline still being built: {ind['count']}"
               + (f" (frames {ind['first_frame']}-{ind['last_frame']})" if ind["count"] else ""))

    hw = report["hang_window"]
    out.append(f"not ready in the last 1000 frames ({hw['from_frame']}-{hw['to_frame']}):")
    if hw["not_ready"]:
        for row in hw["not_ready"]:
            out.append(_fmt_pipeline(row))
    else:
        out.append("  none")
    out.append(f"  pipelines still being built when skipped (no address): {hw['still_being_built_skipped']}")
    if hw["skipped_dispatch"]:
        out.append(f"  pipelines whose dispatch was skipped for a missing translation: "
                   f"{len(hw['skipped_dispatch'])}")
        for row in hw["skipped_dispatch"]:
            out.append(_fmt_pipeline(row))

    out.append(f"last {len(report['pipelines_before_removal'])} pipelines seen before the device removal:")
    for row in report["pipelines_before_removal"]:
        out.append(_fmt_pipeline(row))
    if not report["pipelines_before_removal"]:
        out.append("  none")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Certify pipeline readiness and cache state from a run log.")
    ap.add_argument("log", help="a console capture (UTF-16 or UTF-8, NUL bytes and CR tolerated)")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--out", help="write the report to this file instead of stdout")
    a = ap.parse_args(argv)

    try:
        with open(a.log, "rb") as f:
            data = f.read()
    except OSError as e:
        print(f"pso_report: cannot read {a.log}: {e}", file=sys.stderr)
        return 2

    report = analyse(decode_log(data), a.log)
    text = json.dumps(report, indent=2) + "\n" if a.json else render(report) + "\n"
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
