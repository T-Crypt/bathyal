"""Tests for tools/pso_report.py on hand-written synthetic log fragments.

No real log text is used: every fragment below is written by hand, and its encoding (UTF-16,
BOM, NUL bytes, CR, console word wrap) is applied here so the reader is tested too.

Run: python -m unittest discover -s tests -p "test_pso_report.py" -v
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOL = os.path.join(ROOT, "tools", "pso_report.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import pso_report  # noqa: E402

PIPE_A = "0x1111000"
PIPE_B = "0x2222000"


def utf16(lines, crlf=True, bom=True):
    """A synthetic console capture: UTF-16LE, optional BOM, CRLF line ends."""
    text = ("\r\n" if crlf else "\n").join(lines) + "\r\n"
    data = text.encode("utf-16-le")
    return (b"\xff\xfe" + data) if bom else data


def wrap(line, width=60):
    """Split a line at spaces into fragments no longer than width, as a console does."""
    out, cur = [], ""
    for word in line.split(" "):
        if cur and len(cur) + 1 + len(word) > width:
            out.append(cur)
            cur = word
        else:
            cur = word if not cur else cur + " " + word
    out.append(cur)
    return out


FRAG_HEAD = [
    "Bathyal 0.1.0 - starting 'TestTitle'",
    "xcl: D3D12.X: pipelines whose shaders are not compiled yet get a stand-in and are built in the background",
    "xcl: XGameRuntime: QueryApiImpl 8836fe87-edb9-4fe3-8dad-05f0d2cd5b40/8836fe87-edb9-4fe3-8dad-05f0d2cd5b40 -> XCL",
    "xcl: D3D12.X: 600 frames presented, 30.0 fps over the last 300",
]

FRAG_TAIL = [
    "xcl: D3D12.X: pipeline cache: 900 hits, 12 misses",
    "xcl: D3D12.X: driver pipeline cache: 950 pipelines reused",
    "xcl: D3D12 device removed: 0x887a0006 The GPU will not respond to more commands",
    "Game exited with code 0x887a0006 (2289696774)",
]


def activity(draws, dskip, pc, disp, fskip=None):
    tail = "" if fskip is None else f" ({fskip} skipped)"
    return (f"xcl: D3D12.X: console code so far: {draws} draws with translated pipelines "
            f"({dskip} skipped, {pc} with PC pipelines), {disp} dispatches{tail}; coherent 10/20 MB")


class DecodeTest(unittest.TestCase):
    def test_utf16_bom_nul_and_cr_become_lines(self):
        data = utf16(["alpha", "beta"])
        self.assertIn(b"\x00", data)
        text = pso_report.decode_log(data)
        self.assertEqual([l for l in text.split("\n") if l], ["alpha", "beta"])

    def test_utf16_without_bom_is_detected_by_its_nuls(self):
        text = pso_report.decode_log(utf16(["gamma"], bom=False))
        self.assertEqual(text.strip(), "gamma")

    def test_utf8_log_passes_through(self):
        self.assertEqual([l for l in pso_report.decode_log(b"one\r\ntwo\n").split("\n") if l],
                         ["one", "two"])


class UnwrapTest(unittest.TestCase):
    def test_wrapped_fragment_is_joined_before_matching(self):
        long_line = activity(2048, 0, 0, 61, 0)
        self.assertGreater(len(long_line), 60)
        lines = FRAG_HEAD + wrap(long_line, 60) + FRAG_TAIL
        report = pso_report.analyse(pso_report.decode_log(utf16(lines)))
        self.assertEqual(len(report["activity"]["samples"]), 1)
        sample = report["activity"]["samples"][0]
        self.assertEqual(sample["dispatches"], 61)
        self.assertEqual(sample["dispatches_skipped"], 0)

    def test_a_record_that_starts_a_new_line_is_not_joined(self):
        lines = FRAG_HEAD + [activity(64, 0, 0, 0), activity(128, 0, 0, 0)]
        report = pso_report.analyse(pso_report.decode_log(utf16(lines)))
        self.assertEqual(len(report["activity"]["samples"]), 2)


class ReportTest(unittest.TestCase):
    def analyse(self, lines):
        return pso_report.analyse(pso_report.decode_log(utf16(lines)), "synthetic.log")

    def test_frames_and_story_start(self):
        report = self.analyse(FRAG_HEAD + FRAG_TAIL)
        self.assertEqual(report["frames_presented"], 600)
        self.assertEqual(report["story_start"]["record"], 2)
        self.assertEqual(report["story_start"]["frame"], 0)

    def test_pipeline_mentions_grouped_with_count_and_frames(self):
        lines = (FRAG_HEAD + [
            "xcl: D3D12.X: compute pipeline %s still building after 1297 ms - proceeding with stand-in" % PIPE_A,
            "xcl: D3D12.X: 900 frames presented, 20.0 fps over the last 300",
            "xcl: D3D12.X: compute pipeline %s still building after 900 ms - proceeding with stand-in" % PIPE_A,
            "xcl: D3D12.X: graphics pipeline %s not built after 4.0 s of waiting since present 900" % PIPE_B,
        ] + FRAG_TAIL)
        report = self.analyse(lines)
        by_pso = {p["pso"]: p for p in report["named_pipelines"]}
        self.assertEqual(by_pso[PIPE_A]["count"], 2)
        self.assertEqual(by_pso[PIPE_A]["first_frame"], 600)
        self.assertEqual(by_pso[PIPE_A]["last_frame"], 900)
        self.assertEqual(by_pso[PIPE_A]["kinds"], {"still building": 2})
        self.assertEqual(by_pso[PIPE_B]["kinds"], {"not built": 1})
        # sorted by last frame seen: B was seen after A's last mention
        self.assertEqual(report["named_pipelines"][-1]["pso"], PIPE_B)

    def test_execute_indirect_skip_has_a_count_but_no_address(self):
        lines = FRAG_HEAD + [
            "xcl: D3D12.X: ExecuteIndirect with a pipeline that is still being built skipped",
            "xcl: D3D12.X: ExecuteIndirect with a pipeline that is still being built skipped",
        ] + FRAG_TAIL
        report = self.analyse(lines)
        self.assertEqual(report["indirect_skipped"]["count"], 2)
        self.assertEqual(report["named_pipelines"], [])

    def test_skipped_dispatch_pipeline_is_named(self):
        lines = FRAG_HEAD + [
            "xcl: D3D12.X: Dispatch with a pipeline that has no translated console shader (%s) skipped" % PIPE_A,
        ] + FRAG_TAIL
        report = self.analyse(lines)
        self.assertEqual([p["pso"] for p in report["named_pipelines"]], [PIPE_A])

    def test_cache_and_driver_counts_use_the_last_report(self):
        lines = FRAG_HEAD + [
            "xcl: D3D12.X: pipeline cache: 100 hits, 5 misses",
            "xcl: D3D12.X: driver pipeline cache: 700 pipelines reused",
            "xcl: D3D12.X: driver pipeline cache: 500 pipelines saved",
            "xcl: D3D12.X: pipeline cache: 2100 hits, 40 misses",
            "xcl: D3D12.X: driver pipeline cache: 800 pipelines reused",
        ] + FRAG_TAIL
        report = self.analyse(lines)
        # the tail's line is the last report in this fragment
        self.assertEqual((report["cache"]["hits"], report["cache"]["misses"]), (900, 12))
        # the tail's line is the last report here too
        self.assertEqual(report["driver_pipelines"]["reused"], 950)

    def test_end_status_names_the_hang(self):
        report = self.analyse(FRAG_HEAD + FRAG_TAIL)
        self.assertEqual(report["end_status"]["device_removed"]["reason"], "DXGI_ERROR_DEVICE_HUNG")
        self.assertEqual(report["end_status"]["exit"]["code"], "0x887a0006")
        self.assertEqual(report["end_status"]["exit"]["decimal"], 2289696774)

    def test_slope_of_skipped_dispatches_in_the_last_1000_frames(self):
        lines = FRAG_HEAD + [
            "xcl: D3D12.X: 900 frames presented, 20.0 fps over the last 300",
            activity(100, 0, 0, 10, 0),
            "xcl: D3D12.X: 1200 frames presented, 20.0 fps over the last 300",
            activity(200, 0, 0, 20, 300),
            "xcl: D3D12.X: 1500 frames presented, 20.0 fps over the last 300",
            activity(300, 0, 0, 30, 600),
        ] + FRAG_TAIL
        report = self.analyse(lines)
        slope = report["slope"]["dispatches_skipped"]
        # (900, 0), (1200, 300), (1500, 600): one skip per frame
        self.assertAlmostEqual(slope["per_1000_frames"], 1000.0, places=1)
        self.assertEqual((slope["first"], slope["last"]), (0, 600))

    def test_last_step_runs_to_the_final_sample_not_the_first_of_its_frame(self):
        lines = FRAG_HEAD + [
            "xcl: D3D12.X: 900 frames presented, 20.0 fps over the last 300",
            activity(100, 0, 0, 10, 64),
            "xcl: D3D12.X: 1200 frames presented, 20.0 fps over the last 300",
            activity(200, 0, 0, 20, 128),
            activity(300, 0, 0, 30, 512),
        ] + FRAG_TAIL
        report = self.analyse(lines)
        step = report["slope"]["dispatches_skipped"]["last_step"]
        self.assertEqual(step, {"from_frame": 900, "from_value": 64,
                                "to_frame": 1200, "to_value": 512})

    def test_slope_is_absent_when_every_sample_shares_a_frame(self):
        lines = FRAG_HEAD + [
            activity(100, 0, 0, 10, 0),
            activity(200, 0, 0, 20, 128),
            activity(300, 0, 0, 30, 256),
        ] + FRAG_TAIL
        report = self.analyse(lines)
        slope = report["slope"]["dispatches_skipped"]
        self.assertIsNone(slope["per_1000_frames"])
        self.assertEqual((slope["first"], slope["last"]), (0, 256))

    def test_twenty_pipelines_last_seen_before_the_removal(self):
        lines = list(FRAG_HEAD)
        for k in range(25):
            lines.append("xcl: D3D12.X: compute pipeline 0x%x still building after 500 ms" % (0xA000 + k))
        lines += FRAG_TAIL
        report = self.analyse(lines)
        self.assertEqual(len(report["pipelines_before_removal"]), 20)
        self.assertEqual(report["pipelines_before_removal"][-1]["pso"], "0xa018")
        self.assertEqual(report["pipelines_before_removal"][0]["pso"], "0xa005")

    def test_hang_window_separates_not_ready_from_permanent_skips(self):
        lines = FRAG_HEAD + [
            "xcl: D3D12.X: compute pipeline %s still building after 1297 ms" % PIPE_A,
            "xcl: D3D12.X: Dispatch with a pipeline that has no translated console shader (%s) skipped" % PIPE_B,
            "xcl: D3D12.X: ExecuteIndirect with a pipeline that is still being built skipped",
        ] + FRAG_TAIL
        report = self.analyse(lines)
        self.assertEqual([p["pso"] for p in report["hang_window"]["not_ready"]], [PIPE_A])
        self.assertEqual([p["pso"] for p in report["hang_window"]["skipped_dispatch"]], [PIPE_B])
        self.assertEqual(report["hang_window"]["still_being_built_skipped"], 1)

    def test_empty_log_reports_nothing_rather_than_failing(self):
        report = pso_report.analyse("")
        self.assertEqual(report["frames_presented"], 0)
        self.assertIsNone(report["story_start"])


class CliTest(unittest.TestCase):
    def run_tool(self, path, *args):
        return subprocess.run([sys.executable, "-I", TOOL, path, *args],
                              capture_output=True, text=True)

    def test_json_output_and_exit_zero(self):
        lines = FRAG_HEAD + [activity(64, 0, 0, 0)] + FRAG_TAIL
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "synthetic.log")
            with open(path, "wb") as f:
                f.write(utf16(lines))
            run = self.run_tool(path, "--json")
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads(run.stdout)
            self.assertEqual(report["frames_presented"], 600)
            self.assertEqual(report["cache"]["hits"], 900)

    def test_text_output_names_the_hang(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "synthetic.log")
            with open(path, "wb") as f:
                f.write(utf16(FRAG_HEAD + FRAG_TAIL))
            run = self.run_tool(path)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("DXGI_ERROR_DEVICE_HUNG", run.stdout)
            self.assertIn("pipeline cache (last report): 900 hits, 12 misses", run.stdout)

    def test_missing_file_is_a_clean_error(self):
        run = self.run_tool(os.path.join(HERE, "no-such-log.log"))
        self.assertEqual(run.returncode, 2)
        self.assertIn("cannot read", run.stderr)


if __name__ == "__main__":
    unittest.main()
