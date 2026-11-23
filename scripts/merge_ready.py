"""Lists and merges pull requests that are ready: open, not draft, mergeable, every check passed or skipped,
no review asking for changes. Needs the GitHub CLI (gh) logged in with write access. Standard library only.

  python -I scripts/merge_ready.py                      # dry run: the ready list and why the rest wait
  python -I scripts/merge_ready.py --merge 50           # squash-merge up to 50 ready PRs, oldest first
  python -I scripts/merge_ready.py --merge 50 --approved  # only PRs with an approving review

A PR whose merge fails (a conflict that appeared after an earlier merge) is skipped and reported; the run goes on.
Every decision goes to merge_ready.log (append) as one line: time, PR, action, reason.
"""
import argparse
import datetime
import json
import subprocess
import sys
import time

OK_STATES = {"SUCCESS", "SKIPPED", "NEUTRAL"}
FIELDS = "number,title,author,isDraft,mergeable,reviewDecision,statusCheckRollup,baseRefName,createdAt,labels"


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def check_state(pr: dict) -> tuple[bool, str]:
    checks = pr.get("statusCheckRollup") or []
    if not checks:
        return False, "no checks yet"
    for c in checks:
        state = (c.get("conclusion") or c.get("state") or c.get("status") or "").upper()
        name = c.get("name") or c.get("context") or "?"
        if state in ("", "PENDING", "QUEUED", "IN_PROGRESS", "EXPECTED", "WAITING", "ACTION_REQUIRED"):
            return False, f"check {name} {state.lower() or 'pending'}"
        if state not in OK_STATES:
            return False, f"check {name} {state.lower()}"
    return True, "checks green"


def readiness(pr: dict, base: str, approved_only: bool) -> tuple[bool, str]:
    if pr["baseRefName"] != base:
        return False, f"targets {pr['baseRefName']}"
    if pr["isDraft"]:
        return False, "draft"
    if any(l["name"] in ("do-not-merge", "blocked") for l in pr.get("labels") or []):
        return False, "labelled do-not-merge"
    if pr["mergeable"] != "MERGEABLE":
        return False, f"mergeable={pr['mergeable'].lower()}"
    if pr["reviewDecision"] == "CHANGES_REQUESTED":
        return False, "changes requested"
    if approved_only and pr["reviewDecision"] != "APPROVED":
        return False, "not approved"
    return check_state(pr)


def log(line: str) -> None:
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    with open("merge_ready.log", "a", encoding="utf-8") as f:
        f.write(f"{stamp} {line}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default="T-Crypt/bathyal")
    ap.add_argument("--base", default="main")
    ap.add_argument("--merge", type=int, default=0, metavar="N", help="merge up to N ready PRs (default: dry run)")
    ap.add_argument("--approved", action="store_true", help="only PRs with an approving review")
    ap.add_argument("--limit", type=int, default=2000, help="open PRs to read")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between merges (API rate)")
    a = ap.parse_args()

    prs = json.loads(gh("pr", "list", "-R", a.repo, "--state", "open", "--limit", str(a.limit), "--json", FIELDS))
    prs.sort(key=lambda p: p["createdAt"])
    ready, waiting = [], {}
    for pr in prs:
        ok, why = readiness(pr, a.base, a.approved)
        if ok:
            ready.append(pr)
        else:
            waiting.setdefault(why.split(" ")[0] if why.startswith("check") else why, []).append(pr["number"])
    print(f"{len(prs)} open, {len(ready)} ready")
    for why, nums in sorted(waiting.items(), key=lambda kv: -len(kv[1])):
        print(f"  waiting ({why}): {len(nums)}  e.g. #{', #'.join(map(str, nums[:5]))}")
    if not a.merge:
        for pr in ready[:20]:
            print(f"  ready #{pr['number']} {pr['author']['login']}: {pr['title'][:80]}")
        return 0

    merged, failed = 0, 0
    for pr in ready[: a.merge]:
        n = str(pr["number"])
        try:
            gh("pr", "merge", n, "-R", a.repo, "--squash")
            merged += 1
            log(f"#{n} merged ({pr['author']['login']}: {pr['title'][:80]})")
            print(f"merged #{n}")
        except subprocess.CalledProcessError as e:
            failed += 1
            reason = (e.stderr or "").strip().splitlines()[-1:] or ["unknown"]
            log(f"#{n} skipped: {reason[0][:160]}")
            print(f"skipped #{n}: {reason[0][:160]}")
        time.sleep(a.pause)
    print(f"merged {merged}, skipped {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
