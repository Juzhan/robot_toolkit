"""Offline solve/verify entry point. Never connects to hardware."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .quality import normalize, solve, verify, write_json, save_overlays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['solve', 'verify'])
    parser.add_argument('views', help='native or legacy UR5 views.json')
    parser.add_argument('--solution', help='solution.json (required for verify)')
    parser.add_argument('--out', required=True, help='new output directory')
    parser.add_argument('--intrinsics', choices=['auto', 'factory', 'calibrated'], default='auto')
    parser.add_argument('--min-views', type=int, default=None)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        parser.error('output directory already exists; use a new directory')
    doc = normalize(json.loads(Path(args.views).read_text()))
    if args.action == 'solve':
        state, report = solve(doc, args.intrinsics, args.min_views or 27)
    else:
        if not args.solution:
            parser.error('verify requires --solution')
        state = json.loads(Path(args.solution).read_text())
        report = verify(state, doc, args.min_views or 8)
    write_json(out / ('report.json' if args.action == 'solve' else 'verify.json'), report)
    if args.action == 'solve':
        write_json(out / 'solution.json', state)
    save_overlays(state, doc, out / 'overlays')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if args.action == 'solve' or report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
