"""Generate and preview a fixed calibration view plan without motion."""
import argparse
import json
from pathlib import Path

from ..profiles import build_robot
from .quality import validate_identity, write_json
from .view_plan import generate, save_preview
from .identity import calibration_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--solution', required=True)
    parser.add_argument('--anchor', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--candidates', type=int, default=4096)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        parser.error('use a new output directory')
    task = json.loads(Path(args.task).read_text())
    state = json.loads(Path(args.solution).read_text())
    cfg = task['calibrate']
    camera = next(c for c in task['collect']['cameras'] if c['name'] == cfg['camera'])
    validate_identity(state, calibration_identity(camera, cfg))
    profile = task['robot']
    robot = build_robot(profile)
    anchor_doc = json.loads(Path(args.anchor).read_text())
    if anchor_doc['robot'] != profile['name']:
        parser.error('anchor robot does not match task')
    try:
        robot.connect_read_only()
        plan = generate(state, anchor_doc['q_rad'], robot.forward_kinematics,
                        robot.guard.limits, robot.workspace,
                        count=cfg['min_samples'], verify_count=cfg['verify_samples'],
                        candidate_count=args.candidates, delta_rad=cfg['pose_delta_rad'],
                        clearance_m=cfg.get('path_clearance_m', .01))
    finally:
        robot.disconnect()
    out.mkdir(parents=True)
    write_json(out/'plan.json', plan)
    write_json(out/'coarse_solution.json', state)
    save_preview(plan, out/'preview.svg')
    print(json.dumps({k: plan[k] for k in ('plan_id', 'accepted_candidates', 'rejected',
                                          'train_count', 'verify_count')}, indent=2))
    print(f'Plan: {out / "plan.json"}')


if __name__ == '__main__':
    main()
