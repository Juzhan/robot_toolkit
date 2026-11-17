"""Replay the supplied UR5 archive without importing its code or using hardware."""
import argparse
import json
from pathlib import Path
import zipfile

from robot_toolkit.calibrate.quality import normalize, solve, verify, write_json, save_overlays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        parser.error('use a new output directory')
    with zipfile.ZipFile(args.archive) as archive:
        results = {}
        for name in ('wrist_camera_wrist', 'external_camera_external'):
            prefix = f'data\\calib\\{name}\\'
            def read(relative):
                return json.loads(archive.read(prefix + relative))
            doc = read('views.json')
            heldout = read('verify\\views.json')
            old = read('verify\\verify.json')
            state, report = solve(doc)
            validation = verify(state, heldout)
            # Pixel/depth equations and intrinsic selection reproduce legacy
            # results. Drift now holds the selected K fixed in both solves.
            for key in ('mean_rms_px', 'worst_rms_px', 'depth_error_mm_median'):
                if abs(validation[key] - old[key]) > 1e-3:
                    raise AssertionError(f'{name}: {key} differs from archive')
            root = output / name
            write_json(root / 'views.json', normalize(doc))
            write_json(root / 'verify_views.json', normalize(heldout))
            write_json(root / 'solution.json', state)
            write_json(root / 'report.json', report)
            write_json(root / 'verify.json', validation)
            save_overlays(state, normalize(heldout), root / 'overlays')
            results[name] = dict(source=report['chosen_source'],
                                 cv_rms_px=report['candidates'][report['chosen_source']]['cv_rms_px'],
                                 verify_mean_px=validation['mean_rms_px'],
                                 depth_mm=validation['depth_error_mm_median'],
                                 status=validation['status'])
        write_json(output / 'summary.json', results)
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
