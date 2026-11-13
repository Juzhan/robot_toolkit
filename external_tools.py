"""Explicit external tool configuration; no downloads, startup, or hardware I/O."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent


def _outside_repo(path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved == _REPO_ROOT or _REPO_ROOT in resolved.parents:
        raise ValueError('external tool paths must be outside the robot_toolkit repository')
    return resolved


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    capability: str
    repository: str
    revision: str
    service_path: str | None
    adapter: str | None
    notes: str


@dataclass(frozen=True)
class ToolRuntime:
    name: str
    root: Path
    endpoint: str | None = None
    timeout_s: float = 30.0
    enabled: bool = False

    def __post_init__(self):
        if not self.name or any(character not in 'abcdefghijklmnopqrstuvwxyz0123456789_-'
                    for character in self.name):
            raise ValueError('tool name must be a simple lowercase identifier')
        object.__setattr__(self, 'root', _outside_repo(self.root))
        if not isinstance(self.enabled, bool):
            raise ValueError('enabled must be a boolean')
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError('timeout_s must be positive and finite')
        if self.endpoint is not None:
            parsed = urlsplit(self.endpoint)
            if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                    or parsed.username is not None or parsed.password is not None
                    or parsed.query or parsed.fragment):
                raise ValueError('endpoint must be HTTP(S) without credentials, query or fragment')
            if parsed.port is not None and not 1 <= parsed.port <= 65535:
                raise ValueError('endpoint port must be between 1 and 65535')

    @property
    def source_dir(self) -> Path:
        return _outside_repo(self.root / self.name / 'source')

    @property
    def environment_dir(self) -> Path:
        return _outside_repo(self.root / self.name / 'environment')

    @property
    def models_dir(self) -> Path:
        return _outside_repo(self.root / self.name / 'models')


class ExternalTools:
    """Catalogue plus machine-local configuration, without owning any processes."""

    def __init__(self, definitions, *, root=None, runtimes=None):
        definitions = tuple(definitions)
        self.definitions = {item.name: item for item in definitions}
        if len(self.definitions) != len(definitions):
            raise ValueError('duplicate tool name')
        for name in self.definitions:
            if not name or any(character not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for character in name):
                raise ValueError('tool names must be simple lowercase identifiers')
        self.root = _outside_repo(root or os.environ.get(
            'ROBOT_TOOLKIT_TOOLS_HOME', '~/.local/share/robot-toolkit-tools'))
        self.runtimes = dict(runtimes or {})
        if set(self.runtimes) - set(self.definitions):
            raise ValueError('runtime config contains unknown tools')
        for name, runtime in self.runtimes.items():
            if name != runtime.name:
                raise ValueError('runtime key and tool name disagree')

    @classmethod
    def load(cls, *, config_path=None, root=None) -> 'ExternalTools':
        catalog = json.loads((_REPO_ROOT / 'configs' / 'external_tools.json').read_text())
        if catalog['schema_version'] != 1:
            raise ValueError('unsupported tool catalogue version')
        definitions = [ToolDefinition(**entry) for entry in catalog['tools']]
        runtimes = {}
        if config_path is not None:
            config = json.loads(_outside_repo(config_path).read_text())
            if config['schema_version'] != 1:
                raise ValueError('unsupported runtime config version')
            root = root or config.get('root')
            for name, values in config.get('tools', {}).items():
                values = dict(values)
                runtime_root = values.pop('root', root or os.environ.get(
                    'ROBOT_TOOLKIT_TOOLS_HOME', '~/.local/share/robot-toolkit-tools'))
                runtimes[name] = ToolRuntime(name=name, root=runtime_root, **values)
        return cls(definitions, root=root, runtimes=runtimes)

    def runtime(self, name: str) -> ToolRuntime:
        self.definitions[name]
        return self.runtimes.get(name, ToolRuntime(name, self.root))

    def describe(self, name: str) -> dict:
        definition = self.definitions[name]
        runtime = self.runtime(name)
        return {
            **asdict(definition), 'enabled': runtime.enabled, 'endpoint': runtime.endpoint,
            'source_dir': str(runtime.source_dir),
            'environment_dir': str(runtime.environment_dir),
            'models_dir': str(runtime.models_dir),
            'source_present': runtime.source_dir.is_dir(),
            'environment_present': runtime.environment_dir.is_dir(),
            'models_present': runtime.models_dir.is_dir(),
            'process_owner': 'external', 'ready': None,
            'status': 'disabled' if not runtime.enabled else 'not_checked',
        }

    def sam3(self, name: str = 'sam3') -> 'Sam3Client':
        if self.definitions[name].adapter != 'enpire_sam3_http':
            raise ValueError(f'{name} has no implemented SAM3 adapter')
        return Sam3Client(self.runtime(name))

    def anygrasp(self):
        from .perception import AnyGraspClient

        return AnyGraspClient(self.runtime('anygrasp'))

    def stereo_depth(self, name: str = 'fast_foundationstereo'):
        from .perception import StereoDepthClient

        if self.definitions[name].adapter != 'stereo_depth_http':
            raise ValueError(f'{name} has no stereo depth adapter')
        return StereoDepthClient(self.runtime(name))

    def bundlesdf(self):
        from .perception import BundleSdfClient

        return BundleSdfClient(self.runtime('bundlesdf'))

    def launch_plan(self, name: str) -> dict:
        """Describe, but never execute, the supported loopback SAM3 startup command."""
        definition = self.definitions[name]
        if definition.adapter != 'enpire_sam3_http' or definition.service_path is None:
            raise ValueError(f'{name} has no verified launch protocol')
        runtime = self.runtime(name)
        endpoint = urlsplit(runtime.endpoint or 'http://127.0.0.1:6767')
        if endpoint.scheme != 'http' or endpoint.hostname != '127.0.0.1' or endpoint.path not in ('', '/'):
            raise ValueError('local launch plans require a plain HTTP 127.0.0.1 endpoint')
        cache = _outside_repo(runtime.root / name / 'cache' / 'huggingface')
        return {
            'cwd': str(runtime.source_dir),
            'argv': [str(runtime.environment_dir / 'bin' / 'python'),
                     str(runtime.source_dir / definition.service_path),
                     '--host', '127.0.0.1', '--port', str(endpoint.port or 80)],
            'env_overrides': {'PYTHONPATH': str(runtime.source_dir), 'HF_HOME': str(cache),
                              'HUGGINGFACE_HUB_CACHE': str(cache / 'hub'),
                              'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                              'YAM_UV_ABS_BOOTSTRAPPED': '1'},
            'required_source_revision': definition.revision,
            'validated': False, 'executed': False,
            'prerequisites': ['verify source checkout revision', 'install and lock isolated dependencies',
                              'approve model license and stage a pinned SAM3 snapshot in HF_HOME/hub',
                              'verify CUDA compatibility and free GPU memory'],
        }


class ExternalToolError(RuntimeError):
    """Transport or service-contract failure, never an empty detection result."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        raise ExternalToolError('external tool redirects are disabled')


@dataclass
class Segmentation:
    mask: np.ndarray
    bbox_xywh: tuple[int, int, int, int]
    score: float
    area_px: int


@dataclass
class SegmentationResult:
    """Image-space candidates only; not an object identity, 3D pose or safety result."""

    observation_id: str
    camera: str
    query: str
    detections: tuple[Segmentation, ...]
    provider: str = 'enpire_sam3_http'


class _HttpToolClient:
    """Shared explicit HTTP transport, without process or device ownership."""

    def __init__(self, runtime: ToolRuntime):
        self.runtime = runtime

    def _request(self, path: str, payload=None) -> dict:
        if not self.runtime.enabled or self.runtime.endpoint is None:
            raise ExternalToolError('tool is disabled or has no endpoint; configure it explicitly')
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode('utf-8')
        request = Request(self.runtime.endpoint.rstrip('/') + path, data=body,
                          headers={'Content-Type': 'application/json'},
                          method='GET' if payload is None else 'POST')
        opener = build_opener(ProxyHandler({}), _NoRedirect())
        try:
            with opener.open(request, timeout=self.runtime.timeout_s) as response:
                raw = response.read(64 * 1024 * 1024 + 1)
            if len(raw) > 64 * 1024 * 1024:
                raise ExternalToolError('service response exceeds size limit')
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ExternalToolError('service response must be a JSON object')
            return result
        except HTTPError as exc:
            raise ExternalToolError(f'external tool returned HTTP {exc.code}') from exc
        except (URLError, OSError, ValueError) as exc:
            raise ExternalToolError(f'external tool request failed: {type(exc).__name__}') from exc

class Sam3Client(_HttpToolClient):
    """ENPIRE SAM3 wire adapter; explicit calls only, no warmup/start/stop/retry."""

    def health(self) -> dict:
        """Query service/model status; a running server can still have an unloaded model."""
        response = self._request('/health')
        return {'reachable': True,
                'ready': response.get('status') == 'ok' and response.get('model_loaded') is True,
                'model_loaded': response.get('model_loaded'),
                'model_ref': response.get('model_ref'),
                'process_owner': 'external'}

    def segment(self, rgb: np.ndarray, query: str, *, observation_id: str,
                camera: str, score_threshold: float = 0.2) -> SegmentationResult:
        """Send an existing RGB frame and retain all returned candidates.

        Base64-wrapped NPY is required by the pinned ENPIRE HTTP protocol. It is
        confined to this transport boundary; masks returned to callers are arrays.
        """
        image = np.asarray(rgb)
        if (image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8
                or min(image.shape[:2]) == 0 or image.shape[0] * image.shape[1] > 8_000_000):
            raise ValueError('image must be uint8 HxWx3 RGB with at most eight million pixels')
        if not query.strip() or not observation_id or not camera:
            raise ValueError('query, observation_id and camera are required')
        if not math.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
            raise ValueError('score_threshold must be between zero and one')
        buffer = io.BytesIO()
        np.save(buffer, image, allow_pickle=False)
        response = self._request('/segment_all', {
            'text': query, 'image_b64': base64.b64encode(buffer.getvalue()).decode('ascii'),
            'score_threshold': score_threshold})
        height, width = image.shape[:2]
        try:
            entries = response['detections']
            if (response['height'] != height or response['width'] != width
                    or not isinstance(entries, list) or response['count'] != len(entries)
                    or len(entries) > 64):
                raise ValueError('response dimensions or count disagree')
            detections = []
            for entry in entries:
                encoded = entry['mask_b64']
                if not isinstance(encoded, str) or len(encoded) > (height * width + 4096) * 2:
                    raise ValueError('invalid mask payload size')
                encoded_mask = base64.b64decode(encoded, validate=True)
                stream = io.BytesIO(encoded_mask)
                version = np.lib.format.read_magic(stream)
                if version != (1, 0):
                    raise ValueError('expected NPY v1 mask')
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                if shape != (height, width) or dtype != np.dtype('uint8'):
                    raise ValueError('mask must be uint8 with image dimensions')
                stream.seek(0)
                mask = np.load(stream, allow_pickle=False)
                if not np.all((mask == 0) | (mask == 1)):
                    raise ValueError('mask must be binary')
                rows, columns = np.where(mask)
                if not len(rows):
                    raise ValueError('empty mask in detection')
                bbox = (int(columns.min()), int(rows.min()),
                        int(columns.max() - columns.min() + 1), int(rows.max() - rows.min() + 1))
                score = float(entry['score'])
                if (tuple(entry['bbox_xywh']) != bbox or entry['mask_area'] != len(rows)
                        or not math.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError('mask geometry or score disagrees')
                if score >= score_threshold:
                    detections.append(Segmentation(mask.astype(bool), bbox, score, len(rows)))
        except (KeyError, TypeError, ValueError, EOFError, OSError) as exc:
            raise ExternalToolError(f'invalid SAM3 response: {exc}') from exc
        return SegmentationResult(observation_id, camera, query, tuple(detections))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('list', 'health', 'launch-plan'))
    parser.add_argument('tool', nargs='?')
    parser.add_argument('--config', help='Machine-local JSON file outside this repository')
    parser.add_argument('--root', help='External runtime root; no directories are created')
    args = parser.parse_args(argv)
    try:
        tools = ExternalTools.load(config_path=args.config, root=args.root)
        if args.action == 'list':
            names = [args.tool] if args.tool else sorted(tools.definitions)
            result = [tools.describe(name) for name in names]
        elif args.action == 'health':
            if args.tool is None:
                parser.error('health requires a tool name')
            factories = {'sam3': tools.sam3, 'bundlesdf': tools.bundlesdf, 'anygrasp': tools.anygrasp,
                         'fast_foundationstereo': tools.stereo_depth}
            if args.tool not in factories:
                raise ValueError(f'{args.tool} has no implemented health protocol')
            result = factories[args.tool]().health()
        else:
            if args.tool is None:
                parser.error('launch-plan requires a tool name')
            result = tools.launch_plan(args.tool)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 1 if args.action == 'health' and not result['ready'] else 0
    except (ExternalToolError, KeyError, ValueError, TypeError, OSError) as exc:
        print(json.dumps({'error': str(exc)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())