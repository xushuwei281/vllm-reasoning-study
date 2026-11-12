"""Explicitly package a stable delivery tree; preserve every raw evidence file.

No work occurs on import. Invocation regenerates the top-level SHA-256 manifest
and an archive outside the project, then verifies all CRCs and manifest hashes.
Only generated caches, Python egg-info, lock files and OS cache metadata are
excluded. Base-model downloads or credentials must be audited before invocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile


CACHE_DIRECTORIES = {'__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.cache'}
CACHE_FILES = {'.DS_Store', 'Thumbs.db'}
MANIFEST = 'MANIFEST.sha256'
CHUNK_BYTES = 1024 * 1024


def inventory(root: Path):
    included, excluded = [], []
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        parent = Path(directory)
        for name in list(directories):
            path = parent / name
            if path.is_symlink():
                raise ValueError('Symlink directories require explicit review: ' + str(path.relative_to(root)))
            if name in CACHE_DIRECTORIES or name.endswith('.egg-info'):
                directories.remove(name)
                excluded.append({'path': path.relative_to(root).as_posix() + '/', 'reason': 'generated cache or egg-info'})
        for name in files:
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise ValueError('Symlink files require explicit review: ' + relative)
            if path == root / MANIFEST:
                continue  # Regenerated, included in ZIP, excluded from its own hash list.
            if name in CACHE_FILES or path.suffix in {'.pyc', '.pyo', '.lock'}:
                excluded.append({'path': relative, 'reason': 'generated cache metadata or lock'})
                continue
            if not path.is_file() or '\n' in relative or '\r' in relative or '\\' in relative:
                raise ValueError('Unsupported delivery path: ' + relative)
            included.append(path)
    return sorted(included), sorted(excluded, key=lambda item: item['path'])


def checksum(path: Path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''):
            value.update(chunk)
    return value.hexdigest()


def file_state(path: Path):
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def package(root: Path, output: Path, *, replace: bool = False):
    root, output = root.resolve(), output.resolve()
    if root == output or root in output.parents:
        raise ValueError('Archive must be outside the project tree')
    if output.exists() and not replace:
        raise FileExistsError('Archive exists; explicit --replace is required')
    if not root.is_dir() or not (root / 'README.md').is_file():
        raise ValueError('Expected a complete project root with README.md')
    output.parent.mkdir(parents=True, exist_ok=True)
    files, excluded = inventory(root)
    states = {path: file_state(path) for path in files}
    hashes = {path.relative_to(root).as_posix(): checksum(path) for path in files}
    if any(file_state(path) != states[path] for path in files):
        raise RuntimeError('Delivery files changed during hashing; retry after they are stable')
    manifest = ''.join(f'{hashes[name]}  {name}\n' for name in sorted(hashes)).encode()
    prefix = root.name + '/'
    print(json.dumps({'stage': 'hashed', 'payload_files': len(files),
                      'payload_bytes': sum(states[path][0] for path in files),
                      'excluded_entries': len(excluded)}), flush=True)
    with tempfile.TemporaryDirectory(prefix='.delivery-package-', dir=output.parent) as temporary:
        staging = Path(temporary)
        archive = staging / output.name
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                             allowZip64=True, strict_timestamps=False) as bundle:
            for path in files:
                bundle.write(path, prefix + path.relative_to(root).as_posix())
            bundle.writestr(prefix + MANIFEST, manifest)
        print(json.dumps({'stage': 'compressed', 'archive_bytes': archive.stat().st_size}), flush=True)
        with zipfile.ZipFile(archive) as bundle:
            expected = {prefix + name for name in hashes} | {prefix + MANIFEST}
            names = bundle.namelist()
            if len(names) != len(set(names)) or set(names) != expected:
                raise RuntimeError('ZIP contents differ from the intended manifest inventory')
            bad = bundle.testzip()
            if bad is not None:
                raise RuntimeError('ZIP CRC verification failed: ' + bad)
            if bundle.read(prefix + MANIFEST) != manifest:
                raise RuntimeError('Archived manifest bytes differ')
            for name, expected_hash in hashes.items():
                value = hashlib.sha256()
                with bundle.open(prefix + name) as stream:
                    for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''):
                        value.update(chunk)
                if value.hexdigest() != expected_hash:
                    raise RuntimeError('ZIP member SHA-256 mismatch: ' + name)
        current, _ = inventory(root)
        if current != files or any(file_state(path) != states[path] for path in files):
            raise RuntimeError('Delivery tree changed while packaging; no output published')
        result = {'archive': str(output), 'archive_bytes': archive.stat().st_size,
                  'archive_sha256': checksum(archive), 'archive_entries': len(files) + 1,
                  'manifest': str(root / MANIFEST), 'manifest_entries': len(files),
                  'manifest_sha256': hashlib.sha256(manifest).hexdigest(),
                  'crc_verified': True, 'all_member_sha256_verified': True,
                  'excluded': excluded}
        manifest_temp = staging / MANIFEST
        manifest_temp.write_bytes(manifest)
        manifest_temp.replace(root / MANIFEST)
        archive.replace(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    print(json.dumps(package(root, args.output or root.with_suffix('.zip'), replace=args.replace),
                     indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
