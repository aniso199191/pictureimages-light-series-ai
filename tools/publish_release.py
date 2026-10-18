#!/usr/bin/env python3
"""Publish a checkpoint as a GitOpen release asset (chunked upload).

    python tools/publish_release.py --file models/sft.pt --name "Larme Debug v0.1" --body "..."
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = os.environ.get('GITOPEN_API', 'http://127.0.0.1:13030')
OWNER = os.environ.get('GITOPEN_OWNER', 'pictureimages')
REPO = os.environ.get('GITOPEN_REPO', 'pictureimages-light-series-ai')
TOKEN_FILE = Path('/home/agentuser/gitopen/.admin/pictureimages.token')
CHUNK = 2 * 1024 * 1024 - 4096


def api(method, path, payload=None, raw=None):
    headers = {'Authorization': 'Bearer ' + TOKEN_FILE.read_text().strip()}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
        headers['Content-Type'] = 'application/json'
    elif raw is not None:
        data = raw
        headers['Content-Type'] = 'application/octet-stream'
    req = urllib.request.Request(API + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        print('HTTP', exc.code, exc.read().decode(errors='replace')[:300])
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--body', default='')
    parser.add_argument('--tag', default='')
    args = parser.parse_args()

    path = Path(args.file)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    repo = api('GET', f'/api/v1/repos/{OWNER}/{REPO}')
    tag = args.tag or repo['head']

    releases = api('GET', f'/api/v1/repos/{OWNER}/{REPO}/releases')['releases']
    existing = next((r for r in releases if r['tag'] == tag and r['name'] == args.name), None)
    if existing:
        release_id = existing['id']
        print('reusing release', release_id)
    else:
        created = api('POST', f'/api/v1/repos/{OWNER}/{REPO}/releases',
                      {'tag': tag, 'name': args.name, 'body': args.body})
        release_id = created['id']
        print('created release', release_id)

    begin = api('POST', f'/api/v1/repos/{OWNER}/{REPO}/releases/{release_id}/assets/begin',
                {'filename': path.name, 'total_size': len(data), 'sha256': digest})
    upload_id = begin['upload_id']
    offset = 0
    while offset < len(data):
        chunk = data[offset:offset + CHUNK]
        api('PUT', f'/api/v1/repos/{OWNER}/{REPO}/releases/{release_id}/assets/{upload_id}/chunk?offset={offset}', raw=chunk)
        offset += len(chunk)
        print(f'{offset}/{len(data)}', flush=True)
    done = api('POST', f'/api/v1/repos/{OWNER}/{REPO}/releases/{release_id}/assets/{upload_id}/finish')
    public = os.environ.get('GITOPEN_PUBLIC', 'https://gitopen.we-strikes.top')
    url = f'{public}/{OWNER}/{REPO}/releases/{release_id}/download/{path.name}'
    print(json.dumps({'release_id': release_id, 'asset': done, 'url': url,
                      'sha256_ok': done['sha256'] == digest}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
