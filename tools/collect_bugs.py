#!/usr/bin/env python3
"""Collect bug reports from GitOpen issues into docs/BUGS.md and bugs/issues.jsonl.

An issue counts as a bug when its title or body matches a bug keyword. The
collector is read-only against GitOpen: it never edits or closes issues, so a
human (or the maintainer agent) decides when something is actually fixed.

Usage:
    python tools/collect_bugs.py            # collect and write files
    python tools/collect_bugs.py --check    # exit 1 when new bugs appeared
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
API = os.environ.get('GITOPEN_API', 'http://127.0.0.1:13030')
OWNER = os.environ.get('GITOPEN_OWNER', 'pictureimages')
REPO = os.environ.get('GITOPEN_REPO', 'pictureimages-light-series-ai')
TOKEN_FILE = Path(os.environ.get('GITOPEN_TOKEN_FILE', '/home/agentuser/gitopen/.admin/pictureimages.token'))

BUG_PATTERNS = [
    r'\bbug\b', r'crash', r'error', r'exception', r'traceback', r'fail', r'broken',
    r'崩溃', r'报错', r'错误', r'异常', r'失败', r'闪退', r'不能用', r'无法运行', r'跑不起来',
    r'复现', r'不对', r'有问题',
]


def fetch_issues():
    headers = {'User-Agent': 'bug-collector/1.0'}
    if TOKEN_FILE.is_file():
        headers['Authorization'] = 'Bearer ' + TOKEN_FILE.read_text().strip()
    url = f'{API}/api/v1/repos/{OWNER}/{REPO}/issues'
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)['issues']


def is_bug(issue):
    text = (issue.get('title', '') + '\n' + issue.get('body', '')).lower()
    return any(re.search(pattern, text) for pattern in BUG_PATTERNS)


def render(issues):
    lines = ['# Collected bug reports', '',
             f'Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")} from GitOpen issues.', '',
             '| # | Status | Title | Author | Created |', '|---:|---|---|---|---|']
    for issue in issues:
        title = issue['title'].replace('|', '\\|')
        lines.append(f"| {issue['id']} | {issue['status']} | {title} | {issue['author']} | {issue['created']} |")
    lines += ['', 'Bodies are stored in `bugs/issues.jsonl` for triage and regression tests.', '']
    return '\n'.join(lines)


def publish() -> None:
    """Commit the generated files to GitOpen (create or update automatically)."""
    token = TOKEN_FILE.read_text().strip()
    headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}

    def call(method, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(API + path, data=data, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)

    info = call('GET', f'/api/v1/repos/{OWNER}/{REPO}')
    head = info['head']
    tree = call('GET', f'/api/v1/repos/{OWNER}/{REPO}/files')
    paths = {entry['path']: entry for entry in tree.get('files', tree.get('entries', []))}
    changes = []
    for rel in ('docs/BUGS.md', 'bugs/issues.jsonl'):
        content = (ROOT / rel).read_text(encoding='utf-8')
        changes.append({'op': 'update' if rel in paths else 'create', 'path': rel, 'content': content})
    call('POST', f'/api/v1/repos/{OWNER}/{REPO}/files/batch',
         {'files': changes, 'message': 'Update collected bug reports', 'expected_commit': head})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='exit 1 if any open bug exists')
    parser.add_argument('--publish', action='store_true', help='commit generated files to GitOpen')
    args = parser.parse_args()

    issues = [issue for issue in fetch_issues() if is_bug(issue)]
    issues.sort(key=lambda item: item['id'])

    out_dir = ROOT / 'bugs'
    out_dir.mkdir(exist_ok=True)
    (out_dir / 'issues.jsonl').write_text(
        '\n'.join(json.dumps(issue, ensure_ascii=False) for issue in issues) + ('\n' if issues else ''),
        encoding='utf-8')
    (ROOT / 'docs' / 'BUGS.md').write_text(render(issues), encoding='utf-8')

    open_bugs = [issue for issue in issues if issue.get('status') == 'open']
    print(json.dumps({'total': len(issues), 'open': len(open_bugs),
                      'titles': [issue['title'] for issue in open_bugs]}, ensure_ascii=False))
    if args.publish:
        publish()
    if args.check and open_bugs:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
