"""Keeps only the current build in a Hugging Face Space: squashes the repo's history to one commit and deletes every stored large (LFS)
file that commit does not use. Hugging Face counts replaced files against the Space's storage for ever otherwise (1 GB on the free tier:
five 200 MB builds filled it on 2026-11-24 and uploads were refused with "Storage limit reached").

    python keep_only_current.py <owner/space>

Uses the token `hf auth login` stored (~/.cache/huggingface/token) or HF_TOKEN. Irreversible by design: the Space's earlier commits go.
"""
import json, os, sys, urllib.request

space = sys.argv[1]
token = os.environ.get('HF_TOKEN')
if not token:
    with open(os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'token')) as f:
        token = f.read().strip()
api = 'https://huggingface.co/api/spaces/' + space


def call(url, body=None):
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'},
                                     method='GET' if body is None else 'POST')
    with urllib.request.urlopen(request) as response:
        text = response.read().decode()
        try:
            return json.loads(text), response.headers
        except ValueError:
            return text, response.headers


call(api + '/super-squash/main', {'message': 'SpawnGen Studio (only the current build is kept)'})
info, _ = call(api + '?blobs=true')
current = {s['lfs']['sha256'] for s in info['siblings'] if s.get('lfs')}
stored, url = [], api + '/lfs-files?limit=1000'
while url:
    page, headers = call(url)
    stored += page
    url = None
    for part in (headers.get('Link') or '').split(','):
        if 'rel="next"' in part:
            url = part[part.index('<') + 1:part.index('>')]
unused = [f['fileOid'] for f in stored if f['fileOid'] not in current]
for at in range(0, len(unused), 1000):
    call(api + '/lfs-files/batch', {'deletions': {'sha': unused[at:at + 1000], 'rewriteHistory': False}})
freed = sum(int(f['size']) for f in stored if f['fileOid'] not in current)
print(f'{space}: history squashed; {len(unused)} unused stored files deleted ({freed / 1e6:.0f} MB); '
      f'{len(stored) - len(unused)} in use ({sum(int(f["size"]) for f in stored if f["fileOid"] in current) / 1e6:.0f} MB)')
