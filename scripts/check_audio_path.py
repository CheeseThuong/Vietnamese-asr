import json
from pathlib import Path

path = Path('processed_data_merged/train.jsonl')
if not path.exists():
    print('Missing:', path)
    raise SystemExit(1)

bad = 0
abs_paths = 0
missing_data_prefix = 0
sample = None
lines = 0

with path.open('r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        lines += 1
        obj = json.loads(line)
        ap = obj.get('audio_path', '')
        if sample is None:
            sample = ap
        if not ap:
            bad += 1
            continue
        if ap.startswith('/') or ':' in ap[:3]:
            abs_paths += 1
        if not ap.replace('\\', '/').startswith('Data/'):
            missing_data_prefix += 1

print('Sample audio_path:', sample)
print('Total lines checked:', lines)
print('Empty audio_path:', bad)
print('Absolute paths:', abs_paths)
print('Not starting with Data/:', missing_data_prefix)
