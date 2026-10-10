#!/usr/bin/env python3
"""Static release gate for Phoenix Mobile M3.
Run from the project root: python3 tools/audit/m3_audit.py
This does not claim to replace on-device visual/latency testing.
"""
from pathlib import Path
import json, re, sys

ROOT=Path(__file__).resolve().parents[2]
checks=[]
def check(name, ok, detail):
    checks.append((name, bool(ok), detail))

manifest=ROOT/'PHOENIX_BUILD_MANIFEST.json'
check('Manifest present', manifest.exists(), str(manifest))
if manifest.exists():
    data=json.loads(manifest.read_text())
    check('Phoenix manifest version', data.get('version') in {'0.3.0-m3-rc1','1.0.0','1.1.0'}, data.get('version'))
    check('Dual modes', set(data.get('modes',[]))=={'local','huggingface'}, str(data.get('modes')))

app=ROOT/'app/src/main/java/com/swamitech/phoenix/PhoenixViewModel.kt'
ui=ROOT/'app/src/main/java/com/swamitech/phoenix/MainActivity.kt'
cpp=ROOT/'app/src/main/cpp/native-lib.cpp'
for p in (app,ui,cpp): check(f'File {p.name}', p.exists(), str(p))

if app.exists():
    s=app.read_text()
    check('No fake local success', 'next milestone' not in s.lower() and 'no video was modified' in s.lower(), 'local status strings')
    check('Remote ETA not synthetic', 'elapsed / 300f' not in s, 'remote progress formula')
    check('Explicit local engine error path', 'Local face engine is not available' in s, 'local guard')

if cpp.exists():
    s=cpp.read_text()
    check('Native engine contract is explicit', 'PHOENIX_ENGINE_NOT_READY' in s, 'native return code')
    check('No silent fake processing', 'return 1001' not in s, 'native processing stub removed')

# Source-level sanity: no accidental secrets.
secret_pat=re.compile(r'(hf_[A-Za-z0-9]{20,}|api[_-]?key\s*=\s*["\'][^"\']+["\'])',re.I)
for p in ROOT.rglob('*'):
    if p.is_file() and '.git' not in p.parts and p.suffix in {'.kt','.kts','.cpp','.md','.json'}:
        try: s=p.read_text(errors='ignore')
        except: continue
        check(f'No obvious secret in {p.relative_to(ROOT)}', not secret_pat.search(s), 'scan')

failed=[x for x in checks if not x[1]]
for n,ok,d in checks: print(('PASS' if ok else 'FAIL'), n, '-', d)
print(f'\n{len(checks)-len(failed)}/{len(checks)} static checks passed')
if failed:
    sys.exit(1)
