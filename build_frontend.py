"""
build_frontend.py
Creates the static /frontend/ folder for Vercel deployment.
Run from project root: python build_frontend.py
"""
import shutil
import os
import re
from pathlib import Path

ROOT  = Path(__file__).resolve().parent
APP   = ROOT / "app"
FRONT = ROOT / "frontend"

# ── scaffold dirs ────────────────────────────────────────────────
for d in [FRONT, FRONT / "css", FRONT / "js"]:
    d.mkdir(parents=True, exist_ok=True)

# ── copy static assets ───────────────────────────────────────────
shutil.copytree(APP / "static" / "css", FRONT / "css", dirs_exist_ok=True)
shutil.copytree(APP / "static" / "js",  FRONT / "js",  dirs_exist_ok=True)
img_src = APP / "static" / "img"
if img_src.exists():
    shutil.copytree(img_src, FRONT / "img", dirs_exist_ok=True)
    print("✓ Copied img/")
print("✓ Copied css/ and js/")

# ── patch index.html ─────────────────────────────────────────────
html = (APP / "templates" / "index.html").read_text(encoding="utf-8")

# 1. De-Jinja2: replace url_for() calls with relative paths
html = html.replace(
    "{{ url_for('static', filename='css/style.css') }}",
    "./css/style.css"
)
html = html.replace(
    "{{ url_for('static', filename='js/speaker_profile.js') }}",
    "./js/speaker_profile.js"
)
html = html.replace(
    "{{ url_for('static', filename='js/main.js') }}",
    "./js/main.js"
)

# 2. Insert BASE_URL config block immediately before the main.js <script> tag
BASE_BLOCK = (
    '<script>\n'
    '    // Set this to your Render backend URL after deployment\n'
    "    // Leave empty string '' for local development (Flask serves both)\n"
    '    window.VIETASR_BASE_URL = "";\n'
    '  </script>\n'
)
html = html.replace(
    '<script src="./js/main.js"></script>',
    BASE_BLOCK + '<script src="./js/main.js"></script>'
)

# 3. Add <audio> preview element right after the hidden file input
html = html.replace(
    '<input type="file" id="fileInput" hidden accept=".wav,.mp3,.flac,.m4a,.ogg,.mp4,.aac,.wma,.opus">',
    (
        '<input type="file" id="fileInput" hidden accept=".wav,.mp3,.flac,.m4a,.ogg,.mp4,.aac,.wma,.opus">\n'
        '                    <audio id="audioPreview" controls '
        'style="display:none; margin-top:12px; width:100%;"></audio>'
    )
)

# Sanity check
remaining = re.findall(r"url_for", html)
if remaining:
    print(f"⚠ WARNING: {len(remaining)} url_for() call(s) still present!")
else:
    print("✓ index.html: no Jinja2 url_for remaining")

(FRONT / "index.html").write_text(html, encoding="utf-8")
print("✓ Written frontend/index.html")

# ── patch frontend/js/main.js ────────────────────────────────────
js_path = FRONT / "js" / "main.js"
js = js_path.read_text(encoding="utf-8")

# Add BASE_URL at the very top (before 'use strict')
if "const BASE_URL" not in js:
    js = "const BASE_URL = window.VIETASR_BASE_URL || '';\n" + js
    print("✓ Added BASE_URL const to top of main.js")
else:
    print("  BASE_URL already present in main.js")

# Replace all relative /api/ fetch() calls
replacements = [
    ("fetch('/api/status')",      "fetch(`${BASE_URL}/api/status`)"),
    ('fetch("/api/status")',      "fetch(`${BASE_URL}/api/status`)"),
    ("fetch('/api/summarize',",   "fetch(`${BASE_URL}/api/summarize`,"),
    ('fetch("/api/summarize",',   "fetch(`${BASE_URL}/api/summarize`,"),
    ("fetch('/api/transcribe',",  "fetch(`${BASE_URL}/api/transcribe`,"),
    ('fetch("/api/transcribe",',  "fetch(`${BASE_URL}/api/transcribe`,"),
    ("fetch('/api/post-process'", "fetch(`${BASE_URL}/api/post-process`"),
    ('fetch("/api/post-process"', "fetch(`${BASE_URL}/api/post-process`"),
    ("fetch('/api/normalize'",    "fetch(`${BASE_URL}/api/normalize`"),
    ('fetch("/api/normalize"',    "fetch(`${BASE_URL}/api/normalize`"),
    ("xhr.open('POST','/api/upload')",  "xhr.open('POST',`${BASE_URL}/api/upload`)"),
    ('xhr.open("POST","/api/upload")',  "xhr.open('POST',`${BASE_URL}/api/upload`)"),
    ("xhr.open('POST', '/api/upload')", "xhr.open('POST', `${BASE_URL}/api/upload`)"),
    ('xhr.open("POST", "/api/upload")', "xhr.open('POST', `${BASE_URL}/api/upload`)"),
]
count = 0
for old, new in replacements:
    if old in js:
        js = js.replace(old, new)
        count += 1
        print(f"  Replaced: {old[:55]}")

# Catch-all: any remaining fetch('/api/ or fetch("/api/ patterns
remaining_api = re.findall(r"fetch\(['\"]\/api\/", js)
if remaining_api:
    print(f"⚠ {len(remaining_api)} unhandled fetch('/api/...) patterns remaining — check manually")

# Add audio preview logic before xhr.send(formData)
PREVIEW_CODE = (
    "\n        // Show audio preview\n"
    "        const preview = document.getElementById('audioPreview');\n"
    "        if (preview && file) {\n"
    "            preview.src = URL.createObjectURL(file);\n"
    "            preview.style.display = 'block';\n"
    "        }\n"
)
# Insert before xhr.send(formData)
if "xhr.send(formData)" in js and "audioPreview" not in js:
    js = js.replace("        xhr.send(formData);", PREVIEW_CODE + "        xhr.send(formData);")
    print("✓ Added audio preview logic before xhr.send")
elif "audioPreview" in js:
    print("  Audio preview logic already present in main.js")

js_path.write_text(js, encoding="utf-8")
print("✓ Written frontend/js/main.js")

# ── create vercel.json ───────────────────────────────────────────
vercel_json = '{\n  "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }]\n}\n'
(FRONT / "vercel.json").write_text(vercel_json, encoding="utf-8")
print("✓ Written frontend/vercel.json")

# ── final structure report ───────────────────────────────────────
print("\n── Frontend folder structure ──")
for p in sorted(FRONT.rglob("*")):
    rel = p.relative_to(ROOT)
    indicator = "/" if p.is_dir() else ""
    print(f"  {rel}{indicator}")

print("\n✅ All done! frontend/ is ready for Vercel deployment.")
