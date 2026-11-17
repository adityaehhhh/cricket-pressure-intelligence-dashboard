"""
CAPTURE DASHBOARD SCREENSHOTS
=============================
Starts the local Dash server in-process, renders views via Headless Chrome,
and captures genuine high-resolution screenshots for presentation.
"""

import os
import sys
import time
import threading
import subprocess
import urllib.request
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
SCREENSHOTS_DIR = os.path.join(PROJECT_ROOT, 'Screenshots')
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 8058
URL = f"http://127.0.0.1:{PORT}"

print("=" * 70)
print("CAPTURING DASHBOARD SCREENSHOTS VIA HEADLESS CHROME")
print("=" * 70)

# 1. Load app module and start server in background thread
app_script = os.path.join(BASE_DIR, 'app.py')
spec = importlib.util.spec_from_file_location('dashboard_app', app_script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

server_thread = threading.Thread(
    target=lambda: mod.app.run(host='127.0.0.1', port=PORT, debug=False),
    daemon=True
)
server_thread.start()
print(f"[*] Started Dash application server on {URL} in background thread.")

# Wait for server to become responsive
started = False
for attempt in range(15):
    try:
        req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                started = True
                print(f"[*] Server is responsive on {URL} after {attempt+1} attempts.")
                break
    except Exception:
        time.sleep(1)

if not started:
    print("[!] Server failed to start in time. Aborting.")
    sys.exit(1)

# Extra grace period for client-side JS & Plotly charts to mount
time.sleep(4)

# 2. Capture Screenshots using Chrome Headless
targets = [
    {
        'name': '16_dashboard_player_radar.png',
        'url': f"{URL}/?tab=tab-profile",
        'desc': 'Player Scouting Profile & Football-style 5-axis Radar'
    },
    {
        'name': '17_dashboard_player_comparison.png',
        'url': f"{URL}/?tab=tab-comparison",
        'desc': 'Multi-Player Head-to-Head Comparison with Overlaid Radars'
    },
    {
        'name': '18_dashboard_pressure_vs_normal.png',
        'url': f"{URL}/?tab=tab-context",
        'desc': 'Pressure vs Normal Contextual Quadrant Scatter Analysis'
    },
    {
        'name': '19_dashboard_ml_intelligence.png',
        'url': f"{URL}/?tab=tab-ml",
        'desc': 'GPU-Accelerated XGBoost ML Architecture & Feature Importance'
    },
    {
        'name': '20_dashboard_fixed19_view.png',
        'url': f"{URL}/?tab=tab-leaderboards",
        'desc': 'Locked 19-Player Challenge Shortlist & Scouting Justifications'
    }
]

for idx, t in enumerate(targets, 1):
    out_file = os.path.join(SCREENSHOTS_DIR, t['name'])
    print(f"[{idx}/{len(targets)}] Capturing {t['name']} ({t['desc']})...")
    
    cmd = [
        CHROME_EXE,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--window-size=1920,1200",
        "--virtual-time-budget=8000",
        f"--screenshot={out_file}",
        t['url']
    ]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if os.path.exists(out_file) and os.path.getsize(out_file) > 1000:
            print(f"      SUCCESS: {t['name']} ({os.path.getsize(out_file):,} bytes)")
        else:
            print(f"      WARNING: File not generated or small ({res.stderr})")
    except Exception as e:
        print(f"      ERROR capturing {t['name']}: {e}")

print("=" * 70)
print("SUCCESS: DASHBOARD CAPTURE RUN COMPLETE!")
print("=" * 70)
