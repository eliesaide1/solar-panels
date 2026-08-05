"""Verify Google Earth Pro is installed and reachable over COM.

Run this straight after installing and launching Earth Pro, before doing
anything else. The COM automation API is undocumented these days, so this
confirms your particular build actually exposes it.

    python scripts/check_earth.py
"""

import sys
import winreg
from pathlib import Path

import _bootstrap  # noqa: F401

INSTALL_PATHS = [
    Path(r"C:\Program Files\Google\Google Earth Pro\client\googleearth.exe"),
    Path(r"C:\Program Files (x86)\Google\Google Earth Pro\client\googleearth.exe"),
]
PROGID = "GoogleEarth.ApplicationGE"

# Methods the capture pipeline depends on. Losing any one of these breaks it.
REQUIRED = [
    ("IsInitialized", "readiness check"),
    ("SetCameraParams", "move the camera"),
    ("GetStreamingProgressPercentage", "wait for imagery to load"),
    ("GetViewExtents", "georeference the capture"),
    ("SaveScreenShot", "write the image"),
]

ok = True


def report(passed: bool, label: str, detail: str = "") -> bool:
    global ok
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}{'  -- ' + detail if detail else ''}")
    ok = ok and passed
    return passed


print("1. Installation")
exe = next((p for p in INSTALL_PATHS if p.exists()), None)
if not report(exe is not None, "Google Earth Pro installed", str(exe) if exe else
              "not found; get it from https://www.google.com/earth/about/versions/"):
    sys.exit(1)

print("\n2. COM registration")
found = False
for hive, view in ((winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_64KEY),
                   (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_32KEY),
                   (winreg.HKEY_CLASSES_ROOT, 0)):
    try:
        winreg.OpenKey(hive, rf"SOFTWARE\Classes\{PROGID}" if hive != winreg.HKEY_CLASSES_ROOT
                       else PROGID, 0, winreg.KEY_READ | view).Close()
        found = True
        break
    except OSError:
        continue
report(found, f"ProgID {PROGID}",
       "" if found else "launch Google Earth Pro once, then re-run this")

print("\n3. pywin32")
try:
    import win32com.client
    report(True, "pywin32 importable")
except ImportError:
    report(False, "pywin32 importable", "pip install pywin32")
    sys.exit(1)

print("\n4. Live COM connection  (this may launch Google Earth)")
try:
    app = win32com.client.Dispatch(PROGID)
    report(True, "Dispatch succeeded")
except Exception as exc:
    report(False, "Dispatch succeeded", f"{type(exc).__name__}: {exc}")
    sys.exit(1)

print("\n5. Required methods")
for name, why in REQUIRED:
    try:
        getattr(app, name)
        report(True, name, why)
    except Exception:
        report(False, name, f"missing -- needed to {why}")

print("\n6. Live call")
try:
    import time
    for _ in range(60):
        if app.IsInitialized():
            break
        time.sleep(1)
    app.SetCameraParams(37.7749, -122.4194, 0.0, 1, 800.0, 0.0, 0.0, 5.0)
    time.sleep(3)
    e = app.GetViewExtents()
    span = float(e.North) - float(e.South)
    report(span > 0, "GetViewExtents returns a real view",
           f"N={float(e.North):.6f} S={float(e.South):.6f} span={span:.6f} deg")
except Exception as exc:
    report(False, "live camera + extents call", f"{type(exc).__name__}: {exc}")

print("\n" + ("ALL GOOD -- run scripts/calibrate.py next."
              if ok else
              "SOMETHING FAILED -- see README 'Imagery licensing' for the "
              "tile-source fallback, which skips Google Earth entirely."))
sys.exit(0 if ok else 1)
