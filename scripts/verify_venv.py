"""Verify the venv is correctly using packages from D drive (not Windows C:)."""
import sys

print(f"Python executable: {sys.executable}")
print(f"Python version:    {sys.version.split()[0]}")
print(f"Platform:          {sys.platform}")
print()

print("=== Critical packages ===")
for pkg_name in ["numpy", "pandas", "pytest", "hypothesis", "fire", "dotenv", "responses"]:
    try:
        mod = __import__(pkg_name)
        version = getattr(mod, "__version__", "?")
        location = getattr(mod, "__file__", "?")
        # Verify location is on D: not C:
        is_d_drive = "/mnt/d/" in location.lower() or location.startswith("/mnt/d")
        marker = "OK D:" if is_d_drive else "WARN"
        print(f"  [{marker}] {pkg_name:15s} {version:10s} -> {location}")
    except ImportError as e:
        print(f"  [MISS] {pkg_name:15s} not installed: {e}")

print()
print("=== Path validation ===")
expected_prefix = "/mnt/d/Duy/Docs/School/CS106"
if sys.executable.startswith(expected_prefix):
    print(f"  [OK] venv is on D drive at: {sys.executable}")
else:
    print(f"  [FAIL] venv is NOT on D drive! Found: {sys.executable}")
    sys.exit(1)
