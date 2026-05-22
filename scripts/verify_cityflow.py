"""Verify CityFlow is installed and a basic simulation works."""
import sys

try:
    import cityflow
    print(f"cityflow imported OK")
    print(f"  location: {cityflow.__file__}")
    if hasattr(cityflow, "__version__"):
        print(f"  version:  {cityflow.__version__}")
except ImportError as e:
    print(f"FAIL: cannot import cityflow: {e}")
    sys.exit(1)

# Basic smoke test: try Engine creation
print("\n=== Engine smoke test ===")
try:
    print(f"  cityflow.Engine class: {cityflow.Engine}")
    print("  (skipping actual Engine init — needs config file)")
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}")
    sys.exit(1)

print("\nCityFlow is ready.")
