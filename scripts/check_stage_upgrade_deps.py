import importlib.util

for name in ("datasets", "timm", "transformers"):
    print(f"{name}: {bool(importlib.util.find_spec(name))}")
