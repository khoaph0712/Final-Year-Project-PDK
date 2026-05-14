# Model assets

Drop the TFLite files exported from `scripts/export_model.py` here:

```text
assets/model/
  best_float16.tflite    <- preferred for live preview (faster, smaller)
  best_float32.tflite    <- fallback for maximum accuracy
  best_metadata.json     <- classes + input/output shapes (optional at runtime)
```

The app `require()`s these files at build time via `metro.config.js`, where `.tflite` is registered as an asset extension.

## Why both?

- **Float16** is smaller and faster, and is the default for live preview.
- **Float32** is larger, but useful when checking maximum exported-model accuracy.

Toggle between them in **Settings -> Detection -> Use Float16 model**.
