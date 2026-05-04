# Model assets

Drop the TFLite files exported from `scripts/export_model.py` here:

```
assets/model/
  best_int8.tflite       ← preferred (fast, small)
  best_float16.tflite    ← fallback (more accurate)
  best_metadata.json     ← classes + input/output shapes (optional at runtime)
```

The app will `require()` these files at build time via `metro.config.js`
(`.tflite` is registered as an asset extension).

## Why both?

- **INT8** is ~4× smaller and much faster on mobile NPUs/GPUs, but can lose a few
  accuracy points. Good default for real-time preview.
- **Float16** is a balanced middle ground if you want extra accuracy at similar
  speed on modern devices.

Toggle between them in **Settings → Detection → Use INT8 quantised model**.
