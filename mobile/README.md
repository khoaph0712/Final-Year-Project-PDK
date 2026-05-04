# WasteWise mobile app

Expo Dev Client + TypeScript + `expo-router` + `react-native-vision-camera` + `react-native-fast-tflite`. On-device, offline-first, light/dark themed.

## Stack

| Concern | Choice |
|---|---|
| Framework | Expo SDK 51 (Dev Client) |
| Language | TypeScript (strict) |
| Routing | expo-router (file-system) |
| Camera | react-native-vision-camera v4 + vision-camera-resize-plugin |
| Inference | react-native-fast-tflite |
| Animations | react-native-reanimated + moti |
| State | zustand + AsyncStorage |
| Theming | custom (`src/theme`) with light/dark + system |

## Structure

```
mobile/
├── app/                     expo-router screens
│   ├── _layout.tsx          root providers + stack
│   ├── index.tsx            redirect (onboarded? → tabs, else → onboarding)
│   ├── onboarding.tsx       3-slide intro
│   ├── result.tsx           per-scan result with tip + points
│   ├── bin-locator.tsx      nearest bins (mock; swap for real API)
│   └── (tabs)/
│       ├── _layout.tsx      bottom tabs
│       ├── home.tsx         eco-points, streak, recent scans
│       ├── scan.tsx         live camera + real-time detection
│       ├── history.tsx      filterable scan history
│       ├── leaderboard.tsx  mock friends leaderboard
│       └── settings.tsx     theme, thresholds, clear data
├── src/
│   ├── components/          Screen, Text, Card, Button, ConfidenceBar
│   ├── lib/
│   │   ├── classes.ts       7 classes + bin mapping + tips + points
│   │   ├── yolo.ts          decode output + NMS
│   │   └── useDetector.ts   TFLite loader hook
│   ├── store/useAppStore.ts zustand + AsyncStorage
│   └── theme/               tokens + ThemeProvider (light/dark)
└── assets/
    ├── images/              icon, splash, adaptive icon (add your own)
    └── model/               best_int8.tflite, best_float16.tflite
```

## Setup

Prereqs: **Node 20+**, **Android Studio** (for Android) or **Xcode** (iOS), and
a working JDK 17.

```bash
cd mobile
npm install
```

Place the exported TFLite model files in `assets/model/` (see the step in the
root `README.md`).

### Add basic app images

Create 1024×1024 PNGs at:

- `mobile/assets/images/icon.png`
- `mobile/assets/images/adaptive-icon.png`
- `mobile/assets/images/splash.png`

(Any placeholders are fine for development.)

### Build the Dev Client

Vision Camera + fast-tflite need native code, so you must prebuild once:

```bash
npx expo prebuild --clean
```

### Run on device / emulator

```bash
npm run android     # Android (physical or emulator)
npm run ios         # macOS only
```

For hot-reload during development:

```bash
npm run start       # opens Expo dev tools; scan QR in Expo Dev Client
```

## How detection works

1. `useCameraDevice("back")` provides the camera stream.
2. `useFrameProcessor` runs on the GPU/worklet thread.
3. `vision-camera-resize-plugin` resizes the frame to `640×640 RGB float32`.
4. `TensorflowModel.runSync([input])` returns the YOLOv8 tensor.
5. `decodeYoloOutput` + `nms` (in `src/lib/yolo.ts`) extract boxes.
6. We summarise to the top class and update the UI (throttled to ~5 fps).

## Switching model variants

Open **Settings → Detection → Use INT8 quantised model** to toggle between
`best_int8.tflite` (fast) and `best_float16.tflite` (accurate).

## Troubleshooting

- **"Failed to load TFLite model"** — the two `.tflite` files are missing from
  `assets/model/`. Re-run `python scripts/export_model.py` and copy them over.
- **Black camera on Android emulator** — use a physical device (emulator camera
  input is limited).
- **`new-arch` issues** — set `"newArchEnabled": false` in `app.json` if your
  device or dependencies are behind.
- **Out-of-memory at runtime** — try `--imgsz 320` when exporting, or use the
  INT8 model.
