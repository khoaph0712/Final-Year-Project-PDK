import { useEffect, useState } from "react";
import { loadTensorflowModel, type TensorflowModel } from "react-native-fast-tflite";

import { useAppStore } from "@/store/useAppStore";

export type DetectorState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "ready"; model: TensorflowModel; inputSize: number; numClasses: number }
  | { status: "error"; error: string };

/**
 * Loads the bundled YOLOv8n TFLite model.
 *
 * Drop the exported file into `mobile/assets/model/`:
 *   - best_float16.tflite (faster live preview)
 *   - best_float32.tflite (maximum accuracy)
 */
export function useDetector(): DetectorState {
  const [state, setState] = useState<DetectorState>({ status: "idle" });
  const useFloat16 = useAppStore((s) => s.settings.useFloat16);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setState({ status: "loading" });
      try {
        const modelAsset = useFloat16
          ? require("../../assets/model/best_float16.tflite")
          : require("../../assets/model/best_float32.tflite");

        const model = await loadTensorflowModel(modelAsset, "default");

        const inputShape = model.inputs[0].shape; // [1, H, W, 3] for TFLite NHWC
        const inputSize = inputShape[1] ?? 640;
        const outShape = model.outputs[0].shape;
        const stride = outShape[1] === 8400 ? outShape[2] : outShape[1];
        const numClasses = Math.max(0, (stride ?? 11) - 4);

        if (!cancelled) setState({ status: "ready", model, inputSize, numClasses });
      } catch (err: any) {
        if (!cancelled)
          setState({
            status: "error",
            error:
              err?.message ??
              "Failed to load TFLite model. Make sure best_float16.tflite / best_float32.tflite " +
                "is placed in mobile/assets/model/.",
          });
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [useFloat16]);

  return state;
}
