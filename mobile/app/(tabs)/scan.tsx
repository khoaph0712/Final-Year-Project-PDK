import React, { useEffect, useMemo, useRef, useState } from "react";
import { ActivityIndicator, StyleSheet, View, Linking } from "react-native";
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
  type Frame,
} from "react-native-vision-camera";
import { useResizePlugin } from "vision-camera-resize-plugin";
import { useSharedValue, runOnJS, Worklets } from "react-native-worklets-core";
import { router } from "expo-router";
import * as Haptics from "expo-haptics";

import { Screen } from "@/components/Screen";
import { Text } from "@/components/Text";
import { Button } from "@/components/Button";
import { Card } from "@/components/Card";
import { useTheme } from "@/theme";
import { useDetector } from "@/lib/useDetector";
import { decodeYoloOutput, nms, summariseDetections, type RawBox } from "@/lib/yolo";
import { BIN_FOR, EMOJI, type WasteClass } from "@/lib/classes";
import { useAppStore } from "@/store/useAppStore";

type LiveResult = { topClass: WasteClass; topScore: number; count: number } | null;

export default function Scan() {
  const { colors, spacing, radius } = useTheme();
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice("back");
  const detector = useDetector();
  const { resize } = useResizePlugin();

  const { settings, addScan } = useAppStore();
  const [live, setLive] = useState<LiveResult>(null);
  const [capturing, setCapturing] = useState(false);
  const lastUpdate = useSharedValue(0);

  useEffect(() => {
    if (!hasPermission) requestPermission();
  }, [hasPermission, requestPermission]);

  const ready = detector.status === "ready" && hasPermission && device;
  const inputSize = detector.status === "ready" ? detector.inputSize : 640;
  const numClasses = detector.status === "ready" ? detector.numClasses : 7;

  const setLiveJS = Worklets.createRunOnJS(setLive);

  const frameProcessor = useFrameProcessor(
    (frame: Frame) => {
      "worklet";
      if (detector.status !== "ready") return;
      const now = Date.now();
      if (now - lastUpdate.value < 200) return; // throttle ~5 fps
      lastUpdate.value = now;

      const input = resize(frame, {
        scale: { width: inputSize, height: inputSize },
        pixelFormat: "rgb",
        dataType: "float32",
      });

      const outputs = detector.model.runSync([input]);
      const raw = outputs[0] as Float32Array;
      const shape = detector.model.outputs[0].shape;

      const boxes = decodeYoloOutput(raw, shape, numClasses, settings.confThreshold);
      const kept = nms(boxes, settings.iouThreshold, 20);
      const summary = summariseDetections(kept);

      if (summary) {
        setLiveJS({ topClass: summary.topClass, topScore: summary.topScore, count: kept.length });
      } else {
        setLiveJS(null);
      }
    },
    [detector, numClasses, inputSize, settings.confThreshold, settings.iouThreshold],
  );

  const capture = async () => {
    if (!live) return;
    setCapturing(true);
    try {
      if (settings.hapticsEnabled) Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
      const dummyBox: RawBox = {
        x: 0.5,
        y: 0.5,
        w: 0.4,
        h: 0.4,
        score: live.topScore,
        classIdx: 0,
        cls: live.topClass,
      };
      const record = addScan({
        topClass: live.topClass,
        topConfidence: live.topScore,
        detections: [
          { cls: live.topClass, confidence: live.topScore, box: [dummyBox.x, dummyBox.y, dummyBox.w, dummyBox.h] },
        ],
      });
      router.push({ pathname: "/result", params: { id: record.id } });
    } finally {
      setCapturing(false);
    }
  };

  return (
    <Screen padded={false} edges={["top"]}>
      <View style={{ flex: 1, backgroundColor: "#000" }}>
        {ready && device ? (
          <Camera
            style={StyleSheet.absoluteFill}
            device={device}
            isActive
            frameProcessor={frameProcessor}
            pixelFormat="yuv"
          />
        ) : (
          <View style={styles.center}>
            {detector.status === "loading" ? (
              <>
                <ActivityIndicator color="#fff" />
                <Text color="#fff" style={{ marginTop: 12 }}>
                  Loading detector…
                </Text>
              </>
            ) : detector.status === "error" ? (
              <Card style={{ margin: 24 }}>
                <Text variant="h3">Model not loaded</Text>
                <Text variant="bodySm" muted style={{ marginTop: 8 }}>
                  {detector.error}
                </Text>
              </Card>
            ) : !hasPermission ? (
              <Card style={{ margin: 24 }}>
                <Text variant="h3">Camera permission needed</Text>
                <Text variant="bodySm" muted style={{ marginVertical: 8 }}>
                  Please grant camera access to scan waste items.
                </Text>
                <Button title="Open settings" onPress={() => Linking.openSettings()} />
              </Card>
            ) : (
              <Text color="#fff">No camera found</Text>
            )}
          </View>
        )}

        <View style={[styles.topBar, { paddingTop: spacing.lg + 24, paddingHorizontal: spacing.lg }]}>
          <View
            style={{
              backgroundColor: "rgba(0,0,0,0.45)",
              paddingHorizontal: spacing.md,
              paddingVertical: 8,
              borderRadius: radius.pill,
            }}
          >
            <Text variant="caption" color="#fff">
              {detector.status === "ready" ? "● LIVE" : "● OFFLINE"}
            </Text>
          </View>
        </View>

        <View style={[styles.bottomPanel, { padding: spacing.lg }]}>
          <View
            style={{
              backgroundColor: "rgba(15,23,42,0.78)",
              padding: spacing.lg,
              borderRadius: radius.xl,
              marginBottom: spacing.md,
            }}
          >
            {live ? (
              <View style={{ flexDirection: "row", alignItems: "center", gap: spacing.md }}>
                <Text style={{ fontSize: 36 }}>{EMOJI[live.topClass]}</Text>
                <View style={{ flex: 1 }}>
                  <Text variant="h2" color="#fff" style={{ textTransform: "capitalize" }}>
                    {live.topClass}
                  </Text>
                  <Text variant="bodySm" color="rgba(255,255,255,0.75)">
                    Goes in <Text color={colors.bin[BIN_FOR[live.topClass]]} weight="700">{BIN_FOR[live.topClass]}</Text> · {(live.topScore * 100).toFixed(0)}%
                  </Text>
                </View>
              </View>
            ) : (
              <Text color="rgba(255,255,255,0.8)" align="center">
                Point the camera at an item…
              </Text>
            )}
          </View>
          <Button
            title={live ? `Confirm & save` : "Waiting…"}
            variant="primary"
            fullWidth
            loading={capturing}
            disabled={!live}
            onPress={capture}
          />
        </View>
      </View>
    </Screen>
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, alignItems: "center", justifyContent: "center" },
  topBar: { position: "absolute", top: 0, left: 0, right: 0, flexDirection: "row" },
  bottomPanel: { position: "absolute", bottom: 0, left: 0, right: 0 },
});
