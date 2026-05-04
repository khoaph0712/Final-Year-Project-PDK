import React from "react";
import { ScrollView, Switch, View } from "react-native";

import { Screen } from "@/components/Screen";
import { Text } from "@/components/Text";
import { Card } from "@/components/Card";
import { Button } from "@/components/Button";
import { useTheme } from "@/theme";
import { useAppStore } from "@/store/useAppStore";

function Row({
  label,
  description,
  right,
}: {
  label: string;
  description?: string;
  right: React.ReactNode;
}) {
  const { spacing } = useTheme();
  return (
    <View
      style={{
        flexDirection: "row",
        alignItems: "center",
        gap: spacing.md,
        paddingVertical: spacing.sm,
      }}
    >
      <View style={{ flex: 1 }}>
        <Text variant="body" weight="600">
          {label}
        </Text>
        {description ? (
          <Text variant="bodySm" muted>
            {description}
          </Text>
        ) : null}
      </View>
      {right}
    </View>
  );
}

export default function Settings() {
  const { colors, spacing, mode, setMode, preferred } = useTheme();
  const { settings, setSetting, clearScans } = useAppStore();

  return (
    <Screen>
      <Text variant="display" style={{ marginBottom: spacing.lg }}>
        Settings
      </Text>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 40 }}>
        <Card style={{ marginBottom: spacing.lg }}>
          <Text variant="caption" muted style={{ marginBottom: spacing.sm }}>
            APPEARANCE
          </Text>
          <View style={{ flexDirection: "row", gap: spacing.sm }}>
            {(["light", "dark", "system"] as const).map((m) => {
              const active = preferred === m;
              return (
                <Button
                  key={m}
                  title={m}
                  variant={active ? "primary" : "ghost"}
                  onPress={() => setMode(m)}
                  style={{ flex: 1, textTransform: "capitalize" as any }}
                />
              );
            })}
          </View>
          <Text variant="bodySm" muted style={{ marginTop: spacing.sm }}>
            Currently using {mode} mode.
          </Text>
        </Card>

        <Card style={{ marginBottom: spacing.lg }}>
          <Text variant="caption" muted style={{ marginBottom: spacing.sm }}>
            DETECTION
          </Text>
          <Row
            label="Use INT8 quantised model"
            description="Faster and smaller — slightly lower accuracy."
            right={
              <Switch
                value={settings.useInt8}
                onValueChange={(v) => setSetting("useInt8", v)}
                thumbColor={settings.useInt8 ? colors.primary : colors.border}
              />
            }
          />
          <Row
            label="Confidence threshold"
            description={`${(settings.confThreshold * 100).toFixed(0)}% — lower catches more, with more false positives.`}
            right={
              <View style={{ flexDirection: "row", gap: 6 }}>
                <Button
                  title="−"
                  variant="ghost"
                  onPress={() =>
                    setSetting("confThreshold", Math.max(0.1, +(settings.confThreshold - 0.05).toFixed(2)))
                  }
                />
                <Button
                  title="+"
                  variant="ghost"
                  onPress={() =>
                    setSetting("confThreshold", Math.min(0.9, +(settings.confThreshold + 0.05).toFixed(2)))
                  }
                />
              </View>
            }
          />
          <Row
            label="Haptic feedback"
            description="Vibrate on scan capture."
            right={
              <Switch
                value={settings.hapticsEnabled}
                onValueChange={(v) => setSetting("hapticsEnabled", v)}
                thumbColor={settings.hapticsEnabled ? colors.primary : colors.border}
              />
            }
          />
        </Card>

        <Card style={{ marginBottom: spacing.lg }}>
          <Text variant="caption" muted style={{ marginBottom: spacing.sm }}>
            DATA
          </Text>
          <Button title="Clear scan history" variant="danger" onPress={clearScans} />
        </Card>

        <Card>
          <Text variant="caption" muted>
            ABOUT
          </Text>
          <Text variant="h3" style={{ marginTop: 6 }}>
            WasteWise
          </Text>
          <Text variant="bodySm" muted>
            On-device waste classification with YOLOv8n. Final-year project.
          </Text>
        </Card>
      </ScrollView>
    </Screen>
  );
}
