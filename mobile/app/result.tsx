import React from "react";
import { ScrollView, View } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { MotiView } from "moti";
import { router, useLocalSearchParams } from "expo-router";

import { Screen } from "@/components/Screen";
import { Text } from "@/components/Text";
import { Card } from "@/components/Card";
import { Button } from "@/components/Button";
import { ConfidenceBar } from "@/components/ConfidenceBar";
import { useTheme } from "@/theme";
import { useAppStore } from "@/store/useAppStore";
import { BIN_FOR, BIN_LABEL, EMOJI, TIPS } from "@/lib/classes";

export default function Result() {
  const { colors, spacing, radius } = useTheme();
  const { id } = useLocalSearchParams<{ id: string }>();
  const scan = useAppStore((s) => s.scans.find((x) => x.id === id));

  if (!scan) {
    return (
      <Screen>
        <Text variant="h2">Scan not found</Text>
        <Button title="Back" onPress={() => router.back()} style={{ marginTop: 16 }} />
      </Screen>
    );
  }

  const bin = BIN_FOR[scan.topClass];
  const tip = TIPS[scan.topClass];

  return (
    <Screen padded={false}>
      <LinearGradient
        colors={[colors.bin[bin], colors.primary]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={{ padding: spacing.xl, paddingTop: spacing.xxxl * 1.5, paddingBottom: spacing.xxl }}
      >
        <MotiView
          from={{ scale: 0.6, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ type: "spring", damping: 12 }}
          style={{ alignItems: "center" }}
        >
          <Text style={{ fontSize: 96 }}>{EMOJI[scan.topClass]}</Text>
          <Text variant="display" color="#fff" style={{ textTransform: "capitalize" }}>
            {scan.topClass}
          </Text>
          <Text color="rgba(255,255,255,0.9)">
            Confidence {(scan.topConfidence * 100).toFixed(0)}%
          </Text>
        </MotiView>
      </LinearGradient>

      <ScrollView
        contentContainerStyle={{ padding: spacing.lg, paddingBottom: 48 }}
        style={{ flex: 1, backgroundColor: colors.background }}
      >
        <Card style={{ marginBottom: spacing.lg }}>
          <Text variant="caption" muted>
            THIS BELONGS IN
          </Text>
          <View style={{ flexDirection: "row", alignItems: "center", gap: spacing.md, marginTop: 6 }}>
            <View
              style={{
                width: 16,
                height: 16,
                borderRadius: radius.pill,
                backgroundColor: colors.bin[bin],
              }}
            />
            <Text variant="h2">{BIN_LABEL[bin]}</Text>
          </View>
          <Text variant="bodySm" muted style={{ marginTop: spacing.md }}>
            {tip}
          </Text>
        </Card>

        <Card style={{ marginBottom: spacing.lg }}>
          <Text variant="h3" style={{ marginBottom: spacing.md }}>
            Detected items
          </Text>
          {scan.detections.map((d, i) => (
            <ConfidenceBar
              key={i}
              label={d.cls}
              value={d.confidence}
              bin={BIN_FOR[d.cls]}
            />
          ))}
        </Card>

        <Card style={{ marginBottom: spacing.lg, backgroundColor: colors.primaryMuted, borderColor: "transparent" }}>
          <Text variant="caption" color={colors.primary}>
            EARNED
          </Text>
          <Text variant="h1" color={colors.primary}>
            +{scan.pointsEarned} eco-points
          </Text>
        </Card>

        <View style={{ flexDirection: "row", gap: spacing.md }}>
          <Button
            title="Find nearest bin"
            variant="secondary"
            fullWidth
            onPress={() => router.push("/bin-locator")}
            style={{ flex: 1 }}
          />
          <Button
            title="Scan another"
            fullWidth
            onPress={() => router.replace("/(tabs)/scan")}
            style={{ flex: 1 }}
          />
        </View>
      </ScrollView>
    </Screen>
  );
}
