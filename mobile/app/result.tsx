import React from "react";
import { ScrollView, View, StyleSheet } from "react-native";
import { MotiView } from "moti";
import { router, useLocalSearchParams } from "expo-router";

import { Screen } from "@/components/Screen";
import { Text } from "@/components/Text";
import { Card } from "@/components/Card";
import { Button } from "@/components/Button";
import { ConfidenceBar } from "@/components/ConfidenceBar";
import { useTheme } from "@/theme";
import { useAppStore } from "@/store/useAppStore";
import { BIN_FOR, BIN_LABEL, EMOJI, DISPOSAL_INSTRUCTIONS } from "@/lib/classes";

export default function Result() {
  const { colors, spacing, radius, mode } = useTheme();
  const { id } = useLocalSearchParams<{ id: string }>();
  const scan = useAppStore((s) => s.scans.find((x) => x.id === id));
  
  const isDark = mode === "dark";

  if (!scan) {
    return (
      <Screen>
        <Card style={styles.errorCard}>
          <Text variant="h2" align="center">Scan not found</Text>
          <Button title="Go back" onPress={() => router.back()} style={{ marginTop: 20, alignSelf: "center" }} />
        </Card>
      </Screen>
    );
  }

  const bin = BIN_FOR[scan.topClass];
  const instructions = DISPOSAL_INSTRUCTIONS[scan.topClass];
  const instructionLines = instructions.split("\n");
  const binTitle = instructionLines[0] || "";
  const binDetails = instructionLines[1] || "";

  return (
    <Screen padded={true}>
      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={{ paddingVertical: spacing.xl, paddingBottom: 60, gap: spacing.lg }}
      >
        {/* Dynamic Frosted Glass Hero Header Card */}
        <MotiView
          from={{ scale: 0.9, opacity: 0, translateY: 15 }}
          animate={{ scale: 1, opacity: 1, translateY: 0 }}
          transition={{ type: "spring", damping: 15, delay: 100 }}
        >
          <Card style={styles.heroCard}>
            <MotiView
              from={{ scale: 0.5, rotate: "-15deg" }}
              animate={{ scale: 1, rotate: "0deg" }}
              transition={{ type: "spring", damping: 10, delay: 300 }}
              style={styles.emojiContainer}
            >
              <Text style={{ fontSize: 90 }}>{EMOJI[scan.topClass]}</Text>
            </MotiView>
            
            <Text variant="display" style={[styles.mainClassText, { color: isDark ? "#fff" : colors.text }]}>
              {scan.topClass}
            </Text>
            
            <View style={[styles.badge, { backgroundColor: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.05)" }]}>
              <Text variant="bodySm" weight="600" color={isDark ? "rgba(255,255,255,0.85)" : colors.text}>
                Confidence: {(scan.topConfidence * 100).toFixed(0)}%
              </Text>
            </View>
          </Card>
        </MotiView>

        {/* High-Priority AI Guided Disposal Card */}
        <MotiView
          from={{ opacity: 0, translateY: 15 }}
          animate={{ opacity: 1, translateY: 0 }}
          transition={{ type: "timing", duration: 400, delay: 200 }}
        >
          <Card style={[styles.disposalCard, { borderLeftColor: colors.bin[bin] }]}>
            <Text variant="caption" muted style={{ marginBottom: 4 }}>
              AI DISPOSAL GUIDANCE
            </Text>
            <Text variant="h2" style={[styles.binTitle, { color: isDark ? "#fff" : colors.text }]}>
              {binTitle}
            </Text>
            <Text variant="body" style={[styles.binDetails, { color: isDark ? "rgba(255,255,255,0.85)" : colors.text }]}>
              {binDetails}
            </Text>
          </Card>
        </MotiView>

        {/* Breakdown of Detected Classes */}
        <MotiView
          from={{ opacity: 0, translateY: 15 }}
          animate={{ opacity: 1, translateY: 0 }}
          transition={{ type: "timing", duration: 400, delay: 300 }}
        >
          <Card style={styles.listCard}>
            <Text variant="h3" style={{ marginBottom: spacing.md, color: isDark ? "#fff" : colors.text }}>
              Composition breakdown
            </Text>
            <View style={{ gap: spacing.sm }}>
              {scan.detections.map((d, i) => (
                <ConfidenceBar
                  key={i}
                  label={d.cls}
                  value={d.confidence}
                  bin={BIN_FOR[d.cls]}
                />
              ))}
            </View>
          </Card>
        </MotiView>

        {/* Glowing Rewards Card */}
        <MotiView
          from={{ opacity: 0, translateY: 15 }}
          animate={{ opacity: 1, translateY: 0 }}
          transition={{ type: "timing", duration: 400, delay: 400 }}
        >
          <Card style={[styles.pointsCard, { borderColor: colors.primary }]}>
            <Text variant="caption" color={colors.primary} style={{ fontWeight: "700" }}>
              ECO-CREDIT AWARDED
            </Text>
            <Text variant="h1" color={colors.primary} style={styles.pointsText}>
              +{scan.pointsEarned} Points
            </Text>
            <Text variant="caption" muted>
              Added instantly to your profile score
            </Text>
          </Card>
        </MotiView>

        {/* Action Buttons */}
        <MotiView
          from={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ type: "timing", duration: 300, delay: 500 }}
          style={styles.actionsContainer}
        >
          <Button
            title="Find nearest bin"
            variant="secondary"
            fullWidth
            onPress={() => router.push("/bin-locator")}
            style={{ flex: 1 }}
          />
          <Button
            title="Scan another"
            variant="primary"
            fullWidth
            onPress={() => router.replace("/(tabs)/scan")}
            style={{ flex: 1 }}
          />
        </MotiView>
      </ScrollView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  errorCard: {
    padding: 30,
    marginVertical: 40,
    alignItems: "center",
    justifyContent: "center",
  },
  heroCard: {
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 32,
    gap: 12,
  },
  emojiContainer: {
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.15,
    shadowRadius: 15,
    elevation: 3,
  },
  mainClassText: {
    textTransform: "capitalize",
    fontWeight: "800",
    textAlign: "center",
  },
  badge: {
    paddingHorizontal: 16,
    paddingVertical: 6,
    borderRadius: 99,
  },
  disposalCard: {
    borderLeftWidth: 6,
    paddingLeft: 20,
    gap: 4,
  },
  binTitle: {
    fontWeight: "700",
    fontSize: 20,
  },
  binDetails: {
    fontSize: 15,
    lineHeight: 22,
    marginTop: 4,
  },
  listCard: {
    paddingVertical: 20,
  },
  pointsCard: {
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 2,
    backgroundColor: "rgba(13, 148, 136, 0.05)",
    paddingVertical: 20,
    gap: 2,
  },
  pointsText: {
    fontSize: 32,
    fontWeight: "800",
  },
  actionsContainer: {
    flexDirection: "row",
    gap: 12,
    marginTop: 4,
  },
});
