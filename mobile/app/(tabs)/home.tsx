import React, { useMemo } from "react";
import { ScrollView, View } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { MotiView } from "moti";
import { router } from "expo-router";

import { Screen } from "@/components/Screen";
import { Text } from "@/components/Text";
import { Card } from "@/components/Card";
import { Button } from "@/components/Button";
import { useTheme } from "@/theme";
import { useAppStore } from "@/store/useAppStore";
import { BIN_LABEL, EMOJI } from "@/lib/classes";

export default function Home() {
  const { colors, spacing, radius } = useTheme();
  const { ecoPoints, level, streakDays, scans } = useAppStore();

  const progressToNext = useMemo(() => {
    const nextLevelPoints = level * level * 20;
    const prevLevelPoints = (level - 1) * (level - 1) * 20;
    const span = Math.max(1, nextLevelPoints - prevLevelPoints);
    return Math.min(1, (ecoPoints - prevLevelPoints) / span);
  }, [ecoPoints, level]);

  return (
    <Screen>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 40 }}>
        <View style={{ marginTop: spacing.md, marginBottom: spacing.xl }}>
          <Text variant="caption" muted>
            HELLO, ECO HERO
          </Text>
          <Text variant="display">Let's sort some waste.</Text>
        </View>

        <MotiView
          from={{ opacity: 0, translateY: 8 }}
          animate={{ opacity: 1, translateY: 0 }}
          transition={{ type: "timing", duration: 450 }}
        >
          <Card padded={false} style={{ overflow: "hidden", marginBottom: spacing.lg }}>
            <LinearGradient
              colors={[colors.primary, colors.accent]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={{ padding: spacing.xl }}
            >
              <Text variant="caption" color="rgba(255,255,255,0.85)">
                ECO-POINTS
              </Text>
              <Text variant="display" color="#fff">
                {ecoPoints}
              </Text>
              <View
                style={{
                  height: 8,
                  backgroundColor: "rgba(255,255,255,0.25)",
                  borderRadius: radius.pill,
                  marginVertical: spacing.md,
                  overflow: "hidden",
                }}
              >
                <MotiView
                  from={{ width: "0%" as any }}
                  animate={{ width: `${progressToNext * 100}%` as any }}
                  transition={{ type: "timing", duration: 700 }}
                  style={{ height: "100%", backgroundColor: "#fff" }}
                />
              </View>
              <View style={{ flexDirection: "row", justifyContent: "space-between" }}>
                <Text variant="bodySm" color="#fff">
                  Level {level}
                </Text>
                <Text variant="bodySm" color="rgba(255,255,255,0.9)">
                  🔥 {streakDays}-day streak
                </Text>
              </View>
            </LinearGradient>
          </Card>
        </MotiView>

        <View style={{ flexDirection: "row", gap: spacing.md, marginBottom: spacing.xl }}>
          <Card style={{ flex: 1 }}>
            <Text variant="caption" muted>
              TODAY
            </Text>
            <Text variant="h1">
              {scans.filter((s) => isToday(s.createdAt)).length}
            </Text>
            <Text variant="bodySm" muted>
              items scanned
            </Text>
          </Card>
          <Card style={{ flex: 1 }}>
            <Text variant="caption" muted>
              TOTAL
            </Text>
            <Text variant="h1">{scans.length}</Text>
            <Text variant="bodySm" muted>
              items sorted
            </Text>
          </Card>
        </View>

        <Button
          title="Scan an item"
          fullWidth
          onPress={() => router.push("/(tabs)/scan")}
          style={{ marginBottom: spacing.xl }}
        />

        <Text variant="h2" style={{ marginBottom: spacing.md }}>
          Recent scans
        </Text>
        {scans.slice(0, 5).map((s) => (
          <Card key={s.id} style={{ marginBottom: spacing.sm, flexDirection: "row", alignItems: "center", gap: spacing.md }}>
            <Text style={{ fontSize: 28 }}>{EMOJI[s.topClass]}</Text>
            <View style={{ flex: 1 }}>
              <Text variant="h3" style={{ textTransform: "capitalize" }}>
                {s.topClass}
              </Text>
              <Text variant="bodySm" muted>
                {BIN_LABEL[binOf(s.topClass)]} · {new Date(s.createdAt).toLocaleString()}
              </Text>
            </View>
            <Text variant="bodySm" weight="700" color={colors.primary}>
              +{s.pointsEarned}
            </Text>
          </Card>
        ))}
        {scans.length === 0 && (
          <Card>
            <Text variant="h3">No scans yet</Text>
            <Text variant="bodySm" muted>
              Tap "Scan an item" to start sorting.
            </Text>
          </Card>
        )}
      </ScrollView>
    </Screen>
  );
}

function isToday(ts: number) {
  const d = new Date(ts);
  const n = new Date();
  return d.toDateString() === n.toDateString();
}

import { BIN_FOR } from "@/lib/classes";
function binOf(c: keyof typeof BIN_FOR) {
  return BIN_FOR[c];
}
