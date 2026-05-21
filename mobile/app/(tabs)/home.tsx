import React, { useMemo } from "react";
import { ScrollView, View, StyleSheet } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { MotiView } from "moti";
import { router } from "expo-router";

import { Screen } from "@/components/Screen";
import { Text } from "@/components/Text";
import { Card } from "@/components/Card";
import { Button } from "@/components/Button";
import { useTheme } from "@/theme";
import { useAppStore } from "@/store/useAppStore";
import { BIN_LABEL, EMOJI, BIN_FOR } from "@/lib/classes";

export default function Home() {
  const { colors, spacing, radius, mode } = useTheme();
  const { ecoPoints, level, streakDays, scans } = useAppStore();

  const isDark = mode === "dark";

  const progressToNext = useMemo(() => {
    const nextLevelPoints = level * level * 20;
    const prevLevelPoints = (level - 1) * (level - 1) * 20;
    const span = Math.max(1, nextLevelPoints - prevLevelPoints);
    return Math.min(1, (ecoPoints - prevLevelPoints) / span);
  }, [ecoPoints, level]);

  const itemsScannedToday = useMemo(() => {
    return scans.filter((s) => isToday(s.createdAt)).length;
  }, [scans]);

  return (
    <Screen>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 50 }}>
        {/* Dashboard Title Block */}
        <MotiView
          from={{ opacity: 0, translateY: -10 }}
          animate={{ opacity: 1, translateY: 0 }}
          transition={{ type: "timing", duration: 400 }}
          style={{ marginTop: spacing.md, marginBottom: spacing.lg }}
        >
          <Text variant="caption" muted style={{ fontWeight: "700" }}>
            HELLO, ECO HERO
          </Text>
          <Text variant="display" style={{ fontWeight: "800", color: isDark ? "#fff" : colors.text }}>
            Let's sort some waste.
          </Text>
        </MotiView>

        {/* Liquid Glass ECO-POINTS Score Widget */}
        <MotiView
          from={{ opacity: 0, translateY: 15 }}
          animate={{ opacity: 1, translateY: 0 }}
          transition={{ type: "spring", damping: 15, delay: 100 }}
        >
          <Card padded={false} style={styles.pointsCard}>
            {/* Soft Translucent Backdrop Gradient inside Glass */}
            <LinearGradient
              colors={
                isDark 
                  ? ["rgba(13, 148, 136, 0.25)", "rgba(132, 204, 22, 0.05)"] 
                  : ["rgba(13, 148, 136, 0.15)", "rgba(132, 204, 22, 0.05)"]
              }
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={{ padding: spacing.xl }}
            >
              <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start" }}>
                <View>
                  <Text variant="caption" color={isDark ? "rgba(255,255,255,0.7)" : colors.textMuted} style={{ fontWeight: "700" }}>
                    TOTAL ECO-POINTS
                  </Text>
                  <Text variant="display" style={[styles.pointsDisplayValue, { color: isDark ? "#fff" : colors.text }]}>
                    {ecoPoints}
                  </Text>
                </View>
                <View style={[styles.streakBadge, { backgroundColor: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.05)" }]}>
                  <Text variant="bodySm" style={{ fontSize: 14 }}>
                    🔥 {streakDays} days
                  </Text>
                </View>
              </View>

              {/* Glowing Level Progress Bar */}
              <View style={[styles.progressBarBg, { backgroundColor: isDark ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.08)" }]}>
                <MotiView
                  from={{ width: "0%" as any }}
                  animate={{ width: `${progressToNext * 100}%` as any }}
                  transition={{ type: "timing", duration: 800, delay: 300 }}
                  style={[styles.progressBarFill, { backgroundColor: colors.primary }]}
                />
              </View>

              <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
                <Text variant="bodySm" weight="700" color={colors.primary}>
                  Level {level}
                </Text>
                <Text variant="caption" muted>
                  {Math.round(progressToNext * 100)}% to Level {level + 1}
                </Text>
              </View>
            </LinearGradient>
          </Card>
        </MotiView>

        {/* Dual Stat frosted Cards */}
        <MotiView
          from={{ opacity: 0, translateY: 15 }}
          animate={{ opacity: 1, translateY: 0 }}
          transition={{ type: "timing", duration: 400, delay: 200 }}
          style={styles.statsContainer}
        >
          <Card style={styles.statSubCard}>
            <Text variant="caption" muted style={{ fontWeight: "700" }}>
              SCANNED TODAY
            </Text>
            <Text variant="h1" style={[styles.statValue, { color: isDark ? "#fff" : colors.text }]}>
              {itemsScannedToday}
            </Text>
            <Text variant="caption" muted>
              items registered
            </Text>
          </Card>
          
          <Card style={styles.statSubCard}>
            <Text variant="caption" muted style={{ fontWeight: "700" }}>
              LIFETIME TOTAL
            </Text>
            <Text variant="h1" style={[styles.statValue, { color: isDark ? "#fff" : colors.text }]}>
              {scans.length}
            </Text>
            <Text variant="caption" muted>
              items sorted safely
            </Text>
          </Card>
        </MotiView>

        {/* Main Floating Call to Action */}
        <MotiView
          from={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ type: "spring", damping: 12, delay: 250 }}
        >
          <Button
            title="Scan & Classify Item"
            variant="primary"
            fullWidth
            onPress={() => router.push("/(tabs)/scan")}
            style={styles.scanButton}
          />
        </MotiView>

        {/* Recent Scans Section */}
        <View style={{ marginTop: spacing.md }}>
          <Text variant="h2" style={{ marginBottom: spacing.md, fontWeight: "800", color: isDark ? "#fff" : colors.text }}>
            Recent Scans
          </Text>

          <View style={{ gap: spacing.sm }}>
            {scans.slice(0, 5).map((s, i) => {
              const bin = BIN_FOR[s.topClass];
              return (
                <MotiView
                  key={s.id}
                  from={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ type: "timing", duration: 300, delay: i * 80 }}
                >
                  <Card padded={false} style={styles.scanItemRow}>
                    <View style={styles.emojiBadge}>
                      <Text style={{ fontSize: 26 }}>{EMOJI[s.topClass]}</Text>
                    </View>
                    <View style={{ flex: 1 }}>
                      <Text variant="body" weight="700" style={[styles.itemClassName, { color: isDark ? "#fff" : colors.text }]}>
                        {s.topClass}
                      </Text>
                      <Text variant="caption" muted>
                        {BIN_LABEL[bin]} · {new Date(s.createdAt).toLocaleDateString()}
                      </Text>
                    </View>
                    <View style={[styles.pointsBadge, { backgroundColor: isDark ? "rgba(13, 148, 136, 0.08)" : "rgba(13, 148, 136, 0.05)" }]}>
                      <Text variant="bodySm" weight="700" color={colors.primary}>
                        +{s.pointsEarned} pts
                      </Text>
                    </View>
                  </Card>
                </MotiView>
              );
            })}

            {scans.length === 0 && (
              <Card style={styles.emptyCard}>
                <Text style={{ fontSize: 40, marginBottom: 8 }}>🌱</Text>
                <Text variant="h3" align="center" style={{ color: isDark ? "#fff" : colors.text }}>No scans registered yet</Text>
                <Text variant="bodySm" muted align="center" style={{ marginTop: 4 }}>
                  Tap the button above to capture your first sorted item and earn eco-points!
                </Text>
              </Card>
            )}
          </View>
        </View>
      </ScrollView>
    </Screen>
  );
}

function isToday(ts: number) {
  const d = new Date(ts);
  const n = new Date();
  return d.toDateString() === n.toDateString();
}

const styles = StyleSheet.create({
  pointsCard: {
    borderWidth: 1.5,
    overflow: "hidden",
  },
  pointsDisplayValue: {
    fontSize: 52,
    fontWeight: "900",
    lineHeight: 58,
    marginVertical: 4,
  },
  streakBadge: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 99,
    alignItems: "center",
    justifyContent: "center",
  },
  progressBarBg: {
    height: 10,
    borderRadius: 99,
    marginVertical: 14,
    overflow: "hidden",
  },
  progressBarFill: {
    height: "100%",
    borderRadius: 99,
  },
  statsContainer: {
    flexDirection: "row",
    gap: 12,
    marginTop: 12,
    marginBottom: 16,
  },
  statSubCard: {
    flex: 1,
    padding: 16,
    gap: 2,
  },
  statValue: {
    fontSize: 32,
    fontWeight: "800",
    lineHeight: 36,
  },
  scanButton: {
    paddingVertical: 16,
    marginBottom: 20,
    shadowColor: "#0d9488",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.25,
    shadowRadius: 15,
    elevation: 4,
  },
  scanItemRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 12,
    paddingHorizontal: 16,
    gap: 12,
  },
  emojiBadge: {
    width: 46,
    height: 46,
    borderRadius: 14,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(255, 255, 255, 0.05)",
  },
  itemClassName: {
    textTransform: "capitalize",
  },
  pointsBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
  },
  emptyCard: {
    paddingVertical: 36,
    alignItems: "center",
    justifyContent: "center",
  },
});
