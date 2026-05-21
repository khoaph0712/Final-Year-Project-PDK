import React from "react";
import { View, StyleSheet, ScrollView } from "react-native";
import { MotiView } from "moti";

import { Screen } from "@/components/Screen";
import { Text } from "@/components/Text";
import { Card } from "@/components/Card";
import { useTheme } from "@/theme";
import { useAppStore } from "@/store/useAppStore";

type Row = { id: string; name: string; points: number; self?: boolean; rank?: number };

const MOCK: Row[] = [
  { id: "1", name: "Ayesha Khan", points: 1480 },
  { id: "2", name: "Hana Nguyen", points: 1122 },
  { id: "3", name: "Ben Johnson", points: 980 },
  { id: "4", name: "Kim Young", points: 612 },
  { id: "5", name: "Priya Sharma", points: 540 },
];

export default function Leaderboard() {
  const { colors, spacing, radius, mode } = useTheme();
  const { ecoPoints, level } = useAppStore();
  
  const isDark = mode === "dark";

  // Combine and sort scores
  const rows: Row[] = [...MOCK, { id: "self", name: "You (Eco Hero)", points: ecoPoints, self: true }]
    .sort((a, b) => b.points - a.points)
    .map((r, i) => ({ ...r, rank: i + 1 }));

  const selfIndex = rows.findIndex((r) => r.self);
  const selfRank = selfIndex + 1;

  // Render Rank Crown/Badge Emoji
  const getRankBadge = (rank: number) => {
    if (rank === 1) return "👑";
    if (rank === 2) return "🥈";
    if (rank === 3) return "🥉";
    return null;
  };

  return (
    <Screen>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ paddingBottom: 50 }}>
        {/* Screen Title Block */}
        <View style={{ marginTop: spacing.md, marginBottom: spacing.lg }}>
          <Text variant="display" style={{ fontWeight: "800", color: isDark ? "#fff" : colors.text }}>
            Scoreboard
          </Text>
          <Text variant="bodySm" muted style={{ marginTop: 2 }}>
            Weekly Competition · Eco Heroes Group
          </Text>
        </View>

        {/* User High-fidelity Frosted Stat Card */}
        <MotiView
          from={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ type: "spring", damping: 15 }}
        >
          <Card style={[styles.statsCard, { borderColor: isDark ? "rgba(13, 148, 136, 0.4)" : "rgba(13, 148, 136, 0.25)" }]}>
            <View style={{ flex: 1 }}>
              <Text variant="caption" color={colors.primary} style={{ fontWeight: "700" }}>
                YOUR POSITION
              </Text>
              <Text variant="display" style={[styles.bigRankText, { color: isDark ? "#fff" : colors.text }]}>
                #{selfRank.toString().padStart(2, "0")}
              </Text>
              <Text variant="bodySm" color={isDark ? "rgba(255,255,255,0.75)" : colors.textMuted}>
                Level {level} · {ecoPoints} Total Points
              </Text>
            </View>
            <View style={[styles.avatarGlow, { backgroundColor: colors.primaryMuted, borderColor: colors.primary }]}>
              <Text style={{ fontSize: 32 }}>🌿</Text>
            </View>
          </Card>
        </MotiView>

        {/* Spacer */}
        <View style={{ height: spacing.lg }} />

        {/* Leaderboard Table Header */}
        <View style={styles.tableHeader}>
          <Text variant="caption" muted style={{ width: 45, textAlign: "center" }}>
            RANK
          </Text>
          <Text variant="caption" muted style={{ flex: 1, marginLeft: spacing.sm }}>
            HERO NAME
          </Text>
          <Text variant="caption" muted style={{ textAlign: "right" }}>
            SCORE
          </Text>
        </View>

        {/* Leaderboard Rows */}
        <View style={{ gap: spacing.sm }}>
          {rows.map((r, i) => {
            const isTop3 = r.rank && r.rank <= 3;
            const badge = r.rank ? getRankBadge(r.rank) : null;
            
            // Custom premium styling configurations
            let borderStyle = {};
            let textWeight: "700" | "600" | "400" = "600";
            let rowBg = isDark ? "rgba(30, 41, 59, 0.4)" : "rgba(255, 255, 255, 0.6)";

            if (r.self) {
              // Standout neon gold/teal border style
              borderStyle = {
                borderColor: isDark ? "#eab308" : "#d97706", // Gold neon border
                borderWidth: 2,
                backgroundColor: isDark ? "rgba(234, 179, 8, 0.08)" : "rgba(234, 179, 8, 0.06)",
                shadowColor: "#eab308",
                shadowOpacity: 0.15,
                shadowRadius: 10,
              };
              textWeight = "700";
            } else if (isTop3) {
              borderStyle = {
                borderColor: r.rank === 1 
                  ? "rgba(234, 179, 8, 0.4)" // Gold hint
                  : r.rank === 2 
                    ? "rgba(148, 163, 184, 0.4)" // Silver hint
                    : "rgba(180, 83, 9, 0.3)", // Bronze hint
              };
            }

            return (
              <MotiView
                key={r.id}
                from={{ opacity: 0, translateY: 10 }}
                animate={{ opacity: 1, translateY: 0 }}
                transition={{ type: "timing", duration: 350, delay: i * 80 }}
              >
                <Card
                  padded={false}
                  elevated={!r.self}
                  style={[
                    styles.rankRow,
                    { backgroundColor: rowBg },
                    borderStyle,
                  ]}
                >
                  {/* Rank Column */}
                  <View style={styles.rankCol}>
                    {badge ? (
                      <Text style={{ fontSize: 22 }}>{badge}</Text>
                    ) : (
                      <View style={[styles.rankCircle, { backgroundColor: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)" }]}>
                        <Text variant="bodySm" weight="700" color={isDark ? "rgba(255,255,255,0.85)" : colors.text}>
                          {r.rank}
                        </Text>
                      </View>
                    )}
                  </View>

                  {/* Name Column */}
                  <Text
                    variant="body"
                    weight={textWeight}
                    style={{ flex: 1, color: r.self ? (isDark ? "#eab308" : "#d97706") : (isDark ? "#fff" : colors.text) }}
                    numberOfLines={1}
                  >
                    {r.name} {r.self && "🌟"}
                  </Text>

                  {/* Points Column */}
                  <View style={styles.pointsCol}>
                    <Text
                      variant="body"
                      weight="700"
                      color={r.self ? (isDark ? "#eab308" : "#d97706") : colors.primary}
                    >
                      {r.points}
                    </Text>
                    <Text variant="caption" muted style={{ fontSize: 10 }}>
                      pts
                    </Text>
                  </View>
                </Card>
              </MotiView>
            );
          })}
        </View>
      </ScrollView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  statsCard: {
    flexDirection: "row",
    alignItems: "center",
    padding: 24,
    borderWidth: 1.5,
  },
  bigRankText: {
    fontSize: 42,
    fontWeight: "900",
    lineHeight: 46,
    marginVertical: 4,
  },
  avatarGlow: {
    width: 68,
    height: 68,
    borderRadius: 99,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 2,
    shadowColor: "#0d9488",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.15,
    shadowRadius: 10,
    elevation: 3,
  },
  tableHeader: {
    flexDirection: "row",
    paddingHorizontal: 16,
    marginBottom: 8,
    alignItems: "center",
  },
  rankRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 14,
    paddingHorizontal: 16,
    borderRadius: 18,
    gap: 12,
  },
  rankCol: {
    width: 40,
    alignItems: "center",
    justifyContent: "center",
  },
  rankCircle: {
    width: 28,
    height: 28,
    borderRadius: 99,
    alignItems: "center",
    justifyContent: "center",
  },
  pointsCol: {
    alignItems: "flex-end",
    justifyContent: "center",
  },
});
