import React from "react";
import { View } from "react-native";

import { Screen } from "@/components/Screen";
import { Text } from "@/components/Text";
import { Card } from "@/components/Card";
import { useTheme } from "@/theme";
import { useAppStore } from "@/store/useAppStore";

type Row = { id: string; name: string; points: number; self?: boolean };

const MOCK: Row[] = [
  { id: "1", name: "Ayesha", points: 1480 },
  { id: "2", name: "Hana", points: 1122 },
  { id: "3", name: "Ben", points: 980 },
  { id: "4", name: "Kim", points: 612 },
  { id: "5", name: "Priya", points: 540 },
];

export default function Leaderboard() {
  const { colors, spacing, radius } = useTheme();
  const { ecoPoints, level } = useAppStore();

  const rows: Row[] = [...MOCK, { id: "self", name: "You", points: ecoPoints, self: true }]
    .sort((a, b) => b.points - a.points)
    .map((r, i) => ({ ...r, rank: i + 1 } as any));

  return (
    <Screen>
      <Text variant="display" style={{ marginBottom: spacing.md }}>
        Leaderboard
      </Text>
      <Text variant="bodySm" muted style={{ marginBottom: spacing.lg }}>
        This week · Friends group
      </Text>

      <Card style={{ marginBottom: spacing.lg }}>
        <Text variant="caption" muted>
          YOUR POSITION
        </Text>
        <Text variant="h1">
          #{(rows.findIndex((r) => r.self) + 1).toString().padStart(2, "0")}
        </Text>
        <Text variant="bodySm" muted>
          Level {level} · {ecoPoints} pts
        </Text>
      </Card>

      {rows.map((r: any) => (
        <Card
          key={r.id}
          style={{
            flexDirection: "row",
            alignItems: "center",
            gap: spacing.md,
            marginBottom: spacing.sm,
            backgroundColor: r.self ? colors.primaryMuted : colors.surface,
            borderColor: r.self ? "transparent" : colors.border,
          }}
        >
          <View
            style={{
              width: 40,
              height: 40,
              borderRadius: radius.pill,
              backgroundColor: r.self ? colors.primary : colors.border,
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <Text variant="h3" color={r.self ? colors.primaryContrast : colors.text}>
              {r.rank}
            </Text>
          </View>
          <Text variant="h3" style={{ flex: 1 }}>
            {r.name}
          </Text>
          <Text variant="h3" color={colors.primary}>
            {r.points}
          </Text>
        </Card>
      ))}
    </Screen>
  );
}
