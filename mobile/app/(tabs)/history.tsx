import React, { useMemo, useState } from "react";
import { FlatList, Pressable, View } from "react-native";
import { router } from "expo-router";

import { Screen } from "@/components/Screen";
import { Text } from "@/components/Text";
import { Card } from "@/components/Card";
import { useTheme } from "@/theme";
import { useAppStore } from "@/store/useAppStore";
import { BIN_FOR, BIN_LABEL, CLASSES, EMOJI, type WasteClass } from "@/lib/classes";

export default function History() {
  const { colors, spacing, radius } = useTheme();
  const scans = useAppStore((s) => s.scans);
  const [filter, setFilter] = useState<"all" | WasteClass>("all");

  const filtered = useMemo(
    () => (filter === "all" ? scans : scans.filter((s) => s.topClass === filter)),
    [scans, filter],
  );

  return (
    <Screen>
      <Text variant="display" style={{ marginBottom: spacing.sm }}>
        History
      </Text>
      <Text variant="bodySm" muted style={{ marginBottom: spacing.lg }}>
        {scans.length} total scans
      </Text>

      <FlatList
        horizontal
        showsHorizontalScrollIndicator={false}
        data={["all", ...CLASSES] as const}
        keyExtractor={(x) => x}
        contentContainerStyle={{ gap: spacing.sm, paddingBottom: spacing.md }}
        renderItem={({ item }) => {
          const active = filter === item;
          return (
            <Pressable
              onPress={() => setFilter(item as any)}
              style={{
                paddingHorizontal: spacing.md,
                paddingVertical: 8,
                borderRadius: radius.pill,
                backgroundColor: active ? colors.primary : colors.surface,
                borderColor: colors.border,
                borderWidth: 1,
              }}
            >
              <Text
                variant="bodySm"
                weight="600"
                color={active ? colors.primaryContrast : colors.text}
                style={{ textTransform: "capitalize" }}
              >
                {item}
              </Text>
            </Pressable>
          );
        }}
      />

      <FlatList
        data={filtered}
        keyExtractor={(x) => x.id}
        contentContainerStyle={{ paddingVertical: spacing.md }}
        ItemSeparatorComponent={() => <View style={{ height: spacing.sm }} />}
        ListEmptyComponent={
          <Card>
            <Text variant="h3">No scans match this filter</Text>
          </Card>
        }
        renderItem={({ item }) => (
          <Pressable onPress={() => router.push({ pathname: "/result", params: { id: item.id } })}>
            <Card style={{ flexDirection: "row", alignItems: "center", gap: spacing.md }}>
              <Text style={{ fontSize: 28 }}>{EMOJI[item.topClass]}</Text>
              <View style={{ flex: 1 }}>
                <Text variant="h3" style={{ textTransform: "capitalize" }}>
                  {item.topClass}
                </Text>
                <Text variant="bodySm" muted>
                  {BIN_LABEL[BIN_FOR[item.topClass]]} ·{" "}
                  {new Date(item.createdAt).toLocaleString()}
                </Text>
              </View>
              <View style={{ alignItems: "flex-end" }}>
                <Text variant="bodySm" weight="700" color={colors.primary}>
                  +{item.pointsEarned}
                </Text>
                <Text variant="caption" muted>
                  {(item.topConfidence * 100).toFixed(0)}%
                </Text>
              </View>
            </Card>
          </Pressable>
        )}
      />
    </Screen>
  );
}
