import React from "react";
import { View } from "react-native";
import { MotiView } from "moti";

import { useTheme } from "@/theme";
import { Text } from "./Text";
import type { BinKey } from "@/theme/tokens";

type Props = {
  label: string;
  value: number; // 0..1
  bin?: BinKey;
};

export function ConfidenceBar({ label, value, bin }: Props) {
  const { colors, radius, spacing } = useTheme();
  const pct = Math.max(0, Math.min(1, value));
  const fill = bin ? colors.bin[bin] : colors.primary;
  return (
    <View style={{ marginBottom: spacing.md }}>
      <View style={{ flexDirection: "row", justifyContent: "space-between", marginBottom: 6 }}>
        <Text variant="bodySm" weight="600">
          {label}
        </Text>
        <Text variant="bodySm" muted>
          {(pct * 100).toFixed(0)}%
        </Text>
      </View>
      <View
        style={{
          height: 10,
          backgroundColor: colors.border,
          borderRadius: radius.pill,
          overflow: "hidden",
        }}
      >
        <MotiView
          from={{ width: "0%" as any }}
          animate={{ width: `${pct * 100}%` as any }}
          transition={{ type: "timing", duration: 600 }}
          style={{ height: "100%", backgroundColor: fill, borderRadius: radius.pill }}
        />
      </View>
    </View>
  );
}
