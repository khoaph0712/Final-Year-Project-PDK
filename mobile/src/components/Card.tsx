import React from "react";
import { View, ViewProps, ViewStyle } from "react-native";

import { useTheme } from "@/theme";

type Props = ViewProps & {
  padded?: boolean;
  elevated?: boolean;
  style?: ViewStyle;
};

export function Card({ padded = true, elevated = true, style, children, ...rest }: Props) {
  const { colors, radius, spacing, shadows } = useTheme();
  return (
    <View
      {...rest}
      style={[
        {
          backgroundColor: colors.surface,
          borderRadius: radius.lg,
          borderWidth: 1,
          borderColor: colors.border,
        },
        padded && { padding: spacing.lg },
        elevated && shadows.card,
        style,
      ]}
    >
      {children}
    </View>
  );
}
