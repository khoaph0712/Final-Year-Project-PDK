import React from "react";
import { View, ViewProps, ViewStyle } from "react-native";

import { useTheme } from "@/theme";

type Props = ViewProps & {
  padded?: boolean;
  elevated?: boolean;
  style?: ViewStyle;
};

export function Card({ padded = true, elevated = true, style, children, ...rest }: Props) {
  const { colors, radius, spacing, shadows, mode } = useTheme();
  
  const isDark = mode === "dark";

  return (
    <View
      {...rest}
      style={[
        {
          // Frosted Glassmorphism Background Color
          backgroundColor: isDark ? "rgba(30, 41, 59, 0.45)" : "rgba(255, 255, 255, 0.65)",
          borderRadius: radius.xl, // Premium larger rounded corners
          borderWidth: 1.5,
          // Glowing translucent borders
          borderColor: isDark ? "rgba(255, 255, 255, 0.12)" : "rgba(255, 255, 255, 0.45)",
          overflow: "hidden",
        },
        padded && { padding: spacing.lg },
        elevated && {
          // Deep soft shadow to complement glass depth
          shadowColor: isDark ? "#020617" : "#64748b",
          shadowOffset: { width: 0, height: 10 },
          shadowOpacity: isDark ? 0.35 : 0.08,
          shadowRadius: 20,
          elevation: 5,
        },
        style,
      ]}
    >
      {children}
    </View>
  );
}
