import React from "react";
import { Text as RNText, TextProps as RNTextProps, TextStyle } from "react-native";

import { useTheme } from "@/theme";

type Variant = "display" | "h1" | "h2" | "h3" | "body" | "bodySm" | "caption";

type Props = RNTextProps & {
  variant?: Variant;
  muted?: boolean;
  color?: string;
  align?: TextStyle["textAlign"];
  weight?: TextStyle["fontWeight"];
};

export function Text({
  variant = "body",
  muted,
  color,
  align,
  weight,
  style,
  ...rest
}: Props) {
  const { colors, typography } = useTheme();
  const base = typography[variant];
  return (
    <RNText
      {...rest}
      style={[
        base,
        { color: color ?? (muted ? colors.textMuted : colors.text) },
        align ? { textAlign: align } : null,
        weight ? { fontWeight: weight } : null,
        style,
      ]}
    />
  );
}
