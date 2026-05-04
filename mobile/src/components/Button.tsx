import React from "react";
import { ActivityIndicator, Pressable, PressableProps, View, ViewStyle } from "react-native";
import * as Haptics from "expo-haptics";

import { useTheme } from "@/theme";
import { Text } from "./Text";

type Variant = "primary" | "secondary" | "ghost" | "danger";

type Props = Omit<PressableProps, "children"> & {
  title: string;
  variant?: Variant;
  loading?: boolean;
  fullWidth?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  style?: ViewStyle;
};

export function Button({
  title,
  variant = "primary",
  loading,
  fullWidth,
  leftIcon,
  rightIcon,
  style,
  onPress,
  disabled,
  ...rest
}: Props) {
  const { colors, radius, spacing } = useTheme();

  const bg =
    variant === "primary"
      ? colors.primary
      : variant === "secondary"
        ? colors.primaryMuted
        : variant === "danger"
          ? colors.danger
          : "transparent";

  const fg =
    variant === "primary" || variant === "danger"
      ? colors.primaryContrast
      : variant === "secondary"
        ? colors.primary
        : colors.text;

  return (
    <Pressable
      {...rest}
      disabled={disabled || loading}
      onPress={(e) => {
        Haptics.selectionAsync().catch(() => {});
        onPress?.(e);
      }}
      style={({ pressed }) => [
        {
          backgroundColor: bg,
          opacity: disabled ? 0.5 : pressed ? 0.9 : 1,
          borderRadius: radius.pill,
          paddingVertical: spacing.md,
          paddingHorizontal: spacing.xl,
          alignSelf: fullWidth ? "stretch" : "flex-start",
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "row",
          gap: spacing.sm,
          borderWidth: variant === "ghost" ? 1 : 0,
          borderColor: colors.border,
        },
        style,
      ]}
    >
      {loading ? (
        <ActivityIndicator color={fg} />
      ) : (
        <>
          {leftIcon ? <View>{leftIcon}</View> : null}
          <Text weight="700" color={fg}>
            {title}
          </Text>
          {rightIcon ? <View>{rightIcon}</View> : null}
        </>
      )}
    </Pressable>
  );
}
