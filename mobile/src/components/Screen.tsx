import React from "react";
import { StyleSheet, View, ViewStyle } from "react-native";
import { SafeAreaView, useSafeAreaInsets } from "react-native-safe-area-context";
import { StatusBar } from "expo-status-bar";

import { useTheme } from "@/theme";

type Props = {
  children: React.ReactNode;
  style?: ViewStyle;
  edges?: ("top" | "bottom" | "left" | "right")[];
  padded?: boolean;
};

export function Screen({ children, style, edges = ["top", "bottom"], padded = true }: Props) {
  const { colors, spacing, mode } = useTheme();
  const insets = useSafeAreaInsets();
  return (
    <View style={[{ flex: 1, backgroundColor: colors.background }, style]}>
      <StatusBar style={mode === "dark" ? "light" : "dark"} />
      <SafeAreaView
        style={[
          styles.inner,
          padded && {
            paddingHorizontal: spacing.lg,
            paddingTop: edges.includes("top") ? 0 : insets.top,
            paddingBottom: edges.includes("bottom") ? 0 : insets.bottom,
          },
        ]}
        edges={edges}
      >
        {children}
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  inner: { flex: 1 },
});
