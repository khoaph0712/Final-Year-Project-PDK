import React from "react";
import { StyleSheet, View, ViewStyle, Dimensions } from "react-native";
import { SafeAreaView, useSafeAreaInsets } from "react-native-safe-area-context";
import { StatusBar } from "expo-status-bar";
import { LinearGradient } from "expo-linear-gradient";

import { useTheme } from "@/theme";

type Props = {
  children: React.ReactNode;
  style?: ViewStyle;
  edges?: ("top" | "bottom" | "left" | "right")[];
  padded?: boolean;
};

const { width, height } = Dimensions.get("window");

export function Screen({ children, style, edges = ["top", "bottom"], padded = true }: Props) {
  const { colors, spacing, mode } = useTheme();
  const insets = useSafeAreaInsets();
  
  const isDark = mode === "dark";

  return (
    <View style={[{ flex: 1, backgroundColor: isDark ? "#080c14" : "#f1f5f9" }, style]}>
      <StatusBar style={isDark ? "light" : "dark"} />
      
      {/* Decorative Liquid Glass Glowing Background Circles */}
      <View style={[StyleSheet.absoluteFillObject, { overflow: "hidden" }]}>
        {/* Glow Circle 1: Top-Right (Teal/Cyan) */}
        <LinearGradient
          colors={isDark ? ["rgba(13, 148, 136, 0.22)", "rgba(13, 148, 136, 0.0)"] : ["rgba(13, 148, 136, 0.12)", "rgba(13, 148, 136, 0.0)"]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={[styles.glowBall, {
            top: -height * 0.1,
            right: -width * 0.2,
            width: width * 0.9,
            height: width * 0.9,
            borderRadius: width * 0.45,
          }]}
        />

        {/* Glow Circle 2: Middle-Left (Lime/Green) */}
        <LinearGradient
          colors={isDark ? ["rgba(132, 204, 22, 0.18)", "rgba(132, 204, 22, 0.0)"] : ["rgba(132, 204, 22, 0.10)", "rgba(132, 204, 22, 0.0)"]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={[styles.glowBall, {
            top: height * 0.35,
            left: -width * 0.3,
            width: width * 0.8,
            height: width * 0.8,
            borderRadius: width * 0.4,
          }]}
        />

        {/* Glow Circle 3: Bottom-Right (Rose/Coral) */}
        <LinearGradient
          colors={isDark ? ["rgba(244, 63, 94, 0.16)", "rgba(244, 63, 94, 0.0)"] : ["rgba(244, 63, 94, 0.08)", "rgba(244, 63, 94, 0.0)"]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={[styles.glowBall, {
            bottom: -height * 0.1,
            right: -width * 0.15,
            width: width * 0.85,
            height: width * 0.85,
            borderRadius: width * 0.425,
          }]}
        />
      </View>

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
  inner: { flex: 1, zIndex: 1 },
  glowBall: {
    position: "absolute",
    pointerEvents: "none",
  }
});
