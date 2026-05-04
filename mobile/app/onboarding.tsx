import React, { useRef, useState } from "react";
import {
  Dimensions,
  FlatList,
  NativeScrollEvent,
  NativeSyntheticEvent,
  View,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { MotiView } from "moti";
import { router } from "expo-router";

import { Screen } from "@/components/Screen";
import { Text } from "@/components/Text";
import { Button } from "@/components/Button";
import { useTheme } from "@/theme";
import { useAppStore } from "@/store/useAppStore";

const { width } = Dimensions.get("window");

const slides = [
  {
    emoji: "♻️",
    title: "Sort smarter.",
    desc: "Point your camera at any item. WasteWise instantly tells you which bin it belongs in — on-device, fully offline.",
  },
  {
    emoji: "📸",
    title: "AI in your pocket.",
    desc: "A YOLOv8n model trained on 15k+ waste images classifies plastic, glass, metal, paper, cardboard, organic and more.",
  },
  {
    emoji: "🌱",
    title: "Earn eco-points.",
    desc: "Every correct scan counts. Build streaks, level up, and climb the leaderboard with friends.",
  },
];

export default function Onboarding() {
  const { colors, spacing, radius } = useTheme();
  const setOnboarded = useAppStore((s) => s.setOnboarded);
  const [index, setIndex] = useState(0);
  const ref = useRef<FlatList>(null);

  const onScroll = (e: NativeSyntheticEvent<NativeScrollEvent>) => {
    const i = Math.round(e.nativeEvent.contentOffset.x / width);
    if (i !== index) setIndex(i);
  };

  const next = () => {
    if (index < slides.length - 1) {
      ref.current?.scrollToIndex({ index: index + 1, animated: true });
    } else {
      setOnboarded(true);
      router.replace("/(tabs)/home");
    }
  };

  return (
    <Screen padded={false}>
      <LinearGradient
        colors={[colors.primary, colors.accent]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={{ position: "absolute", inset: 0 as any, opacity: 0.12 }}
      />
      <FlatList
        ref={ref}
        data={slides}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        onScroll={onScroll}
        scrollEventThrottle={16}
        keyExtractor={(item) => item.title}
        renderItem={({ item, index: i }) => (
          <View style={{ width, paddingHorizontal: spacing.xl, paddingTop: spacing.xxxl * 2 }}>
            <MotiView
              from={{ opacity: 0, translateY: 10 }}
              animate={{ opacity: 1, translateY: 0 }}
              transition={{ type: "timing", duration: 500, delay: i * 60 }}
            >
              <Text style={{ fontSize: 96, marginBottom: spacing.xl }}>{item.emoji}</Text>
              <Text variant="display" style={{ marginBottom: spacing.md }}>
                {item.title}
              </Text>
              <Text variant="body" muted>
                {item.desc}
              </Text>
            </MotiView>
          </View>
        )}
      />

      <View
        style={{
          position: "absolute",
          bottom: 48,
          left: 0,
          right: 0,
          paddingHorizontal: spacing.xl,
        }}
      >
        <View style={{ flexDirection: "row", gap: spacing.sm, marginBottom: spacing.xl }}>
          {slides.map((_, i) => (
            <View
              key={i}
              style={{
                height: 6,
                flex: 1,
                backgroundColor: i === index ? colors.primary : colors.border,
                borderRadius: radius.pill,
              }}
            />
          ))}
        </View>
        <Button
          title={index === slides.length - 1 ? "Get started" : "Next"}
          fullWidth
          onPress={next}
        />
      </View>
    </Screen>
  );
}
