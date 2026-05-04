import React from "react";
import { Tabs } from "expo-router";
import { View } from "react-native";

import { useTheme } from "@/theme";
import { Text } from "@/components/Text";

function TabIcon({ focused, emoji, label }: { focused: boolean; emoji: string; label: string }) {
  const { colors, radius, spacing } = useTheme();
  return (
    <View
      style={{
        alignItems: "center",
        justifyContent: "center",
        paddingVertical: 6,
        paddingHorizontal: spacing.md,
        backgroundColor: focused ? colors.primaryMuted : "transparent",
        borderRadius: radius.pill,
        minWidth: 52,
      }}
    >
      <Text style={{ fontSize: 20 }}>{emoji}</Text>
      <Text
        variant="caption"
        color={focused ? colors.primary : colors.textMuted}
        style={{ marginTop: 2 }}
      >
        {label}
      </Text>
    </View>
  );
}

export default function TabLayout() {
  const { colors } = useTheme();
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarShowLabel: false,
        tabBarStyle: {
          backgroundColor: colors.surface,
          borderTopColor: colors.border,
          height: 72,
          paddingTop: 10,
          paddingBottom: 12,
        },
      }}
    >
      <Tabs.Screen
        name="home"
        options={{
          tabBarIcon: ({ focused }) => <TabIcon focused={focused} emoji="🏠" label="Home" />,
        }}
      />
      <Tabs.Screen
        name="scan"
        options={{
          tabBarIcon: ({ focused }) => <TabIcon focused={focused} emoji="📷" label="Scan" />,
        }}
      />
      <Tabs.Screen
        name="history"
        options={{
          tabBarIcon: ({ focused }) => <TabIcon focused={focused} emoji="🗂️" label="History" />,
        }}
      />
      <Tabs.Screen
        name="leaderboard"
        options={{
          tabBarIcon: ({ focused }) => <TabIcon focused={focused} emoji="🏆" label="Ranks" />,
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          tabBarIcon: ({ focused }) => <TabIcon focused={focused} emoji="⚙️" label="Settings" />,
        }}
      />
    </Tabs>
  );
}
