import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { useColorScheme } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";

import { palette, radius, shadows, spacing, typography, binColors } from "./tokens";

type ThemeMode = "light" | "dark" | "system";

export type ThemeColors = {
  background: string;
  surface: string;
  surfaceElevated: string;
  border: string;
  text: string;
  textMuted: string;
  textInverse: string;
  primary: string;
  primaryMuted: string;
  primaryContrast: string;
  accent: string;
  success: string;
  warning: string;
  danger: string;
  bin: typeof binColors;
};

const lightColors: ThemeColors = {
  background: palette.gray50,
  surface: "#FFFFFF",
  surfaceElevated: "#FFFFFF",
  border: palette.gray200,
  text: palette.gray900,
  textMuted: palette.gray500,
  textInverse: "#FFFFFF",
  primary: palette.teal600,
  primaryMuted: palette.teal100,
  primaryContrast: "#FFFFFF",
  accent: palette.lime500,
  success: palette.teal500,
  warning: palette.amber400,
  danger: palette.rose500,
  bin: binColors,
};

const darkColors: ThemeColors = {
  background: palette.slate950,
  surface: palette.gray900,
  surfaceElevated: palette.gray800,
  border: "#1F2937",
  text: "#F8FAFC",
  textMuted: palette.gray400,
  textInverse: palette.gray900,
  primary: palette.teal400,
  primaryMuted: palette.teal900,
  primaryContrast: palette.gray900,
  accent: palette.lime400,
  success: palette.teal400,
  warning: palette.amber400,
  danger: palette.rose500,
  bin: binColors,
};

export type Theme = {
  colors: ThemeColors;
  spacing: typeof spacing;
  radius: typeof radius;
  shadows: typeof shadows;
  typography: typeof typography;
  mode: "light" | "dark";
  setMode: (mode: ThemeMode) => void;
  preferred: ThemeMode;
};

const ThemeContext = createContext<Theme | null>(null);
const STORAGE_KEY = "wastewise.theme_mode";

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const systemScheme = useColorScheme();
  const [preferred, setPreferred] = useState<ThemeMode>("system");

  useEffect(() => {
    AsyncStorage.getItem(STORAGE_KEY).then((saved) => {
      if (saved === "light" || saved === "dark" || saved === "system") setPreferred(saved);
    });
  }, []);

  const resolved: "light" | "dark" =
    preferred === "system" ? (systemScheme === "dark" ? "dark" : "light") : preferred;

  const setMode = (mode: ThemeMode) => {
    setPreferred(mode);
    AsyncStorage.setItem(STORAGE_KEY, mode).catch(() => {});
  };

  const value = useMemo<Theme>(
    () => ({
      colors: resolved === "dark" ? darkColors : lightColors,
      spacing,
      radius,
      shadows,
      typography,
      mode: resolved,
      setMode,
      preferred,
    }),
    [resolved, preferred],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): Theme {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within ThemeProvider");
  return ctx;
}
