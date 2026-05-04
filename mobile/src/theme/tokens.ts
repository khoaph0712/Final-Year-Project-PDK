export const palette = {
  teal50: "#F0FDFA",
  teal100: "#CCFBF1",
  teal200: "#99F6E4",
  teal400: "#2DD4BF",
  teal500: "#14B8A6",
  teal600: "#0D9488",
  teal700: "#0F766E",
  teal800: "#115E59",
  teal900: "#134E4A",

  lime400: "#A3E635",
  lime500: "#84CC16",
  amber400: "#FBBF24",
  rose500: "#F43F5E",
  sky500: "#0EA5E9",
  violet500: "#8B5CF6",
  slate950: "#0B1220",

  gray50: "#F8FAFC",
  gray100: "#F1F5F9",
  gray200: "#E2E8F0",
  gray300: "#CBD5E1",
  gray400: "#94A3B8",
  gray500: "#64748B",
  gray600: "#475569",
  gray700: "#334155",
  gray800: "#1E293B",
  gray900: "#0F172A",
};

export const binColors = {
  recycling: palette.teal500,
  compost: palette.lime500,
  general: palette.gray500,
  hazardous: palette.rose500,
} as const;

export type BinKey = keyof typeof binColors;

export const radius = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  xxl: 32,
  pill: 999,
};

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  xxl: 32,
  xxxl: 48,
};

export const typography = {
  display: { fontSize: 34, lineHeight: 40, fontWeight: "800" as const, letterSpacing: -0.5 },
  h1: { fontSize: 28, lineHeight: 34, fontWeight: "800" as const, letterSpacing: -0.4 },
  h2: { fontSize: 22, lineHeight: 28, fontWeight: "700" as const, letterSpacing: -0.2 },
  h3: { fontSize: 18, lineHeight: 24, fontWeight: "700" as const },
  body: { fontSize: 16, lineHeight: 22, fontWeight: "500" as const },
  bodySm: { fontSize: 14, lineHeight: 20, fontWeight: "500" as const },
  caption: { fontSize: 12, lineHeight: 16, fontWeight: "600" as const, letterSpacing: 0.3 },
};

export const shadows = {
  card: {
    shadowColor: "#0F172A",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.08,
    shadowRadius: 16,
    elevation: 4,
  },
  float: {
    shadowColor: "#0F172A",
    shadowOffset: { width: 0, height: 12 },
    shadowOpacity: 0.18,
    shadowRadius: 24,
    elevation: 10,
  },
};
