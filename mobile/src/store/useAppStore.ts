import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { nanoid } from "nanoid/non-secure";

import type { WasteClass } from "@/lib/classes";
import { POINTS } from "@/lib/classes";

export type Detection = {
  cls: WasteClass;
  confidence: number;
  box: [number, number, number, number];
};

export type ScanRecord = {
  id: string;
  createdAt: number;
  topClass: WasteClass;
  topConfidence: number;
  detections: Detection[];
  imageUri?: string;
  pointsEarned: number;
};

type Settings = {
  useFloat16: boolean;
  confThreshold: number;
  iouThreshold: number;
  hapticsEnabled: boolean;
};

type AppState = {
  onboarded: boolean;
  setOnboarded: (v: boolean) => void;
  scans: ScanRecord[];
  addScan: (s: Omit<ScanRecord, "id" | "createdAt" | "pointsEarned">) => ScanRecord;
  clearScans: () => void;
  ecoPoints: number;
  level: number;
  streakDays: number;
  lastScanDay: string | null;
  settings: Settings;
  setSetting: <K extends keyof Settings>(key: K, value: Settings[K]) => void;
};

const dayKey = (ts: number) => new Date(ts).toISOString().slice(0, 10);
const levelFromPoints = (p: number) => Math.max(1, Math.floor(Math.sqrt(p / 20)) + 1);

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      onboarded: false,
      setOnboarded: (v) => set({ onboarded: v }),

      scans: [],
      addScan: (s) => {
        const id = nanoid();
        const createdAt = Date.now();
        const pointsEarned = POINTS[s.topClass] ?? 5;
        const record: ScanRecord = { id, createdAt, pointsEarned, ...s };

        const today = dayKey(createdAt);
        const last = get().lastScanDay;
        const yesterday = dayKey(createdAt - 86400000);
        let streak = get().streakDays;
        if (last === today) {
          // same day, streak unchanged
        } else if (last === yesterday) {
          streak += 1;
        } else {
          streak = 1;
        }

        const nextPoints = get().ecoPoints + pointsEarned;
        set({
          scans: [record, ...get().scans].slice(0, 200),
          ecoPoints: nextPoints,
          level: levelFromPoints(nextPoints),
          streakDays: streak,
          lastScanDay: today,
        });
        return record;
      },
      clearScans: () =>
        set({ scans: [], ecoPoints: 0, level: 1, streakDays: 0, lastScanDay: null }),

      ecoPoints: 0,
      level: 1,
      streakDays: 0,
      lastScanDay: null,

      settings: {
        useFloat16: true,
        confThreshold: 0.35,
        iouThreshold: 0.45,
        hapticsEnabled: true,
      },
      setSetting: (key, value) =>
        set({ settings: { ...get().settings, [key]: value } }),
    }),
    {
      name: "wastewise-store",
      storage: createJSONStorage(() => AsyncStorage),
      version: 1,
    },
  ),
);
