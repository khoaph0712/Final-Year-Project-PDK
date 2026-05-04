import type { BinKey } from "@/theme/tokens";

export const CLASSES = [
  "plastic",
  "glass",
  "metal",
  "paper",
  "cardboard",
  "organic",
  "other",
] as const;

export type WasteClass = (typeof CLASSES)[number];

export const BIN_FOR: Record<WasteClass, BinKey> = {
  plastic: "recycling",
  glass: "recycling",
  metal: "recycling",
  paper: "recycling",
  cardboard: "recycling",
  organic: "compost",
  other: "general",
};

export const BIN_LABEL: Record<BinKey, string> = {
  recycling: "Recycling",
  compost: "Compost",
  general: "General waste",
  hazardous: "Hazardous",
};

export const TIPS: Record<WasteClass, string> = {
  plastic: "Rinse bottles and remove caps. Most PET and HDPE plastics are recyclable.",
  glass: "Remove lids, rinse, and place whole (unbroken) glass in the recycling bin.",
  metal: "Empty, rinse, and crush aluminium cans to save space. Avoid mixing with food.",
  paper: "Keep paper dry. Remove plastic windows from envelopes before recycling.",
  cardboard: "Flatten boxes, remove tape, and keep dry for clean recycling.",
  organic: "Compostable. Includes food scraps, coffee grounds, and yard waste.",
  other: "Unclear category — check your local council guidelines before disposing.",
};

export const POINTS: Record<WasteClass, number> = {
  plastic: 10,
  glass: 10,
  metal: 12,
  paper: 8,
  cardboard: 8,
  organic: 6,
  other: 3,
};

export const EMOJI: Record<WasteClass, string> = {
  plastic: "🧴",
  glass: "🫙",
  metal: "🥫",
  paper: "📄",
  cardboard: "📦",
  organic: "🍎",
  other: "🗑️",
};
