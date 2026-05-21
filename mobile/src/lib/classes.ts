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

export const DISPOSAL_INSTRUCTIONS: Record<WasteClass, string> = {
  plastic: "🟢 Place in the GREEN Recycling Bin\nMake sure to empty, rinse, and squeeze the plastic bottle to conserve space.",
  glass: "🔵 Place in the BLUE Glass Bin\nRemove metal caps, rinse thoroughly, and place carefully to prevent shattering.",
  metal: "🔴 Place in the RED Metal Bin\nRinse out residues from aluminum/tin cans and flatten them completely.",
  paper: "🟡 Place in the YELLOW Paper Bin\nEnsure the paper is clean and dry. Avoid greasy boxes or shredded fragments.",
  cardboard: "📦 Place in the CARDBOARD Bin\nBreak down and flatten the cardboard shipping boxes before sliding them in.",
  organic: "🟤 Place in the BROWN Compost Bin\nDispose of fruit/vegetable scraps and organic waste here. No plastic wrappers!",
  other: "🗑️ Place in the BLACK General Waste Bin\nPlace miscellaneous items and non-recyclable composite materials here.",
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
