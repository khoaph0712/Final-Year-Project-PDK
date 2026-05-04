import React, { useEffect, useState } from "react";
import { ScrollView, View } from "react-native";
import * as Location from "expo-location";
import { router } from "expo-router";

import { Screen } from "@/components/Screen";
import { Text } from "@/components/Text";
import { Card } from "@/components/Card";
import { Button } from "@/components/Button";
import { useTheme } from "@/theme";
import { BIN_LABEL } from "@/lib/classes";

type Bin = {
  id: string;
  name: string;
  type: "recycling" | "compost" | "general";
  lat: number;
  lon: number;
  note: string;
};

// Mock bins — swap for a real API call (e.g. municipal open-data) later.
const MOCK_BINS: Bin[] = [
  { id: "1", name: "Main Street Recycling Hub", type: "recycling", lat: 0, lon: 0, note: "Plastic, glass, metal, paper" },
  { id: "2", name: "Community Compost Point", type: "compost", lat: 0, lon: 0, note: "Food & garden waste" },
  { id: "3", name: "Park General Bin", type: "general", lat: 0, lon: 0, note: "Non-recyclables only" },
];

function haversine(a: { lat: number; lon: number }, b: { lat: number; lon: number }) {
  const R = 6371;
  const dLat = ((b.lat - a.lat) * Math.PI) / 180;
  const dLon = ((b.lon - a.lon) * Math.PI) / 180;
  const s =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((a.lat * Math.PI) / 180) *
      Math.cos((b.lat * Math.PI) / 180) *
      Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.asin(Math.sqrt(s));
}

export default function BinLocator() {
  const { colors, spacing, radius } = useTheme();
  const [coords, setCoords] = useState<{ lat: number; lon: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== "granted") {
        setError("Location permission denied — showing mock bins.");
        return;
      }
      const loc = await Location.getCurrentPositionAsync({});
      setCoords({ lat: loc.coords.latitude, lon: loc.coords.longitude });
      MOCK_BINS.forEach((b, i) => {
        b.lat = loc.coords.latitude + (i + 1) * 0.001;
        b.lon = loc.coords.longitude + (i + 1) * 0.001;
      });
    })();
  }, []);

  const bins = coords
    ? [...MOCK_BINS].sort(
        (a, b) => haversine(coords, a) - haversine(coords, b),
      )
    : MOCK_BINS;

  return (
    <Screen>
      <View style={{ flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: spacing.lg }}>
        <Text variant="display">Bins near you</Text>
        <Button title="Back" variant="ghost" onPress={() => router.back()} />
      </View>

      {error ? (
        <Card style={{ marginBottom: spacing.md }}>
          <Text variant="bodySm" muted>
            {error}
          </Text>
        </Card>
      ) : null}

      <ScrollView showsVerticalScrollIndicator={false}>
        {bins.map((b) => (
          <Card
            key={b.id}
            style={{
              marginBottom: spacing.md,
              flexDirection: "row",
              alignItems: "center",
              gap: spacing.md,
            }}
          >
            <View
              style={{
                width: 48,
                height: 48,
                borderRadius: radius.md,
                backgroundColor: colors.bin[b.type],
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Text color="#fff" variant="h2">
                ♻︎
              </Text>
            </View>
            <View style={{ flex: 1 }}>
              <Text variant="h3">{b.name}</Text>
              <Text variant="bodySm" muted>
                {BIN_LABEL[b.type]} · {b.note}
              </Text>
              {coords ? (
                <Text variant="caption" muted style={{ marginTop: 2 }}>
                  ≈ {(haversine(coords, b) * 1000).toFixed(0)} m away
                </Text>
              ) : null}
            </View>
          </Card>
        ))}
      </ScrollView>
    </Screen>
  );
}
