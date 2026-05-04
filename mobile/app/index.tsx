import { Redirect } from "expo-router";

import { useAppStore } from "@/store/useAppStore";

export default function Index() {
  const onboarded = useAppStore((s) => s.onboarded);
  return onboarded ? <Redirect href="/(tabs)/home" /> : <Redirect href="/onboarding" />;
}
