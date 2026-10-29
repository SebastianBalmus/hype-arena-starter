import { useSyncExternalStore } from "react";

export function create<T>(initial: T) {
  let v = initial;
  const subs = new Set<() => void>();
  return {
    get: () => v,
    set: (n: T) => { v = n; subs.forEach((s) => s()); },
    subscribe: (s: () => void) => { subs.add(s); return () => subs.delete(s); },
  };
}

export function useStore<T>(s: ReturnType<typeof create<T>>) {
  return useSyncExternalStore(s.subscribe, s.get, s.get);
}
