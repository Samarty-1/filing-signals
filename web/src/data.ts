import type { Firm, FirmIndexRow, Summary } from "./types";

const base = import.meta.env.BASE_URL;
const cache = new Map<string, Promise<unknown>>();

function get<T>(path: string): Promise<T> {
  if (!cache.has(path)) {
    cache.set(path, fetch(`${base}data/${path}`).then((r) => {
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
      return r.json();
    }));
  }
  return cache.get(path) as Promise<T>;
}

export const loadSummary = () => get<Summary>("summary.json");
export const loadIndex = () => get<FirmIndexRow[]>("firms.json");
export const loadFirm = (cik: number) => get<Firm>(`firms/${cik}.json`);
