export type Route = { view: "overview" } | { view: "firms" } | { view: "firm"; cik: number };

export function parseRoute(hash: string): Route {
  const firm = /^#\/firm\/(\d+)$/.exec(hash);
  if (firm) return { view: "firm", cik: Number(firm[1]) };
  if (hash === "#/firms") return { view: "firms" };
  return { view: "overview" };
}
