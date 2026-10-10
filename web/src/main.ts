import "./style.css";
import { parseRoute } from "./router";
import { renderFirm } from "./views/firm";
import { renderFirms } from "./views/firms";
import { renderOverview } from "./views/overview";

const root = document.querySelector<HTMLElement>("#app")!;

async function route(): Promise<void> {
  const r = parseRoute(location.hash);
  document.querySelectorAll<HTMLAnchorElement>("nav a").forEach((a) => {
    a.classList.toggle("active", a.dataset.view === (r.view === "firm" ? "firms" : r.view));
  });
  root.setAttribute("aria-busy", "true");
  try {
    if (r.view === "firm") await renderFirm(root, r.cik);
    else if (r.view === "firms") await renderFirms(root);
    else await renderOverview(root);
  } catch (err) {
    root.innerHTML = `<section class="card"><h2>Couldn't load that</h2><p class="muted">${String(err)}</p>
      <p><a href="#/">Back to the overview</a></p></section>`;
  } finally {
    root.removeAttribute("aria-busy");
    window.scrollTo(0, 0);
  }
}

window.addEventListener("hashchange", route);
route();
