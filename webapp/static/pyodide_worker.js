// Web worker : fait tourner le backend Python (river_web.py) dans un thread à
// part via Pyodide, pour que l'interface reste fluide pendant les calculs.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

let pyodide = null;
let traiterFn = null; // proxy de la fonction Python traiter(), appelée directement

async function init() {
  pyodide = await loadPyodide();
  await pyodide.loadPackage(["numpy", "pandas", "scikit-learn"]);
  const urlPy = new URL("river_web.py?v=" + Date.now(), self.location.href);
  const src = await (await fetch(urlPy)).text();
  await pyodide.runPythonAsync(src);
  traiterFn = pyodide.globals.get("traiter");
  postMessage({ type: "pret" });
}
const initProm = init().catch((e) => postMessage({ type: "erreur-init", error: String(e) }));

onmessage = async (e) => {
  const { id, method, path, body } = e.data;
  try {
    await initProm;
    console.log("[worker] →", method, path);
    // On passe les args à l'appel : chaque requête a ses propres valeurs
    // (plus de globals partagés qui se marchaient dessus quand 2 appels se croisaient).
    const res = await traiterFn(method, path, body ? JSON.stringify(body) : null);
    console.log("[worker] ←", method, path, "(" + String(res).length + " o)");
    postMessage({ id, ok: true, result: res });
  } catch (err) {
    console.error("[worker] échec", method, path, err);
    postMessage({ id, ok: false, error: String((err && err.message) || err) });
  }
};
