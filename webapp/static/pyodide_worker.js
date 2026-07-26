// Web worker : fait tourner le backend Python (river_web.py) dans un thread à
// part via Pyodide, pour que l'interface reste fluide pendant les calculs.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

let pyodide = null;
let traiterFn = null; // proxy de la fonction Python traiter(), appelée directement
let besoinSyncFn = null;

async function init() {
  pyodide = await loadPyodide();
  await pyodide.loadPackage(["numpy", "pandas", "scikit-learn"]);
  // « Disque » persistant du navigateur (IndexedDB) monté sur /persist :
  // les modèles + points survivent d'une visite à l'autre.
  try {
    pyodide.FS.mkdir("/persist");
    pyodide.FS.mount(pyodide.FS.filesystems.IDBFS, {}, "/persist");
    await new Promise((res) => pyodide.FS.syncfs(true, () => res()));  // charge l'existant
  } catch (e) {
    console.warn("[worker] stockage persistant indisponible :", e);
  }
  const urlPy = new URL("river_web.py?v=" + Date.now(), self.location.href);
  const src = await (await fetch(urlPy)).text();
  await pyodide.runPythonAsync(src);   // river_web.py recharge les stations sauvegardées
  traiterFn = pyodide.globals.get("traiter");
  besoinSyncFn = pyodide.globals.get("besoin_sync");
  postMessage({ type: "pret" });
}

function graverSiBesoin() {
  try {
    if (besoinSyncFn && besoinSyncFn()) pyodide.FS.syncfs(false, () => {});
  } catch (e) { /* pas bloquant */ }
}
const initProm = init().catch((e) => postMessage({ type: "erreur-init", error: String(e) }));

onmessage = async (e) => {
  const { id, type } = e.data;
  // --- chemins BINAIRES (export/import de modèle) : octets bruts, transférés
  //     sans copie, pour ne pas saturer la mémoire (téléphones). ---
  if (type === "exporter" || type === "importer") {
    try {
      await initProm;
      if (type === "exporter") {
        const fn = pyodide.globals.get("exporter_bytes");
        const proxy = fn(e.data.code);
        fn.destroy();
        const u8 = proxy.toJs ? proxy.toJs() : proxy;
        if (proxy.destroy) proxy.destroy();
        postMessage({ id, ok: true, bytes: u8 }, [u8.buffer]);
      } else {
        const fn = pyodide.globals.get("importer_bytes");
        const res = await fn(new Uint8Array(e.data.buffer), e.data.mode || "demander");
        fn.destroy();
        postMessage({ id, ok: true, result: res });   // res = chaîne JSON
        graverSiBesoin();
      }
    } catch (err) {
      postMessage({ id, ok: false, error: String((err && err.message) || err) });
    }
    return;
  }

  const { method, path, body } = e.data;
  try {
    await initProm;
    console.log("[worker] →", method, path);
    // On passe les args à l'appel : chaque requête a ses propres valeurs
    // (plus de globals partagés qui se marchaient dessus quand 2 appels se croisaient).
    const res = await traiterFn(method, path, body ? JSON.stringify(body) : null);
    console.log("[worker] ←", method, path, "(" + String(res).length + " o)");
    postMessage({ id, ok: true, result: res });
    graverSiBesoin();   // grave dans IndexedDB si une station a été sauvegardée
  } catch (err) {
    console.error("[worker] échec", method, path, err);
    postMessage({ id, ok: false, error: String((err && err.message) || err) });
  }
};
