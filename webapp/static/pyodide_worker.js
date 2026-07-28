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

// ---- pool Monte Carlo : parallélisme CPU optionnel (plusieurs sous-workers) ----
// Chaque sous-worker (mc_worker.js) a sa propre instance Pyodide. Exposé à Python
// via self.mcPoolActif()/self.entrainerMCParallele() ; river_web.py replie en
// séquentiel si mcPoolActif() < 2 ou si un appel échoue.
let mcTaille = 0, mcPool = [], mcSeq = 0;
const mcCb = {};
self.mcPoolActif = () => mcTaille;
self.mcConfigPool = (t) => { mcTaille = (t | 0); };

function mcSpawn() {
  while (mcPool.length < mcTaille) {
    const w = new Worker(new URL("mc_worker.js", self.location.href));
    w.onmessage = (e) => {
      const d = e.data;
      if (d.type === "pret" || d.type === "erreur") return;
      const cb = mcCb[d.id]; if (!cb) return; delete mcCb[d.id];
      d.ok ? cb.res(d.modeles) : cb.rej(new Error(d.error));
    };
    mcPool.push(w);
  }
}
function mcTache(w, xb, yb, seeds) {
  const id = ++mcSeq;
  return new Promise((res, rej) => {
    mcCb[id] = { res, rej };
    w.postMessage({ id, xb, yb, seeds });   // clone (pas de transfert : mêmes données pour tous les workers)
  });
}
// Appelée depuis Python : renvoie une liste (aplatie, dans l'ordre) de modèles (Uint8Array).
self.entrainerMCParallele = async (xb, yb, n) => {
  if (mcTaille < 2) throw new Error("pool inactif");
  mcSpawn();
  const k = Math.min(mcTaille, n);
  const paquets = Array.from({ length: k }, () => []);
  for (let i = 0; i < n; i++) paquets[i % k].push(i);     // répartit les membres sur les workers
  const parPaquet = await Promise.all(paquets.map((seeds, j) => mcTache(mcPool[j], xb, yb, seeds)));
  const out = new Array(n);
  paquets.forEach((seeds, j) => seeds.forEach((idx, t) => { out[idx] = parPaquet[j][t]; }));
  return out;
};

onmessage = async (e) => {
  const { id, type } = e.data;
  // --- configuration du pool parallèle (taille décidée par l'appareil + réglage) ---
  if (type === "config-pool") { self.mcConfigPool(e.data.taille); return; }
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
