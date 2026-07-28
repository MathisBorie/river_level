// Démarre le backend « dans le navigateur » (Pyodide dans un web worker) et
// expose window.RIVER_BACKEND, que app.js utilise à la place des appels à Flask.
(function () {
  const worker = new Worker(new URL("pyodide_worker.js", document.currentScript.src));
  let seq = 0;
  const enAttente = {};
  let resoudrePret;
  const pret = new Promise((r) => (resoudrePret = r));

  worker.onmessage = (e) => {
    const d = e.data;
    if (d.type === "pret") {
      resoudrePret();
      // Le moteur est prêt, mais la carte serait vide : on prolonge l'écran de
      // chargement le temps de télécharger les stations (app.js le lèvera quand
      // elles sont prêtes). Filet de sécurité au cas où ça traîne.
      majMessageOverlay("Chargement des stations…");
      setTimeout(masquerOverlay, 20000);
      return;
    }
    if (d.type === "erreur-init") { montrerErreurOverlay(d.error); return; }
    const cb = enAttente[d.id];
    if (!cb) return;
    delete enAttente[d.id];
    if (d.ok) cb.resolve(d.bytes !== undefined ? d.bytes : d.result);
    else cb.reject(new Error(d.error));
  };

  // Export/import de modèle en BINAIRE (octets bruts, pas de base64) — léger en mémoire.
  window.RIVER_EXPORT = async (code) => {
    await pret;
    const id = ++seq;
    return new Promise((resolve, reject) => {
      enAttente[id] = { resolve, reject };
      worker.postMessage({ id, type: "exporter", code });
    });
  };
  window.RIVER_IMPORT = async (arrayBuffer, mode) => {
    await pret;
    const id = ++seq;
    const brut = await new Promise((resolve, reject) => {
      enAttente[id] = { resolve, reject };
      worker.postMessage({ id, type: "importer", buffer: arrayBuffer, mode }, [arrayBuffer]);
    });
    const data = JSON.parse(brut);
    if (data && data.erreur) throw new Error(data.erreur);
    return data;
  };

  // Configure le pool de calcul parallèle (0 = désactivé). Envoyé au worker qui
  // gère les sous-workers ; sans effet tant que Python ne le lit pas.
  window.RIVER_CONFIG_POOL = (taille) => worker.postMessage({ type: "config-pool", taille: taille | 0 });

  window.RIVER_BACKEND = async (method, path, body) => {
    await pret;
    const id = ++seq;
    const brut = await new Promise((resolve, reject) => {
      enAttente[id] = { resolve, reject };
      worker.postMessage({ id, method, path, body });
    });
    const data = JSON.parse(brut);
    if (data && data.erreur) throw new Error(data.erreur);
    return data;
  };

  function masquerOverlay() {
    const o = document.getElementById("overlay-chargement");
    if (o) o.classList.add("fini");
  }
  function majMessageOverlay(txt) {
    const o = document.getElementById("overlay-chargement");
    const m = o && o.querySelector(".msg");
    if (m) m.textContent = txt;
  }
  function montrerErreurOverlay(msg) {
    const o = document.getElementById("overlay-chargement");
    if (o) o.querySelector(".msg").textContent = "Erreur au démarrage : " + msg;
  }
})();
