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
    if (d.type === "pret") { resoudrePret(); masquerOverlay(); return; }
    if (d.type === "erreur-init") { montrerErreurOverlay(d.error); return; }
    const cb = enAttente[d.id];
    if (!cb) return;
    delete enAttente[d.id];
    if (d.ok) cb.resolve(d.result); else cb.reject(new Error(d.error));
  };

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
  function montrerErreurOverlay(msg) {
    const o = document.getElementById("overlay-chargement");
    if (o) o.querySelector(".msg").textContent = "Erreur au démarrage : " + msg;
  }
})();
