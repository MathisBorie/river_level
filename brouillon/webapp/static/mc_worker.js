// Sous-worker "muet" du pool Monte Carlo : entraîne des membres MLP de l'ensemble
// à partir de tableaux bruts (npy) et renvoie les modèles sérialisés (joblib).
// Sert au calcul PARALLÈLE : plusieurs de ces workers tournent en même temps.
// La config d'un membre doit rester IDENTIQUE à _mc_membre_neuf() de river_web.py.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

let pyodide = null, fn = null;

async function init() {
  pyodide = await loadPyodide();
  await pyodide.loadPackage(["numpy", "scikit-learn"]);
  await pyodide.runPythonAsync([
    "import io, numpy as np, joblib",
    "from sklearn.neural_network import MLPRegressor",
    "from sklearn.pipeline import make_pipeline",
    "from sklearn.preprocessing import StandardScaler",
    "from sklearn.compose import TransformedTargetRegressor",
    "def _membre(X, Y, i):",
    "    idx = np.random.default_rng(1000 + i).integers(0, len(X), len(X))",
    "    net = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64,32), alpha=1e-3,",
    "        learning_rate_init=0.005, max_iter=300, early_stopping=True, n_iter_no_change=12, random_state=i))",
    "    mdl = TransformedTargetRegressor(regressor=net, transformer=StandardScaler())",
    "    mdl.fit(X[idx], Y[idx])",
    "    b = io.BytesIO(); joblib.dump(mdl, b); return b.getvalue()",
    "def entrainer_membres(xb, yb, seeds):",
    "    X = np.load(io.BytesIO(bytes(xb))); Y = np.load(io.BytesIO(bytes(yb)))",
    "    return [ _membre(X, Y, int(i)) for i in seeds ]",
  ].join("\n"));
  fn = pyodide.globals.get("entrainer_membres");
  postMessage({ type: "pret" });
}
const prom = init().catch((e) => postMessage({ type: "erreur", error: String(e) }));

onmessage = async (e) => {
  const { id, xb, yb, seeds } = e.data;
  try {
    await prom;
    const res = fn(xb, yb, seeds);          // xb/yb = Uint8Array (npy) ; seeds = liste d'indices
    const arr = res.toJs();                  // liste d'octets (modèles joblib) -> Array<Uint8Array>
    res.destroy();
    postMessage({ id, ok: true, modeles: arr }, arr.map((u) => u.buffer));
  } catch (err) {
    postMessage({ id, ok: false, error: String((err && err.message) || err) });
  }
};
