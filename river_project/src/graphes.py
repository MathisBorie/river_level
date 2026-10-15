"""
================================================================================
MODULE graphes.py
================================================================================
Graphiques interactifs à partir des modèles enregistrés (`modeles.resultats`).

- `explorer` : figure Plotly pour parcourir le test : débit réel et prévisions des modèles,
  un curseur pour faire glisser une fenêtre de durée choisie, un curseur pour l'échéance, la
  liste des modèles (cliquer pour masquer / montrer) avec leur R². S'affiche dans le notebook
  (VS Code l'affiche sans rien installer) et s'enregistre en page HTML autonome (à envoyer
  telle quelle : elle s'ouvre dans un navigateur, sans Python ni internet).
================================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .modeles import resultats

_COULEURS = ["#2563eb", "#ea580c", "#16a34a", "#9333ea", "#dc2626", "#0891b2", "#ca8a04", "#db2777"]


def _valeurs(serie, index):
    """Valeurs (1 décimale) sur l'heure complète `index` ; NaN là où il n'y a rien (trou)."""
    return serie.reindex(index).to_numpy(float).round(1)


def explorer(cles, fenetre="30D", noms=None, titre="Prévisions sur le test", fichier=None):
    """
    Figure interactive pour parcourir le test de plusieurs modèles enregistrés.

    Deux curseurs sous le graphique : « Début » fait glisser la fenêtre jour par jour (l'axe
    vertical se règle sur la fenêtre), « Échéance » choisit l'échéance affichée. Les courbes
    sont tracées à l'heure PRÉVUE (t + échéance) : le débit réel reste la même rivière quelle
    que soit l'échéance, on voit la prévision s'en éloigner quand l'échéance grandit. La liste
    des modèles (dans le coin) donne leur R² à l'échéance choisie ; un clic masque ou montre
    un modèle.

    Parameters
    ----------
    cles : dict of str to str, or list of str
        Modèles du cache : ``{nom affiché: clé}`` (ex. ``CLES2``) ou liste de clés (le nom
        affiché est alors l'étiquette du modèle).
    fenetre : str, default "30D"
        Durée affichée (format pandas : ``"7D"``, ``"30D"``, ``"90D"``).
    noms : list of str, optional
        Noms affichés, dans l'ordre de `cles` (remplace les noms par défaut).
    titre : str, default "Prévisions sur le test"
        Titre.
    fichier : str or Path, optional
        Si donné, enregistre aussi une page HTML autonome (plotly.js inclus, ≈ 5 Mo + données).

    Returns
    -------
    plotly.graph_objects.Figure
        Dernière ligne d'une cellule : la figure s'affiche dans le notebook.

    Raises
    ------
    ValueError
        Si les modèles n'ont aucune échéance en commun.

    Notes
    -----
    Les données de toutes les échéances sont dans la figure : ≈ 5 Mo pour 3 modèles, 15
    échéances et 2 ans de test. Dans un notebook, la sortie de la cellule pèse autant :
    l'effacer avant d'envoyer le notebook. L'axe vertical de chaque position de la fenêtre
    est le plus haut débit (réel ou prévu, toutes échéances) de cette fenêtre.

    Examples
    --------
    >>> from src.graphes import explorer
    >>> explorer(CLES2)                                                          # dans le notebook
    >>> explorer(CLES2, fenetre="7D", fichier="../data/outputs/clapiere.html")   # + page HTML autonome
    """
    if isinstance(cles, dict):
        noms_defaut, cles = list(cles), list(cles.values())
    else:
        cles, noms_defaut = list(cles), None
    R = [resultats(c) for c in cles]
    noms = noms or noms_defaut or [r["fiche"]["etiquette"] or r["fiche"]["nom"] for r in R]
    communes = [c for c in R[0]["reel"].columns if all(c in r["reel"].columns for r in R)]
    if not communes:
        raise ValueError("ces modèles n'ont aucune échéance en commun")
    index = pd.date_range(min(r["reel"].index.min() for r in R), max(r["reel"].index.max() for r in R), freq="1h")
    heures = {c: int(c.split("t+")[1][:-1]) for c in communes}
    reel = {c: _valeurs(R[0]["reel"][c], index) for c in communes}
    prevu = [{c: _valeurs(r["prevu"][c], index) for c in communes} for r in R]
    debut0 = index[0].tz_localize(None)
    x0 = {c: (debut0 + pd.Timedelta(hours=heures[c])).isoformat() for c in communes}
    heure = 3600 * 1000                                              # pas de l'axe, en millisecondes

    def nom_modele(i, c):
        r2 = R[i]["r2"].get(c)
        return noms[i] + ("" if r2 is None or pd.isna(r2) else f"  ·  R² {r2:.3f}")

    c0 = communes[0]
    fig = go.Figure([go.Scatter(x0=x0[c0], dx=heure, y=reel[c0], name="réel", mode="lines",
                                line=dict(color="#1f2933", width=1.6))] +
                    [go.Scatter(x0=x0[c0], dx=heure, y=prevu[i][c0], name=nom_modele(i, c0), mode="lines",
                                line=dict(color=_COULEURS[i % len(_COULEURS)], width=1.1)) for i in range(len(R))])

    etapes_ech = [dict(method="restyle", label=f"+{heures[c]} h ({heures[c] / 24:.2g} j)",
                       args=[{"y": [reel[c]] + [prevu[i][c] for i in range(len(R))],
                              "x0": [x0[c]] * (len(R) + 1),
                              "name": ["réel"] + [nom_modele(i, c) for i in range(len(R))]}])
                  for c in communes]

    largeur = int(pd.Timedelta(fenetre) / pd.Timedelta("1h"))
    tout = np.vstack(list(reel.values()) + [v for d in prevu for v in d.values()])
    max_horaire = np.nanmax(np.where(np.isnan(tout), 0.0, tout), axis=0)   # par heure, toutes courbes
    decalage = max(heures.values())
    etapes_debut = []
    for jour in range(0, max(1, (len(index) + decalage - largeur) // 24 + 1)):
        a = debut0 + pd.Timedelta(days=jour)
        haut = float(max_horaire[max(0, jour * 24 - decalage):jour * 24 + largeur].max(initial=0.0)) or 1.0
        etapes_debut.append(dict(method="relayout", label=a.strftime("%d/%m/%Y"),
                                 args=[{"xaxis.range": [a.isoformat(), (a + pd.Timedelta(hours=largeur)).isoformat()],
                                        "yaxis.range": [0, 1.08 * haut]}]))

    fig.update_layout(
        title=dict(text=f"{titre}<br><sup>tracé à l'heure prévue (t + échéance) ; cliquer sur un modèle pour le masquer</sup>"),
        xaxis=dict(type="date", range=etapes_debut[0]["args"][0]["xaxis.range"]),
        yaxis=dict(title="débit (m³/s)", range=etapes_debut[0]["args"][0]["yaxis.range"]),
        legend=dict(x=1, xanchor="right", y=1, bgcolor="rgba(255,255,255,0.85)"),
        hovermode="x unified", template="plotly_white", height=620, margin=dict(l=60, r=20, t=70, b=160),
        sliders=[dict(active=0, steps=etapes_debut, currentvalue=dict(prefix="Début : "), pad=dict(t=40), y=0),
                 dict(active=0, steps=etapes_ech, currentvalue=dict(prefix="Échéance : t "), pad=dict(t=40), y=-0.22)],
    )
    if fichier:
        fichier = Path(fichier)
        fichier.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(fichier, include_plotlyjs=True, full_html=True)
        print(f"page écrite : {fichier} ({fichier.stat().st_size / 1e6:.1f} Mo) — à ouvrir dans un navigateur")
    return fig


_PAGE_PLUIE = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Erreurs de pluie</title>
<script>__PLOTLY__</script>
<style>
  :root { --fond: #ffffff; --texte: #1f2933; --doux: #5f6b78; --cadre: #e3e7eb; --accent: #b91c1c; }
  @media (prefers-color-scheme: dark) { :root { --fond: #111418; --texte: #e6e9ec; --doux: #9aa5b1; --cadre: #2e353d; --accent: #f87171; } }
  body { margin: 0; background: var(--fond); color: var(--texte); font: 14px/1.45 system-ui, -apple-system, sans-serif; }
  main { max-width: 1150px; margin: 0 auto; padding: 16px; }
  h1 { font-size: 19px; margin: 0 0 2px; } p.sous { margin: 0 0 14px; color: var(--doux); }
  .reglages { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px 20px; padding: 12px; border: 1px solid var(--cadre); border-radius: 8px; }
  label { display: block; color: var(--doux); font-size: 12.5px; margin-bottom: 4px; }
  label b { color: var(--texte); }
  select, input[type=number] { width: 100%; box-sizing: border-box; padding: 5px; background: var(--fond); color: var(--texte); border: 1px solid var(--cadre); border-radius: 5px; font: inherit; }
  input[type=range] { width: 100%; accent-color: var(--accent); }
  .resume { margin: 12px 0 0; display: flex; flex-wrap: wrap; gap: 8px 22px; color: var(--doux); }
  .resume b { color: var(--texte); font-variant-numeric: tabular-nums; }
  #graphe { width: 100%; height: 60vh; min-height: 380px; }
  @media (max-width: 800px) { .reglages { grid-template-columns: 1fr 1fr; } }
</style></head>
<body><main>
  <h1>Erreurs de la prévision de pluie : histogramme et loi gamma, en direct</h1>
  <p class="sous">__SOUS_TITRE__</p>
  <div class="reglages">
    <div><label>Sens</label><select id="sens">
      <option value="tombe">prévision sachant la pluie tombée</option>
      <option value="prevu">pluie tombée sachant la prévision</option></select></div>
    <div><label>Pas du cumul : <b id="txt-pas"></b></label><input id="pas" type="range" min="0" max="5" step="1" value="4"></div>
    <div><label>Début du cumul : <b id="txt-debut"></b></label><input id="debut" type="range" min="0" step="1" value="48"></div>
    <div><label><input id="sec" type="checkbox"> condition « sèche » (&lt; 0,1 mm) au lieu de la fourchette</label></div>
    <div><label>Début de la fourchette : <b id="txt-de"></b></label><input id="de" type="range" min="0" max="60" step="0.1" value="2"></div>
    <div><label>Taille de la fourchette : <b id="txt-fourchette"></b></label><input id="taille" type="range" min="0.1" max="30" step="0.1" value="2"></div>
    <div><label>Barres de l'histogramme : <b id="txt-barres"></b></label><input id="barres" type="range" min="5" max="120" step="1" value="40"></div>
  </div>
  <div class="resume" id="resume"></div>
  <div id="graphe"></div>
</main>
<script>
const D = __DONNEES__;
const SEC = 0.1, H = D.heures, R = D.runs, C = D.carreaux.length;
function decode(b64) { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new Uint16Array(u.buffer); }
const P = decode(D.prevu), O = decode(D.tombe), VIDE = 65535;
const $ = id => document.getElementById(id);

// --- loi gamma : log Γ (Lanczos), digamma et trigamma (récurrence + développement asymptotique)
function lgamma(x) {
  const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
                    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1; let a = c[0]; const t = x + g + 0.5;
  for (let i = 1; i < 9; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
function digamma(x) { let r = 0; while (x < 6) { r -= 1 / x; x += 1; } const f = 1 / (x * x);
  return r + Math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f / 252)); }
function trigamma(x) { let r = 0; while (x < 6) { r += 1 / (x * x); x += 1; } const f = 1 / (x * x);
  return r + 1 / x + f / 2 + f / x * (1 / 6 - f * (1 / 30 - f / 42)); }
function ajusterGamma(v) {                         // maximum de vraisemblance : θ = moyenne / k, ln k − ψ(k) = ln moyenne − moyenne(ln x)
  const n = v.length, m = v.reduce((s, x) => s + x, 0) / n, ml = v.reduce((s, x) => s + Math.log(x), 0) / n, s = Math.log(m) - ml;
  let k = (3 - s + Math.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s);   // départ (Minka), puis Newton
  for (let i = 0; i < 8; i++) k -= (Math.log(k) - digamma(k) - s) / (1 / k - trigamma(k));
  return {k, theta: m / k};
}
const densite = (x, k, th) => Math.exp((k - 1) * Math.log(x) - x / th - lgamma(k) - k * Math.log(th));

// --- modèle d'erreur ajusté (notebook 06, partie 8) : ne connaît que la pluie tombée o (mm sur 24 h) et l'échéance t (jours)
const M = D.modele;
function modele(o, t) {
  const sig = x => 1 / (1 + Math.exp(-x)), dot = (c, z) => c.reduce((s, ci, i) => s + ci * z[i], 0);
  if (o < SEC) { const z = [1, t, t * t];
    return {p0: sig(dot(M.p0_sec, z)), mu: Math.exp(dot(M.g_sec.slice(0, 3), z)), k: Math.exp(dot(M.g_sec.slice(3), z))}; }
  const l = Math.log(o), z = [1, l, l * l, t, t * l, t * t];
  return {p0: sig(dot(M.p0_pluie, z)), mu: Math.exp(dot(M.g_pluie.slice(0, 6), z)), k: Math.exp(dot(M.g_pluie.slice(6), z))};
}

// --- cumuls (pluie prévue, pluie tombée) sur ]début, début + pas] heures après l'émission, pour chaque carreau et prévision
function cumuls(debut, pas) {
  const pr = [], to = [];
  for (let c = 0; c < C; c++) for (let r = 0; r < R; r++) {
    const base = (c * R + r) * H; let sp = 0, so = 0, ok = true;
    for (let h = debut + 1; h <= debut + pas; h++) { const p = P[base + h], o = O[base + h];   // heures qui FINISSENT dans ]début, début + pas]
      if (p === VIDE || o === VIDE) { ok = false; break; } sp += p; so += o; }
    if (ok) { pr.push(sp / 10); to.push(so / 10); }
  }
  return {prevu: pr, tombe: to};
}
const sombre = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
const PAS = [1, 3, 6, 12, 24, 48];
function maj() {
  const pas = PAS[+$("pas").value]; $("txt-pas").textContent = `${pas} h`; $("debut").max = H - 1 - pas;
  $("txt-barres").textContent = $("barres").value; $("txt-de").textContent = `${(+$("de").value).toFixed(1)} mm`;
  if (+$("debut").value > H - 1 - pas) $("debut").value = H - 1 - pas;
  const debut = +$("debut").value, fin = debut + pas, sens = $("sens").value, autre = sens === "tombe" ? "prevu" : "tombe";
  $("txt-debut").textContent = `de +${debut} h à +${fin} h (jour ${(fin / 24).toFixed(2).replace(".", ",")})`;
  const nom = {tombe: "pluie tombée", prevu: `pluie prévue (émise ${debut} à ${fin} h avant)`};
  const z = cumuls(debut, pas), sec = $("sec").checked, a = +$("de").value, b = +(a + +$("taille").value).toFixed(2);
  $("txt-fourchette").textContent = sec ? "(condition sèche)" : `de ${a} à ${b} mm`;
  const v = [], conds = [];
  for (let i = 0; i < z[sens].length; i++) { const c = z[sens][i]; if (sec ? c < SEC : (c >= a && c < b)) { v.push(z[autre][i]); conds.push(c); } }
  const avecModele = M && sens === "tombe" && pas === 24 && fin >= 24 && fin <= 216;
  const cas = avecModele ? conds.map(o => modele(o, fin / 24)) : [];
  const pluie = v.filter(x => x >= SEC), psec = v.length ? 1 - pluie.length / v.length : NaN;
  const cond = sec ? `${nom[sens]} sèche (< 0,1 mm)` : `${nom[sens]} entre ${a} et ${b} mm`;
  const traces = [];
  let info = `<span>cas : <b>${v.length}</b></span><span>secs : <b>${(100 * psec).toFixed(1)} %</b></span>`;
  if (pluie.length >= 20) {
    const tri = [...pluie].sort((x, y) => x - y), haut = tri[Math.floor(0.99 * (tri.length - 1))] || tri[tri.length - 1];
    const nb = +$("barres").value, l = haut / nb, comptes = new Array(nb).fill(0);
    pluie.forEach(x => { if (x <= haut) comptes[Math.min(nb - 1, Math.floor(x / l))]++; });
    const g = ajusterGamma(pluie), moy = pluie.reduce((s, x) => s + x, 0) / pluie.length, tot = v.reduce((s, x) => s + x, 0) / v.length;
    const xs = Array.from({length: 200}, (_, i) => SEC + i * (haut - SEC) / 199);
    traces.push({type: "bar", x: comptes.map((_, i) => (i + 0.5) * l), y: comptes.map(n => n / (pluie.length * l)), width: l,
                 name: `${pluie.length} cas non secs`, marker: {color: "rgba(59,130,246,0.55)"}});
    traces.push({x: xs, y: xs.map(x => densite(x, g.k, g.theta)), mode: "lines", line: {color: "#dc2626", width: 2.5},
                 name: `gamma ajustée : k = ${g.k.toFixed(2)}, θ = ${g.theta.toFixed(2)} mm`});
    traces.push({x: [moy, moy], y: [0, 0], mode: "lines", line: {color: sombre ? "#e6e9ec" : "#111", dash: "dot", width: 1}, name: `moyenne des non secs ${moy.toFixed(2)} mm`, yaxis: "y", hoverinfo: "skip"});
    if (cas.length) {                                         // densité du modèle : moyenne des gammas des mêmes cas, pondérées par 1 − p0
      const poids = cas.map(c => 1 - c.p0), somme = poids.reduce((s, w) => s + w, 0);
      traces.push({x: xs, y: xs.map(x => cas.reduce((s, c, i) => s + poids[i] * densite(x, c.k, c.mu / c.k), 0) / somme),
                   mode: "lines", line: {color: "#16a34a", width: 2.5, dash: "dash"},
                   name: "modèle (ne connaît que la pluie tombée et l'échéance)"});
    }
    info += `<span>k = <b>${g.k.toFixed(2)}</b></span><span>θ = <b>${g.theta.toFixed(2)} mm</b></span>` +
            `<span>moyenne des non secs : <b>${moy.toFixed(2)} mm</b></span><span>moyenne totale (secs compris) : <b>${tot.toFixed(2)} mm</b></span>`;
    if (cas.length) {
      const p0m = cas.reduce((s, c) => s + c.p0, 0) / cas.length, poids = cas.map(c => 1 - c.p0), sw = poids.reduce((s, w) => s + w, 0);
      const mum = cas.reduce((s, c, i) => s + poids[i] * c.mu, 0) / sw;
      info += `<span style="color:#16a34a">modèle : secs <b>${(100 * p0m).toFixed(1)} %</b>, moyenne des non secs <b>${mum.toFixed(2)} mm</b></span>`;
    } else if (M) info += `<span>(modèle affiché seulement pour : sens « prévision sachant la pluie tombée », pas de 24 h, fin du cumul entre +24 h et +216 h)</span>`;
  } else info += `<span><b>trop peu de cas non secs pour ajuster une gamma</b></span>`;
  $("resume").innerHTML = info;
  const lay = {title: {text: `Loi de la ${nom[autre]} sachant la ${cond} (cumuls de ${pas} h)`, font: {size: 15}},
    margin: {l: 64, r: 16, t: 50, b: 50}, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: {color: sombre ? "#e6e9ec" : "#1f2933"}, bargap: 0, showlegend: true,
    legend: {x: 1, xanchor: "right", y: 1, bgcolor: sombre ? "rgba(17,20,24,0.85)" : "rgba(255,255,255,0.85)"},
    xaxis: {title: {text: `${nom[autre]} (mm sur ${pas} h)`}, gridcolor: sombre ? "#2e353d" : "#e5e7eb", rangemode: "tozero"},
    yaxis: {title: {text: "densité de probabilité (1/mm)"}, gridcolor: sombre ? "#2e353d" : "#e5e7eb", rangemode: "tozero"}};
  if (traces.length) {                                       // ligne de la moyenne : jusqu'en haut de l'histogramme
    const ymax = Math.max(...traces[0].y, ...traces[1].y, ...(traces[3] ? traces[3].y : [0])); traces[2].y = [0, ymax * 1.05]; }
  Plotly.react("graphe", traces, lay, {responsive: true, displaylogo: false});
}
["sens", "pas", "debut", "sec", "de", "taille", "barres"].forEach(id => $(id).addEventListener("input", maj));
maj();
</script></body></html>
"""


def explorer_pluie(carreaux, fichier, debut="2024-03-14", fin=None, heures=240, modele=None):
    """
    Page HTML interactive : loi de la pluie prévue sachant la pluie tombée (ou l'inverse), en direct.

    Réglages dans la page : sens, pas du cumul (1 à 48 h), début du cumul en heures après
    l'émission de la prévision, fourchette de la condition (début + taille, ou condition sèche),
    nombre de barres. Avec `modele`, la densité du modèle d'erreur est tracée à côté de la gamma
    ajustée sur les cas affichés.
    Tout est calculé dans la page (cumuls, gamma ajustée par maximum de vraisemblance), sans
    Python ni internet. Mêmes calculs que le notebook 06 (histogramme des cas non secs, part des
    cas secs, k, θ, moyennes).

    Parameters
    ----------
    carreaux : list of int
        Carreaux à inclure (prévisions archivées complètes ; ex. les 7 choisis de La Clapière).
    fichier : str or Path
        Où écrire la page.
    debut, fin : str, optional
        Période des prévisions (jours d'émission). Par défaut toute l'archive.
    heures : int, default 240
        Heures de chaque prévision gardées (10 jours).
    modele : dict, optional
        Coefficients du modèle d'erreur (notebook 06, partie 8) : ``p0_pluie``, ``p0_sec`` (logistiques,
        constante puis coefficients), ``g_pluie``, ``g_sec`` (coefficients de ln μ puis de ln k). Si donné,
        la page trace aussi la densité du modèle, qui ne connaît que la pluie tombée et l'échéance
        (cumuls de 24 h, sens « prévision sachant la pluie tombée »).

    Returns
    -------
    Path
        Le fichier écrit.

    Notes
    -----
    Les valeurs sont rangées en dixièmes de millimètre sur 2 octets (l'arrondi d'Open-Meteo) :
    ≈ 4,7 Mo (plotly.js) + 5,4 octets par heure, carreau et prévision (≈ 13 Mo pour 7 carreaux).

    Examples
    --------
    >>> explorer_pluie([835931, 841112, 838522, 830767, 846307, 841113, 833346], "../data/outputs/pluie.html")
    """
    import base64
    import json
    import plotly
    from . import config

    runs_dir = config.CACHE_PREVISION / "single_runs" / "ecmwf_ifs" / "runs"
    carreaux = [int(c) for c in carreaux]
    prev = pd.read_parquet(runs_dir, columns=["carreau", "run", "time", "precipitation"],
                           filters=[("carreau", "in", carreaux)])
    prev = prev[(prev["run"] >= pd.Timestamp(debut, tz="UTC")) & ((fin is None) | (prev["run"] <= pd.Timestamp(fin or "2100-01-01", tz="UTC")))]
    prev["h"] = ((prev["time"] - prev["run"]) / pd.Timedelta("1h")).astype(int)
    prev = prev[prev["h"] < heures]
    runs = np.sort(prev["run"].unique())
    obs = pd.concat({c: pd.read_parquet(config.CACHE_METEO / "carreaux" / f"{c}_horaire.parquet", columns=["precipitation"])["precipitation"]
                     for c in carreaux}, names=["carreau", "time"])
    prev["tombe"] = obs.reindex(pd.MultiIndex.from_arrays([prev["carreau"], prev["time"]])).to_numpy()
    i_c = pd.Index(carreaux).get_indexer(prev["carreau"])
    i_r = pd.Index(runs).get_indexer(prev["run"])
    tableaux = {}
    for nom, col in (("prevu", "precipitation"), ("tombe", "tombe")):
        t = np.full((len(carreaux), len(runs), heures), 65535, dtype="<u2")
        v = prev[col].to_numpy(float)
        ok = ~np.isnan(v)
        t[i_c[ok], i_r[ok], prev["h"].to_numpy()[ok]] = np.clip(np.round(v[ok] * 10), 0, 65534).astype("<u2")
        tableaux[nom] = base64.b64encode(t.tobytes()).decode()
    donnees = {"carreaux": carreaux, "runs": len(runs), "heures": heures, "modele": modele, **tableaux}
    sous_titre = (f"Prévisions ECMWF IFS 9 km émises chaque jour à 0 h UTC, du {pd.Timestamp(runs[0]):%d/%m/%Y} au "
                  f"{pd.Timestamp(runs[-1]):%d/%m/%Y} ({len(runs)} prévisions), contre la pluie observée (IFS 9 km) ; {len(carreaux)} carreaux "
                  f"mélangés. « Sec » = moins de 0,1 mm. Gamma ajustée par maximum de vraisemblance sur les cas non secs.")
    page = (_PAGE_PLUIE.replace("__PLOTLY__", plotly.offline.get_plotlyjs()).replace("__SOUS_TITRE__", sous_titre)
            .replace("__DONNEES__", json.dumps(donnees, separators=(",", ":"))))
    fichier = Path(fichier)
    fichier.parent.mkdir(parents=True, exist_ok=True)
    fichier.write_text(page, encoding="utf-8")
    print(f"page écrite : {fichier} ({fichier.stat().st_size / 1e6:.1f} Mo) — à ouvrir dans un navigateur")
    return fichier
