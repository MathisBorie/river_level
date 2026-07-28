/* River Lab — front vanilla JS.
   Vue 1 : carte de toutes les stations Hub'Eau (clusters).
   Vue 2 : tableau de bord d'une station, qui pilote l'API Flask
   (jobs de fond avec log en polling + actions rapides synchrones). */

"use strict";

// ---------------------------------------------------------------- état global
const etat = {
  stations: [],
  code: null,           // station affichée
  riviere: null,        // dernier état renvoyé par /api/riviere/<code>
  jobId: null,
  jobLignesLues: 0,
  modeSelection: false,
  pointsChoisis: [],    // [{lat, lon, marker}]
};

let carteStations = null;
let clusterStations = null;
let carteZone = null;
let couchesZone = [];   // couches à nettoyer à chaque rendu

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- utilitaires
let _toastTimer = null;
function toast(message, erreur = false) {
  const t = $("toast");
  t.textContent = message;
  t.className = erreur ? "erreur" : "";
  if (erreur) console.error("[toast]", message);
  clearTimeout(_toastTimer);
  // les erreurs restent affichées (clic pour fermer) ; le reste disparaît après 6 s
  if (!erreur) _toastTimer = setTimeout(() => t.classList.add("cache"), 6000);
  t.onclick = () => t.classList.add("cache");
  t.classList.remove("cache");
}

async function api(chemin, options = {}) {
  // Version « site statique » : le backend tourne dans le navigateur (Pyodide).
  if (window.RIVER_BACKEND) {
    const method = options.method || "GET";
    const body = options.body ? JSON.parse(options.body) : null;
    return window.RIVER_BACKEND(method, chemin, body);
  }
  const reponse = await fetch(chemin, options);
  const corps = await reponse.json().catch(() => ({}));
  if (!reponse.ok) throw new Error(corps.erreur || `Erreur HTTP ${reponse.status}`);
  return corps;
}

function post(chemin, donnees) {
  return api(chemin, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(donnees || {}),
  });
}

const octetsLisibles = (o) =>
  o > 1e9 ? (o / 1e9).toFixed(2) + " Go" : o > 1e6 ? (o / 1e6).toFixed(1) + " Mo" : Math.round(o / 1e3) + " ko";

// Date ISO "AAAA-MM-JJ" -> "JJ/MM/AAAA".
function dateFr(iso) {
  if (!iso) return "";
  const m = String(iso).slice(0, 10).match(/^(\d{4})-(\d{2})-(\d{2})$/);
  return m ? `${m[3]}/${m[2]}/${m[1]}` : iso;
}
// Débit L/s -> m³/s, formaté (2 décimales, séparateur français).
function debitM3(lps) {
  return (lps / 1000).toLocaleString("fr-FR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
// Valeur déjà en m³/s -> texte français.
const fmtM3 = (v) => Number(v).toLocaleString("fr-FR", { maximumFractionDigits: 2 });

// ---------------------------------------------------------------- graphiques (Chart.js)
const charts = {};
function detruireChart(id) { if (charts[id]) { charts[id].destroy(); delete charts[id]; } }

// Plugin : trait vertical pointillé « aujourd'hui / départ » + libellés passé/futur.
const pluginPivot = {
  id: "lignePivot",
  afterDatasetsDraw(chart, args, opts) {
    if (!opts || opts.index == null) return;
    const { ctx, chartArea: { top, bottom }, scales: { x } } = chart;
    const px = x.getPixelForValue(opts.index);
    if (px == null || isNaN(px)) return;
    ctx.save();
    ctx.beginPath();
    ctx.setLineDash([5, 4]);
    ctx.moveTo(px, top); ctx.lineTo(px, bottom);
    ctx.lineWidth = 1.4; ctx.strokeStyle = "rgba(90, 100, 112, 0.7)"; ctx.stroke();
    ctx.setLineDash([]);
    ctx.font = "650 11px ui-sans-serif, system-ui, sans-serif";
    ctx.fillStyle = "rgba(80, 92, 104, 0.9)";
    ctx.textBaseline = "top";
    ctx.textAlign = "right"; ctx.fillText("◀ passé", px - 7, top + 3);
    ctx.textAlign = "left"; ctx.fillText("futur ▶", px + 7, top + 3);
    ctx.restore();
  },
};
if (window.Chart) Chart.register(pluginPivot);
// Info-bulle épinglée en haut du graphe (ne cache plus la courbe, surtout sur mobile).
if (window.Chart) {
  Chart.Tooltip.positioners.hautCentre = function (elements, pos) {
    const a = this.chart.chartArea;
    return { x: a.left + a.width / 2, y: a.top + 2 };
  };
}

// Options communes + création d'un graphique linéaire interactif.
// pivotIndex : index (catégorie) où tracer le trait « aujourd'hui » (ou null).
function dessinerCourbe(canvasId, labels, datasets, pivotIndex = null) {
  detruireChart(canvasId);
  const el = document.getElementById(canvasId);
  if (!el) return;
  charts[canvasId] = new Chart(el.getContext("2d"), {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 300 },
      interaction: { mode: "index", intersect: false },
      plugins: {
        lignePivot: { index: pivotIndex },
        legend: { labels: { filter: (it) => !it.text.startsWith("_"), boxWidth: 12, font: { size: 11 } } },
        tooltip: {
          position: "hautCentre", yAlign: "top", caretSize: 0,
          filter: (it) => !it.dataset.label.startsWith("_") && it.parsed.y != null,
          callbacks: {
            title: (items) => (items.length ? items[0].label : ""),
            label: (it) => {
              const v = it.parsed.y;
              if (it.dataset._bas) {
                const bas = it.dataset._bas[it.dataIndex];
                return `${it.dataset.label} : ${fmtM3(bas)} – ${fmtM3(v)} m³/s`;
              }
              return `${it.dataset.label} : ${fmtM3(v)} m³/s`;
            },
          },
        },
      },
      scales: {
        y: { beginAtZero: true, title: { display: true, text: "Débit (m³/s)" }, ticks: { font: { size: 11 } } },
        x: { ticks: { maxRotation: 45, autoSkip: true, maxTicksLimit: 11, font: { size: 10 } }, grid: { display: false } },
      },
    },
  });
}

// Niveaux d'IC cochés pour un graphe ("backtest" / "prevision").
function nivIC(graphe) {
  return Array.from(document.querySelectorAll(`.ic-niv[data-graphe="${graphe}"]:checked`)).map((c) => c.value);
}

// Tableau R² par horizon (J+0 … J+n).
function tableauR2(sd, nbJours) {
  if (!sd || !sd.length) return "";
  const scores = sd.slice(0, nbJours + 1);
  return "<table><tr><th>Horizon</th>" +
    scores.map((_, i) => `<th>J+${i}</th>`).join("") +
    "</tr><tr><td>R²</td>" +
    scores.map((s) => `<td>${(s * 100).toFixed(0)}%</td>`).join("") +
    "</tr></table>";
}

// Fan chart : passé observé + prévision + bandes IC (+ réel pour le backtest).
// niveaux : liste des IC à dessiner (["50","95","99"] filtrés) ; par défaut ["95"].
function rendreFanChart(canvasId, data, avecReel, niveaux) {
  niveaux = niveaux || ["95"];
  const obs = data.observe || [];
  const pts = data.points || [];
  if (!pts.length) return;
  const labels = obs.map((o) => dateFr(o.date)).concat(pts.slice(1).map((p) => dateFr(p.date)));
  const N = labels.length;
  const base = obs.length - 1;            // index du pivot (dernière date observée = 1re prévue)
  const vide = (n) => new Array(n).fill(null);

  const obsData = obs.map((o) => o.debit).concat(vide(N - obs.length));
  const prevData = vide(N), reelData = vide(N);
  pts.forEach((p, i) => { prevData[base + i] = p.prev; if (avecReel) reelData[base + i] = p.reel; });

  const datasets = [];
  // Dégradé « froid » du plus large (périwinkle clair) au plus resserré (teal) :
  // lisible et élégant, chaque bande se distingue de la suivante.
  const bandes = [
    ["99", "rgba(124, 152, 240, 0.16)"],  // périwinkle
    ["95", "rgba(56, 158, 205, 0.22)"],   // cyan
    ["50", "rgba(13, 148, 136, 0.30)"],   // teal
  ];
  if (data.hybride) {
    bandes.filter(([niv]) => niveaux.includes(niv)).forEach(([niv, couleur]) => {
      const bas = vide(N), haut = vide(N);
      pts.forEach((p, i) => { bas[base + i] = p[`ic${niv}_bas`]; haut[base + i] = p[`ic${niv}_haut`]; });
      datasets.push({ label: `_bas${niv}`, data: bas, borderColor: "transparent", pointRadius: 0, fill: false });
      datasets.push({ label: `IC ${niv}%`, data: haut, borderColor: "transparent", backgroundColor: couleur, pointRadius: 0, fill: "-1", _bas: bas });
    });
  }
  datasets.push({ label: "Passé (observé)", data: obsData, borderColor: "#233240", borderWidth: 2.4, pointRadius: 0, tension: 0.2 });
  datasets.push({ label: "Prévision", data: prevData, borderColor: "#0a6b62", backgroundColor: "#0a6b62", borderWidth: 2.6, pointRadius: 2.4, pointBorderColor: "#fff", pointBorderWidth: 1, tension: 0.2 });
  if (avecReel) datasets.push({ label: "Réel", data: reelData, borderColor: "#e0533f", borderWidth: 2, borderDash: [5, 3], pointRadius: 2.4, tension: 0.2 });

  dessinerCourbe(canvasId, labels, datasets, base);
}

// Ligne de fiabilité (R²) affichée à CHAQUE prévision / test.
const NOMS_MODELES = {
  gradient_boosting: "Gradient Boosting", ridge: "Ridge", lineaire: "Régression linéaire",
  // (noms de la version serveur, conservés pour compat)
  ridge_causal: "Ridge causal", ridge_brut: "Ridge", lineaire_pca: "Linéaire (PCA)", keras_brut: "Réseau de neurones",
};
function classeR2(pct) { return pct >= 70 ? "" : pct >= 50 ? "moyen" : "faible"; }
function rendreFiabilite(elemId, data) {
  const el = $(elemId);
  if (!el || data.score == null) { if (el) el.classList.add("cache"); return; }
  const pct = Math.round(data.score * 100);
  const nom = NOMS_MODELES[data.modele] || data.modele;
  let detail = "";
  const sd = data.scores_detail;
  if (sd && sd.length) {
    detail = `<span class="details-r2">(J0 ${Math.round(sd[0] * 100)}% → J+${sd.length - 1} ${Math.round(sd[sd.length - 1] * 100)}%)</span>`;
  }
  // Fidélité des intervalles : couverture réelle + note (précision + finesse).
  let couv = "";
  if (data.couverture && data.couverture["95"] != null) {
    const c = data.couverture;
    const note = c.note != null ? ` · <b title="Qualité des intervalles sur tous les horizons : précision ET finesse, 0-100">note ${c.note}/100</b>` : "";
    const noteP = c.note_proche != null ? ` <span title="Même note mais sur la 1re semaine (J+1→J+7), sans être plombée par les horizons lointains">(J+1→7 : ${c.note_proche})</span>` : "";
    couv = ` <span class="details-r2" title="Couverture réelle mesurée sur des données jamais vues (cibles 50/95/99).">· IC 95% couvre <b>${c["95"]}%</b>${note}${noteP}</span>`;
  }
  el.innerHTML = `<span class="details-r2">${nom} — fiabilité globale</span> ` +
    `<span class="badge-r2 ${classeR2(pct)}">R² ${pct}%</span> ${detail}${couv}`;
  el.classList.remove("cache");
}

function rendreHistoChart(canvasId, serie) {
  const labels = serie.map((p) => dateFr(p.date));
  const data = serie.map((p) => p.debit);
  dessinerCourbe(canvasId, labels, [{
    label: "Débit", data, borderColor: "#0f9488", backgroundColor: "rgba(15,148,136,0.14)",
    borderWidth: 1.6, pointRadius: 0, fill: true, tension: 0.2,
  }]);
}

// Popovers « ⓘ »
document.querySelectorAll(".info-btn").forEach((btn) => {
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const pop = document.getElementById(btn.dataset.popover);
    document.querySelectorAll(".popover").forEach((p) => { if (p !== pop) p.classList.add("cache"); });
    pop.classList.toggle("cache");
  });
});
document.addEventListener("click", (e) => {
  if (!e.target.closest(".popover") && !e.target.closest(".info-btn"))
    document.querySelectorAll(".popover").forEach((p) => p.classList.add("cache"));
});

// ---------------------------------------------------------------- vue 1 : stations
async function initStations() {
  carteStations = L.map("carte-stations").setView([46.6, 2.4], 6);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    attribution: "© OpenStreetMap, © CartoDB",
  }).addTo(carteStations);

  try {
    etat.stations = await api("/api/stations");
  } catch (e) {
    toast("Impossible de charger les stations : " + e.message, true);
    masquerOverlayDemarrage();   // ne pas rester bloqué sur l'écran de chargement
    return;
  }
  await rafraichirCodesAvecModeles();
  rendreStations();
  masquerOverlayDemarrage();     // stations prêtes → on lève l'écran de chargement

  $("recherche-station").addEventListener("input", rendreStations);
  $("filtre-service").addEventListener("change", rendreStations);
  $("filtre-mes-modeles").addEventListener("change", rendreStations);
}

// Ensemble des codes de stations qui ont DÉJÀ un modèle sur cet appareil.
async function rafraichirCodesAvecModeles() {
  try {
    const inv = await api("/api/stockage");
    etat.codesAvecModeles = new Set(
      (inv.stations || []).filter((s) => (s.modeles || []).length).map((s) => s.code));
  } catch (e) {
    etat.codesAvecModeles = etat.codesAvecModeles || new Set();
  }
}

function masquerOverlayDemarrage() {
  const o = document.getElementById("overlay-chargement");
  if (o) o.classList.add("fini");
}

function rendreStations() {
  if (!Array.isArray(etat.stations)) {
    console.error("[stations] réponse inattendue :", etat.stations);
    toast("Réponse stations inattendue (voir console)", true);
    return;
  }
  const requete = $("recherche-station").value.trim().toLowerCase();
  const seulementService = $("filtre-service").checked;
  const seulementMesModeles = $("filtre-mes-modeles").checked;
  const codesModeles = etat.codesAvecModeles || new Set();

  if (clusterStations) carteStations.removeLayer(clusterStations);
  // Options économes en mémoire (mobile) : pas d'animation de cluster, on ne garde
  // dans le DOM que les marqueurs visibles, clustering plus large.
  clusterStations = L.markerClusterGroup({
    chunkedLoading: true, removeOutsideVisibleBounds: true,
    animate: false, animateAddingMarkers: false, maxClusterRadius: 90,
  });

  const marqueurs = [];
  let visibles = 0;
  for (const s of etat.stations) {
    if (seulementService && !s.en_service) continue;
    if (seulementMesModeles && !codesModeles.has(s.code)) continue;
    if (requete) {
      const texte = `${s.nom} ${s.cours_eau || ""} ${s.code}`.toLowerCase();
      if (!texte.includes(requete)) continue;
    }
    visibles++;
    const marqueur = L.marker([s.lat, s.lon]);
    // Popup construit à la DEMANDE (au clic) — évite de générer des milliers de
    // chaînes HTML d'un coup, ce qui saturait la mémoire sur téléphone.
    marqueur.bindPopup(() =>
      `<b>${s.nom}</b><br>${s.cours_eau || ""}<br><code>${s.code}</code><br>` +
      `<button class="primaire" onclick="ouvrirStation('${s.code}')">Ouvrir cette station →</button>`);
    marqueurs.push(marqueur);
  }
  clusterStations.addLayers(marqueurs);   // ajout en bloc (plus efficace que un par un)
  carteStations.addLayer(clusterStations);
  $("compteur-stations").textContent = `${visibles} stations`;
}

// ---------------------------------------------------------------- navigation
function reinitialiserAffichageStation() {
  // Vide tout ce qui vient de la station précédente : graphiques, tableaux, dates.
  $("image-pca").classList.add("cache");
  $("image-pca").removeAttribute("src");
  ["chart-prevision", "chart-backtest", "chart-historique", "chart-pca"].forEach(detruireChart);
  etat.dernierBacktest = null; etat.dernierePrevision = null;
  if (chartPca) { chartPca = null; }
  $("zone-pca").style.display = "none";
  $("pca-resultat").innerHTML = "";
  ["vide-prevision", "vide-backtest", "vide-historique"].forEach((id) => { const e = $(id); if (e) e.style.display = ""; });
  ["fiab-prevision", "fiab-backtest"].forEach((id) => { const e = $(id); if (e) { e.classList.add("cache"); e.innerHTML = ""; } });
  document.querySelectorAll(".popover").forEach((p) => p.classList.add("cache"));
  $("scores-detail").innerHTML = "";
  $("tableau-prevision").innerHTML = "";
  $("stats-historique").innerHTML = "";
  $("tableau-modeles").innerHTML = "";
  $("backtest-date").value = "";
  $("periode-dispo").textContent = "période : …";
  $("banniere-avertissement").classList.add("cache");
  $("banniere-avertissement").textContent = "";
  fermerReglages();
  $("cout-estimation").textContent = "";
  $("analyse-diagnostic").textContent = "";
  const leg = $("legende-carte");
  if (leg) { leg.innerHTML = ""; leg.style.display = "none"; }
  etat.carteAjustee = false;   // nouvelle station : on autorise un recadrage
  etat.datesTest = null;
  if (etat.modeSelection) quitterModeSelection();
}

window.ouvrirStation = async function (code, depuisHistorique) {
  etat.code = code;
  etat.recordCharge = false;
  if (!depuisHistorique) history.pushState({ station: code }, "", "#station=" + code);
  reinitialiserAffichageStation();
  $("vue-stations").classList.add("cache");
  $("vue-station").classList.remove("cache");
  $("btn-retour").classList.remove("cache");
  $("titre-station").textContent = `Chargement de ${code}…`;
  activerPanneau(etat.panneauActif || "carte");
  await rafraichirEtat();
  chargerPeriodeDisponible();
  rendreRecord(code);
  compterVue("station/" + code, (etat.riviere && etat.riviere.nom_station) || code);
};

// ---------------------------------------------------------- layout « focus »
// Un panneau en grand (actif), les autres en vignettes. Clic pour permuter.
function activerPanneau(nom) {
  etat.panneauActif = nom;
  document.querySelectorAll(".panneau").forEach((p) => {
    const actif = p.dataset.panneau === nom;
    p.classList.toggle("actif", actif);
    p.classList.toggle("miniature", !actif);
  });
  // Les graphiques et la carte doivent se recalculer quand leur conteneur grandit.
  setTimeout(() => {
    if (nom === "carte" && carteZone) carteZone.invalidateSize();
    Object.values(charts).forEach((c) => c.resize());
  }, 90);
}

document.querySelectorAll(".panneau-tete").forEach((tete) => {
  tete.addEventListener("click", () => {
    const p = tete.closest(".panneau");
    if (p) activerPanneau(p.dataset.panneau);
  });
});

async function chargerPeriodeDisponible() {
  const code = etat.code;
  try {
    const p = await api(`/api/riviere/${code}/periode`);
    if (etat.code === code) {
      $("periode-dispo").textContent = `Données disponibles : du ${dateFr(p.debut)} au ${dateFr(p.fin)}` +
        (p.nb_annees ? ` (${p.nb_annees} ans)` : "");
      etat.periode = p;
      if (p.avertissement) {
        $("banniere-avertissement").textContent = "⚠️ " + p.avertissement;
        $("banniere-avertissement").classList.remove("cache");
      }
      if (!$("historique-debut").value) {
        const finD = new Date(p.fin);
        const debutD = new Date(finD);
        debutD.setFullYear(finD.getFullYear() - 1);
        $("historique-fin").value = p.fin;
        $("historique-debut").value = debutD.toISOString().slice(0, 10);
        $("historique-debut").min = p.debut;
        $("historique-fin").max = p.fin;
      }
      // Fenêtre d'apprentissage : par défaut toute la période disponible.
      for (const id of ["opt-train-debut", "opt-train-fin"]) {
        $(id).min = p.debut;
        $(id).max = p.fin;
      }
      // Par défaut : on démarre au max(plancher, début des données) — les années
      // trop anciennes sont lacunaires et peu représentatives du climat actuel.
      // Sur MOBILE, plancher plus récent : moins de données = moins de mémoire
      // (le téléchargement + les matrices d'une longue période font planter le tél.).
      const plancher = (etat.capacites && etat.capacites.mobile) ? "2010-01-01" : "1980-01-01";
      if (!$("opt-train-debut").value) $("opt-train-debut").value = (p.debut < plancher) ? plancher : p.debut;
      if (!$("opt-train-fin").value) $("opt-train-fin").value = p.fin;
      estimerCout();
    }
  } catch (e) {
    $("periode-dispo").textContent = "période : inconnue";
  }
}

// -------------------------------------------------------- quotas & coût API
function classeJauge(pct) {
  return pct >= 85 ? "haut" : pct >= 60 ? "moyen" : "";
}

async function rendreQuota() {
  let q;
  try {
    q = await api("/api/quota");
  } catch (e) {
    return;
  }
  $("tableau-quota").innerHTML = q.fenetres
    .map((f) => {
      const pct = Math.min(100, f.pct);
      return (
        `<div class="quota-ligne">` +
        `<span class="quota-nom">${f.libelle}</span>` +
        `<div class="quota-jauge ${classeJauge(f.pct)}"><span style="width:${pct}%"></span></div>` +
        `<span class="quota-chiffres"><b>${f.utilise}</b> / ` +
        `<input class="quota-limite-input" data-fenetre="${f.fenetre}" type="number" min="1" value="${f.limite}">` +
        (f.reset_dans_s ? `<br><span class="quota-reset">−1 dans ${f.reset_texte}</span>` : "") +
        `</span></div>`
      );
    })
    .join("");
}

$("btn-quota-enregistrer").addEventListener("click", async () => {
  const limites = {};
  document.querySelectorAll(".quota-limite-input").forEach((inp) => {
    const v = parseInt(inp.value);
    if (v > 0) limites[inp.dataset.fenetre] = v;
  });
  try {
    await post("/api/quota/limites", limites);
    toast("Limites de quota enregistrées.");
    rendreQuota();
  } catch (e) {
    toast(e.message, true);
  }
});

$("btn-quota-reinit").addEventListener("click", async () => {
  if (!confirm("Remettre le compteur d'appels Open-Meteo à zéro ? (Les limites restent.)")) return;
  try {
    await post("/api/quota/reinitialiser");
    toast("Compteur remis à zéro.");
    rendreQuota();
  } catch (e) {
    toast(e.message, true);
  }
});

const joursEntre = (a, b) => {
  if (!a || !b) return 0;
  return Math.max(1, Math.round((new Date(b) - new Date(a)) / 86400000) + 1);
};

async function estimerCout() {
  const debut = $("opt-train-debut").value;
  const fin = $("opt-train-fin").value;
  const jours = joursEntre(debut, fin);
  if (!jours) {
    $("cout-estimation").textContent = "";
    return;
  }
  // 2 variables horaires (neige, température) + 1 quotidienne (pluie) par défaut.
  const nVars = 3;
  try {
    const q = await api(`/api/quota?jours=${jours}&vars=${nVars}&points=5`);
    if (q.estimation) {
      const e = q.estimation;
      $("cout-estimation").textContent = (e.tient_dans_le_jour ? "✅ " : "⚠️ ") + e.message;
    }
  } catch (e) {}
}

["opt-train-debut", "opt-train-fin"].forEach((id) => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("change", estimerCout);
});

function revenirCarte() {
  $("vue-station").classList.add("cache");
  $("vue-stations").classList.remove("cache");
  $("btn-retour").classList.add("cache");
  fermerReglages();
  etat.code = null;
  // la station qu'on quitte a pu gagner un modèle → rafraîchir le filtre « Mes modèles »
  rafraichirCodesAvecModeles().then(() => { if ($("filtre-mes-modeles").checked) rendreStations(); });
  setTimeout(() => carteStations && carteStations.invalidateSize(), 100);
}

// Bouton retour ET flèche « précédent » du navigateur reviennent à la carte
// (au lieu de quitter le site) : chaque station est une entrée d'historique.
$("btn-retour").addEventListener("click", () => {
  if (history.state && history.state.station) history.back();
  else revenirCarte();
});
window.addEventListener("popstate", (e) => {
  const code = e.state && e.state.station;
  if (code) ouvrirStation(code, true);
  else if (etat.code) revenirCarte();
});

// ---------------------------------------------------------------- tiroir réglages
function ouvrirReglages() {
  $("drawer-reglages").classList.remove("cache");
  $("drawer-fond").classList.remove("cache");
}
function fermerReglages() {
  $("drawer-reglages").classList.add("cache");
  $("drawer-fond").classList.add("cache");
}
$("btn-ouvrir-reglages").addEventListener("click", ouvrirReglages);
$("btn-fermer-reglages").addEventListener("click", fermerReglages);
$("drawer-fond").addEventListener("click", fermerReglages);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") fermerReglages(); });

// ---------------------------------------------------------------- vue 2 : état
async function rafraichirEtat() {
  try {
    etat.riviere = await api(`/api/riviere/${etat.code}`);
  } catch (e) {
    toast(e.message, true);
    return;
  }
  const r = etat.riviere;

  $("titre-station").textContent = `${r.nom_station || "Station"} (${r.code_station})`;

  const badges = [];
  if (r.zones_definies) badges.push("🗺️ zones ok");
  if (r.coords_finales) badges.push(`📍 ${r.coords_finales.length} points météo`);
  if (r.modeles.length) badges.push(`🧠 ${r.modeles.length} modèle(s)`);
  if (r.sauvegarde_existante) badges.push("💾 sauvegardé");
  $("badges-station").innerHTML = badges.map((b) => `<span class="badge">${b}</span>`).join("");

  $("btn-mode-selection").classList.toggle("cache", !r.zones_definies);

  if (r.predict_day) $("opt-predict-day").value = r.predict_day;
  if (r.past_day) $("opt-past-day").value = r.past_day;

  rendreCarteZone();
  rendreModeles();
  rendreStockage();
  rendreLiensFolium();
  rendreQuota();
  majStatutsPanneaux(r);
  await preparerBacktest();
}

// Petits états affichés sur chaque vignette de panneau.
function majStatutsPanneaux(r) {
  const pret = r.modeles.length > 0;
  const set = (id, txt) => { const el = $(id); if (el) el.textContent = txt; };
  set("etat-carte", r.coords_finales ? `${r.coords_finales.length} points`
                    : r.zones_definies ? "à choisir" : "à définir");
  set("etat-analyse", pret ? "modèle prêt" : "à lancer");
  set("etat-prediction", pret ? "prête" : "—");
  set("etat-test", pret ? "prêt" : "—");
  set("etat-historique", "dispo");
}

// Rafraîchit UNIQUEMENT la carte pendant un job (throttlé à ~2,5 s), pour la
// voir se remplir en direct : candidats → présélection (orange) → finaux (rouge).
async function rafraichirCartePendantJob() {
  const maintenant = Date.now();
  if (etat._dernierRefreshCarte && maintenant - etat._dernierRefreshCarte < 2500) return;
  etat._dernierRefreshCarte = maintenant;
  const code = etat.code;
  try {
    const r = await api(`/api/riviere/${code}`);
    if (etat.code !== code) return;   // l'utilisateur a changé de station entre-temps
    etat.riviere = r;
    rendreCarteZone();
  } catch (e) {}
}

// ---------------------------------------------------------------- carte de zone
function rendreCarteZone() {
  const r = etat.riviere;
  if (!carteZone) {
    carteZone = L.map("carte-zone");
    L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
      attribution: "© OpenStreetMap, © CartoDB",
    }).addTo(carteZone);
    carteZone.on("click", clicCarteZone);
  }
  couchesZone.forEach((c) => carteZone.removeLayer(c));
  couchesZone = [];

  if (r.lat_station != null) {
    // On ne recadre la vue qu'UNE fois par station (sinon chaque rafraîchissement
    // live pendant l'analyse remettrait le zoom/centre à zéro).
    if (!etat.carteAjustee) carteZone.setView([r.lat_station, r.lon_station], 11);
    const st = L.marker([r.lat_station, r.lon_station]).bindTooltip("📍 Station Hub'Eau");
    st.addTo(carteZone);
    couchesZone.push(st);
  }

  if (r.geojson_bassins) {
    const zones = L.geoJSON(r.geojson_bassins, {
      style: { color: "#006400", weight: 2, fillColor: "#2ca02c", fillOpacity: 0.12 },
      onEachFeature: (f, couche) => {
        const nom = f.properties && f.properties.LbZoneHydro;
        if (nom) couche.bindTooltip(nom);
      },
    });
    zones.addTo(carteZone);
    couchesZone.push(zones);
    if (!etat.carteAjustee) {
      try { carteZone.fitBounds(zones.getBounds().pad(0.1)); etat.carteAjustee = true; } catch (e) {}
    }
  }

  // Trois couches distinctes de la sélection en deux temps :
  //   1. tous les candidats de la grille (petits points gris translucides)
  //   2. la présélection ~30 (points orange moyens, altitude en info-bulle)
  //   3. les points finaux ~5 (gros points rouges)
  const cle = (lat, lon) => `${(+lat).toFixed(4)},${(+lon).toFixed(4)}`;
  const ensPre = new Set((r.points_preselectionnes || []).map(([a, b]) => cle(a, b)));
  const ensFin = new Set((r.coords_finales || []).map(([a, b]) => cle(a, b)));

  let nbCandidats = 0;
  if (r.points_par_zone) {
    Object.entries(r.points_par_zone).forEach(([nom, points]) => {
      points.forEach(([lat, lon]) => {
        const k = cle(lat, lon);
        if (ensPre.has(k) || ensFin.has(k)) return; // dessinés au-dessus, plus gros
        nbCandidats++;
        const c = L.circleMarker([lat, lon], {
          radius: 2.5, color: "#9aa7b2", fillColor: "#c3ced8", fillOpacity: 0.45, weight: 0.5,
        }).bindTooltip(`candidat — ${nom}`);
        c.addTo(carteZone);
        couchesZone.push(c);
      });
    });
  }

  if (r.points_preselectionnes) {
    r.points_preselectionnes.forEach(([lat, lon], i) => {
      if (ensFin.has(cle(lat, lon))) return; // les finaux sont dessinés en rouge
      const alt = r.altitudes_preselection ? r.altitudes_preselection[i] : null;
      const c = L.circleMarker([lat, lon], {
        radius: 5, color: "#b25f00", fillColor: "#ffa733", fillOpacity: 0.85, weight: 1,
      }).bindTooltip(`présélection (étape 1)${alt != null ? ` — ${Math.round(alt)} m` : ""}`);
      c.addTo(carteZone);
      couchesZone.push(c);
    });
  }

  if (r.coords_finales) {
    r.coords_finales.forEach(([lat, lon]) => {
      const c = L.circleMarker([lat, lon], {
        radius: 9, color: "#c40000", fillColor: "#ff3333", fillOpacity: 0.9, weight: 2,
      }).bindTooltip("🏆 Point retenu (étape 2)");
      c.addTo(carteZone);
      couchesZone.push(c);
    });
  }

  majLegendeCarte(nbCandidats, ensPre.size, ensFin.size);
  setTimeout(() => carteZone.invalidateSize(), 100);
}

// Légende de la carte (couches de la sélection en deux temps).
function majLegendeCarte(nbCandidats, nbPre, nbFin) {
  const conteneur = $("legende-carte");
  if (!conteneur) return;
  const lignes = [];
  if (nbCandidats) lignes.push(`<span class="pastille" style="background:#c3ced8"></span> ${nbCandidats} candidats`);
  if (nbPre) lignes.push(`<span class="pastille" style="background:#ffa733"></span> ${nbPre} présélectionnés (altitude+couverture)`);
  if (nbFin) lignes.push(`<span class="pastille" style="background:#ff3333"></span> ${nbFin} retenus (pluie+neige)`);
  conteneur.innerHTML = lignes.join(" &nbsp; ");
  conteneur.style.display = lignes.length ? "" : "none";
}

function rendreLiensFolium() {
  const r = etat.riviere;
  $("liens-folium").innerHTML = (r.cartes_folium || [])
    .map((f) => `<a href="/carte/${r.code_station}/${f}" target="_blank">🗺️ ${f}</a>`)
    .join(" ");
}

// ---------------------------------------------------------------- sélection manuelle
function clicCarteZone(evenement) {
  if (!etat.modeSelection) return;
  const { lat, lng } = evenement.latlng;
  const marqueur = L.marker([lat, lng], { draggable: true }).addTo(carteZone);
  marqueur.bindTooltip("Point choisi (cliquer pour retirer)");
  const point = { lat, lon: lng, marker: marqueur };
  marqueur.on("click", () => {
    carteZone.removeLayer(marqueur);
    etat.pointsChoisis = etat.pointsChoisis.filter((p) => p !== point);
    majCompteurPoints();
  });
  marqueur.on("dragend", () => {
    const pos = marqueur.getLatLng();
    point.lat = pos.lat;
    point.lon = pos.lng;
  });
  etat.pointsChoisis.push(point);
  majCompteurPoints();
}

function majCompteurPoints() {
  $("nb-points-choisis").textContent = etat.pointsChoisis.length;
}

$("btn-mode-selection").addEventListener("click", () => {
  etat.modeSelection = true;
  $("btn-mode-selection").classList.add("cache");
  $("btn-lancer-points").classList.remove("cache");
  $("btn-annuler-selection").classList.remove("cache");
  toast("Clique sur la carte pour placer tes points météo (dans ou autour des zones vertes).");
});

function quitterModeSelection() {
  etat.modeSelection = false;
  etat.pointsChoisis.forEach((p) => carteZone.removeLayer(p.marker));
  etat.pointsChoisis = [];
  majCompteurPoints();
  $("btn-mode-selection").classList.remove("cache");
  $("btn-lancer-points").classList.add("cache");
  $("btn-annuler-selection").classList.add("cache");
}

$("btn-annuler-selection").addEventListener("click", quitterModeSelection);

$("btn-lancer-points").addEventListener("click", async () => {
  if (!etat.pointsChoisis.length) return toast("Place au moins un point sur la carte.", true);
  const points = etat.pointsChoisis.map((p) => [p.lat, p.lon]);
  const corps = { points, modele: $("opt-modele").value, ...paramsDonnees() };
  try {
    const { job_id } = await post(`/api/riviere/${etat.code}/points`, corps);
    quitterModeSelection();
    suivreJob(job_id, "Entraînement sur tes points…");
  } catch (e) {
    toast(e.message, true);
  }
});

// ---------------------------------------------------------------- actions jobs
$("btn-zones").addEventListener("click", async () => {
  try {
    const { job_id } = await post(`/api/riviere/${etat.code}/zones`);
    suivreJob(job_id, "Détermination des bassins versants");
  } catch (e) {
    toast(e.message, true);
  }
});

function fenetreApprentissage() {
  return { start_train: $("opt-train-debut").value || null, end_train: $("opt-train-fin").value || null };
}

// Paramètres communs au pipeline / points manuels (tous connectés au moteur).
function paramsDonnees() {
  return {
    past_day: parseInt($("opt-past-day").value) || 20,
    predict_day: parseInt($("opt-predict-day").value) || 15,
    mode_split: $("opt-mode-split").value,
    part_sigma: parseFloat($("opt-part-sigma").value) || 0.3,
    part_eval: parseFloat($("opt-part-eval").value) || 0.2,
    temp_mode: $("opt-temp-mode").value,
    var_modele: $("opt-var-modele").value,
    ...fenetreApprentissage(),
  };
}

// 999 dans le sélecteur = 99,9 %.
function seuilEnergie() {
  const v = $("opt-pca-energie").value;
  return v === "999" ? 99.9 : parseInt(v);
}

// ---------------------------------------------- meilleur score LOCAL (par station + horizon)
// Stocké dans le navigateur (localStorage) : aucun partage entre utilisateurs
// pour l'instant. Un bouton propose de recharger les réglages du meilleur essai.
const CHAMPS_PARAMS = ["opt-modele", "opt-predict-day", "opt-past-day", "opt-temp-mode", "opt-mode-split",
  "opt-part-sigma", "opt-part-eval", "opt-var-modele", "opt-n-preselection", "opt-n-final",
  "opt-poids-pluie", "opt-poids-neige", "opt-poids-altitude", "opt-pca-energie"];
let PARAMS_DEFAUT = null;   // valeurs initiales des champs, capturées au chargement

function snapshotParams() {
  const s = {};
  for (const id of CHAMPS_PARAMS) { const el = $(id); if (el) s[id] = el.value; }
  return s;
}
function appliquerParams(snap) {
  for (const id of CHAMPS_PARAMS) { const el = $(id); if (el && snap[id] != null) el.value = snap[id]; }
  estimerCout();
}
function lireRecords() {
  try { return JSON.parse(localStorage.getItem("riverlab:records") || "{}"); } catch (e) { return {}; }
}
function cleRecord(code, horizon) { return `${code}|${horizon}`; }

// Appelé à la fin d'un entraînement : mémorise les réglages si le score bat le record.
function enregistrerRecord(code, horizon, score) {
  if (!code || score == null || !isFinite(score)) return;
  const recs = lireRecords();
  const cle = cleRecord(code, horizon);
  if (!recs[cle] || score > recs[cle].score) {
    recs[cle] = { score, params: snapshotParams(), date: new Date().toISOString().slice(0, 10) };
    try { localStorage.setItem("riverlab:records", JSON.stringify(recs)); } catch (e) {}
  }
  rendreRecord(code);
}

// Affiche le bouton "meilleur score" pour la station + l'horizon courants (s'il existe).
function rendreRecord(code) {
  const el = $("record-params");
  if (!el) return;
  const horizon = parseInt($("opt-predict-day").value) || 15;
  const rec = lireRecords()[cleRecord(code, horizon)];
  if (!rec) { el.classList.add("cache"); el.innerHTML = ""; return; }
  el.classList.remove("cache");
  if (etat.recordCharge) {
    el.innerHTML = `<button class="mini" id="btn-record-defaut">↩ Revenir aux réglages par défaut</button>`;
    $("btn-record-defaut").onclick = () => {
      if (PARAMS_DEFAUT) appliquerParams(PARAMS_DEFAUT);
      etat.recordCharge = false; rendreRecord(code);
    };
  } else {
    el.innerHTML = `<button class="mini" id="btn-record-charger" title="Charge les réglages qui ont donné ce score sur cet appareil, puis relance l'analyse">` +
      `🏆 Meilleur ici : R² ${(rec.score * 100).toFixed(0)} % à J+${horizon} — charger ces réglages</button>`;
    $("btn-record-charger").onclick = () => {
      appliquerParams(rec.params); etat.recordCharge = true; rendreRecord(code);
      toast("Réglages du meilleur score chargés. Clique sur « Analyser cette rivière » pour relancer.");
    };
  }
}

$("btn-pipeline").addEventListener("click", async () => {
  const corps = {
    modele: $("opt-modele").value,
    seuil_energie: seuilEnergie(),
    ...paramsDonnees(),
    selection: {
      n_preselection: parseInt($("opt-n-preselection").value) || 30,
      n_final: parseInt($("opt-n-final").value) || 5,
      poids_pluie: parseFloat($("opt-poids-pluie").value),
      poids_neige: parseFloat($("opt-poids-neige").value),
      poids_altitude: parseFloat($("opt-poids-altitude").value),
    },
  };
  try {
    const { job_id } = await post(`/api/riviere/${etat.code}/pipeline`, corps);
    activerPanneau("carte");   // on regarde les points apparaître en direct
    suivreJob(job_id, "Analyse de la rivière en cours…");
  } catch (e) {
    toast(e.message, true);
  }
});

$("btn-donnees").addEventListener("click", async () => {
  try {
    const { job_id } = await post(`/api/riviere/${etat.code}/donnees`, paramsDonnees());
    suivreJob(job_id, "Téléchargement des données complètes (sans entraînement)");
  } catch (e) {
    toast(e.message, true);
  }
});

let chartPca = null;
$("btn-pca").addEventListener("click", async () => {
  $("btn-pca").disabled = true;
  try {
    const res = await api(`/api/riviere/${etat.code}/pca?seuil=${seuilEnergie()}`);
    dessinerPca(res);
  } catch (e) {
    toast(e.message, true);
  } finally {
    $("btn-pca").disabled = false;
  }
});

function dessinerPca(res) {
  $("pca-resultat").innerHTML =
    `<b>${res.n_composantes}</b> composantes suffisent pour garder <b>${res.energie}%</b> de l'information ` +
    `(sur ${res.n_features} variables) — soit ${Math.round((1 - res.n_composantes / res.n_features) * 100)}% de compression.`;
  $("zone-pca").style.display = "";
  if (chartPca) chartPca.destroy();
  chartPca = new Chart($("chart-pca"), {
    type: "line",
    data: {
      labels: res.variance_cumulee.map((_, i) => i + 1),
      datasets: [{
        data: res.variance_cumulee, borderColor: "#0f9488",
        backgroundColor: "rgba(15,148,136,.12)", fill: true, pointRadius: 0, borderWidth: 2, tension: 0.2,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { title: (t) => `${t[0].label} composantes`, label: (c) => `${c.parsed.y.toFixed(1)} % d'énergie` } },
      },
      scales: {
        x: { title: { display: true, text: "nombre de composantes" }, ticks: { maxTicksLimit: 10 } },
        y: { min: 0, max: 100, title: { display: true, text: "énergie conservée (%)" } },
      },
    },
  });
}

$("btn-entrainer").addEventListener("click", async () => {
  const modeles = Array.from(document.querySelectorAll(".case-modele:checked")).map((c) => c.value);
  if (!modeles.length) return toast("Coche au moins un modèle à entraîner.", true);
  const corps = { modeles, seuil_energie: seuilEnergie(), var_modele: $("opt-var-modele").value };
  try {
    const { job_id } = await post(`/api/riviere/${etat.code}/entrainer`, corps);
    suivreJob(job_id, `Entraînement de ${modeles.length} modèle(s)`);
  } catch (e) {
    toast(e.message, true);
  }
});

document.querySelectorAll(".btn-un-modele").forEach((bouton) => {
  bouton.addEventListener("click", async () => {
    const modele = bouton.dataset.modele;
    const corps = { modele, seuil_energie: seuilEnergie(), var_modele: $("opt-var-modele").value };
    try {
      const { job_id } = await post(`/api/riviere/${etat.code}/entrainer-modele`, corps);
      suivreJob(job_id, `Entraînement : ${NOMS_MODELES[modele] || modele}`);
    } catch (e) {
      toast(e.message, true);
    }
  });
});

// ---------------------------------------------------------------- suivi de job
function suivreJob(jobId, titre) {
  etat.jobId = jobId;
  etat.jobLignesLues = 0;
  $("barre-job").classList.remove("cache");
  $("btn-arreter-job").classList.remove("cache");
  $("btn-arreter-job").disabled = false;
  $("btn-arreter-job").textContent = "Arrêter";
  $("btn-plier-log").classList.remove("cache");
  $("job-log").classList.add("cache");
  $("job-titre").textContent = titre;
  $("job-statut").textContent = "En cours…";
  $("job-log").textContent = "";
  rendreProgres(null);
  bouclePollingJob();
}

// Affiche l'avancement de l'étape en cours (barre déterminée ou indéterminée).
function rendreProgres(p) {
  const conteneur = $("job-progres");
  const piste = $("job-progres-piste");
  if (!p || !p.phase) {
    conteneur.classList.add("cache");
    return;
  }
  conteneur.classList.remove("cache");
  $("job-progres-phase").textContent = p.phase;
  if (p.pct == null) {
    // étape à durée inconnue → bande animée, pas de %
    piste.classList.add("indetermine");
    $("job-progres-pct").textContent = "";
    $("job-progres-barre").style.width = "";
  } else {
    piste.classList.remove("indetermine");
    $("job-progres-pct").textContent = `${p.pct}%`;
    $("job-progres-barre").style.width = p.pct + "%";
  }
}

$("btn-arreter-job").addEventListener("click", async () => {
  if (etat.jobId == null) return;
  $("btn-arreter-job").disabled = true;
  $("btn-arreter-job").textContent = "Arrêt en cours…";
  $("job-statut").textContent = "Arrêt en cours (fin de l'étape en cours)…";
  $("job-progres-piste").classList.add("indetermine");
  try {
    await post(`/api/jobs/${etat.jobId}/arreter`);
  } catch (e) {
    toast(e.message, true);
  }
});

async function bouclePollingJob() {
  if (etat.jobId == null) return;
  let job;
  try {
    job = await api(`/api/jobs/${etat.jobId}?depuis=${etat.jobLignesLues}`);
  } catch (e) {
    $("job-statut").textContent = "⚠️ suivi interrompu : " + e.message;
    return;
  }

  if (job.log.length) {
    $("job-log").textContent += job.log.join("\n") + "\n";
    $("job-log").scrollTop = $("job-log").scrollHeight;
    etat.jobLignesLues = job.nb_lignes_log;
  }

  rendreProgres(job.progression);

  if (job.statut === "en_cours" || job.statut === "en_attente") {
    rendreQuota();   // un téléchargement en cours consomme du quota : on le voit bouger
    rafraichirCartePendantJob();   // la carte se remplit au fil de l'analyse (candidats → présélection → finaux)
    setTimeout(bouclePollingJob, 1200);
    return;
  }

  $("btn-arreter-job").classList.add("cache");
  $("job-progres").classList.add("cache");
  $("job-progres-piste").classList.remove("indetermine");

  if (job.statut === "termine") {
    $("job-statut").textContent = "Terminé ✅";
    const r = job.resultat || {};
    if (r.score_gradient_boosting != null) {
      const parts = [`✅ Modèle prêt — fiabilité ${(r.score_gradient_boosting * 100).toFixed(0)} %`];
      if (r.temps_reponse_jours != null) parts.push(`temps de réponse du bassin ~${r.temps_reponse_jours} j`);
      $("analyse-diagnostic").textContent = parts.join(" · ");
      toast("Analyse terminée : le modèle de prévision est prêt.");
    } else if (r.temps_reponse_jours != null) {
      toast(`Terminé ! Temps de réponse du bassin : ~${r.temps_reponse_jours} j.`);
    } else {
      toast("Terminé !");
    }
    fermerFenetreJobBientot();
  } else if (job.statut === "arrete") {
    $("job-statut").textContent = "Arrêté.";
    toast("Calcul arrêté.");
    fermerFenetreJobBientot();
  } else {
    $("job-statut").textContent = "Erreur : " + (job.erreur || "inconnue");
    toast("Échec : " + (job.erreur || ""), true);
  }
  etat.jobId = null;
  rendreQuota();
  await rafraichirEtat();
  // mémorise localement les réglages si ce nouvel entraînement bat le record de la station+horizon
  if (job.statut === "termine" && etat.riviere && (etat.riviere.modeles || []).length) {
    const best = Math.max(...etat.riviere.modeles.map((m) => m.score || 0));
    enregistrerRecord(etat.code, parseInt($("opt-predict-day").value) || 15, best);
  }
}

// Masque la fenêtre quelques secondes après la fin (sauf si un nouveau job a démarré).
function fermerFenetreJobBientot() {
  setTimeout(() => {
    if (etat.jobId == null) $("barre-job").classList.add("cache");
  }, 6000);
}

$("btn-plier-log").addEventListener("click", () => {
  const cache = $("job-log").classList.toggle("cache");
  $("btn-plier-log").textContent = cache ? "Voir le détail" : "Masquer le détail";
});

$("btn-reduire-job").addEventListener("click", () => {
  const reduit = $("barre-job").classList.toggle("reduit");
  $("btn-reduire-job").textContent = reduit ? "▢" : "–";
  $("btn-reduire-job").title = reduit ? "Agrandir" : "Réduire";
});

// Choix du coin d'ancrage : bas-droite → bas-gauche → haut-gauche → haut-droite
const COINS_JOB = ["", "coin-bl", "coin-tl", "coin-tr"];
let iCoinJob = 0;
$("btn-coin-job").addEventListener("click", () => {
  const b = $("barre-job");
  COINS_JOB.filter(Boolean).forEach((c) => b.classList.remove(c));
  iCoinJob = (iCoinJob + 1) % COINS_JOB.length;
  if (COINS_JOB[iCoinJob]) b.classList.add(COINS_JOB[iCoinJob]);
});

// ---------------------------------------------------------------- modèles
function rendreModeles() {
  const r = etat.riviere;
  const conteneur = $("tableau-modeles");
  let html = "";
  if (!r.modeles.length) {
    html = "<p class='aide'>Aucun modèle entraîné : lance le pipeline (ou choisis tes points à la main).</p>";
  } else {
    html =
      "<table class='tab-modeles'><tr><th>Modèle</th><th>R²</th>" +
      "<th title='Couverture réelle de l’IC 95% sur données jamais vues'>IC 95%</th>" +
      "<th title='Qualité des intervalles : précision ET finesse (0-100)'>note</th><th></th></tr>" +
      r.modeles.map((m) => {
        const cov = m.couverture || {};
        const ic = cov["95"] != null ? `${cov["95"]}%` : "—";
        const note = cov.note != null
          ? `${cov.note}${cov.note_proche != null ? ` <span class="aide" title="1re semaine J+1→J+7">(${cov.note_proche})</span>` : ""}`
          : "—";
        return `<tr><td>${NOMS_MODELES[m.nom] || m.nom}</td><td><b>${(m.score * 100).toFixed(0)} %</b></td>` +
          `<td>${ic}</td><td>${note}</td><td class="actions-modele">` +
          (r.donnees_en_memoire ? `<button class="mini" data-incert="${m.nom}" title="Ré-entraîner l'incertitude de ce modèle">🎯</button>` : "") +
          `<button class="mini-del" data-delmod="${m.nom}" title="Supprimer ce modèle">✕</button></td></tr>`;
      }).join("") + "</table>";
  }
  // Données d'entraînement en mémoire : réutilisables (ajout de modèle, PCA, incertitude), supprimables.
  if (r.donnees_en_memoire) {
    html += `<p class="aide donnees-mem">🗃️ <b>Données d'entraînement</b> gardées en mémoire — ${octetsLisibles(r.octets_donnees || 0)} ` +
      `<span class="ind">(pour ajouter un modèle, refaire une PCA, ou ré-entraîner l'incertitude 🎯)</span> ` +
      `<button class="mini-del" data-deldata="1" title="Libérer la mémoire">✕</button></p>`;
  } else if (r.modeles.length) {
    html += `<p class="aide ind">Données d'entraînement libérées : ré-analyse la rivière pour ajouter d'autres modèles.</p>`;
  }
  conteneur.innerHTML = html;
  conteneur.querySelectorAll("[data-incert]").forEach((b) => b.addEventListener("click", () => reentrainerIncertitude(b.dataset.incert)));
  conteneur.querySelectorAll("[data-delmod]").forEach((b) => b.addEventListener("click", () => supprimerStockage(etat.code, "modele:" + b.dataset.delmod)));
  conteneur.querySelectorAll("[data-deldata]").forEach((b) => b.addEventListener("click", () => supprimerStockage(etat.code, "donnees")));

  const options = r.modeles.map((m) => `<option value="${m.nom}">${NOMS_MODELES[m.nom] || m.nom}</option>`).join("");
  $("backtest-modele").innerHTML = options;
  $("prevision-modele").innerHTML = options;
  const prefere = ["gradient_boosting", "ridge", "lineaire"].find((n) => r.modeles.some((m) => m.nom === n));
  if (prefere) { $("backtest-modele").value = prefere; $("prevision-modele").value = prefere; }

  // Les actions qui réutilisent les données d'entraînement dépendent de leur présence en mémoire.
  const sansData = !r.donnees_en_memoire;
  $("btn-pca").disabled = sansData;
  $("btn-entrainer").disabled = sansData;
  document.querySelectorAll(".btn-un-modele").forEach((b) => (b.disabled = sansData));
  $("btn-donnees").disabled = !r.coords_finales;
}

async function reentrainerIncertitude(nom) {
  try {
    const { job_id } = await post(`/api/riviere/${etat.code}/incertitude`,
      { modele: nom, var_modele: $("opt-var-modele").value });
    suivreJob(job_id, `Incertitude de ${NOMS_MODELES[nom] || nom} (${NOMS_MODELES[$("opt-var-modele").value] || $("opt-var-modele").value})`);
  } catch (e) {
    toast(e.message, true);
  }
}

// ---------------------------------------------------------------- backtest
async function preparerBacktest() {
  const r = etat.riviere;
  const possible = r.modeles.length > 0;
  $("btn-backtest").disabled = !possible;
  $("btn-prevision").disabled = !possible;
  if (!possible) return;
  try {
    const d = await api(`/api/riviere/${etat.code}/dates-test`);
    const champ = $("backtest-date");
    champ.min = d.min;
    champ.max = d.max;
    if (!champ.value) champ.value = d.dates[Math.floor(d.dates.length / 2)];
    etat.datesTest = d.dates;
    etat.anneesTest = d.annees_test || [];
    $("backtest-annees").textContent = etat.anneesTest.length
      ? `années de test : ${etat.anneesTest.join(", ")}` : "";
  } catch (e) {
    $("btn-backtest").disabled = true;
  }
}

$("btn-backtest").addEventListener("click", async () => {
  const modele = $("backtest-modele").value;
  let date = $("backtest-date").value;
  if (!date) return toast("Choisis une date de test.", true);
  // Recale sur la date de test disponible la plus proche (si on a la liste).
  if (etat.datesTest && etat.datesTest.length && !etat.datesTest.includes(date)) {
    const cible = new Date(date).getTime();
    date = etat.datesTest.reduce((a, b) =>
      Math.abs(new Date(a) - cible) < Math.abs(new Date(b) - cible) ? a : b
    );
    $("backtest-date").value = date;
    toast(`Date ajustée à la plus proche disponible : ${dateFr(date)}`);
  }
  const nbJours = parseInt($("backtest-jours").value) || 15;
  // Si la date n'est pas déjà dans le jeu stocké, le moteur télécharge la fenêtre.
  const stockee = etat.datesTest && etat.datesTest.includes(date);
  $("btn-backtest").disabled = true;
  $("btn-backtest").textContent = stockee ? "Test…" : "Téléchargement…";
  try {
    const res = await api(`/api/riviere/${etat.code}/backtest?modele=${modele}&date=${date}&hybride=1&nb_jours=${nbJours}`);
    etat.dernierBacktest = res;   // gardé pour re-dessiner quand on change les IC cochés
    rendreFanChart("chart-backtest", res, true, nivIC("backtest"));
    rendreFiabilite("fiab-backtest", res);
    $("vide-backtest").style.display = "none";
    $("scores-detail").innerHTML = tableauR2(res.scores_detail, nbJours);
  } catch (e) {
    toast(e.message, true);
  } finally {
    $("btn-backtest").disabled = false;
    $("btn-backtest").textContent = "Tester";
  }
});

// ---------------------------------------------------------------- prévision réelle
$("btn-prevision").addEventListener("click", async () => {
  const modele = $("prevision-modele").value;
  const nbJours = parseInt($("prevision-jours").value) || 15;
  $("btn-prevision").disabled = true;
  $("btn-prevision").textContent = "Téléchargement météo prévisionnelle…";
  try {
    const res = await api(`/api/riviere/${etat.code}/prevision?modele=${modele}&hybride=1&nb_jours=${nbJours}`);
    etat.dernierePrevision = res;   // gardé pour re-dessiner quand on change les IC cochés
    rendreFanChart("chart-prevision", res, false, nivIC("prevision"));
    rendreFiabilite("fiab-prevision", res);
    $("vide-prevision").style.display = "none";
    $("scores-detail-prevision").innerHTML = tableauR2(res.scores_detail, nbJours);
    // Tableau détaillé (dans le popover ⓘ), à partir des points prévus (hors pivot).
    const prev = (res.points || []).slice(1);
    const avecIC = res.hybride;
    $("tableau-prevision").innerHTML =
      "<table><tr><th>Date</th><th>Débit (m³/s)</th>" +
      (avecIC ? "<th>IC 95% (m³/s)</th>" : "") + "</tr>" +
      prev
        .map(
          (p) =>
            `<tr><td>${dateFr(p.date)}</td><td><b>${fmtM3(p.prev)}</b></td>` +
            (avecIC ? `<td>${fmtM3(p.ic95_bas)} – ${fmtM3(p.ic95_haut)}</td>` : "") + "</tr>"
        )
        .join("") +
      "</table>";
  } catch (e) {
    toast(e.message, true);
  } finally {
    $("btn-prevision").disabled = false;
    $("btn-prevision").textContent = "Prédire le futur";
  }
});

// Changer les IC cochés re-dessine le graphe concerné (sans refaire de calcul).
document.querySelectorAll(".ic-niv").forEach((c) => {
  c.addEventListener("change", () => {
    if (c.dataset.graphe === "backtest" && etat.dernierBacktest) {
      rendreFanChart("chart-backtest", etat.dernierBacktest, true, nivIC("backtest"));
    } else if (c.dataset.graphe === "prevision" && etat.dernierePrevision) {
      rendreFanChart("chart-prevision", etat.dernierePrevision, false, nivIC("prevision"));
    }
  });
});

// ---------------------------------------------------------------- historique
$("btn-historique").addEventListener("click", async () => {
  const debut = $("historique-debut").value;
  const fin = $("historique-fin").value;
  if (!debut || !fin) return toast("Choisis les deux dates.", true);
  $("btn-historique").disabled = true;
  try {
    const res = await api(`/api/riviere/${etat.code}/historique?debut=${debut}&fin=${fin}`);
    rendreHistoChart("chart-historique", res.serie || []);
    $("vide-historique").style.display = "none";
    // Le backend renvoie déjà les débits en m³/s et les dates en JJ/MM/AAAA.
    const m3 = (v) => v.toLocaleString("fr-FR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    $("stats-historique").innerHTML =
      `<table><tr><th>Jours</th><th>Moyenne</th><th>Min</th><th>Max</th></tr>` +
      `<tr><td>${res.nb_jours}</td><td>${m3(res.debit_moyen)} m³/s</td>` +
      `<td>${m3(res.debit_min)} m³/s (${res.date_min})</td>` +
      `<td>${m3(res.debit_max)} m³/s (${res.date_max})</td></tr></table>`;
  } catch (e) {
    toast(e.message, true);
  } finally {
    $("btn-historique").disabled = false;
  }
});

// ---------------------------------------------------------------- stockage
function rendreStockage() {
  $("btn-sauvegarder").disabled = !(etat.riviere && etat.riviere.modeles.length);
}

// Inventaire complet : toutes les stations, avec suppression fine.
async function rendreInventaireStockage() {
  const conteneur = $("inventaire-stockage");
  conteneur.innerHTML = "<p class='aide'>Chargement…</p>";
  let inv;
  try {
    inv = await api("/api/stockage");
  } catch (e) {
    conteneur.innerHTML = "<p class='aide'>Inventaire indisponible.</p>";
    return;
  }
  $("stockage-total").textContent = octetsLisibles(inv.octets_total);
  if (!inv.stations.length) {
    conteneur.innerHTML = "<p class='aide'>Aucune station stockée sur cet appareil. ✨</p>";
    return;
  }
  const del = (code, cible, label) =>
    `<button class="mini-del" data-code="${code}" data-cible="${cible}" title="Supprimer ${label}">✕</button>`;

  conteneur.innerHTML = inv.stations.map((s) => {
    const lignes = [];
    for (const m of s.modeles) {
      const r2 = m.score != null ? ` · R² ${(m.score * 100).toFixed(0)}%` : "";
      lignes.push(`<li><span>🧠 ${m.nom} — <b>${octetsLisibles(m.octets)}</b>${r2}</span>${del(s.code, "modele:" + m.nom, "ce modèle")}</li>`);
    }
    if (s.octets_test > 0)
      lignes.push(`<li><span>🎯 jeu de test — <b>${octetsLisibles(s.octets_test)}</b> <span class="ind">(pour les tests sur le passé)</span></span>${del(s.code, "test", "le jeu de test")}</li>`);
    if (s.octets_travail > 0)
      lignes.push(`<li><span>🧰 fichiers de travail — <b>${octetsLisibles(s.octets_travail)}</b> <span class="ind">(régénérables)</span></span>${del(s.code, "travail", "les fichiers de travail")}</li>`);
    return (
      `<div class="bloc-station">` +
      `<div class="bloc-station-tete"><b>${s.nom || s.code}</b> <span class="ind">${s.code}</span>` +
      `<span class="bloc-station-taille">${octetsLisibles(s.octets_total)}</span>` +
      `<button class="secondaire mini" data-export="${s.code}" data-nom="${(s.nom || s.code).replace(/"/g, "")}" title="Télécharger ce modèle dans un fichier">⤓ Exporter</button>` +
      `<button class="secondaire danger mini" data-code="${s.code}" data-cible="station">Supprimer</button></div>` +
      (lignes.length ? `<ul class="liste-stockage">${lignes.join("")}</ul>` : "") +
      `</div>`
    );
  }).join("");

  conteneur.querySelectorAll("[data-cible]").forEach((btn) => {
    btn.addEventListener("click", () => supprimerStockage(btn.dataset.code, btn.dataset.cible));
  });
  conteneur.querySelectorAll("[data-export]").forEach((btn) => {
    btn.addEventListener("click", () => exporterModele(btn.dataset.export, btn.dataset.nom));
  });
}

// ---- partage de modèles : export/import en BINAIRE (léger, pas de base64) ----
async function exporterModele(code, nom) {
  try {
    const bytes = await window.RIVER_EXPORT(code);   // Uint8Array
    const blob = new Blob([bytes], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${(nom || code).replace(/[^\w\-]+/g, "_")}.riverlab`;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    toast(`Modèle exporté (${octetsLisibles(blob.size)}). Tu peux envoyer ce fichier.`);
  } catch (e) {
    toast("Export impossible : " + e.message, true);
  }
}

$("btn-importer-modele").addEventListener("click", () => $("fichier-modele").click());
$("fichier-modele").addEventListener("change", async (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  ev.target.value = "";  // permet de ré-importer le même fichier
  try {
    // Le buffer est « transféré » (détaché) à chaque appel -> on relit le fichier au besoin.
    let res = await window.RIVER_IMPORT(await file.arrayBuffer(), "demander");
    if (res.conflit) {
      const infos = (x) => `${x.modeles.join(", ")} — R² <b>${x.r2}%</b>, prévoit <b>${x.horizon} j</b>`;
      const html =
        `<p>Un modèle existe déjà pour <b>${res.nom}</b>. Que faire ?</p>` +
        `<ul class="liste-choix"><li>📍 Actuel : ${infos(res.actuel)}</li>` +
        `<li>📥 Importé : ${infos(res.importe)}</li></ul>`;
      const mode = await demanderChoix("Modèle déjà présent", html, [
        { label: "Garder le meilleur", valeur: "meilleur", classe: "primaire" },
        { label: "Garder les deux", valeur: "fusionner" },
        { label: "Remplacer", valeur: "remplacer" },
        { label: "Annuler", valeur: null },
      ]);
      if (!mode) return;
      res = await window.RIVER_IMPORT(await file.arrayBuffer(), mode);
    }
    if (res.erreur) throw new Error(res.erreur);
    toast(`Importé : ${res.nom} — ${res.modeles.join(", ")} (R² ${res.r2}%, ${res.horizon} j).`);
    await rendreInventaireStockage();
    if (res.code) {
      $("drawer-reglages").classList.add("cache");
      $("drawer-fond").classList.add("cache");
      ouvrirStation(res.code);
    }
  } catch (e) {
    toast("Import impossible : " + e.message, true);
  }
});

// Petite fenêtre de choix (renvoie la valeur du bouton cliqué, ou null).
function demanderChoix(titre, html, boutons) {
  return new Promise((resolve) => {
    const fond = document.createElement("div");
    fond.className = "modal-fond";
    fond.innerHTML =
      `<div class="modal-boite"><h3>${titre}</h3><div class="modal-corps">${html}</div>` +
      `<div class="modal-actions"></div></div>`;
    const actions = fond.querySelector(".modal-actions");
    boutons.forEach((b) => {
      const btn = document.createElement("button");
      btn.textContent = b.label;
      btn.className = b.classe || "secondaire";
      btn.addEventListener("click", () => { fond.remove(); resolve(b.valeur); });
      actions.appendChild(btn);
    });
    fond.addEventListener("click", (e) => { if (e.target === fond) { fond.remove(); resolve(null); } });
    document.body.appendChild(fond);
  });
}

async function supprimerStockage(code, cible) {
  const nom = cible === "station" ? "TOUTE la station " + code
    : cible === "test" ? "le jeu de test de " + code + " (les tests sur le passé ne marcheront plus)"
    : cible === "travail" ? "les fichiers de travail de " + code
    : cible === "donnees" ? "les données d'entraînement en mémoire (tu ne pourras plus ajouter de modèle ni ré-entraîner l'incertitude sans ré-analyser)"
    : "le modèle « " + (NOMS_MODELES[cible.split(":")[1]] || cible.split(":")[1]) + " » de " + code;
  if (!confirm("Supprimer " + nom + " ?")) return;
  try {
    const res = await post("/api/stockage/supprimer", { code, cible });
    toast(`Supprimé : ${res.libelle} — ${octetsLisibles(res.octets_liberes)} libérés.`);
    await rendreInventaireStockage();
    if (code === etat.code) await rafraichirEtat();
  } catch (e) {
    toast(e.message, true);
  }
}

$("btn-rafraichir-stockage").addEventListener("click", rendreInventaireStockage);

$("btn-sauvegarder").addEventListener("click", async () => {
  try {
    await post(`/api/riviere/${etat.code}/sauvegarder`);
    toast("Modèles et points sauvegardés. 💾");
    await rafraichirEtat();
    rendreInventaireStockage();
  } catch (e) {
    toast(e.message, true);
  }
});

// Charge l'inventaire à l'ouverture du tiroir Réglages.
$("btn-ouvrir-reglages").addEventListener("click", rendreInventaireStockage);

// -------------------------------------------------- compteur de visites (GoatCounter)
function chargerAnalytics(code) {
  // `code` = sous-domaine GoatCounter (ex. "riverlab") ou URL /count complète.
  const url = code.includes("://") ? code : `https://${code}.goatcounter.com/count`;
  const s = document.createElement("script");
  s.async = true;
  s.setAttribute("data-goatcounter", url);
  s.src = "//gc.zgo.at/count.js";
  document.body.appendChild(s);
  etat.analytics = true;   // compte la page ; les ouvertures de station sont comptées à part
}
// Compte une "page" logique (SPA) — ex. l'ouverture d'une station.
function compterVue(chemin, titre) {
  if (etat.analytics && window.goatcounter && window.goatcounter.count) {
    window.goatcounter.count({ path: chemin, title: titre || chemin });
  }
}

// ---------------------------------------------------------------- mode public
async function chargerConfig() {
  let cfg = {};
  try { cfg = await api("/api/config"); } catch (e) {}
  if (cfg.analytics) chargerAnalytics(cfg.analytics);
  if (!cfg.public) return;
  document.body.classList.add("public");
  // Masque tout ce qui entraîne / modifie (le backend le bloque de toute façon).
  ["btn-ouvrir-reglages", "btn-pipeline", "btn-zones", "btn-mode-selection"].forEach((id) => {
    const el = $(id); if (el) el.classList.add("cache");
  });
  const note = $("analyse-diagnostic");
  if (note) note.innerHTML =
    "🌍 <b>Démo en ligne</b> — rivières déjà analysées, prêtes à explorer (prévision, test sur le passé, historique). " +
    "Pour analyser <i>tes</i> rivières, installe le projet en local depuis le dépôt GitHub.";
}

// -------------------------------------------------- détection des capacités de l'appareil
// Sert à décider si le calcul parallèle (pool de workers Pyodide) vaut le coup.
function estMobile() {
  return /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent || "") ||
    (navigator.maxTouchPoints > 1 && window.matchMedia && window.matchMedia("(pointer: coarse)").matches);
}
function capacitesAppareil() {
  const coeurs = navigator.hardwareConcurrency || null;   // nb de threads logiques
  const memGo = navigator.deviceMemory || null;           // Go approx (Chrome/Edge seulement)
  const mobile = estMobile();
  // Un pool coûte ~1 instance Pyodide (~100 Mo + démarrage) par worker. Sur mobile
  // c'est la mort assurée (l'onglet est tué par manque de RAM) -> jamais de pool.
  let pool = 0, verdict;
  if (mobile) {
    verdict = "mobile — parallélisme désactivé (mémoire limitée)";
  } else if (coeurs && memGo) {
    if (coeurs >= 6 && memGo >= 8) { pool = Math.min(coeurs - 2, Math.floor(memGo / 3), 4); }
    verdict = pool >= 2
      ? `parallélisme envisageable (jusqu'à ${pool} workers)`
      : "parallélisme peu rentable ici (démarrage/mémoire > gain)";
  } else if (coeurs) {
    if (coeurs >= 6) { pool = Math.min(coeurs - 2, 4); verdict = `parallélisme envisageable (${pool} workers), mémoire inconnue`; }
    else verdict = "parallélisme peu rentable ici";
  } else {
    verdict = "capacités non détectables — parallélisme non recommandé";
  }
  return { coeurs, memGo, pool, mobile, verdict };
}
function rendreInfosAppareil() {
  const el = $("infos-appareil");
  if (!el) return;
  const c = capacitesAppareil();
  const cœurs = c.coeurs ? `${c.coeurs} cœurs` : "cœurs inconnus";
  const mem = c.memGo ? `~${c.memGo} Go RAM` : "RAM inconnue";
  el.innerHTML = `<b>${cœurs}</b>, <b>${mem}</b> → ${c.verdict}.`;
  etat.capacites = c;
}

// Applique le réglage de calcul parallèle : taille effective = pool de l'appareil
// SEULEMENT si l'utilisateur l'a activé, sinon 0 (séquentiel).
function appliquerParallele() {
  const sel = $("opt-parallele");
  const active = sel && sel.value === "on";
  const taille = active ? (etat.capacites && etat.capacites.pool) || 0 : 0;
  try { localStorage.setItem("riverlab:parallele", active ? "on" : "off"); } catch (e) {}
  if (window.RIVER_CONFIG_POOL) window.RIVER_CONFIG_POOL(taille);
  if (active && taille < 2) toast("Ton appareil n'a pas assez de marge : le calcul restera séquentiel.");
}
function initParallele() {
  const sel = $("opt-parallele");
  if (!sel) return;
  let pref = "off";
  try { pref = localStorage.getItem("riverlab:parallele") || "off"; } catch (e) {}
  sel.value = pref;
  sel.addEventListener("change", appliquerParallele);
  appliquerParallele();
}

// ---------------------------------------------------------------- démarrage
rendreInfosAppareil();
initParallele();
PARAMS_DEFAUT = snapshotParams();   // réglages par défaut (pour le bouton "revenir aux réglages par défaut")
$("opt-predict-day").addEventListener("change", () => { if (etat.code) rendreRecord(etat.code); });
initStations();
chargerConfig();
