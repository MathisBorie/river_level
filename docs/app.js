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

// Fan chart : passé observé + prévision + bandes IC (+ réel pour le backtest).
function rendreFanChart(canvasId, data, avecReel) {
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
    bandes.forEach(([niv, couleur]) => {
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
  gradient_boosting: "Gradient Boosting", ridge_causal: "Ridge causal",
  ridge_brut: "Ridge", lineaire_pca: "Linéaire (PCA)", keras_brut: "Réseau de neurones",
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
  el.innerHTML = `<span class="details-r2">${nom} — fiabilité globale</span> ` +
    `<span class="badge-r2 ${classeR2(pct)}">R² ${pct}%</span> ${detail}`;
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
    return;
  }
  rendreStations();

  $("recherche-station").addEventListener("input", rendreStations);
  $("filtre-service").addEventListener("change", rendreStations);
}

function rendreStations() {
  if (!Array.isArray(etat.stations)) {
    console.error("[stations] réponse inattendue :", etat.stations);
    toast("Réponse stations inattendue (voir console)", true);
    return;
  }
  const requete = $("recherche-station").value.trim().toLowerCase();
  const seulementService = $("filtre-service").checked;

  if (clusterStations) carteStations.removeLayer(clusterStations);
  clusterStations = L.markerClusterGroup({ chunkedLoading: true });

  let visibles = 0;
  for (const s of etat.stations) {
    if (seulementService && !s.en_service) continue;
    if (requete) {
      const texte = `${s.nom} ${s.cours_eau || ""} ${s.code}`.toLowerCase();
      if (!texte.includes(requete)) continue;
    }
    visibles++;
    const marqueur = L.marker([s.lat, s.lon]);
    marqueur.bindPopup(
      `<b>${s.nom}</b><br>${s.cours_eau || ""}<br><code>${s.code}</code><br>` +
      `<button class="primaire" onclick="ouvrirStation('${s.code}')">Ouvrir cette station →</button>`
    );
    clusterStations.addLayer(marqueur);
  }
  carteStations.addLayer(clusterStations);
  $("compteur-stations").textContent = `${visibles} stations`;
}

// ---------------------------------------------------------------- navigation
function reinitialiserAffichageStation() {
  // Vide tout ce qui vient de la station précédente : graphiques, tableaux, dates.
  $("image-pca").classList.add("cache");
  $("image-pca").removeAttribute("src");
  ["chart-prevision", "chart-backtest", "chart-historique"].forEach(detruireChart);
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

window.ouvrirStation = async function (code) {
  etat.code = code;
  reinitialiserAffichageStation();
  $("vue-stations").classList.add("cache");
  $("vue-station").classList.remove("cache");
  $("btn-retour").classList.remove("cache");
  $("titre-station").textContent = `Chargement de ${code}…`;
  activerPanneau(etat.panneauActif || "carte");
  await rafraichirEtat();
  chargerPeriodeDisponible();
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
      $("periode-dispo").textContent = `période : ${dateFr(p.debut)} → ${dateFr(p.fin)}` +
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
      if (!$("opt-train-debut").value) $("opt-train-debut").value = p.debut;
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

$("btn-retour").addEventListener("click", () => {
  $("vue-station").classList.add("cache");
  $("vue-stations").classList.remove("cache");
  $("btn-retour").classList.add("cache");
  fermerReglages();
  etat.code = null;
  setTimeout(() => carteStations && carteStations.invalidateSize(), 100);
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

  // Pré-remplit les choix d'agrégation depuis les params sauvegardés.
  const agg = (r.params && r.params.agregations) || {};
  if (agg.temperature_2m) $("opt-agg-temp").value = agg.temperature_2m;
  if (agg.snow_depth) $("opt-agg-neige").value = agg.snow_depth;
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
  const corps = { points, ...paramsDonnees() };
  try {
    const { job_id } = await post(`/api/riviere/${etat.code}/points`, corps);
    quitterModeSelection();
    suivreJob(job_id, "Points manuels → données complètes → Gradient Boosting");
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

function agregationsChoisies() {
  return { snow_depth: $("opt-agg-neige").value, temperature_2m: $("opt-agg-temp").value };
}

function fenetreApprentissage() {
  return { start_train: $("opt-train-debut").value || null, end_train: $("opt-train-fin").value || null };
}

// Paramètres de construction des données communs au pipeline / données / points.
function paramsDonnees() {
  return {
    past_day: parseInt($("opt-past-day").value) || 20,
    predict_day: parseInt($("opt-predict-day").value) || 15,
    mode_split: $("opt-mode-split").value,
    part_test: parseFloat($("opt-part-test").value) || 0.2,
    agregations: agregationsChoisies(),
    ...fenetreApprentissage(),
  };
}

// 999 dans le sélecteur = 99,9 %.
function seuilEnergie() {
  const v = $("opt-pca-energie").value;
  return v === "999" ? 99.9 : parseInt(v);
}

$("opt-methode-selection").addEventListener("change", () => {
  const genetique = $("opt-methode-selection").value === "genetique";
  $("ligne-genetique").style.display = genetique ? "" : "none";
  $("ligne-deux-temps").style.display = genetique ? "none" : "";
});

$("btn-pipeline").addEventListener("click", async () => {
  const methode = $("opt-methode-selection").value;
  const corps = {
    methode,
    // paramètres de construction des données (communs aux deux méthodes)
    past_day: parseInt($("opt-past-day").value) || 20,
    predict_day: parseInt($("opt-predict-day").value) || 15,
    mode_split: $("opt-mode-split").value,
    part_test: parseFloat($("opt-part-test").value) || 0.2,
    agregations: agregationsChoisies(),
    ...fenetreApprentissage(),
  };
  if (methode === "genetique") {
    corps.ga = {
      taille_population: parseInt($("opt-population").value) || 20,
      nombre_generations: parseInt($("opt-generations").value) || 50,
      predict_day_final: parseInt($("opt-predict-day").value) || 15,
      past_day_final: parseInt($("opt-past-day").value) || 20,
      predict_day_opti: parseInt($("opt-ga-horizon").value) || 3,
      past_day_opti: parseInt($("opt-past-day").value) || 20,
      mode_split: $("opt-mode-split").value,
      part_test: parseFloat($("opt-part-test").value) || 0.2,
    };
  } else {
    corps.selection = {
      n_preselection: parseInt($("opt-n-preselection").value) || 30,
      n_final: parseInt($("opt-n-final").value) || 5,
      poids_pluie: parseFloat($("opt-poids-pluie").value),
      poids_neige: parseFloat($("opt-poids-neige").value),
      poids_altitude: parseFloat($("opt-poids-altitude").value),
    };
  }
  try {
    const { job_id } = await post(`/api/riviere/${etat.code}/pipeline`, corps);
    const titre = methode === "genetique"
      ? "Pipeline (sélection génétique)"
      : "Pipeline (sélection 2 temps : altitude+couverture → corrélation+neige)";
    activerPanneau("carte");   // on regarde les points apparaître en direct
    suivreJob(job_id, titre);
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

$("btn-pca").addEventListener("click", async () => {
  try {
    const { job_id } = await post(`/api/riviere/${etat.code}/pca`, { seuil_energie: seuilEnergie() });
    suivreJob(job_id, `Analyse PCA (${seuilEnergie()} % d'énergie)`);
  } catch (e) {
    toast(e.message, true);
  }
});

$("btn-entrainer").addEventListener("click", async () => {
  const modeles = Array.from(document.querySelectorAll(".case-modele:checked")).map((c) => c.value);
  if (!modeles.length) return toast("Coche au moins un modèle à entraîner.", true);
  const corps = { modeles, incertitude: $("opt-incertitude").checked, seuil_energie: seuilEnergie() };
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
    const corps = { modele, incertitude: $("opt-incertitude").checked, seuil_energie: seuilEnergie() };
    try {
      const { job_id } = await post(`/api/riviere/${etat.code}/entrainer-modele`, corps);
      suivreJob(job_id, `Entraînement de ${modele} + incertitude`);
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
  $("btn-arreter-job").textContent = "⏹️ Arrêter";
  $("job-titre").textContent = titre;
  $("job-statut").textContent = "⏳ en cours…";
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
    $("job-progres-pct").textContent = "en cours…";
    $("job-progres-barre").style.width = "";
  } else {
    piste.classList.remove("indetermine");
    $("job-progres-pct").textContent = `${p.pct}%` + (p.total ? ` (${p.courant}/${p.total})` : "");
    $("job-progres-barre").style.width = p.pct + "%";
  }
}

$("btn-arreter-job").addEventListener("click", async () => {
  if (etat.jobId == null) return;
  $("btn-arreter-job").disabled = true;
  $("btn-arreter-job").textContent = "arrêt demandé…";
  $("job-statut").textContent = "⏹️ arrêt demandé, fin de l'étape en cours…";
  try {
    await post(`/api/jobs/${etat.jobId}/arreter`);
    toast("Arrêt demandé : le calcul s'interrompra au prochain point de contrôle.");
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

  if (job.statut === "termine") {
    $("job-statut").textContent = "✅ terminé";
    const r = job.resultat || {};
    if (r.score_gradient_boosting != null) {
      const parts = [`✅ Modèle prêt — fiabilité ${(r.score_gradient_boosting * 100).toFixed(0)} %`];
      if (r.temps_reponse_jours != null) parts.push(`temps de réponse du bassin ~${r.temps_reponse_jours} j`);
      $("analyse-diagnostic").textContent = parts.join(" · ");
      toast("Analyse terminée : le modèle de prévision est prêt.");
    } else if (r.temps_reponse_jours != null) {
      toast(`Terminé ! Temps de réponse du bassin : ~${r.temps_reponse_jours} j.`);
    } else {
      toast("Job terminé !");
    }
    if (r.image) {
      $("image-pca").src = "data:image/png;base64," + r.image;
      $("image-pca").classList.remove("cache");
    }
  } else if (job.statut === "arrete") {
    $("job-statut").textContent = "⏹️ arrêté";
    toast("Calcul arrêté. Tu peux relancer avec les bons paramètres.");
  } else {
    $("job-statut").textContent = "❌ erreur : " + (job.erreur || "inconnue");
    toast("Le job a échoué : " + (job.erreur || ""), true);
  }
  etat.jobId = null;
  rendreQuota();
  await rafraichirEtat();
}

$("btn-plier-log").addEventListener("click", () => {
  $("job-log").classList.toggle("cache");
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
  if (!r.modeles.length) {
    conteneur.innerHTML = "<p class='aide'>Aucun modèle entraîné : lance le pipeline (ou choisis tes points à la main).</p>";
  } else {
    conteneur.innerHTML =
      "<table><tr><th>Modèle</th><th>R²</th><th>Espace</th><th>Incertitude</th></tr>" +
      r.modeles
        .map(
          (m) =>
            `<tr><td>${m.nom}</td><td><b>${(m.score * 100).toFixed(1)} %</b></td>` +
            `<td>${m.espace}</td><td>${m.hybride ? "✅" : "—"}</td></tr>`
        )
        .join("") +
      "</table>";
  }

  const options = r.modeles.map((m) => `<option value="${m.nom}">${m.nom}</option>`).join("");
  $("backtest-modele").innerHTML = options;
  $("prevision-modele").innerHTML = options;
  // Modèle par défaut = le meilleur disponible (le sélecteur est masqué au grand
  // public ; on prédit avec le meilleur sans qu'il ait à choisir).
  const prefere = ["gradient_boosting", "ridge_causal", "keras_brut", "ridge_brut", "lineaire_pca"]
    .find((n) => r.modeles.some((m) => m.nom === n));
  if (prefere) {
    $("backtest-modele").value = prefere;
    $("prevision-modele").value = prefere;
  }

  const sansEntrainement = !r.donnees_train_presentes;
  $("btn-pca").disabled = sansEntrainement;
  $("btn-entrainer").disabled = sansEntrainement;
  document.querySelectorAll(".btn-un-modele").forEach((b) => (b.disabled = sansEntrainement));
  if (sansEntrainement && r.modeles.length) {
    $("btn-pca").title = $("btn-entrainer").title =
      "Les données d'entraînement ne sont plus sur le disque (nettoyées) : relance le téléchargement des données pour ré-entraîner.";
  }
  $("btn-donnees").disabled = !r.coords_finales;
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
  } catch (e) {
    $("btn-backtest").disabled = true;
  }
}

$("btn-backtest").addEventListener("click", async () => {
  const modele = $("backtest-modele").value;
  let date = $("backtest-date").value;
  if (etat.datesTest && !etat.datesTest.includes(date)) {
    // choisit la date de test disponible la plus proche
    const cible = new Date(date).getTime();
    date = etat.datesTest.reduce((a, b) =>
      Math.abs(new Date(a) - cible) < Math.abs(new Date(b) - cible) ? a : b
    );
    $("backtest-date").value = date;
    toast(`Date ajustée à la plus proche disponible : ${dateFr(date)}`);
  }
  const hybride = $("backtest-hybride").checked ? 1 : 0;
  const nbJours = parseInt($("backtest-jours").value) || 15;
  $("btn-backtest").disabled = true;
  try {
    const res = await api(`/api/riviere/${etat.code}/backtest?modele=${modele}&date=${date}&hybride=${hybride}&nb_jours=${nbJours}`);
    rendreFanChart("chart-backtest", res, true);
    rendreFiabilite("fiab-backtest", res);
    $("vide-backtest").style.display = "none";
    if (res.scores_detail) {
      const scores = res.scores_detail.slice(0, nbJours + 1);
      $("scores-detail").innerHTML =
        "<table><tr><th>Horizon</th>" +
        scores.map((_, i) => `<th>J+${i}</th>`).join("") +
        "</tr><tr><td>R²</td>" +
        scores.map((s) => `<td>${(s * 100).toFixed(0)}%</td>`).join("") +
        "</tr></table>";
    }
  } catch (e) {
    toast(e.message, true);
  } finally {
    $("btn-backtest").disabled = false;
  }
});

// ---------------------------------------------------------------- prévision réelle
$("btn-prevision").addEventListener("click", async () => {
  const modele = $("prevision-modele").value;
  const hybride = $("prevision-hybride").checked ? 1 : 0;
  const nbJours = parseInt($("prevision-jours").value) || 15;
  $("btn-prevision").disabled = true;
  $("btn-prevision").textContent = "Téléchargement météo prévisionnelle…";
  try {
    const res = await api(`/api/riviere/${etat.code}/prevision?modele=${modele}&hybride=${hybride}&nb_jours=${nbJours}`);
    rendreFanChart("chart-prevision", res, false);
    rendreFiabilite("fiab-prevision", res);
    $("vide-prevision").style.display = "none";
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
      `<button class="secondaire danger mini" data-code="${s.code}" data-cible="station">Tout supprimer</button></div>` +
      (lignes.length ? `<ul class="liste-stockage">${lignes.join("")}</ul>` : "") +
      `</div>`
    );
  }).join("");

  conteneur.querySelectorAll("[data-cible]").forEach((btn) => {
    btn.addEventListener("click", () => supprimerStockage(btn.dataset.code, btn.dataset.cible));
  });
}

async function supprimerStockage(code, cible) {
  const nom = cible === "station" ? "TOUTE la station " + code
    : cible === "test" ? "le jeu de test de " + code + " (les tests sur le passé ne marcheront plus)"
    : cible === "travail" ? "les fichiers de travail de " + code
    : "le modèle « " + cible.split(":")[1] + " » de " + code;
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

// ---------------------------------------------------------------- démarrage
initStations();
chargerConfig();
