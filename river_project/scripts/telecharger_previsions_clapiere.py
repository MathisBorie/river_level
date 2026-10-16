# Prévisions météo PASSÉES de La Clapière (41 carreaux), API Previous Runs, prévues 1 à 7 jours avant.
# ecmwf_ifs (9 km, même grille que nos carreaux) depuis le 2025-10-01.
# Relance toute seule après QuotaEpuise (toutes les heures) ou une erreur (10 min).
import sys, time, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # le dossier river_project
from src import config
from src.bassins import charger_bassins, ForetBassins, carreaux_amont
from src.meteo import Meteo, QuotaEpuise
FIN = "2026-10-09"
TACHES = [  # (modèle, début, variables de base)
    ("ecmwf_ifs", "2025-10-01", ["precipitation", "temperature_2m", "rain", "snowfall", "shortwave_radiation", "et0_fao_evapotranspiration"]),
]
carr = carreaux_amont(2121189970, ForetBassins.depuis_table(charger_bassins()))
for modele, debut, base in TACHES:
    H = [f"{v}_previous_day{n}" for v in base for n in range(1, 8)]
    m = Meteo(api="previous_runs", modele=modele, dossier=config.CACHE_PREVISION / modele)
    while True:
        reste = m.cout_prevu(carr, H, [], debut, FIN)
        print(dt.datetime.now().strftime("%Y-%m-%d %H:%M"), modele, "reste :", reste, flush=True)
        if reste["requetes"] == 0:
            print(modele, "TERMINÉ", flush=True); break
        try:
            m.telecharger(carr, H, [], debut, FIN)
        except QuotaEpuise as e:
            print(dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "quota :", e, "-> nouvel essai dans 1 h", flush=True)
            time.sleep(3600)
        except Exception as e:
            print(dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "erreur :", repr(e)[:300], "-> nouvel essai dans 10 min", flush=True)
            time.sleep(600)
