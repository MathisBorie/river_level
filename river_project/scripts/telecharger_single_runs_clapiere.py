# Prévisions ECMWF IFS 9 km COMPLÈTES (10 jours) émises chaque jour à 0 h UTC, du 2024-03-14 au 2026-10-10 (API Single Runs).
# La Clapière : d'abord les 7 carreaux choisis par F.choisir_carreaux() (notebook 05), puis les 34 autres.
# Relance toute seule après QuotaEpuise (toutes les heures) ou une erreur (10 min) ; relancer le script reprend où il en était.
import sys, time, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # le dossier river_project
from src.bassins import charger_bassins, ForetBassins, carreaux_amont
from src.meteo import PrevisionsArchivees, QuotaEpuise
DEBUT, FIN = "2024-03-14", "2026-10-10"
V = ["precipitation", "temperature_2m", "snow_depth", "shortwave_radiation", "rain", "soil_moisture_7_to_28cm",
     "snowfall", "et0_fao_evapotranspiration", "soil_moisture_28_to_100cm"]
PREMIERS = [835931, 841112, 838522, 830767, 846307, 841113, 833346]
carr = carreaux_amont(2121189970, ForetBassins.depuis_table(charger_bassins()))
pa = PrevisionsArchivees()
for nom, lot in [("7 carreaux choisis", carr[carr.carreau.isin(PREMIERS)]), ("34 autres", carr[~carr.carreau.isin(PREMIERS)])]:
    while True:
        reste = pa.cout_prevu(lot, DEBUT, FIN)
        print(dt.datetime.now().strftime("%Y-%m-%d %H:%M"), nom, "reste :", reste, flush=True)
        if reste["requetes"] == 0:
            print(nom, "TERMINÉ", flush=True); break
        try:
            pa.telecharger(lot, V, DEBUT, FIN, barre=False)
        except QuotaEpuise as e:
            print(dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "quota :", e, "-> nouvel essai dans 1 h", flush=True)
            time.sleep(3600)
        except Exception as e:
            print(dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "erreur :", repr(e)[:300], "-> nouvel essai dans 10 min", flush=True)
            time.sleep(600)
