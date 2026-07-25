# Image du site River Lab, prête pour Hugging Face Spaces (SDK Docker).
FROM python:3.11-slim

# Utilisateur non-root (recommandé par HF Spaces) qui possède /app -> écritures OK.
RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user river.py .
COPY --chown=user webapp/ ./webapp/

USER user
# App COMPLÈTE, démarrée à vide : chaque visiteur analyse et gère SES rivières.
# (RIVER_PUBLIC reste à 0 -> entraînement autorisé. Le mode lecture seule existe
#  toujours si un jour tu veux figer une démo : passe RIVER_PUBLIC=1.)
ENV PORT=7860 \
    HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/mpl \
    TF_CPP_MIN_LOG_LEVEL=3

EXPOSE 7860
WORKDIR /app/webapp
# 1 seul worker (l'état des jobs/rivières est en mémoire partagée) + threads.
CMD ["gunicorn", "--workers", "1", "--threads", "8", "--timeout", "300", "--bind", "0.0.0.0:7860", "app:app"]
