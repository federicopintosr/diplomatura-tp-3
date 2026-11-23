"""Funciones compartidas del TP3: palabras clave y tópicos en tweets de 2009.

Lectura y fechas viven acá para que todas las notebooks lean los datos igual.
"""
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RAW_DIR = RAIZ / "data" / "raw"
PROCESSED_DIR = RAIZ / "data" / "processed"
MODELS_DIR = RAIZ / "models"

# Link de la consigna (Google Drive). `confirm=t` evita la pantalla de aviso por tamaño.
URL_DATOS = (
    "https://drive.usercontent.google.com/download"
    "?id=0B04GJPshIjmPRnZManQwWEdTZjg&export=download&confirm=t"
)
ZIP_DATOS = RAW_DIR / "trainingandtestdata.zip"
FILE_TRAIN = RAW_DIR / "training.1600000.processed.noemoticon.csv"
FILE_TEST = RAW_DIR / "testdata.manual.2009.06.14.csv"

COLUMNAS = ["polaridad", "id", "fecha", "query", "usuario", "texto"]

# Diferencia de cada zona horaria con UTC. Todas las fechas se llevan a PDT
# (hora del Pacífico, la del archivo principal).
OFFSET_UTC = {"PDT": -7, "UTC": 0}
OFFSET_DESTINO = OFFSET_UTC["PDT"]


def descargar_datos():
    """Descarga y descomprime los CSV en data/raw/ si todavía no están."""
    if FILE_TRAIN.exists() and FILE_TEST.exists():
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_DATOS.exists():
        urllib.request.urlretrieve(URL_DATOS, ZIP_DATOS)
    with zipfile.ZipFile(ZIP_DATOS) as z:
        z.extractall(RAW_DIR)


def leer_tweets(path):
    """Lee un CSV del dataset: sin encabezado y en latin-1."""
    return pd.read_csv(path, encoding="latin-1", header=None, names=COLUMNAS)


def parsear_fecha(s):
    """'Mon Apr 06 22:19:45 PDT 2009' -> datetime en hora PDT (sin zona).

    pandas no interpreta abreviaturas como PDT, así que la zona se separa
    del texto y se aplica como desplazamiento en horas.
    """
    zona = s.str[20:23]
    desconocidas = set(zona.unique()) - set(OFFSET_UTC)
    if desconocidas:
        raise ValueError(f"Zonas horarias sin mapear: {desconocidas}")
    local = pd.to_datetime(s.str[:20] + s.str[24:], format="%a %b %d %H:%M:%S %Y")
    horas = OFFSET_DESTINO - zona.map(OFFSET_UTC)
    return local + pd.to_timedelta(horas, unit="h")
