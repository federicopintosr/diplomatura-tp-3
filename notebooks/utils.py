"""Funciones compartidas del TP3: palabras clave y tópicos en tweets de 2009.

Lectura, fechas, limpieza y stopwords viven acá para que todas las notebooks
procesen el texto igual.
"""
import html
import urllib.request
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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

# Salidas de la notebook 03 (entrada de la 04, 05 y 06)
FILE_MUESTRA = PROCESSED_DIR / "muestra.csv.gz"
FILE_EMBEDDINGS = PROCESSED_DIR / "embeddings.npy"  # no se sube al repo (~330 MB)
MODELO_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"

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


# ---------------------------------------------------------------- texto

# Contracciones y formas de chat que no están en la lista de sklearn
# (en los tweets suelen aparecer con y sin apóstrofe).
STOPWORDS_EXTRA = {
    "i'm", "im", "don't", "dont", "can't", "cant", "it's", "that's", "thats",
    "i've", "ive", "i'll", "ill", "didn't", "didnt", "doesn't", "doesnt",
    "isn't", "isnt", "won't", "wont", "you're", "youre", "u", "ur",
}
STOPWORDS = sorted(ENGLISH_STOP_WORDS | STOPWORDS_EXTRA)

# Palabras: empiezan con letra, mínimo 2 caracteres, admiten dígitos y apóstrofe.
TOKEN_PATTERN = r"(?u)\b[a-z][a-z0-9']+\b"


def limpiar_texto(s):
    """Limpieza liviana: deja el texto legible para un modelo de lenguaje.

    - `ï¿½` (carácter perdido al codificar, casi siempre un apóstrofe) -> `'`
    - entidades HTML (`&amp;`, `&quot;`, `&lt;`) -> carácter real
    - quita URLs y @menciones (el dataset no tiene retweets `RT`)
    - `#hashtag` -> `hashtag` (la palabra se conserva)
    - espacios repetidos -> uno
    No pasa a minúsculas ni quita stopwords: eso lo hace cada vectorizador.
    """
    return (
        s.str.replace("ï¿½", "'", regex=False)
        .map(html.unescape)
        .str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
        .str.replace(r"@\w+", " ", regex=True)
        .str.replace("#", "", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


# ---------------------------------------------------------------- muestra y embeddings

def cargar_muestra():
    """Muestra procesada por la notebook 03, con las fechas ya convertidas."""
    df = pd.read_csv(FILE_MUESTRA, parse_dates=["fecha_pdt", "dia"])
    df["texto_limpio"] = df["texto_limpio"].astype(str)
    return df


def cargar_embeddings(n_esperado):
    """Embeddings de la muestra (mismo orden de filas). Falla si no coinciden."""
    emb = np.load(FILE_EMBEDDINGS)
    if len(emb) != n_esperado:
        raise ValueError("Los embeddings no corresponden a la muestra: correr la notebook 03")
    return emb


# ---------------------------------------------------------------- métricas de tópicos

def coherencia_npmi(topicos, X_binaria, vocab):
    """NPMI promedio de cada tópico, contando co-ocurrencias dentro del mismo tweet.

    topicos: lista de listas de palabras (unigramas o bigramas del vocabulario).
    X_binaria: matriz tweets x vocabulario con 1 si el término aparece.
    NPMI(a, b) = PMI(a, b) / -log P(a, b): va de -1 (nunca juntas) a 1 (siempre juntas).
    Se ignoran las palabras que no están en el vocabulario y los pares que comparten
    una palabra (`swine` y `swine flu` coinciden siempre por construcción).
    """
    indice = {w: i for i, w in enumerate(vocab)}
    n = X_binaria.shape[0]
    resultado = []
    for palabras in topicos:
        terminos = [w for w in palabras if w in indice]
        idx = [indice[w] for w in terminos]
        sub = X_binaria[:, idx]
        conjunta = (sub.T @ sub).toarray() / n
        marginal = np.diag(conjunta)
        valores = []
        for i, j in combinations(range(len(idx)), 2):
            if set(terminos[i].split()) & set(terminos[j].split()):
                continue
            p_ij = conjunta[i, j]
            if p_ij == 0:
                valores.append(-1.0)
            else:
                valores.append(np.log(p_ij / (marginal[i] * marginal[j])) / -np.log(p_ij))
        resultado.append(np.mean(valores) if valores else np.nan)
    return np.array(resultado)


def diversidad(topicos):
    """Proporción de palabras distintas entre las top-n de todos los tópicos (1 = sin repetidas)."""
    todas = [w for t in topicos for w in t]
    return len(set(todas)) / len(todas)


class ReduccionGuardada:
    """Reemplaza a UMAP dentro de BERTopic devolviendo una reducción ya calculada.

    UMAP es el paso más lento y no depende de HDBSCAN: se calcula una vez (con
    semilla) y se reutiliza para comparar varios `min_cluster_size`.
    """

    def __init__(self, reducidos):
        self.reducidos = reducidos

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(X) != len(self.reducidos):
            raise ValueError("Solo sirve para los mismos documentos con los que se calculó")
        return self.reducidos


def coherencia_embeddings(topicos, modelo):
    """Similitud coseno promedio entre los embeddings de las palabras de cada tópico.

    A diferencia de la NPMI, reconoce sinónimos y variantes (`tweet` / `tweets`)
    aunque no aparezcan juntas en un mismo tweet. Mismos pares que la NPMI.
    """
    palabras = sorted({w for t in topicos for w in t})
    vectores = dict(zip(palabras, modelo.encode(palabras, normalize_embeddings=True,
                                                 show_progress_bar=False)))
    resultado = []
    for t in topicos:
        sims = [vectores[a] @ vectores[b] for a, b in combinations(t, 2)
                if not set(a.split()) & set(b.split())]
        resultado.append(np.mean(sims) if sims else np.nan)
    return np.array(resultado)


# ---------------------------------------------------------------- grupos en el tiempo

MIN_TWEETS_DIA = 1_000  # días con menos tweets en la muestra (23, 25 y 27/5) no se comparan


def pmi_grupo_dia(grupos, dias, min_tweets=20, min_tweets_dia=MIN_TWEETS_DIA):
    """PMI(grupo, día) = log2( P(grupo | día) / P(grupo) ), sin outliers (-1).

    Un grupo es un tópico o un cluster. La PMI queda en -inf si el grupo tiene
    menos de `min_tweets` ese día o si el día tiene menos de `min_tweets_dia`
    tweets en total (proporciones poco confiables).
    """
    d = pd.DataFrame({"grupo": grupos, "dia": dias})
    dias_chicos = d["dia"].value_counts().loc[lambda s: s < min_tweets_dia].index
    d = d[d["grupo"] != -1]
    conteo = pd.crosstab(d["grupo"], d["dia"])
    p_dia = conteo / conteo.sum(axis=0)
    p_global = conteo.sum(axis=1) / conteo.values.sum()
    pmi = np.log2(p_dia.div(p_global, axis=0).where(conteo > 0))
    pmi = pmi.where(conteo >= min_tweets, -np.inf)
    pmi.loc[:, pmi.columns.isin(dias_chicos)] = -np.inf
    return pmi, conteo


def evaluar_eventos(grupos, textos, dias, eventos, min_precision=0.5):
    """Para cada evento: grupo que más tweets suyos reúne, precisión, recall y pico.

    - precisión: % de los tweets del grupo que mencionan el evento
    - recall: % de los tweets que mencionan el evento que caen en ese grupo
    - aislado: precisión >= `min_precision` (el grupo trata sobre el evento)
    - pico ok: el día de mayor PMI del grupo coincide con la fecha de la fuente
    """
    from eventos import dias_validos

    grupos = np.asarray(grupos)
    dias = pd.DatetimeIndex(dias)
    pmi, _ = pmi_grupo_dia(grupos, dias, min_tweets=5)
    filas = []
    for clave, ev in eventos.items():
        menciona = textos.str.contains(ev["patron"]).to_numpy()
        por_grupo = pd.Series(grupos[menciona]).value_counts()
        por_grupo = por_grupo[por_grupo.index != -1]
        fila = {"evento": ev["nombre"], "origen": ev["origen"], "tweets": int(menciona.sum())}
        if len(por_grupo) == 0:
            filas.append(fila | {"grupo": None, "precisión": 0.0, "recall": 0.0})
            continue
        g = por_grupo.index[0]
        pico = pmi.loc[g].idxmax()
        fila |= {
            "grupo": int(g),
            "precisión": por_grupo.iloc[0] / (grupos == g).sum(),
            "recall": por_grupo.iloc[0] / menciona.sum(),
            "pico del grupo": pico.strftime("%d/%m"),
            "fecha fuente": " y ".join(pd.to_datetime(ev["fechas"]).strftime("%d/%m")),
            "pico ok": pico in dias_validos(ev["fechas"], dias.unique()),
        }
        filas.append(fila)
    res = pd.DataFrame(filas)
    res["aislado"] = res["precisión"] >= min_precision
    res["confirmado"] = res["aislado"] & res["pico ok"].fillna(False).astype(bool)
    return res.round({"precisión": 2, "recall": 2})
