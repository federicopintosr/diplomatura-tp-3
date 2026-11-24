# TP3 - Palabras clave y tópicos en tweets de 2009

Análisis de tópicos sobre tweets de abril a junio de 2009 con dos enfoques: **BERTopic** y **palabras clave (KeyBERT) + clustering (K-means)**. Los tópicos que reflejan eventos se contrastan con **fuentes externas**.

**Resultado:** BERTopic encuentra **22 de 34 eventos candidatos** como tópicos propios, con pico en la fecha que confirma la fuente. Entre ellos hay 6 que no estaban en las hipótesis iniciales. KeyBERT + K-means describe mejor **de qué se habla en general** (50 temas nombrados a mano), pero diluye los eventos.

---

## Objetivo

- Encontrar los temas de los que se habla en el corpus y analizar si reflejan **eventos** de las fechas del dataset.
- **Validar** cada evento con fuentes confiables: el modelo por sí solo no confirma que algo haya pasado.
- Comparar la **utilidad e interpretabilidad** de los dos enfoques.

La consigna está en [`Consignas_proyecto_NLP.ipynb`](Consignas_proyecto_NLP.ipynb) (opción 2: extracción de palabras clave y tópicos).

## Datos

| Archivo | Filas | Uso |
| --- | --- | --- |
| `training.1600000.processed.noemoticon.csv` | 1.600.000 tweets (6/4 al 25/6/2009) | Análisis de tópicos |
| `testdata.manual.2009.06.14.csv` | 498 tweets etiquetados por búsqueda (`query`) | Control externo (notebook 06) |

- Es **Sentiment140** ([Go, Bhayani y Huang, 2009](https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)). Los tweets se juntaron buscando emoticones, así que predomina la charla personal.
- Columnas: polaridad, id, fecha, query, usuario, texto.
- La notebook 01 los descarga desde el link de la consigna a `data/raw/`. Esa carpeta no se sube (~240 MB).

## Estructura del repositorio

```text
TP3/
├── data/
│   ├── raw/                        # CSV originales (se descargan con la 01, no se suben)
│   └── processed/
│       ├── muestra.csv.gz          # 217.133 tweets limpios (salida de la 03)
│       ├── embeddings.npy          # embeddings de la muestra (333 MB, no se sube: lo genera la 03)
│       ├── umap_5d.npy, umap_2d.npy  # reducción UMAP de la 04
│       ├── topicos_bertopic.csv    # tópico de cada tweet (con y sin outliers)
│       ├── palabras_clave.csv.gz   # 3 palabras clave por tweet (KeyBERT)
│       ├── clusters_kmeans.csv     # cluster y nombre de cada tweet
│       └── validacion_*.csv        # evaluación de eventos de cada enfoque
├── models/
│   └── bertopic/                   # modelo BERTopic (safetensors, 7 MB)
├── notebooks/
│   ├── 01_discovery.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocesamiento.ipynb
│   ├── 04_bertopic.ipynb
│   ├── 05_keywords_y_clustering.ipynb
│   ├── 06_comparacion_y_conclusiones.ipynb
│   ├── utils.py                    # lectura, fechas, limpieza, stopwords y métricas compartidas
│   ├── eventos.py                  # 34 eventos candidatos con patrón, fechas, fuente y etiqueta
│   └── traducciones.csv            # traducción al español de los tweets que se muestran (para presentar)
├── Consignas_proyecto_NLP.ipynb    # consigna
├── requirements.txt
└── README.md
```

## Instalación

Requiere **Python 3.10**. No hace falta GPU: todo corre en CPU.

```bash
git clone https://github.com/federicopintosr/diplomatura-tp-3.git
cd diplomatura-tp-3
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt     # instala torch en su versión solo CPU
```

La primera vez se descarga el modelo de embeddings `all-MiniLM-L6-v2` (~90 MB) desde Hugging Face.

## Ejecución

Las notebooks se ejecutan **desde la carpeta `notebooks/`** y **en orden**. Tiempos medidos en una PC sin GPU (CPU de 16 hilos, 30 GB de RAM):

| Notebook | Qué hace | Guarda | Tiempo |
| --- | --- | --- | --- |
| 01 | Descarga y diagnóstico (no modifica nada) | `data/raw/` | < 1 min |
| 02 | EDA: palabras, hashtags, PMI de bigramas, PMI palabra–día, hipótesis de eventos | — | ~3,5 min |
| 03 | Limpieza, deduplicación, muestra con tope por día, embeddings | `muestra.csv.gz`, `embeddings.npy` | ~15 min |
| 04 | BERTopic: elección de parámetros, tópicos en el tiempo, validación de eventos, fichas de tópicos | modelo, `topicos_bertopic.csv` | ~2,5 min (+7 min si recalcula UMAP) |
| 05 | KeyBERT, representación, K-means, tópicos nombrados a mano con ejemplos | `palabras_clave.csv.gz`, `clusters_kmeans.csv` | ~4,5 min (+3 min si recalcula KeyBERT) |
| 06 | Comparación, concordancia, control con el archivo de test, conclusiones | — | ~1 min |

`embeddings.npy` no está en el repo: para correr de la 04 en adelante hay que correr antes la 01 y la 03.

## Decisiones principales

| Decisión | Por qué |
| --- | --- |
| Muestra de **hasta 5.000 tweets por día** (217 mil) en vez de proporcional | El volumen diario depende de la recolección, no de la actividad real. Con muestra proporcional, junio se lleva el 58% y los eventos de abril casi desaparecen |
| Limpieza liviana: sin URLs ni menciones, entidades HTML corregidas; stopwords solo en el vectorizador | El modelo de embeddings necesita la frase completa |
| No filtrar por idioma | `langdetect` falla con textos cortos: los tweets que marcaba como no inglés estaban en inglés |
| Uni y bigramas en los vectorizadores | La PMI mostró que los temas concretos son nombres de dos palabras (`farrah fawcett`, `swine flu`) |
| BERTopic con `min_cluster_size=50` | Es el que más eventos aísla. Un evento con menos tweets que el mínimo no puede ser tópico |
| K-means sobre el **promedio de embeddings de las palabras clave**, K = 50 | Agrupa por significado. Sin codo claro en la inercia; 50 grupos se pueden nombrar a mano |
| **PMI tópico–día** en lugar de conteos por fecha | Usa proporciones, así que no le afecta el volumen desparejo de cada día |
| Evento confirmado = grupo con precisión ≥ 50% **y** pico ±1 día de la fecha de la fuente | Criterio explícito y reproducible (`utils.evaluar_eventos`) |

## Resultados

| Criterio | BERTopic | KeyBERT + K-means |
| --- | --- | --- |
| Tópicos | 549 | 50 |
| Tweets asignados | 57% (el resto, outliers) | 99% |
| Coherencia NPMI / por embeddings | −0,60 / 0,33 | −0,17 / 0,52 |
| Diversidad | 0,73 | 0,99 |
| Eventos confirmados (de 34) | **22** | 1 |
| Control con el archivo de test (AMI con la búsqueda) | **0,53** | 0,40 |

- **BERTopic** sirve para encontrar **qué pasó cada día**: muertes de Farrah Fawcett, David Carradine y Ed McMahon, Air France 447, protestas en Irán, iPhone OS 3.0, final de *Britain's Got Talent*, estrenos, y eventos que no estaban en las hipótesis (choque del metro de Washington, estreno de *True Blood*). Farrah Fawcett murió a las 9:28 PDT; el pico de tweets arranca a las 9:45.
- **KeyBERT + K-means** sirve para describir **de qué se habla en general**: 50 temas cotidianos nombrados a mano y agrupados en 3 macrotemas. Los eventos quedan dentro de clusters amplios (*Famosos*, *Películas y TV*) y no se distinguen.
- Las métricas de coherencia favorecen a K-means porque sus tópicos son amplios. **Ninguna métrica alcanza sola para elegir: depende del objetivo.**
- Michael Jackson murió el 25/6 a las 14:26 PDT, después del último tweet del dataset (10:28): no aparece.

## Limitaciones

- Tweets elegidos por emoticones y en inglés: no representan "lo que se tuiteaba" en 2009.
- Solo 48 de 81 días tienen datos. Los eventos en huecos (por ejemplo, Pascua) no pueden aparecer.
- Los eventos con menos de ~50 tweets en la muestra no llegan a formar tópico.
- La precisión contra eventos se mide por palabras clave: es una cota inferior.
- Se usó una sola semilla; no se midió la estabilidad de los agrupamientos. Entre corridas, K-means puede mover ~0,05% de los tweets de cluster por diferencias numéricas en CPU, sin cambiar nombres ni resultados.

## Fuentes

- Dataset: Go, A., Bhayani, R. y Huang, L. (2009). *Twitter Sentiment Classification using Distant Supervision*. Stanford.
- Fechas de los eventos: Wikipedia en inglés y los medios que cita (BBC, Variety, CTV, People, entre otros). Cada evento tiene su link en [`notebooks/eventos.py`](notebooks/eventos.py).
