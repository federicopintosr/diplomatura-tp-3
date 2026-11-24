"""Eventos candidatos y su validación con fuentes.

Cada evento tiene:
- `patron`: regex para encontrar los tweets que lo mencionan (texto limpio en minúsculas).
- `fechas`: fechas confirmadas por la fuente (formato AAAA-MM-DD).
- `fuente`: links (markdown) donde se verificó la fecha.
- `origen`: "EDA" si salió como hipótesis en la notebook 02, "BERTopic" si lo encontró el modelo.

Las fuentes se consultaron en noviembre de 2026. Para fechas que siguen una regla
(Día de la Madre, Pascua), la fecha se calculó a partir de esa regla.
"""
import pandas as pd

W = "https://en.wikipedia.org/wiki/"

EVENTOS = {
    # ---- hipótesis de la notebook 02
    "pascua": dict(
        nombre="Pascua", origen="EDA", patron=r"\beaster\b", fechas=["2009-04-12"],
        fuente="Cálculo con `dateutil.easter`; [Wikipedia](" + W + "Easter)"),
    "oracle_sun": dict(
        nombre="Oracle anuncia la compra de Sun", origen="EDA", patron=r"\boracle\b|\bmysql\b",
        fechas=["2009-04-20"], fuente=f"[Wikipedia]({W}Acquisition_of_Sun_Microsystems_by_Oracle_Corporation)"),
    "gripe_porcina": dict(
        nombre="Gripe porcina: la OMS sube la alerta a fase 4 y 5", origen="EDA", patron=r"\bswine\b|h1n1",
        fechas=["2009-04-27", "2009-04-29"], fuente=f"[Wikipedia]({W}2009_swine_flu_pandemic_timeline)"),
    "wolverine": dict(
        nombre="Estreno de X-Men Origins: Wolverine", origen="EDA", patron=r"wolverine",
        fechas=["2009-05-01"], fuente=f"[Wikipedia]({W}X-Men_Origins:_Wolverine)"),
    "pacquiao_hatton": dict(
        nombre="Pelea Pacquiao–Hatton", origen="EDA", patron=r"pacquiao|\bhatton\b",
        fechas=["2009-05-02"],
        fuente=f"[Wikipedia]({W}Manny_Pacquiao_vs._Ricky_Hatton), [BBC](https://news.bbc.co.uk/sport2/hi/boxing/8028250.stm)"),
    "star_trek": dict(
        nombre="Estreno de Star Trek", origen="EDA", patron=r"star ?trek",
        fechas=["2009-05-08"], fuente=f"[Wikipedia]({W}Star_Trek_(2009_film))"),
    "dia_madre": dict(
        nombre="Día de la Madre (EE. UU.)", origen="EDA", patron=r"mother'?s'? ?day",
        fechas=["2009-05-10"], fuente=f"Segundo domingo de mayo: [Wikipedia]({W}Mother%27s_Day_(United_States))"),
    "american_idol": dict(
        nombre="Danny Gokey eliminado de American Idol", origen="EDA",
        patron=r"american idol|\bgokey\b|archuleta|kris allen", fechas=["2009-05-13"],
        fuente=f"[Wikipedia]({W}Danny_Gokey)"),
    "eurovision": dict(
        nombre="Final de Eurovisión", origen="EDA", patron=r"eurovision",
        fechas=["2009-05-16"], fuente=f"[Wikipedia]({W}Eurovision_Song_Contest_2009)"),
    "wolfram_alpha": dict(
        nombre="Lanzamiento de Wolfram Alpha", origen="EDA", patron=r"wolfram",
        fechas=["2009-05-15", "2009-05-18"],
        fuente=f"[Wikipedia]({W}WolframAlpha), [BBC](https://news.bbc.co.uk/1/hi/technology/8052798.stm)"),
    "bgt": dict(
        nombre="Final de Britain's Got Talent", origen="EDA", patron=r"susan boyle|\bbgt\b|got talent|diversity",
        fechas=["2009-05-30"], fuente=f"[Wikipedia]({W}Britain%27s_Got_Talent_series_3)"),
    "mtv": dict(
        nombre="MTV Movie Awards", origen="EDA", patron=r"\bmtv\b", fechas=["2009-05-31"],
        fuente=f"[Wikipedia]({W}2009_MTV_Movie_Awards), [Variety](https://variety.com/2009/film/awards/twilight-tops-mtv-movie-awards-1118004362/)"),
    "new_moon": dict(
        nombre="Adelanto de New Moon (en los MTV Movie Awards)", origen="EDA", patron=r"new moon",
        fechas=["2009-05-31"], fuente=f"[Wikipedia]({W}2009_MTV_Movie_Awards)"),
    "leno_conan": dict(
        nombre="Último Tonight Show de Leno / debut de Conan", origen="EDA", patron=r"\bleno\b|\bconan\b",
        fechas=["2009-05-29", "2009-06-01"], fuente=f"[Wikipedia]({W}The_Tonight_Show_with_Conan_O%27Brien)"),
    "air_france": dict(
        nombre="Air France 447 cae en el Atlántico", origen="EDA", patron=r"air ?france|\baf ?447\b",
        fechas=["2009-06-01"], fuente=f"[Wikipedia]({W}Air_France_Flight_447)"),
    "e3": dict(
        nombre="E3 y Project Natal", origen="EDA", patron=r"\be3\b|project natal",
        fechas=["2009-06-01", "2009-06-02"], fuente=f"[E3 2009]({W}E3_2009), [Kinect]({W}Kinect)"),
    "carradine": dict(
        nombre="Muerte de David Carradine", origen="EDA", patron=r"carradine",
        fechas=["2009-06-03", "2009-06-04"],
        fuente=f"[Wikipedia]({W}David_Carradine), [BBC](http://news.bbc.co.uk/1/hi/entertainment/8083479.stm)"),
    "sims3": dict(
        nombre="Lanzamiento de Los Sims 3", origen="EDA", patron=r"\bsims\b",
        fechas=["2009-06-02"], fuente=f"[Wikipedia]({W}The_Sims_3)"),
    "palm_pre": dict(
        nombre="Lanzamiento del Palm Pre", origen="EDA", patron=r"palm pre",
        fechas=["2009-06-06"], fuente=f"[Wikipedia]({W}Palm_Pre)"),
    "elecciones_ue": dict(
        nombre="Elecciones europeas (el BNP gana sus primeras bancas)", origen="EDA",
        patron=r"\beu09\b|\bbnp\b|european election", fechas=["2009-06-04", "2009-06-07"],
        fuente=f"[Wikipedia]({W}2009_European_Parliament_election_in_the_United_Kingdom), [BBC](https://news.bbc.co.uk/2/hi/europe/8088309.stm)"),
    "nba": dict(
        nombre="Lakers campeones de la NBA", origen="EDA", patron=r"\blakers?\b|nba final",
        fechas=["2009-06-14"], fuente=f"[Wikipedia]({W}2009_NBA_Finals)"),
    "iran": dict(
        nombre="Elecciones y protestas en Irán; muerte de Neda", origen="EDA", patron=r"\biran|tehran|\bneda\b",
        fechas=["2009-06-12", "2009-06-20"],
        fuente=f"[Protestas]({W}2009_Iranian_presidential_election_protests), [Neda]({W}Killing_of_Neda_Agha-Soltan)"),
    "iphone": dict(
        nombre="iPhone OS 3.0 y iPhone 3GS", origen="EDA", patron=r"\bos ?3(?:\.0)?\b|iphone 3\.0|\b3gs\b",
        fechas=["2009-06-17", "2009-06-19"],
        fuente=f"[iPhone OS 3]({W}IPhone_OS_3), [iPhone 3GS]({W}IPhone_3GS)"),
    "dia_padre": dict(
        nombre="Día del Padre (EE. UU.)", origen="EDA", patron=r"father'?s'? ?day",
        fechas=["2009-06-21"], fuente=f"Tercer domingo de junio: [Wikipedia]({W}Father%27s_Day)"),
    "jon_kate": dict(
        nombre="Jon y Kate Gosselin anuncian su separación", origen="EDA", patron=r"jon (?:&|and) kate|gosselin",
        fechas=["2009-06-22"], fuente=f"[Wikipedia]({W}List_of_Kate_Plus_8_episodes)"),
    "mcmahon": dict(
        nombre="Muerte de Ed McMahon", origen="EDA", patron=r"mcmahon", fechas=["2009-06-23"],
        fuente=f"[Wikipedia]({W}Ed_McMahon), [CTV](https://www.ctvnews.ca/legendary-tv-host-ed-mcmahon-dead-at-86-1.410459)"),
    "transformers": dict(
        nombre="Estreno de Transformers 2", origen="EDA", patron=r"transformers", fechas=["2009-06-24"],
        fuente=f"[Wikipedia]({W}Transformers:_Revenge_of_the_Fallen)"),
    "farrah": dict(
        nombre="Muerte de Farrah Fawcett (9:28 PDT)", origen="EDA", patron=r"farrah|fawcett",
        fechas=["2009-06-25"],
        fuente=f"[Wikipedia]({W}Farrah_Fawcett), [People](https://people.com/celebrity/farrah-fawcett-dies-of-cancer-at-62/)"),
    # ---- encontrados por BERTopic (no estaban en las hipótesis)
    "star_wars_day": dict(
        nombre="Día de Star Wars (4 de mayo)", origen="BERTopic",
        patron=r"star ?wars day|may the (?:4th|fourth)|starwarsday", fechas=["2009-05-04"],
        fuente=f"[Wikipedia]({W}Star_Wars_Day)"),
    "bank_holiday": dict(
        nombre="Feriado bancario de mayo (Reino Unido)", origen="BERTopic", patron=r"bank holiday",
        fechas=["2009-05-04"], fuente=f"Primer lunes de mayo: [Wikipedia]({W}Public_holidays_in_the_United_Kingdom)"),
    "top_model": dict(
        nombre="Final de America's Next Top Model (Allison Harvard 2.ª)", origen="BERTopic",
        patron=r"top model|\bantm\b|\btyra\b", fechas=["2009-05-13"],
        fuente=f"[Wikipedia]({W}America%27s_Next_Top_Model_season_12)"),
    "true_blood": dict(
        nombre="Estreno de la temporada 2 de True Blood", origen="BERTopic", patron=r"true ?blood",
        fechas=["2009-06-14"], fuente=f"[Wikipedia]({W}True_Blood_season_2)"),
    "gp_britanico": dict(
        nombre="GP británico de Fórmula 1 (gana Vettel)", origen="BERTopic",
        patron=r"silverstone|british gp|\bf1\b|grand prix", fechas=["2009-06-21"],
        fuente=f"[Wikipedia]({W}2009_British_Grand_Prix), [BBC](https://news.bbc.co.uk/sport1/hi/motorsport/formula_one/8111672.stm)"),
    "metro_dc": dict(
        nombre="Choque de trenes del metro de Washington (9 muertos)", origen="BERTopic",
        patron=r"\bmetro\b|red line|fort totten", fechas=["2009-06-22"],
        fuente=f"[Wikipedia]({W}June_2009_Washington_Metro_train_collision)"),
}


def dias_validos(fechas, dias_con_datos):
    """Días en los que un pico confirma la fecha: la fecha ±1 día, o el primer
    día con datos posterior si la fecha cae en un hueco del dataset."""
    dias_con_datos = pd.DatetimeIndex(sorted(dias_con_datos))
    validos = set()
    for f in pd.to_datetime(fechas):
        validos |= {f + pd.Timedelta(days=d) for d in (-1, 0, 1)}
        posteriores = dias_con_datos[dias_con_datos >= f]
        if len(posteriores):
            validos.add(posteriores[0])
    return validos
