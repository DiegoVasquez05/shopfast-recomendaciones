"""Normalizador de español chileno (modismos, abreviaturas de chat y errores comunes).

IMPORTANTE: este archivo debe seguir siendo compatible con Python 3.8 y no
importar nada del proyecto, porque se empaqueta tal cual en el contenedor
SageMaker scikit-learn (ver aws/sagemaker_deploy.py).

Ejemplo:
    >>> normalize("wena po, kiero cachar cuanto debo de la boleta altiro")
    'hola quiero saber cuanto debo de la boleta ahora'
"""
import re
import unicodedata

# Modismos chilenos / lenguaje de chat -> español estándar.
# Se aplican sobre texto en minúsculas y sin tildes, por palabra completa.
CHILEAN_SLANG = {
    # saludos y muletillas
    "wena": "hola", "wenas": "hola", "holi": "hola", "alo": "hola", "aló": "hola",
    "po": "", "pos": "", "poh": "", "oe": "oye", "oiga": "oye", "ya": "ok", "dale": "ok", "sipo": "si",
    "nopo": "no", "nica": "no", "bkn": "excelente", "bacan": "excelente", "filete": "excelente",
    "chao": "adios", "chau": "adios", "nos vimos": "adios",
    # verbos y expresiones
    "cachai": "entiendes", "cacho": "entiendo", "cachar": "saber", "cache": "entendi",
    "altiro": "ahora", "al tiro": "ahora", "toy": "estoy", "tai": "estas", "ta": "esta", "tan": "estan",
    "pa": "para", "pal": "para el", "na": "nada", "ni ahi": "no me interesa", "achuntar": "acertar",
    "apechugar": "asumir", "pololo": "pareja", "polola": "pareja", "pega": "trabajo",
    "cuatico": "complicado", "cuatica": "complicado", "heavy": "grave", "la raja": "excelente",
    "fome": "malo", "penca": "malo", "como las weas": "muy malo", "chanta": "estafador",
    "pucha": "lamentablemente", "chuta": "lamentablemente", "harto": "mucho", "caleta": "mucho",
    "ene": "mucho", "cualquier cantidad": "mucho", "luca": "mil pesos", "lucas": "mil pesos",
    "plata": "dinero", "billete": "dinero", "gamba": "cien pesos",
    "kiero": "quiero", "kero": "quiero", "nesesito": "necesito", "nesecito": "necesito",
    "porfa": "por favor", "porfavor": "por favor", "xfa": "por favor", "pls": "por favor",
    # abreviaturas de chat
    "q": "que", "k": "que", "ke": "que", "xq": "porque", "pq": "porque", "porq": "porque",
    "tb": "tambien", "tmb": "tambien", "x": "por", "d": "de", "dnd": "donde", "cdo": "cuando",
    "msj": "mensaje", "wsp": "whatsapp", "wasap": "whatsapp", "guasap": "whatsapp",
    # dominio telco
    "cel": "celular", "celu": "celular", "fono": "telefono", "movil": "celular",
    "megas": "datos moviles", "gigas": "datos moviles", "gb": "datos moviles", "chip": "sim",
    "wifi": "internet", "wi fi": "internet", "la net": "internet", "el internet": "internet",
    "señal": "senal", "portarme": "portabilidad", "cambiarme de compania": "portabilidad",
    "la cuenta": "la boleta", "mi cuenta de luz": "mi boleta",
}

# insultos: se reemplazan por un marcador que usa el análisis de sentimiento
INSULTS = {"weon", "weona", "hueon", "huevon", "wn", "aweonao", "ctm", "conchetumare", "csm", "qlo", "ql",
           "culiao", "mierda", "weas", "wea"}

_MULTI_SPACE = re.compile(r"\s+")
_REPEATED = re.compile(r"(\w)\1{2,}")       # holaaaa -> holaa -> hola (abajo)
_NON_TEXT = re.compile(r"[^\w\s$%]")


def strip_accents(text):
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", text)


def _build_regex():
    keys = sorted(CHILEAN_SLANG, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(strip_accents(k)) for k in keys) + r")\b")


_SLANG_RX = _build_regex()
_SLANG_MAP = {strip_accents(k): v for k, v in CHILEAN_SLANG.items()}


def normalize(text):
    """Minúsculas, sin tildes, sin letras repetidas, modismos -> estándar, insultos -> 'insulto'."""
    if not text:
        return ""
    t = strip_accents(str(text).lower())
    t = _REPEATED.sub(r"\1", t)
    t = _NON_TEXT.sub(" ", t)
    t = _SLANG_RX.sub(lambda m: _SLANG_MAP[m.group(1)], t)
    t = " ".join("insulto" if w in INSULTS else w for w in t.split())
    return _MULTI_SPACE.sub(" ", t).strip()
