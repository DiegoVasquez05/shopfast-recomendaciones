# Datos

## Dataset asignado: Bitext Gen AI Chatbot Customer Support Dataset (Kaggle)

<https://www.kaggle.com/datasets/bitext/bitext-gen-ai-chatbot-customer-support-dataset>

~26.900 pares instrucción/respuesta en inglés, 27 intenciones y 11 categorías
(columnas `flags, instruction, category, intent, response`).

```bash
pip install kaggle      # requiere ~/.kaggle/kaggle.json
kaggle datasets download -d bitext/bitext-gen-ai-chatbot-customer-support-dataset -p data --unzip
mv data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv data/bitext.csv
```

Los CSV no se suben al repositorio (`.gitignore`). En AWS se guardan en S3 con cifrado SSE-KMS.

## Archivos versionados

| Archivo | Contenido |
|---|---|
| `kb/telcoaustral_kb.json` | Base de conocimiento para el RAG (33 artículos, ficticios, uso académico) |
| `eval_es_cl.csv` | 67 frases en español chileno escritas a mano para evaluar el NLU (no se usan para entrenar) |

El corpus de adaptación al español chileno se genera con `training/spanish_cl.py` al entrenar.
