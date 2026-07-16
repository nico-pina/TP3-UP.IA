# TP3-UP.IA- Análisis de sentimientos en Twitter + Topicos

El repositorio contiene un proyecto de NLP orientado al análisis de sentimientos sobre un dataset de 1,6 millones de Tweets. Se entrenan y comparan modelos propios (TF-IDF, Bag of Words), un modelo pre-entrenado basado en embeddings (GloVe) y un modelo de reglas/léxico pre-entrenado (TextBlob). Se complementa con un análisis exploratorio de **modelado de tópicos** (BERTopic + sentence-transformers), cruzando los temas detectados con el sentimiento de cada tweet.


## Estructura del repositorio
```
├── Data/
│   └── training.1600000.processed.noemoticon.csv   # Dataset original (Sentiment140)
│
├── Preprocesamiento/
│   └── training_limpio.csv                          # Dataset tras limpieza (notebook 01)
│
├── Clean/
│   ├── X_train_clean.pkl                            
│   ├── X_test_clean.pkl                             
│   ├── y_train_clean.pkl
│   └── y_test_clean.pkl
│
├── Vectorización/
│   ├── tfidf_vectorizer.pkl
│   ├── X_train_tfidf.pkl / X_test_tfidf.pkl
│   ├── bow_vectorizer.pkl
│   ├── X_train_bow.pkl / X_test_bow.pkl
│   └── y_train.pkl / y_test.pkl
│
├── Modelos/
│   ├── logreg_tfidf.pkl              # Regresión Logística + TF-IDF
│   ├── nbayes_bow.pkl                # Naive Bayes + BoW
│   ├── logreg_embeddings.pkl         # Regresión Logística + GloVe
│   └── textblob_accuracy.pkl         # Accuracy de TextBlob (Para la comparación final)
│
└── Notebooks/
    ├── 01_EDA_y_preprocesamiento.ipynb
    ├── 02_Vectorización.ipynb
    ├── 03_Modelos.ipynb
    ├── 04_Modelopreentrenado_GloVe.ipynb
    └── 05_Bonus_Topicos.ipynb
```
## Descripción de cada notebook

### `01_EDA_y_preprocesamiento.ipynb`
Carga el dataset, remapea etiquetas (`0`→`-1`, `4`→`1`) y limpia el texto (HTML, URLs, menciones, puntuación, stopwords). Incluye EDA de balance de clases, longitud de tweets y top de palabras por clase (log-odds ratio). Guarda `Preprocesamiento/training_limpio.csv`.

### `02_Vectorización.ipynb`
Split train/test (80/20, estratificado) antes de vectorizar. Genera TF-IDF y BoW con los mismos filtros de frecuencia, para una comparación justa entre ambas representaciones.

### `03_Modelos.ipynb`
Entrena Regresión Logística + TF-IDF y Naive Bayes +BoW, analiza sus palabras más influyentes, y compara ambos contra TextBlob (evaluado sobre texto original, no limpio, para no penalizarlo).

### `04_Modelopreentrenado_GloVe.ipynb`
Representa cada tweet como el promedio de embeddings GloVe ponderado por TF-IDF; entrena Regresión Logística y LightGBM sobre esa representación. Cierra con la comparación final de los cinco modelos del proyecto.

### `05_Bonus_Topicos.ipynb` (Bonus exploratorio)
Modelado de tópicos con BERTopic + sentence-transformers sobre una muestra de 250.000 tweets, cruzando cada tópico con su sentimiento. Exploratorio e independiente del pipeline de clasificación de sentimientos de los notebook 01 a 04.

## Conclusiones generales del análisis de sentimiento
 
- **El corpus está perfectamente balanceado** (50/50), por lo que no fue necesario aplicar resampling ni ponderación de clases.
- **La longitud del tweet no es un buen discriminador de sentimiento** por sí sola; las distribuciones son casi idénticas entre clases.
- **El modelo más simple fue el mejor:** Regresión Logística + TF-IDF alcanzó el mayor accuracy en test (~0.80), seguido de cerca por Naive Bayes + BoW (~0.77).
- **Los embeddings pre-entrenados (GloVe) rindieron peor que los enfoques simples** (~0.73 tanto con Regresión Logística), pese a ser la aproximación más sofisticada, se cree que el hecho de promediar embeddings con IDF palabra por palabra pierde información de orden y negación (por ejemplo, "not good" ≠ "good"), mientras que TF-IDF con bigramas sí la captura parcialmente; además, GloVe-Twitter es un embedding genérico no ajustado a esta tarea ni a este corpus específico.
- **TextBlob, el modelo de reglas/léxico genérico, obtuvo el desempeño más bajo** (~0.62), como era esperable al no estar entrenado sobre datos ni ajustado al lenguaje informal de Twitter.
- En conjunto, el proyecto muestra que lo más importante no es usar modelos complejos, sino tener una representación adecuada para la tarea. Un TF‑IDF bien ajustado al propio corpus funcionó mejor que un modelo basado en reglas preentrenado y que los embeddings GloVe promediados.

## Conclusiones generales del EDA de topicos + sentimientos

- BERTopic identificó **tópicos coherentes y fáciles de interpretar**, validados con ejemplos reales de tweets.
- Aproximadamente **43% de los tweets quedó sin tópico asignado**, esperable por lo breve y coloquial del corpus.
- Se detectó un tópico compuesto casi por completo por **texto no inglés**, mostrando que el dataset no es totalmente monolingüe.
- El cruce sentimiento–tópico muestra que "her/she/ashley" y "rain" concentran más negatividad, mientras "twitter" y "sun/sunshine" tienden a ser más positivos.
- El análisis agrega una capa temática al estudio de sentimiento, mostrando en qué tipos de contenido se concentra cada polaridad.



## Cómo reproducir el repositorio
 
1. Clonar el repositorio y ubicar `training.1600000.processed.noemoticon.csv` en `Data/`.
2. Ejecutar los notebooks del pipeline principal, ubicados en `Notebooks/`, en orden: `01` → `02` → `03` → `04_Modelopreentrenado_GloVe`. Cada uno crea automáticamente las carpetas de salida que necesita (`Preprocesamiento/`, `Clean/`, `Vectorización/`, `Modelos/`) si no existen. Es importante el orden, ya que, por ejemplo.. el notebook `04_Modelopreentrenado_GloVe` requiere que `03` se haya ejecutado previamente, ya que reutiliza los modelos y el accuracy de TextBlob guardados en `Modelos/`.
3. `04_Topicos.ipynb` (también en `Notebooks/`) es independiente del resto del pipeline (no requiere `02`, `03` ni `04_Modelopreentrenado_GloVe`): solo necesita que `01` se haya ejecutado, para poder leer `Preprocesamiento/training_limpio.csv`.
