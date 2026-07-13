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
│   ├── lgbm_embeddings.pkl           # LightGBM + GloVe
│   └── textblob_accuracy.pkl         # Accuracy de TextBlob (Para la comparación final)
│
└── Notebooks/
    ├── 01_EDA_y_preprocesamiento.ipynb
    ├── 02_Vectorización.ipynb
    ├── 03_Modelos.ipynb
    ├── 04_Modelopreentrenado_GloVe.ipynb
    └── 04.1_Topicos.ipynb
```
