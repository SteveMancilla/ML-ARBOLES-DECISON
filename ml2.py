import nltk
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# Descargar recursos necesarios
nltk.download('stopwords')

# Configuración inicial
stopwords_es = stopwords.words('spanish')
stemmer = SnowballStemmer('spanish')

# Tokenizador básico (no usa punkt)
def tokenizar_basico(texto):
    return texto.split()

# Preprocesamiento de texto
def preprocesar_texto(texto):
    texto = texto.lower()
    texto = ''.join(c for c in texto if c not in string.punctuation)
    tokens = tokenizar_basico(texto)
    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords_es]
    return ' '.join(tokens)

# Dataset original
frases = [
    "Me siento muy ansioso por los exámenes",
    "Tengo mucha presión por las tareas",
    "Estoy muy motivado para seguir adelante",
    "No tengo ganas de hacer nada hoy",
    "Estoy preocupado por mi futuro",
    "Siento que no puedo más con el estrés",
    "Hoy fue un gran día, me siento feliz",
    "Estoy muy triste por lo que pasó",
    "No tengo energía ni motivación para estudiar",
    "Estoy emocionado por el nuevo ciclo",
    "Siento ansiedad cada vez que entro a clase",
    "Estoy estresado por los plazos",
    "Estoy enfocado y listo para avanzar",
    "Siento mucha tristeza al recordar ciertas cosas",
    "Hoy me siento positivo y con energía",
    "No sé qué hacer, estoy confundido",
    "Tengo miedo de reprobar",
    "Hoy fue un día productivo",
    "Siento paz al caminar solo",
    "Estoy muy estresado por mi familia"
]

emociones = [
    "ansiedad",
    "estrés",
    "motivación",
    "tristeza",
    "ansiedad",
    "estrés",
    "motivación",
    "tristeza",
    "tristeza",
    "motivación",
    "ansiedad",
    "estrés",
    "motivación",
    "tristeza",
    "motivación",
    "ansiedad",
    "ansiedad",
    "motivación",
    "tranquilidad",
    "estrés"
]

# AUMENTO DE DATOS: duplicar frases con ligeras variaciones
frases_extra = []
emociones_extra = []
for frase, emocion in zip(frases, emociones):
    frases_extra.append(frase)
    frases_extra.append(frase.replace("siento", "me siento").replace("hoy", "este día"))
    frases_extra.append(frase + ". De verdad me afecta.")
    emociones_extra += [emocion] * 3

# Filtrar clases poco representadas
conteo = Counter(emociones_extra)
frases_filtradas = [f for i, f in enumerate(frases_extra) if conteo[emociones_extra[i]] > 2]
emociones_filtradas = [e for i, e in enumerate(emociones_extra) if conteo[e] > 2]

# Preprocesar
frases_procesadas = [preprocesar_texto(f) for f in frases_filtradas]

# Vectorización TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1500)
X = vectorizer.fit_transform(frases_procesadas)

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, emociones_filtradas, test_size=0.2, random_state=42, stratify=emociones_filtradas
)

# Entrenar modelo Random Forest
modelo = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)

# Resultados
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReporte:")
print(classification_report(y_test, y_pred, zero_division=0))

# Matriz de confusión
labels = unique_labels(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.grid(False)
plt.show()

# Visualización del árbol
plt.figure(figsize=(20, 10))
plot_tree(modelo.estimators_[0],
          feature_names=vectorizer.get_feature_names_out(),
          class_names=modelo.classes_,
          filled=True, rounded=True, fontsize=8)
plt.title("Árbol de decisión - 1er árbol del Random Forest")
plt.show()

# Palabras clave más importantes
importances = modelo.feature_importances_
indices = np.argsort(importances)[::-1]
features = np.array(vectorizer.get_feature_names_out())

top_n = 10
plt.figure(figsize=(10, 5))
plt.barh(range(top_n), importances[indices[:top_n]][::-1])
plt.yticks(range(top_n), features[indices[:top_n]][::-1])
plt.xlabel("Importancia")
plt.title("Top 10 palabras clave según Random Forest")
plt.show()