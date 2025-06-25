import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# 1. Cargar el archivo Excel
df = pd.read_excel("tercer_modelo/datos_emociones_texto.xlsx")  # Asegúrate que esté en el mismo directorio

# 2. Separar características y etiquetas
X = df["texto"]
y = df["emocion"]

# 3. Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Crear pipeline: vectorización TF-IDF + modelo Random Forest
modelo = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Entrenar el modelo
modelo.fit(X_train, y_train)

# 6. Evaluación
y_pred = modelo.predict(X_test)
print("📊 Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# 7. Mostrar matriz de confusión (opcional)
disp = ConfusionMatrixDisplay.from_estimator(
    modelo, X_test, y_test,
    display_labels=modelo.classes_,
    cmap=plt.cm.Blues,
    xticks_rotation=45
)
plt.title("Matriz de Confusión - Detección de Emociones")
plt.grid(False)
plt.tight_layout()
plt.show()

# 8. Guardar el modelo entrenado en un archivo .pkl
joblib.dump(modelo, "modelo_emociones_texto.pkl")
print("✅ Modelo guardado como 'modelo_emociones_texto.pkl'")