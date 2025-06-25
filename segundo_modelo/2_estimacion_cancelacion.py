import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib

# 1. Cargar los datos desde Excel
df = pd.read_excel("segundo_modelo/datos_cancelacion_citas.xlsx")  # Cambia la ruta si es necesario

# 2. Separar variables independientes (X) y variable objetivo (y)
X = df.drop(columns=["resultado"])
y = df["resultado"]

# 3. Especificar las columnas categóricas
categorical_features = ["estudiante_id", "dia_semana", "hora_cita", "tipo_cita", "modalidad", "motivo"]

# 4. Preprocesador con OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder='passthrough'  # conservar columnas numéricas
)

# 5. Definir el pipeline (preprocesamiento + modelo)
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# 6. Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Entrenar el modelo
model.fit(X_train, y_train)

# 8. Evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("📊 Resultados del modelo:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# 9. Matriz de confusión (opcional)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Asistió", "Canceló/Inasistió"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
plt.grid(False)
plt.show()

# 10. Guardar el modelo entrenado
joblib.dump(model, "modelo_cancelacion_citas.pkl")
print("✅ Modelo guardado como 'modelo_cancelacion_citas.pkl'")
