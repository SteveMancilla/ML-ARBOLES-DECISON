import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Cargar el archivo Excel con los datos históricos
df = pd.read_excel("primer_modelo/datos_sesiones_psicologicas.xlsx")  # Cambia la ruta si es necesario

# Separar características y variable objetivo
X = df.drop(columns=["duracion_minutos"])
y = df["duracion_minutos"]

# Definir columnas categóricas que se deben codificar
categorical_features = [
    "estudiante_id", "psicologo_id", "tipo_cita",
    "motivo", "modalidad", "dia_semana", "hora_inicio"
]

# Preprocesador: codificación one-hot
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)]
)

# Pipeline: preprocesamiento + modelo
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📊 Resultados del modelo:")
print(f"MAE (Error absoluto medio): {mae:.2f}")
print(f"MSE (Error cuadrático medio): {mse:.2f}")
print(f"R2 (Coeficiente de determinación): {r2:.2f}")

# Guardar el modelo entrenado (opcional)
joblib.dump(model, "modelo_duracion_sesion.pkl")