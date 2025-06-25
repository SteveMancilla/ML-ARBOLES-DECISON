import pandas as pd
import joblib

# 1. Cargar el modelo entrenado (.pkl)
modelo = joblib.load("segundo_modelo/modelo_cancelacion_citas.pkl")

# 2. Crear un nuevo registro para predecir
nuevo_estudiante = pd.DataFrame([{
    "estudiante_id": "E150",
    "total_citas_previas": 4,
    "total_canceladas": 2,
    "total_reprogramadas": 1,
    "dia_semana": "jueves",
    "hora_cita": "15:30",
    "tipo_cita": "individual",
    "modalidad": "virtual",
    "motivo": "ansiedad"
}])

# 3. Predecir resultado (0 = asistirá, 1 = cancelará o faltará)
prediccion = modelo.predict(nuevo_estudiante)[0]

# 4. Mostrar el resultado interpretado
if prediccion == 1:
    print("❌ El estudiante probablemente **cancelará o no asistirá** a la cita.")
else:
    print("✅ El estudiante probablemente **asistirá** a la cita.")
