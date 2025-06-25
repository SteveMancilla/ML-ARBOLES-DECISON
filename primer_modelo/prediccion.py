import joblib
import pandas as pd

modelo = joblib.load("primer_modelo/modelo_duracion_sesion.pkl")

nuevo = pd.DataFrame([{
    "estudiante_id": "E020",
    "psicologo_id": "P04",
    "tipo_cita": "individual",
    "motivo": "estrés social",
    "modalidad": "presencial",
    "dia_semana": "jueves",
    "hora_inicio": "11:30"
}])

prediccion = modelo.predict(nuevo)
print(f"Duración estimada: {prediccion[0]:.2f} minutos")
