import pandas as pd
import joblib

# 1. Cargar el modelo entrenado
modelo = joblib.load("tercer_modelo/modelo_emociones_texto.pkl")

# 2. Ingresar texto del estudiante (puedes modificar esta parte para que venga desde un formulario o API)
texto_estudiante = "Lloro sin razon clara"

# 3. Crear DataFrame con el nuevo texto
nuevo_input = pd.DataFrame([{"texto": texto_estudiante}])

# 4. Realizar la predicción
emocion_predicha = modelo.predict(nuevo_input)[0]

# 5. Mostrar resultado
print(f"Texto ingresado: {texto_estudiante}")
print(f"Emoción detectada: {emocion_predicha}")
