import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# 1. Crear un conjunto de datos más grande con características y etiquetas emocionales simuladas
np.random.seed(42)

# Generando datos simulados para las características (3 características por fila) y más emociones
emotions = ['Happy', 'Sad', 'Neutral', 'Surprise', 'Anger']

data = {
   'Feature1': np.random.uniform(0, 1, 200),  # Reducir las muestras a 200
   'Feature2': np.random.uniform(1, 2, 200),
   'Feature3': np.random.uniform(0, 0.5, 200),
   'Emotion': [random.choice(emotions) for _ in range(200)]
}

# Crear un DataFrame
df = pd.DataFrame(data)

# Guardar en un archivo Excel
df.to_excel('emotions_data_simulated.xlsx', index=False)

# Mostrar las primeras filas
print(df.head())

# 2. Preprocesamiento de datos
# Convertir las emociones a números
label_encoder = LabelEncoder()
df['Emotion'] = label_encoder.fit_transform(df['Emotion'])

# Selección de características (Feature1, Feature2, Feature3)
X = df[['Feature1', 'Feature2', 'Feature3']]
y = df['Emotion']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calcular los pesos de clase
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# 3. Usar SMOTE (Synthetic Minority Over-sampling Technique) para balancear las clases
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 4. Entrenamiento del modelo usando Gradient Boosting
model = GradientBoostingClassifier(
    n_estimators=300,       # Aumentamos los árboles a 300
    learning_rate=0.05,     # Ajustamos la tasa de aprendizaje
    max_depth=6,            # Aumentamos la profundidad de los árboles
    random_state=42
)

# Entrenar el modelo
model.fit(X_train_res, y_train_res)

# 5. Predicciones
y_pred = model.predict(X_test)

# 6. Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy*100:.2f}%")
print(classification_report(y_test, y_pred))

# 7. Guardar las predicciones en un archivo Excel
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_excel('predictions_results.xlsx', index=False)

# 8. Crear Gráficas para el Análisis de Datos

# Configuración de la visualización
sns.set(style="whitegrid")

# 8.1 Gráfico de dispersión entre Feature1 y Feature2
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Feature1', y='Feature2', hue='Emotion', palette='coolwarm', s=100)
plt.title('Distribución de Emociones según Feature1 y Feature2')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend(title='Emotion')
plt.show()

# 8.2 Crear histogramas para Feature1, Feature2, y Feature3
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df['Feature1'], kde=True, color='skyblue')
plt.title('Distribución de Feature1')

plt.subplot(1, 3, 2)
sns.histplot(df['Feature2'], kde=True, color='lightgreen')
plt.title('Distribución de Feature2')

plt.subplot(1, 3, 3)
sns.histplot(df['Feature3'], kde=True, color='lightcoral')
plt.title('Distribución de Feature3')

plt.tight_layout()
plt.show()

# 8.3 Gráfico de barras para la distribución de emociones
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Emotion', palette='Set2')
plt.title('Distribución de Emociones')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.show()

# 9. Validación cruzada para medir el rendimiento
cross_val_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')  # Validación cruzada con más particiones
print(f"Precisión media en validación cruzada: {cross_val_scores.mean() * 100:.2f}%")