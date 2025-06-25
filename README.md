# 🌱 Clasificador de Emociones con Árboles de Decisión

Este proyecto usa aprendizaje automático para detectar emociones en frases escritas por estudiantes, como parte de una IA que simula a un psicólogo virtual.

---

## 🧠 Algoritmo usado

Se utiliza el modelo **Random Forest**, un conjunto de múltiples **árboles de decisión** que trabajan en conjunto para mejorar la precisión y reducir errores:

- Cada árbol analiza el texto (convertido con TF-IDF) y predice una emoción.
- La predicción final se elige por **votación** entre todos los árboles.
- Es robusto, preciso y adecuado para clasificación de texto en pocas muestras.

---

## ▶️ Cómo ejecutar

1. Clona el repositorio y entra al proyecto:

```bash
https://github.com/SteveMancilla/ML-ARBOLES-DECISON.git
