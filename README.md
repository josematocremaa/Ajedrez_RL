# AlphaChess: Agente de IA con Aprendizaje por Refuerzo y CNN
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
Proyecto desarrollado para la asignatura de **Inteligencia Artificial** del Máster en Automática y Robótica (UPM). 

## 📊 Rendimiento Final
El agente fue evaluado contra un oponente aleatorio en 50 partidas con los siguientes resultados:
- **Victorias:** 47
- **Derrotas:** 0
- **Empates:** 3
- **Tasa de éxito:** 94.0%

## 🧠 Arquitectura y Metodología
* **Modelo:** CNN con entrada de tablero 8x8x12.
* **Búsqueda:** Algoritmo Minimax con poda Alpha-Beta (Profundidad 4).
* **Entrenamiento:** Aprendizaje por Refuerzo híbrido
* **Evaluación:** Sistema de balance 80% Material / 20% Red Neuronal Posicional.

## 🛠️ Estructura del Proyecto
* `model.py`: Definición de la red `AlphaChessNet`.
* `train_loop.py`: Bucle de entrenamiento y generación de experiencia.
* `play.py`: Script para jugar contra la IA (Humano vs IA).
* `utils.py`: Procesamiento de tableros y cálculo de recompensas.

