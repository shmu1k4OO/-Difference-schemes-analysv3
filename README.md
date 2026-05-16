# Difference Schemes Analyser v3

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.0+-orange.svg)](https://matplotlib.org/)

Численное решение волнового уравнения **uₜ + c·uₓ = 0** с использованием различных разностных схем.

## О проекте

Программа реализует три разностные схемы для решения уравнения переноса:
| Схема | Метод | Особенности |
|-------|-------|-------------|
| **Неявный левый уголок** | `implicit_left_corner` | 1-й порядок точности |
| **Схема "квадрат"** | `box_scheme_thomas` | 2-й порядок точности |
| **Схема Федоренко** | `fedorenko_scheme_vectorized` | Адаптивная, с автоматическим выбором вязкости |

## Режимы работы

- `main.py` — анимации решения + сохранение GIF
```bash
python main.py
```

- `main.py experiment` — численный эксперимент на гладком решении для определения порядков сходимости
```bash
python main.py experiment
```

- `main.py figures` — сохранение графиков решений в разные моменты времени
```bash
python main.py figures
```