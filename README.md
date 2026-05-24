
# Повний цикл Data Science проєкту в Python

Підсумкова практична робота з курсу "Data Science in Python". У цьому проєкті реалізовано повний модульний конвеєр (pipeline) для обробки даних, проєктування ознак та класифікації сортів вин за допомогою машинного навчання.

## 📁 Структура проєкту

Проєкт організовано у вигляді структурованого Python-пакета:

```text
ds_final_project/
├── ds_project/               # Головний пакет модуля
│   ├── __init__.py
│   ├── utils.py              # Завантаження та відмовостійкість даних
│   ├── data/                 # Модуль передобробки та збереження даних
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── wine_types.csv
│   ├── features/             # Feature Engineering та Feature Selection
│   │   ├── __init__.py
│   │   └── build_features.py
│   └── models/               # Навчання моделей та розрахунок метрик
│       ├── __init__.py
│       └── train_model.py
├── tests/                    # Автоматичні тести
│   └── test_pipeline.py
├── .flake8                   # Конфігурація лінтера якості коду
├── .gitignore                # Специфікація ігнорування файлів кешу
└── main.py                   # Головний скрипт запуску pipeline
