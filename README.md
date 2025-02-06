# Разработка модели машинного обучения
## Цель: 
Задача состояла в разработке модели машинного обучения, способную эффективно работать с большими данными.
## Исходные данные
Исходными данными были 10 датафреймов: 5 датафреймов признаков (4 тренировочных и один тестовый) и 5 датафреймов целевых переменных (также 4 тренировочных и один тестовый). Общий вес данных составил около 80 Гб.
## Предобработка данных
Предобработка осуществляется в ноутбуке preprocessing.ipynb. 
Здесь происходит:

    Удаление сильно коррелирующих признаков.
    Удаление признаков, не влияющих на целевую переменную.
    Изменение типа данных с np.float64 на np.float32.

Результатом стала экономия места:

    Тренировочные данные: 75% сжатия.
    Целевые переменные: 75% сжатия.

После этого были объединены тренировочные датасеты в один общий набор, а также целевые переменные. Далее они были заархивированы в формате Feather, что привело к дополнительному сжатию:

    Тренировочные данные — 36%.
    Тренировочный таргет — 17%.
    Тестовые данные — 16%.
    Тестовый таргет — 17.5%.

Обучение модели
Обучение моделей происходит в ноутбуке train.ipynb. Используются подготовленные данные для обучения моделей машинного обучения на тестовых наборах.
Характеристики датасетов

    Размеры тренировочного датасета: (120000000,7).
    Размеры тестового датасета: (30000000,7).

Испытанные модели
Были выбраны три модели разной архитектуры:

    Линейная регрессия
    Градиентный бустинг (LGBMRegressor)
    Ансамбль (RandomForest Regressor)

На тренировочных данных была измерена скорость обучения; на тестовых — метрики R2 и RMSE:


**Метрики моделей**
=====================

| Модель | Время обучения | R2 | RMSE |
|--------|---------------|----|------|
| Линейная регрессия    | ~31 секунда   | ~0.432   | 930 |
| LGBRegressor         | ~54 секунды   | ≈0       | ~1235 |
| RandomForest Regressor| ~24 секунды  | ≈0       | ~1235 |

В качестве основной модели была выбрана модель Линейной регрессии, которая предсказала тестовые данные с $R2 = 0.444$ и $RMSE = 921$

# Разработка REST-API и создание Telegram бота

## Настройка приложения FastAPI
Сначала была создана базовая структура приложения FastAPI со всеми необходимыми импортами для API обработки данных.
1. API Эндпоинты

    1. Загрузка файлов через эндпоинт /load.
    2. Предобработка данных через эндпоинт /preprocess, где удаляются определенные столбцы из датафрейма; стандартизируются данные; оптимизируются типы до float32.
    3. Список доступных моделей по эндпоинту /list_models.
    4. Выбор модели через эндпоинт /choose_model.
    5. Предсказание по выбранной модели через эндпоинт /predict.

2. Запуск приложения FastAPI
Приложение запускается на порту 8000 с помощью uvicorn для автоматического обновления кода во время разработки.

## Настройка Telegram бота 
Был создан Telegram бот для взаимодействия пользователей посредством aiogram: 1. Создание клавиатур для удобного взаимодействия:

    1.keyboard_load (загрузить файл/список моделей)
    2.keyboard_process (обработать)
    3.keyboard_predict (прогнозировать)
    4.keyboard_choose_models (выбрать модель)

Команды бота: 
+ /start - приветствие пользователя
+ /load - сообщает о загрузке файлов
+ /predict - выполняет прогнозирование
+ /list_models - список доступных моделей Запущен процесс постоянного опроса сервера Telegram для реагирования на команды пользователей без перезапуска процесса после каждого взаимодействия.
