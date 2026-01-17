Инструкция по запуску приложения

1. Подготовка окружения

Требуется:

Python 3.9
Anaconda
Git

Рекомендуется использовать отдельное Conda-окружение:

conda create -n heart_attack_api python=3.9 -y
conda activate heart_attack_api

2. Установка зависимостей

Склонируйте репозиторий и установите зависимости:

git clone https://github.com/KubasovaAnni/heart_attack_project.git
cd heart_attack_project
pip install -r requirements.txt
(если requirements.txt отсутствует:)
pip install fastapi uvicorn pandas numpy scikit-learn catboost joblib

3. Запуск приложения

Из корня проекта выполните:
uvicorn app.main:app --reload

Приложение будет доступно по адресу:
http://127.0.0.1:8000

4. Использование API

Swagger-документация доступна по адресу:
http://127.0.0.1:8000/docs

Для получения предсказания используйте эндпоинт:
POST /predict

Передайте данные пациента в формате JSON.
В ответ возвращается предсказание риска сердечного приступа.

5. Получение предсказаний для тестовой выборки

Предсказания для тестовых данных формируются в Jupyter Notebook
из папки notebooks.

Результат сохраняется в файл:
submission.csv

Файл содержит столбцы:
id
prediction
и используется для автоматической проверки качества модели.

Файл `submission.csv` формируется отдельно в Jupyter Notebook и
предназначен для автоматической проверки качества модели.

FastAPI используется только для интерактивного получения предсказаний
и не сохраняет результаты на диск.

6. Остановка сервера

Для остановки приложения нажмите:
Ctrl + C
в терминале с запущенным сервером.