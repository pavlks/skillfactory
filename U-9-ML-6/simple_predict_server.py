import pickle

import numpy as np
from flask import Flask, request
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston


# Инициализируем сервер Flask
app = Flask(__name__)

# загружаем признаки и целевую переменную из датасета boston
X, y = load_boston(return_X_y=True)
# возьмём два признака для разнообразия
X = X[:, 4:6].reshape(-1, 2)
# посмотрим первые 5 значений, чтобы понимать, какого рода данные передавать
# для предсказания
print(X[:5])
# инициализируем и обучим модель
regressor = LinearRegression()
regressor.fit(X, y)
# пробные данные для предсказания
value_to_predict = np.array([0.8, 8]).reshape(-1, 2)
# предсказываем и печатаем
predicted = regressor.predict(value_to_predict)
print(f"Predicted result = {predicted}")


# сериализуем модель и сохраняем в файл
with open('trained_model.pkl', 'wb') as file:
   	pickle.dump(regressor, file)

# читаем из файла и десереализуем модель
with open('myfile.pkl', 'rb') as file:
    	regressor_from_file = pickle.load(file)

# предсказываем результат по двум значениям
def model_predict(value1, value2):
    """
    Функция берет на вход 2 значения, которые приблизительно равны [0.5, 7]
    и возвращает результат предсказания на уже обученной модели
    """
    value_to_predict = np.array([value1, value2]).reshape(-1, 2)
    predicted_value = regressor_from_file.predict(value_to_predict)
    return predicted_value

@app.route('/predict')
def predict_page():
    """
    Данный адресное окончание ожидае два значения [value1, value2]
    """
    # получаем переданные значения
    value1 = request.args.get('value1')
    value2 = request.args.get('value2')

    # проверяем, что переданы оба значения
    if value1 is None or value2 is None:
        return "value1 and value2 are expected\n"

    # проверяем, что переданы числовые значения
    try:
        float(value1)
        float(value2)
    except ValueError:
        return "Values must be INTEGERS or FLOATS\n"

    # рассчитываем предсказание по полученным значениям
    prediction = model_predict(value1, value2)
    # перенос строки для сообщения ответа
    nl = '\n'
    return f'The result of prediction is {prediction}!{nl}'

if __name__ == '__main__':
    app.run('localhost', 5000)
