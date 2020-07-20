import numpy as np
import pandas as pd
from sklearn import neural_network
import sklearn
from sklearn import metrics
import pickle

csv1 = "BTC15.csv"
csv2 = "regression15.pickle"    # название лучшего засоленного файла

data = pd.read_csv(csv1, sep=",")

data = data[['open', 'high', 'low', 'close', 'predict']]
"""
в общем у нас есть данные свечного графика актива,в этом случае старые данные BTC/USD
в колонке predict записана цена закрытия следующей свечи,значит на по параменрам текущей свечи надо предсказать закрытие
следующей => у нас есть открытие закрытие высшая и низшая цены
"""
predict = "predict"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

linear = neural_network.multilayer_perceptron.MLPRegressor()    # здесь не линейная регрессия,а многослойный перцептрон
# как видишь в пайтоне можно оооочень просто использовать разные алгоритмы обучения,просто переписала 1 строку и все

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)

best = 0
for _ in range(50):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)

    linear = linear

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print("Accuracy: " + str(acc))

    if acc > best:
        best = acc
        with open(csv2, "wb") as f:
            pickle.dump(linear, f)

# LOAD MODEL
pickle_in = open(csv2, "rb")
linear = pickle.load(pickle_in)
"""
открой файл prediction15min.py , чтобы понять как потом юзать pickle файлы
"""
predicted = linear.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], y_test[x])

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predicted))   # разные методы рассчета ошибки
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predicted))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predicted)))

print("Trained: ", best)
