import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np


# Первое задание без использования keras
def firstTask(dataPath: str):
    # Конвертация в pandas.DataFrame
    def convertToDF(data):
        return pd.DataFrame(data)

    # Функции активации:

    # Сигмоида
    def sigmoidFunc(x):
        return (1 / (1 + np.exp(-x)))

    # Гиперболический тангенс
    def tanFunc(x):
        return np.tanh(x)

    # ReLu функция
    def reluFunc(x):
        if (0 > x.all()):
            return 0
        else:
            return x

    # Чтение данных из файла в dataset
    dataset = pd.read_csv(dataPath, sep=",")

    # Разделение на признаки и метки
    feature = dataset.iloc[:, :-1]
    label = dataset['class']
    # Замена -1 на 0 (требуется для обучения)
    label = label.replace({-1 : 0})

    # Деление данных на обучающие и тестовые
    trainSize = 0.9
    testSize = 1 - trainSize
    feature_train, feature_test, label_train, label_test = train_test_split(feature, label, train_size=trainSize,
                                                                            test_size=testSize, random_state=7)

    feature_train = np.array(feature_train)
    feature_test = np.array(feature_test)
    label_train = np.array([label_train]).T
    label_test = np.array([label_test]).T

    # Задание начальных весов случайным образом
    np.random.seed(25)
    weights = (2 * np.random.random((2, 1)) - 1)

    for func in (sigmoidFunc, tanFunc, reluFunc):
        # Обучение - Метод обратного распространения
        for i in range(1000):  # Приразличных i нейрон может обучиться и переобучиться
            # Получение выходных значений
            inputLayer = feature_train
            outputData = func(np.dot(inputLayer, weights))

            # Корректировка весов, исходя из ошибки
            err = label_train - outputData
            adjustment = np.dot(inputLayer.T, (err * (outputData * (1 - outputData))))
            weights += adjustment

        # Тестирование нейрона
        inputLayer = feature_test
        outputData = func(np.dot(inputLayer, weights))

        resultData = list()
        for i in outputData:
            if (i >= 0.5):
                resultData.append(1)
            else:
                resultData.append(0)

        accuracyTest = accuracy_score(label_test, resultData)
        confMatrix = confusion_matrix(label_test, resultData)
        print("Нейрон на примере %s" %dataPath)
        print("Функция активации -", func.__name__)
        print("Матрица ошибок: \n", confMatrix)
        print("Точность: ", accuracyTest)





firstTask("data\\nn_0.csv")
firstTask("data\\nn_1.csv")
