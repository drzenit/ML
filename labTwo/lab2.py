import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np


def firstTask(dataPath: str):
    # Конвертация в pandas.DataFrame
    def convertToDF(data):
        return pd.DataFrame(data)

    # Функции активации:

    # Сигмоида
    def sigmoidFunc(x):
        return (1 / (1 + np.exp(-x)))

    # Чтение данных из файла в dataset
    dataset = pd.read_csv(dataPath, sep=",")

    # Разделение на признаки и метки
    feature = dataset.iloc[:, :-1]
    label = dataset['class']
    # Замена -1 на 0 (требуется для обучения)
    label = label.replace({-1 : 0})

    # Нормировка
    #TODO: Нормировать

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
    print(weights)

    # Обучение - Метод обратного распространения
    #a = feature_train
    #outputDatas = sigmoidFunc(np.dot(a, weights))
    #err = label_train - outputDatas
    for i in range(20000):
        #print(weights)
        # Получение выходных значений
        inputLayer = feature_train
        outputData = sigmoidFunc(np.dot(inputLayer, weights))

        # Корректировка весов, исходя из ошибки
        err = label_train - outputData
        #print(err)
        adjustment = np.dot(inputLayer.T, (err * (outputData * (1 - outputData))))
        weights += adjustment

    # Тестирование нейрона
    inputLayer = feature_test
    outputData = sigmoidFunc(np.dot(inputLayer, weights))

    print(weights)
    print(outputData)
    resultData = list()
    for i in outputData:
        if (i >= 0.5):
            resultData.append(1)
        else:
            resultData.append(0)

    print(resultData)
    accuracyTest = accuracy_score(label_test, resultData)
    confMatrix = confusion_matrix(label_test, resultData)
    print(confMatrix)
    print(accuracyTest)


firstTask("data\\nn_0.csv")