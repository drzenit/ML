from math import sqrt
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
import pandas as pd
import numpy as np



# Задание 1

def firstTask(dataPath: str, dataType: str):
    # Чтение данных из файла в dataset
    if (dataType == "txt"):
        dataset = pd.read_csv(dataPath, sep=",", header=None)
        dataset = dataset.replace({'x': 1, 'o': 0, 'b': 2})

        # Разделение на признаки и метки
        feature = dataset.iloc[:, :-1]
        label = dataset.iloc[:, 9]
    elif (dataType == "csv"):
        dataset = pd.read_csv(dataPath)

        # Разделение на признаки и метки
        feature = dataset.iloc[:, :-1]
        label = dataset['type']

    Y = list()
    X = list()
    Ytr = list()
    trainSizeArr = np.arange(0.05, 0.95, 0.01)
    for i in trainSizeArr:
        trainSize = i
        testSize = 1 - i
        X.append(trainSize)

        # Деление данных на обучающие и тестовые
        feature_train, feature_test, label_train, label_test = train_test_split(feature, label, train_size=trainSize, test_size=testSize)

        # Создание и обучение экземпляра классификатора
        bayesCl = GaussianNB()
        bayesCl.fit(feature_train, label_train)

        # Тестирование классификатора на тестовой выборке
        bayesResult = bayesCl.predict(feature_test)

        # Тестирование классификатора на обучающей выборке
        bayesResultTr = bayesCl.predict(feature_train)

        # Оценка точности классификатора тестовой выборки
        bayesAccuracy = accuracy_score(bayesResult, label_test)
        Y.append(bayesAccuracy)

        # Оценка точности классификатора обучающей выборки
        bayesAccuracyTr = accuracy_score(bayesResultTr, label_train)
        Ytr.append(bayesAccuracyTr)

    plt.plot(X, Y)
    plt.xlabel("Train_Size")
    plt.ylabel("Accuracy")
    plt.show()

    plt.plot(X, Ytr)
    plt.xlabel("Train_Size")
    plt.ylabel("Accuracy")
    plt.show()

def secondTask():
    # Замена наименований X1 = X, X2 = Y, -1 = 1, 1 = 2 (для удобства)
    pointList1X = list()
    pointList1Y = list()
    pointList2X = list()
    pointList2Y = list()

    # Генерация точек
    for i in range(10):
        pointList1X.append(random.normalvariate(15, sqrt(4)))
        pointList1Y.append(random.normalvariate(18, sqrt(4)))
    for i in range(90):
        pointList2X.append(random.normalvariate(18, sqrt(2)))
        pointList2Y.append(random.normalvariate(18, sqrt(2)))

    # Построение диаграммы (класс -1 зеленые точки, класс 1 красные точки)
    plt.plot(pointList1X, pointList1Y, 'go', pointList2X, pointList2Y, 'ro')
    plt.show()

    # Создание и сборка в один DataFrame
    pointDF1 = pd.DataFrame(pointList1X, columns=['X'])
    pointDF1['Y'] = pointList1Y
    pointDF1['label'] = 'C1'

    pointDF2 = pd.DataFrame(pointList2X, columns=['X'])
    pointDF2['Y'] = pointList2Y
    pointDF2['label'] = 'C2'

    pointDF = pointDF1.append(pointDF2)

    feature = pointDF.iloc[:, :-1]
    label = pointDF.iloc[:, 2]

    # Деление данных на обучающие и тестовые
    trainSize = 0.6
    testSize = 1 - trainSize
    feature_train, feature_test, label_train, label_test = train_test_split(feature, label, train_size=trainSize, test_size=testSize)

    # Создание и обучение экземпляра классификатора
    bayesCl = GaussianNB()
    bayesCl.fit(feature_train, label_train)

    # Тестирование классификатора
    bayesResult = bayesCl.predict(feature_test)

    # Оценка точности классификатора
    bayesAccuracy = accuracy_score(bayesResult, label_test)
    print(bayesAccuracy)

    # Матрица ошибок
    bayesConfMat = confusion_matrix(bayesResult, label_test)
    print(bayesConfMat)

    # ROC-кривая




secondTask()

#firstTask("data\\tic_tac_toe.txt", "txt")
#firstTask("data\spam.csv", "csv")





