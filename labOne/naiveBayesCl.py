from math import sqrt
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
import pandas as pd
import numpy as np


# Задание 1
def firstTask(dataPath: str, dataType: str):
    print("ЗАДАНИЕ_1")
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
    plt.title("ЗАДАНИЕ 1 Тестовая %s" %dataPath)
    plt.xlabel("Train_Size")
    plt.ylabel("Accuracy")
    plt.show()

    plt.plot(X, Ytr)
    plt.title("ЗАДАНИЕ 1 Обучающая %s" %dataPath)
    plt.xlabel("Train_Size")
    plt.ylabel("Accuracy")
    plt.show()

# Задание 2
def secondTask():
    print("ЗАДАНИЕ_2")
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
    plt.title("ЗАДАНИЕ 2 - Распределение точек")
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
    print("Матрица ошибок:")
    print(bayesConfMat)

    # Таблица ошибок
    TP1 = bayesConfMat[0, 0]
    FP1 = bayesConfMat[0, 1]
    FN1 = bayesConfMat[1, 0]
    TN1 = bayesConfMat[1, 1]

    TP2 = bayesConfMat[1, 1]
    FP2 = bayesConfMat[1, 0]
    FN2 = bayesConfMat[0, 1]
    TN2 = bayesConfMat[0, 0]

    # Получение вероятностей классификации
    bayesResultProba = bayesCl.predict_proba(feature_test)
    bayesResultProba = pd.DataFrame(bayesResultProba)

    proba1 = bayesResultProba[0]
    proba2 = bayesResultProba[1]

    # Замена на бинарные значения для удобства
    label_test1 = pd.DataFrame(label_test)
    label_test1 = label_test.replace({'C1': 1, 'C2': 0})

    label_test2 = pd.DataFrame(label_test)
    label_test2 = label_test.replace({'C1': 0, 'C2': 1})

    # Построение ROC-кривой и получение AUC (площади под кривой)
    fpr, tpr, thresholds = roc_curve(label_test1, proba1)
    auc = roc_auc_score(label_test1, proba1)
    print("Площадь AUC = ", auc)
    plt.plot(fpr, tpr, 'g')
    fpr, tpr, thresholds = roc_curve(label_test2, proba2)
    auc = roc_auc_score(label_test2, proba2)
    print("Площадь AUC = ", auc)
    plt.plot(fpr, tpr, 'r')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ЗАДАНИЕ 2 - ROC-кривая")
    plt.show()

    # Получение PR-кривой
    precision, recall, thresholds = precision_recall_curve(label_test1, proba1)
    plt.plot(recall, precision, 'g')
    precision, recall, thresholds = precision_recall_curve(label_test2, proba2)
    plt.plot(recall, precision, 'r')
    plt.plot([0, 1], [0.5, 0.5], 'k--')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("ЗАДАНИЕ 2 - PR-кривая")
    plt.show()





firstTask("data\\tic_tac_toe.txt", "txt")
firstTask("data\spam.csv", "csv")

secondTask()



