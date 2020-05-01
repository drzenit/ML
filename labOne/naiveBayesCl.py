import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
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
        lable = dataset.iloc[:, 9]
    elif (dataType == "csv"):
        dataset = pd.read_csv(dataPath)

        # Разделение на признаки и метки
        feature = dataset.iloc[:, :-1]
        lable = dataset['type']

    Y = list()
    X = list()
    Ytr = list()
    trainSizeArr = np.arange(0.01, 0.99, 0.01)
    for i in trainSizeArr:
        trainSize = i
        testSize = 1 - i
        X.append(trainSize)

        # Деление данных на обучающие и тестовые
        feature_train, feature_test, lable_train, lable_test = train_test_split(feature, lable, train_size=trainSize, test_size=testSize)

        # Создание и обучение экземпляра классификатора
        bayesCl = GaussianNB()
        bayesCl.fit(feature_train, lable_train)

        # Тестирование классификатора на тестовой выборке
        bayesResult = bayesCl.predict(feature_test)

        # Тестирование классификатора на обучающей выборке
        bayesResultTr = bayesCl.predict(feature_train)

        # Оценка точности классификатора тестовой выборки
        bayesAccuracy = accuracy_score(bayesResult, lable_test)
        Y.append(bayesAccuracy)

        # Оценка точности классификатора обучающей выборки
        bayesAccuracyTr = accuracy_score(bayesResultTr, lable_train)
        Ytr.append(bayesAccuracyTr)

    plt.plot(X, Y)
    plt.xlabel("Train_Size")
    plt.ylabel("Accuracy")
    plt.show()

    plt.plot(X, Ytr)
    plt.xlabel("Train_Size")
    plt.ylabel("Accuracy")
    plt.show()

firstTask("data\\tic_tac_toe.txt", "txt")



