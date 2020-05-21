import random
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


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

    # "Выравнивание" данных
    feature_train = np.array(feature_train)
    feature_test = np.array(feature_test)
    label_train = np.array([label_train]).T
    label_test = np.array([label_test]).T

    # Задание начальных весов случайным образом
    np.random.seed(25)
    weights = (2 * np.random.random((2, 1)) - 1)

    # Перебор функций активации
    for func in (sigmoidFunc, tanFunc):
        # Обучение - Метод обратного распространения
        for i in range(EPOCHS):  # Приразличных i нейрон может обучиться и переобучиться
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

        # Приведение к классу 0 или 1
        resultData = list()
        if (func.__name__ == "sigmoidFunc"):
            for res in outputData:
                if (res >= 0.5):
                    resultData.append(1)
                elif (res < 0.5):
                    resultData.append(0)
        elif (func.__name__ == "tanFunc"):
            for res in outputData:
                if (res >= 0):
                    resultData.append(1)
                elif (res < 0):
                    resultData.append(0)

        accuracyTest = accuracy_score(label_test, resultData)
        confMatrix = confusion_matrix(label_test, resultData)

        # Вывод результатов
        print("Нейрон на примере %s" %dataPath)
        print("Функция активации -", func.__name__)
        print("Матрица ошибок: \n", confMatrix)
        print("Точность: ", accuracyTest)

def firstTaskKeras(dataPath: str):
    # Чтение данных из файла в dataset
    dataset = pd.read_csv(dataPath, sep=",")

    # Разделение на признаки и метки
    feature = dataset.iloc[:, :-1]
    label = dataset['class']
    # Замена -1 на 0 (требуется для обучения)
    label = label.replace({-1: 0})

    # Деление данных на обучающие и тестовые
    trainSize = 0.8
    testSize = 1 - trainSize
    feature_train, feature_test, label_train, label_test = train_test_split(feature, label, train_size=trainSize,
                                                                            test_size=testSize, random_state=7)

    # Выравнивание данных
    feature_train = np.array(feature_train)
    feature_test = np.array(feature_test)
    label_train = np.array([label_train]).T
    label_test = np.array([label_test]).T

    # Различные функции активации и оптимизаторы
    for func in ("sigmoid", "tanh", "relu"):
        for optim in ("adagrad", "nadam", "adam", "adamax"):
            # Создаем нейрон из keras
            neuron = Sequential()
            neuron.add(Dense(1, activation=func))
            neuron.compile(loss='binary_crossentropy', optimizer=optim, metrics=['accuracy'])

            # Обучаем нейрон
            neuron.fit(feature_train, label_train, epochs=EPOCHS, use_multiprocessing=True, verbose=0)

            # Тестирование нейрона
            neuronResult = neuron.predict(feature_test)

            # Приведение к классу 0 или 1
            resultData = list()
            if (func == "sigmoid"):
                for res in neuronResult:
                    if (res >= 0.5):
                        resultData.append(1)
                    elif (res < 0.5):
                        resultData.append(0)
            elif (func == "tanh"):
                for res in neuronResult:
                    if (res >= 0):
                        resultData.append(1)
                    elif (res < 0):
                        resultData.append(0)
            elif (func == "relu"):
                for res in neuronResult:
                    if (res >= 5):
                        resultData.append(1)
                    elif (res < 5):
                        resultData.append(0)

            # Оценка
            accuracyTest = accuracy_score(label_test, resultData)
            confMatrix = confusion_matrix(label_test, resultData)

            # Вывод результатов
            print("Нейрон на примере %s" % dataPath)
            print("Функция активации - ", func)
            print("Оптимизатор - ", optim)
            print("Матрица ошибок: \n", confMatrix)
            print("Точность: ", accuracyTest)


EPOCHS = 100  # Количество эпох. Для nn_1.csv, при количесвте >1 наблюжается переобучение, точнсоть на тестовых данных падает

print("******************************************HANDMADE******************************************")
firstTask("data\\nn_0.csv")
firstTask("data\\nn_1.csv")

print("******************************************KERAS******************************************")
firstTaskKeras("data\\nn_0.csv")
firstTaskKeras("data\\nn_1.csv")
