import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from  tensorflow.keras.datasets import mnist


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
        for optim in ("adadelta", "adagrad", "nadam", "adam", "adamax"):
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

def secondTask(dataPath: str):
    def readFile(dataPath: str):
        # Чтение данных из файла в dataset
        dataset = pd.read_csv(dataPath, sep='\t', header=0)

        # Разделение на признаки и метки (для удобства меняем red - 0, green - 1)
        dataset = dataset.replace({'red': 0, 'green': 1})
        feature = dataset.iloc[:, :-1]
        label = dataset.iloc[:, -1]

        return feature, label

    # Чтение данных из файла в dataset
    dataset = pd.read_csv(dataPath, sep=",")

    # Разделение на признаки и метки
    feature = dataset.iloc[:, :-1]
    label = dataset['class']
    # Замена -1 на 0 (требуется для обучения)
    label = label.replace({-1: 0})
    #feature, label = readFile(dataPath)

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

    # Создаем нейрон из keras
    neuron = Sequential()
    neuron.add(Dense(1, activation="tanh"))
    neuron.compile(loss='binary_crossentropy', optimizer="adamax", metrics="accuracy")

    # Обучаем нейрон
    neuron.fit(feature_train, label_train, epochs=EPOCHS2, use_multiprocessing=True, verbose=1)

    # Тестирование нейрона
    neuronResult = neuron.predict(feature_test)

    # Приведение к классу 0 или 1
    resultData = list()
    for res in neuronResult:
        if (res >= 0.5):
            resultData.append(1)
        elif (res < 0.5):
            resultData.append(0)

    # Оценка
    accuracyTest = accuracy_score(label_test, resultData)
    confMatrix = confusion_matrix(label_test, resultData)

    # Вывод результатов
    print("Нейрон на примере %s" % dataPath)
    print("Функция активации - ", "sigmoid")
    print("Оптимизатор - ", "adagrad")
    print("Матрица ошибок: \n", confMatrix)
    print("Точность: ", accuracyTest)

def thirdTask():
    # Задаем пазмерность
    imgRows, imgCols = 28, 28

    # Загружаем разделенные данные из kears.mnist
    (feature_train, label_train), (feature_test, label_test) = mnist.load_data()
    print("Размер тренировачных данных - ", len(feature_train))
    print("Размер тестовых данных - ", len(feature_train))

    # Нормировка данных изображений
    feature_train = feature_train / 255
    feature_test = feature_test / 255

    # Инициализируем нейронную сеть
    model = Sequential()

    # Добавляем и настраиваем слои
    model.add(Flatten(input_shape=(imgRows, imgCols)))  # Создаем входной слой равный размерности изображения (28 * 28 = 784)
    model.add(Dense(128, activation="relu"))  # Добавляем скрытый слой
    model.add(Dense(10, activation="softmax"))  # Добавляем выходной слой на 10 нейронов = количеству типов цифр [0-9]

    # Компилируем модель
    model.compile(optimizer="adadelta", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Обучаем модель
    model.fit(feature_train, label_train, epochs=EPOCHS3)

    # Тестирование модели методами keras
    testLoss, testAccuracy = model.evaluate(feature_test, label_test)
    print("Точность  = ", testAccuracy)


#EPOCHS = 3000  # Количество эпох для 1 задания

#print("******************************************HANDMADE******************************************")
#firstTask("data\\nn_0.csv")
#firstTask("data\\nn_1.csv")

#print("******************************************KERAS******************************************")
#firstTaskKeras("data\\nn_0.csv")
#firstTaskKeras("data\\nn_1.csv")


#EPOCHS2 = 60

#print("******************************************SECOND_TASK******************************************")
#secondTask("data\\nn_1.csv")

#EPOCHS3 = 21

#print("******************************************THIRD_TASK******************************************")
#thirdTask()
