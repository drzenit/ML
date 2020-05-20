import pandas as pd
from sklearn.metrics import plot_confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np


# Визуализация разбиения пространства
def visualize(feature_train, label_test, svc, ax: None = None):
    X0, X1 = feature_train.iloc[:, 0], feature_train.iloc[:, 1]
    xx, yy = np.meshgrid(np.arange(X0.min() - 0.2, X0.max() + 0.2, 0.02),
                         np.arange(X1.min() - 0.2, X1.max() + 0.2, 0.02))
    z = svc.predict((np.c_[xx.ravel(), yy.ravel()]))
    z = z.reshape(xx.shape)
    if ax is None:
        plt.contourf(xx, yy, z)
        plt.scatter(X0, X1, c=label_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    else:
        ax.contourf(xx, yy, z)
        ax.scatter(X0, X1, c=label_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

# Чтение данных из файла svmdata_N.txt и svmdata_N_test.txt
def readFile(dataPath: str):
    # Чтение данных из файла в dataset
    dataset = pd.read_csv(dataPath, sep='\t', header=0)

    # Разделение на признаки и метки (для удобства меняем red - 0, green - 1)
    dataset = dataset.replace({'red': 0, 'green': 1})
    feature = dataset.iloc[:, :-1]
    label = dataset.iloc[:, -1]

    return feature, label


def fourthTaskA(dataPathTest: str, dataPathTrain: str):
    print("ЗАДАНИЕ_4.1")
    # Деление данных на обучающие и тестовые
    C = 1.0
    feature_train, label_train = readFile(dataPathTrain)
    feature_test, label_test = readFile(dataPathTest)

    # Создание и обучение экземпляра классификатора
    svc = SVC(kernel='linear', C=C)
    svc.fit(feature_train, label_train)
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Визуализация разбиения пространства признаков
    visualize(feature_train, label_test, svc, ax[0, 0])
    ax[0, 0].set(xlabel='X1', ylabel='X2', title='SVC linear \n C = 1.0')

    # Графическое представление матрицы ошибок
    plot_confusion_matrix(svc, feature_test, label_test, ax=ax[0, 1])
    ax[0, 1].set_title('Матрица ошибок на тестовых данных')
    plot_confusion_matrix(svc, feature_train, label_train, ax=ax[1, 1])
    ax[1, 1].set_title('Матрица ошибок на обучающих данных')
    plt.suptitle("4.2")
    plt.show()
    print('Кол-во опорных векторов: ', svc.n_support_)

def fourthTaskB(dataPathTest: str, dataPathTrain: str):
    print("ЗАДАНИЕ_4.2")
    # Деление данных на обучающие и тестовые
    feature_train, label_train = readFile(dataPathTrain)
    feature_test, label_test = readFile(dataPathTest)

    # Подборка штрафного параметра для нулевой ошибки (точность = 1) для обучающей выборки
    for i in range(1, 1000, 1):
        svc = SVC(kernel='linear', C=i)
        svc.fit(feature_train, label_train)
        if accuracy_score(label_train, svc.predict(feature_train)) == 1.0:
            print('При штрафном параметре равном ', i, ':')
            print('Точнсть для обучающей выборки = ', accuracy_score(label_train, svc.predict(feature_train)))
            print('Точность для тестовой выборки = ', accuracy_score(label_test, svc.predict(feature_test)))
            break

    # Подборка штрафного параметра для нулевой ошибки (точность = 1) для тестовой выборки
    for i in range(1, 1000, 1):
        svc = SVC(kernel='linear', C=i)
        svc.fit(feature_train, label_train)
        if accuracy_score(label_test, svc.predict(feature_test)) == 1.0:
            print('При штрафном параметре равном ', i, ':')
            print('Точнсть для обучающей выборки = ', accuracy_score(label_train, svc.predict(feature_train)))
            print('Точность для тестовой выборки = ', accuracy_score(label_test, svc.predict(feature_test)))
            break

def fourthTaskC(dataPathTest: str, dataPathTrain: str):
    print("ЗАДАНИЕ_4.3")
    # Деление данных на обучающие и тестовые
    feature_train, label_train = readFile(dataPathTrain)
    feature_test, label_test = readFile(dataPathTest)

    # Ядра
    kernels = ('linear', 'sigmoid', 'rbf')
    x, y = 0, 0
    fig, ax = plt.subplots(3, figsize=(10, 10))

    # Визуализируем ядра из списка kernels
    for i in kernels:
        svc = SVC(kernel=i)
        svc.fit(feature_train, label_train)
        visualize(feature_train, label_test, svc, ax[x])
        str = 'SVC классификатор с ' + i + ' ядром \n Точность = ' + accuracy_score(label_test,
                                                                                    svc.predict(feature_test)).__str__()
        ax[x].set(title=str, xlabel='X1', ylabel='X2')
        ax[x].set_xticks([])
        ax[x].set_yticks([])
        x += 1

    # Отрисовка
    #plt.suptitle("4.3")
    plt.show()
    x, y = 0, 0
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))

    # Визуализируем полиноминальное ядро степеней 1-5
    for i in range(1, 6, 1):
        svc = SVC(kernel='poly', degree=i)
        svc.fit(feature_train, label_train)
        visualize(feature_train, label_test, svc, ax[x, y])
        str = 'SVC с полиноминальным ядром и степенью ' + i.__str__() + '\n' + accuracy_score(label_test,
                                                                                                            svc.predict(
                                                                                                                feature_test)).__str__()
        ax[x, y].set_xticks([])
        ax[x, y].set_yticks([])
        ax[x, y].set(title=str, xlabel='X1', ylabel='X2')
        x += 1
        if x % 3 == 0:
            y += 1
            x = 0
    #plt.suptitle("4.3")
    plt.show()


def fourthTaskD(dataPathTest: str, dataPathTrain: str):
    print("ЗАДАНИЕ_4.4")
    # Деление данных на обучающие и тестовые
    feature_train, label_train = readFile(dataPathTrain)
    feature_test, label_test = readFile(dataPathTest)

    # Ядра
    kernels = ('sigmoid', 'rbf')
    x, y = 0, 0
    fig, ax = plt.subplots(2, figsize=(10, 10))

    # Визуализируем ядра из списка kernels
    for i in kernels:
        svc = SVC(kernel=i)
        svc.fit(feature_train, label_train)
        visualize(feature_train, label_test, svc, ax[x])
        str = 'SVC с ' + i + ' ядром \n Точность = ' + accuracy_score(label_test,
                                                                                    svc.predict(feature_test)).__str__()
        ax[x].set_xticks([])
        ax[x].set_yticks([])
        ax[x].set(title=str, xlabel='X1', ylabel='X2')
        x += 1

    # Отрисовка
    #plt.suptitle("4.4")
    plt.show()
    x, y = 0, 0
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))

    # Визуализируем полиноминальное ядро степеней 1-5
    for i in range(1, 6, 1):
        svc = SVC(kernel='poly', degree=i)
        svc.fit(feature_train, label_train)
        visualize(feature_train, label_test, svc, ax[x, y])
        str = 'SVC с полиноминальным ядром и степенью ' + i.__str__() + '\n' + accuracy_score(label_test,
                                                                                                            svc.predict(
                                                                                                                feature_test)).__str__()
        ax[x, y].set_xticks([])
        ax[x, y].set_yticks([])
        ax[x, y].set(title=str, xlabel='X1', ylabel='X2')
        x += 1
        if x % 3 == 0:
            y += 1
            x = 0
    #plt.suptitle("4.4")
    plt.show()


def fourthTaskE(dataPathTest: str, dataPathTrain: str):
    print("ЗАДАНИЕ_4.5")
    # Деление данных на обучающие и тестовые
    feature_train, label_train = readFile(dataPathTrain)
    feature_test, label_test = readFile(dataPathTest)

    # Ядра
    kernels = ('sigmoid', 'rbf')
    gammas = (0.1, 1, 5, 10)
    x, y = 0, 0
    fig, ax = plt.subplots(2, 4, figsize=(15, 15))

    # Визуализируем ядра из списка kernels с gamma
    for gamma in gammas:
        x = 0
        for i in kernels:
            svc = SVC(kernel=i, gamma=gamma)
            svc.fit(feature_train, label_train)
            visualize(feature_train, label_test, svc, ax[x, y])
            str = 'SVC классификатор с ' + i + ' ядром \n Точность = ' + (
                    "%.3f" % accuracy_score(label_test, svc.predict(
                feature_test))).__str__() + '. Гамма = ' + gamma.__str__()
            ax[x, y].set_xticks([])
            ax[x, y].set_yticks([])
            ax[x, y].set(title=str, xlabel='X1', ylabel='X2')
            x += 1
        y += 1

    # Отрисовка
    #plt.suptitle("4.5")
    plt.show()

    # Визуализируем полиномиальное ядро со степенями 1-5 и gamma
    for i in range(1, 6, 1):
        fig, ax = plt.subplots(4, figsize=(15, 15))
        x = 0
        for gamma in gammas:
            svc = SVC(kernel='poly', degree=i, gamma=gamma)
            svc.fit(feature_train, label_train)
            visualize(feature_train, label_test, svc, ax[x])
            str = 'SVC классификатор с полиноминальным ядром и степенью ' + i.__str__() + '\n' + (
                    "%.3f" % accuracy_score(
                label_test, svc.predict(feature_test))).__str__() + '. Гамма = ' + gamma.__str__()
            ax[x].set(title=str, xlabel='X1', ylabel='X2')
            ax[x].set_xticks([])
            ax[x].set_yticks([])
            x += 1

        # Отрисовка
        #plt.suptitle('4.5')
        plt.show()

    svc = SVC(kernel='rbf', gamma=1000)
    svc.fit(feature_train, label_train)
    visualize(feature_train, label_test, svc)
    plt.title('Демонстрация переобучения на гауссовом ядре с гаммой = 1000. \n Точность:' + "%.3f" %accuracy_score(label_test, svc.predict(feature_test)))
    #plt.suptitle("4.5")
    plt.show()

fourthTaskE("data\svmdata_b.txt", "data\svmdata_b_test.txt")