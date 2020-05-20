from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def fifthTaskA(dataPath: str):
    print("ЗАДАНИЕ_5.1")
    # Чтение данных из файла в dataset
    dataset = pd.read_csv(dataPath, sep=",")

    # Удаление столбца Id
    dataset.drop(['Id'], axis='columns', inplace=True)

    # Разделение на признаки и метки
    feature = dataset.iloc[:, :-1]
    label = dataset['Type']

    # Деление данных на обучающие и тестовые
    trainSize = 0.7
    testSize = 1 - trainSize
    feature_train, feature_test, label_train, label_test = train_test_split(feature, label, train_size=trainSize, test_size=testSize)

    # Влияние максимальной глубины
    maxDepList = list()
    accuracyList = list()
    for maxDepth in range(1, 50):
        maxDepList.append(maxDepth)
        # Создание и обучение классификатора
        treeCl = DecisionTreeClassifier(max_depth=maxDepth)
        treeCl.fit(feature_train, label_train)

        # Тестирование классификатора
        treeResult = treeCl.predict(feature_test)

        # Оценка точности классификатора
        treeAccuracy = accuracy_score(treeResult, label_test)
        accuracyList.append(treeAccuracy)

    # Построение графика зависимости точности классификации от глубины дерева
    plt.plot(maxDepList, accuracyList)
    plt.xlabel("Max_Depth")
    plt.ylabel("Accuracy")
    plt.title("ЗАДАНИЕ 5.1 %s" %dataPath)
    plt.show()

    # Влияние критерия расщепления
    splitList = list()
    accuracyList = list()
    for split in('best', 'random'):
        splitList.append(split)
        # Создание и обучение классификатора
        treeCl = DecisionTreeClassifier(splitter=split)
        treeCl.fit(feature_train, label_train)

        # Тестирование классификатора
        treeResult = treeCl.predict(feature_test)

        # Оценка точности классификатора
        treeAccuracy = accuracy_score(treeResult, label_test)
        accuracyList.append(treeAccuracy)

    # Построение графика зависимости точности классификации от критерия расщепления
    plt.plot(splitList, accuracyList, 'o')
    plt.xlabel("Splitter")
    plt.ylabel("Accuracy")
    plt.title("ЗАДАНИЕ 5.1 %s" %dataPath)
    plt.show()

    # Визуализация дерева решений
    treeCl = DecisionTreeClassifier()
    treeCl.fit(feature_train, label_train)
    plot_tree(treeCl)
    plt.title("ЗАДАНИЕ 5.1 %s" %dataPath)
    plt.show()

def fifthTaskB(dataPath: str):
    print("ЗАДАНИЕ_5.2")
    # Чтение данных из файла в dataset
    dataset = pd.read_csv(dataPath, sep=",")

    # Переменные для выбора наиболее оптимального дерева
    minSplit = -10
    maxDepth = -10
    maxFeature = -10
    criter = -10
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    # Деление данных на обучающие и тестовые
    trainSize = 0.7
    testSize = 1 - trainSize
    feature_train, feature_test, label_train, label_test = train_test_split(dataset.iloc[:, :-1], dataset.iloc[:, -1],
                                                                            train_size=trainSize, test_size=testSize)

    crits = ('gini', 'entropy')
    Accs = []

    # Пройдёмся по 4 параметрам дерева(критей, макс. глубина, мин. кол-ва образцов для расщепления и кол-во признаков для расщепления
    # каждый раз прибавляя самый оптимальный параметр предыдущего прохода. Таким образом под конец получим самое оптимальное дерево
    for i in crits:
        DTC = DecisionTreeClassifier(criterion=i)
        DTC.fit(feature_train, label_train)
        Acc = accuracy_score(label_test, DTC.predict(feature_test))
        if Accs:
            if Acc > max(Accs):
                criter = i
        Accs.append(Acc)

    ax[0, 0].plot(crits, Accs, 'o')
    ax[0, 0].set(xlabel='Критерий', ylabel='Точность', title='Зависимость точности от типа расщепления')

    if criter == -10:
        criter = 'gini'

    # Зависимость точности от макс. глубины
    Accs.clear()
    depths = (1, 2, 3, 5, 7, 10, 15, 20, 50, 100, 300, 500, 1000)
    for i in depths:
        DTC = DecisionTreeClassifier(criterion=criter, max_depth=i)
        DTC.fit(feature_train, label_train)
        Acc = accuracy_score(label_test, DTC.predict(feature_test))
        if Accs:
            if Acc > max(Accs):
                maxDepth = i
        Accs.append(Acc)

    ax[0, 1].plot(depths, Accs, 'r-')
    ax[0, 1].set_xscale('log')
    ax[0, 1].set(xlabel='Макс. глубина', ylabel='Точность', title='Зависимость точности от глубины')

    # The minimum number of samples required to split an internal node
    splits = (0.1, 0.2, 0.5, 2, 5, 7, 10, 20, 50, 100, 200, 500, 1000)
    Accs.clear()
    for i in splits:
        DTC = DecisionTreeClassifier(criterion=criter, max_depth=maxDepth, min_samples_split=i)
        DTC.fit(feature_train, label_train)
        Acc = accuracy_score(label_test, DTC.predict(feature_test))
        if Accs:
            if Acc > max(Accs):
                minSplit = i
        Accs.append(Acc)

    ax[1, 0].plot(splits, Accs, 'r-')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set(xlabel='Мин. кол-во образцов для разделения', ylabel='Точность',
                 title='Зависимость точности от кол-ва образцов для разделения')

    # Кол-во признаков для разделении
    features = (0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6)
    Accs.clear()
    for i in features:
        DTC = DecisionTreeClassifier(criterion=criter, max_depth=maxDepth, min_samples_split=minSplit, max_features=i)
        DTC.fit(feature_train, label_train)
        Acc = accuracy_score(label_test, DTC.predict(feature_test))
        if Accs:
            if Acc > max(Accs):
                maxFeature = i
        Accs.append(Acc)

    ax[1, 1].plot(features, Accs, 'r-')
    ax[1, 1].set_xscale('log')
    ax[1, 1].set(xlabel='Кол-во признаков для разделении', ylabel='Точность',
                 title='Зависимость точности от кол-ва признаков для разделения')
    #plt.suptitle('Выбираем оптимальные параметры для дерева')
    #plt.suptitle("5.2")
    plt.show()

    # Вывод самого оптимального дерева и его точности
    DTC = DecisionTreeClassifier(criterion=criter, max_depth=maxDepth, min_samples_split=minSplit)
    DTC.fit(feature_train, label_train)
    DTCAcc = accuracy_score(label_test, DTC.predict(feature_test))
    print('Точность оптимального дерева:', DTCAcc)
    plot_tree(DTC)
    plt.title(
        'Оптимальное дерево. Точность = ' + DTCAcc.__str__() + '\nКритерий - ' + criter + '. Глубина - ' + maxDepth.__str__() + '.\nМин. кол-во образцов для расщепления - ' + minSplit.__str__())
    #plt.suptitle("5.2")
    plt.show()


#fifthTaskA("data\glass.csv")
fifthTaskB("data\spam7.csv")


