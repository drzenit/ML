from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def fifthTaskA(dataPath: str):
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
    plt.show()

    # Визуализация дерева решений
    treeCl = DecisionTreeClassifier()
    treeCl.fit(feature_train, label_train)
    plot_tree(treeCl, filled=True, fontsize=8, max_depth=3)
    plt.show()

def fifthTaskB(dataPath: str):
    # Чтение данных из файла в dataset
    dataset = pd.read_csv(dataPath, sep=",")

    # Разделение на признаки и метки
    feature = dataset.iloc[:, :-1]
    label = dataset['yesno']
    print(feature)
    print(label)

    # Деление данных на обучающие и тестовые
    trainSize = 0.7
    testSize = 1 - trainSize
    feature_train, feature_test, label_train, label_test = train_test_split(feature, label, train_size=trainSize,
                                                                            test_size=testSize)

    # Создание и обучение классификатора
    treeCl = DecisionTreeClassifier()
    treeCl.fit(feature_train, label_train)

    # Тестирование классификатора
    treeResult = treeCl.predict(feature_test)

    # Оценка точности классификатора
    treeAccuracy = accuracy_score(treeResult, label_test)
    print(treeAccuracy)

    # Визуализация дерева решений
    treeCl = DecisionTreeClassifier()
    treeCl.fit(feature_train, label_train)
    plot_tree(treeCl, filled=True, fontsize=8, max_depth=5)
    plt.show()


#fifthTaskA("data\glass.csv")
fifthTaskB("data\spam7.csv")


