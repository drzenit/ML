from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def fourTask(dataPath: str):
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

    # Обучение классификатора
    treeCl = DecisionTreeClassifier()
    treeCl.fit(feature_train, label_train)

    # Тестирование классификатора
    treeResult = treeCl.predict(feature_test)

    # Оценка точности классификатора
    treeAccuracy = accuracy_score(treeResult, label_test)
    print(treeAccuracy)


    plot_tree(treeCl, filled=True, fontsize=8, max_depth=3)
    plt.show()


fourTask("data\glass.csv")


