import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def thirdTask(dataPath: str):
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

    # Вариация K
    kList = list()
    errList = list()
    for k in range(1, 150):
        # Создание и обучение экземпляра классификатора
        knnCl = KNeighborsClassifier(n_neighbors=k)
        knnCl.fit(feature_train, label_train)

        # Тестирование классификатора
        knnResult = knnCl.predict(feature_test)

        # Оценка точности классификатора
        knnAccuracy = accuracy_score(knnResult, label_test)
        knnError = 1 - knnAccuracy

        # Сбор параметров для графика
        kList.append(k)
        errList.append(knnError)

    # Построение графика зависимости ошибки классификации от количества ближайших соседей
    plt.plot(kList, errList)
    plt.xlabel("K")
    plt.ylabel("Error")
    plt.show()

    # Влияние метрики расстояния
    metricList = list()
    accuracyList = list()
    distanceMetric = ["euclidean", "manhattan", "chebyshev", "minkowski"]
    for metric in distanceMetric:
        # Создание и обучение экземпляра классификатора
        knnCl = KNeighborsClassifier(metric=metric)
        knnCl.fit(feature_train, label_train)

        # Тестирование классификатора
        knnResult = knnCl.predict(feature_test)

        # Оценка точности классификатора
        knnAccuracy = accuracy_score(knnResult, label_test)

        # Сбор параметров для графика
        metricList.append(metric)
        accuracyList.append(knnAccuracy)
    plt.plot(metricList, accuracyList, 'o')
    plt.xlabel("Metric")
    plt.ylabel("Accuracy")
    plt.show()

    # Определение типа стекла
    sample = {"RI": pd.Series([1.516]), "Na": pd.Series([11.7]), "Mg": pd.Series([1.01]), "Al": pd.Series([1.19]),
                "Si": pd.Series([72.59]), "K": pd.Series([0.43]), "Ca": pd.Series([11.44]),
                    "Ba": pd.Series([0.02]), "Fe": pd.Series([0.1])}
    sample = pd.DataFrame(sample)
    knnCl = KNeighborsClassifier()
    knnCl.fit(feature_train, label_train)
    sampleResult = knnCl.predict(sample)
    print(sampleResult)


thirdTask("data\glass.csv")
