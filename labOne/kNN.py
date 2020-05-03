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
        print(knnError)

        # Сбор параметров для графика
        kList.append(k)
        errList.append(knnError)

    # Построение графика зависимости ошибки классификации от количества ближайших соседей
    plt.plot(kList, errList)
    plt.xlabel("K")
    plt.ylabel("Error")
    plt.show()


thirdTask("data\glass.csv")
