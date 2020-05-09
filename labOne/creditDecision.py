import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


def knnSystem(dataPathTrain: str, dataPathTest: str):
    # Чтение данных из файла в dataset
    datasetTrain = pd.read_table(dataPathTrain)
    datasetTest = pd.read_table(dataPathTest)

    # Разделение на признаки и метки
    label_train = datasetTrain['SeriousDlqin2yrs']
    datasetTrain.drop(['SeriousDlqin2yrs'], axis='columns', inplace=True)
    feature_train = datasetTrain
    label_test = datasetTest['SeriousDlqin2yrs']
    datasetTest.drop(['SeriousDlqin2yrs'], axis='columns', inplace=True)
    feature_test = datasetTest

    # Создание и обучение экземпляра классификатора
    knnCl = KNeighborsClassifier()
    knnCl.fit(feature_train, label_train)

    # Тестирование классификатора
    knnResult = knnCl.predict(feature_test)

    # Оценка точности классификатора
    knnAccuracy = accuracy_score(knnResult, label_test)
    print(knnAccuracy)

    # Матрица ошибок
    knnConfMat = confusion_matrix(knnResult, label_test)
    print(knnConfMat)

    # Получение вероятностей классификации
    knnResultProba = knnCl.predict_proba(feature_test)
    knnResultProba = pd.DataFrame(knnResultProba)

    proba = knnResultProba[1]

    # Построение ROC-кривой и получение AUC (площади под кривой)
    fpr, tpr, thresholds = roc_curve(label_test, proba)
    auc = roc_auc_score(label_test, proba)
    print(auc)
    plt.plot(fpr, tpr, 'g')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()

    # Получение PR-кривой
    precision, recall, thresholds = precision_recall_curve(label_test, proba)
    plt.plot(recall, precision, 'g')
    plt.plot([0, 1], [0.5, 0.5], 'k--')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

def decisionTree(dataPathTrain: str, dataPathTest: str):
    # Чтение данных из файла в dataset
    datasetTrain = pd.read_table(dataPathTrain)
    datasetTest = pd.read_table(dataPathTest)

    # Разделение на признаки и метки
    label_train = datasetTrain['SeriousDlqin2yrs']
    datasetTrain.drop(['SeriousDlqin2yrs'], axis='columns', inplace=True)
    feature_train = datasetTrain
    label_test = datasetTest['SeriousDlqin2yrs']
    datasetTest.drop(['SeriousDlqin2yrs'], axis='columns', inplace=True)
    feature_test = datasetTest

    # Создание и обучение экземпляра классификатора
    treeCl = DecisionTreeClassifier()
    treeCl.fit(feature_train, label_train)

    # Тестирование классификатора
    treeResult = treeCl.predict(feature_test)

    # Оценка точности классификатора
    treeAccuracy = accuracy_score(treeResult, label_test)
    print(treeAccuracy)

    # Матрица ошибок
    treeConfMat = confusion_matrix(treeResult, label_test)
    print(treeConfMat)

    # Получение вероятностей классификации
    treeResultProba = treeCl.predict_proba(feature_test)
    treeResultProba = pd.DataFrame(treeResultProba)

    proba = treeResultProba[1]

    # Построение ROC-кривой и получение AUC (площади под кривой)
    fpr, tpr, thresholds = roc_curve(label_test, proba)
    auc = roc_auc_score(label_test, proba)
    print(auc)
    plt.plot(fpr, tpr, 'g')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()

    # Получение PR-кривой
    precision, recall, thresholds = precision_recall_curve(label_test, proba)
    plt.plot(recall, precision, 'g')
    plt.plot([0, 1], [0.5, 0.5], 'k--')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()



knnSystem("data\\bank_scoring_train.csv", "data\\bank_scoring_test.csv")
decisionTree("data\\bank_scoring_train.csv", "data\\bank_scoring_test.csv")
