import csv
import time
import pandas as pd
import numpy as np


from sklearn.model_selection import KFold
from knn.car_evaluation.knn_classifier import KNN
from knn.car_evaluation.car import Car


def prepare_data(data, kfold):
    for train_index, test_index in kfold.split(data):
        data_train, data_test, label_test = [], [], []

        for index in list(train_index):
            data_train.append(data[index])

        for index in list(test_index):
            label_test.append(data[index].klass)
            data[index].klass = None
            data_test.append(data[index])
        yield [data_train, data_test, label_test]


def run_knn(k, data_train, data_test):
    knn = KNN(k=k)
    knn.training(data_train)
    start_time = time.time()
    result = knn.predict_many(data_test)
    result = [o.klass for o in result]
    print(":> %s - [DONE]" % (time.time() - start_time))

    return result



if __name__ == '__main__':
    with open("./data/car.data", "r") as car_data:
        lines = list(csv.reader(car_data, delimiter=","))
        # Step 1: Loading and prepare data
        data = []
        for line in lines:
            # Make Row instance and clean data
            row = Car(buying=line[0],
                      maint=line[1],
                      doors=line[2],
                      persons=line[3],
                      lugboot=line[4],
                      safety=line[5],
                      klass=line[6])
            data.append(row)

        kf3 = KFold(n_splits=3, shuffle=False)
        kf5 = KFold(n_splits=5, shuffle=False)

        skf2 = KFold(n_splits=2, shuffle=True)
        skf3 = KFold(n_splits=3, shuffle=True)
        skf4 = KFold(n_splits=4, shuffle=True)
        skf5 = KFold(n_splits=5, shuffle=True)

        for data_train, data_test, label_test in prepare_data(data=data,
                                                              kfold=skf5):
            # Step 2: Training KNN and Labeling test data
            result = run_knn(k=7,
                             data_test=data_test,
                             data_train=data_train)

            # Step 3: Validate accuracy and make Confusion Matrix
            predicted = pd.Series(result, name="Predicted")
            actual = pd.Series(label_test, name="Actual")
            df_confusion = pd.crosstab(predicted, actual)
            print(df_confusion)

            match = 0
            not_match = 0
            for o in zip(result, label_test):
                if o[0] == o[1]:
                    match += 1
                else:
                    not_match += 1
            print("[TRAIN] %s vs [TEST]: %s" % (len(data_train), len(data_test)))
            print("[MATCH] %s vs [NOT-MATCH] % s" % (match, not_match))
            print("[TOTAL] %s - %s" % (match/len(result), len(result)))
            print("#"*50)
