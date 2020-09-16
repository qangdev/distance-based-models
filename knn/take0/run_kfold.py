import csv
import time
import pandas as pd
import numpy as np

from random import sample
from sklearn.model_selection import KFold
from knn.take0.knn_classifier import KNN
from knn.take0.car import Car


def prepare_data(kfold, lines):
    # Step 1: Loading and prepare data
    data = []
    for train_index, test_index in kfold.split(lines):
        data_train, data_test, label_test = [], [], []
        for index in list(train_index):
            line = lines[index]
            row = Car()  # Make Row instance and clear data
            row.buying = line[0]
            row.maint = line[1]
            row.doors = line[2]
            row.persons = line[3]
            row.lugboot = line[4]
            row.safety = line[5]
            row.klass = line[6]
            data_train.append(row)
        for index in list(test_index):
            line = lines[index]
            row = Car()  # Make Row instance and clear data
            row.buying = line[0]
            row.maint = line[1]
            row.doors = line[2]
            row.persons = line[3]
            row.lugboot = line[4]
            row.safety = line[5]
            data_test.append(row)
            label_test.append(line[6])
        data.append([data_train, data_test, label_test])
    return data


def run_knn(k, data_train, data_test):
    # Step 2: Training KNN and Labling data
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

        kf3 = KFold(n_splits=3, shuffle=False)
        kf5 = KFold(n_splits=5, shuffle=False)

        skf2 = KFold(n_splits=2, shuffle=True)
        skf3 = KFold(n_splits=3, shuffle=True)
        skf4 = KFold(n_splits=4, shuffle=True)
        skf5 = KFold(n_splits=5, shuffle=True)

        for data_train, data_test, label_test in prepare_data(kfold=skf4, lines=lines):
            result = run_knn(k=5,
                             data_test=data_test,
                             data_train=data_train)
            predicted = pd.Series(result, name="Predicted")
            actual = pd.Series(label_test, name="Actual")
            df_confusion = pd.crosstab(predicted, actual)
            print(df_confusion)

            # Step 3: Validate accuracy
            match = 0
            not_match = 0
            for o in zip(result, label_test):
                if o[0] == o[1]:
                    match += 1
                else:
                    not_match += 1
            print("[TRAIN] %s vs [TEST]: %s" % (len(data_train), len(data_test)))
            print("[MATCH] %s" % match)
            print("[NOT-MATCH] %s" % not_match)
            print("[TOTAL] %s - %s" % (match/len(result), len(result)))
            print("#"*50)
