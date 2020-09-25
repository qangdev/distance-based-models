import unittest

import csv
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from random import sample
from knn.car_evaluation.knn_classifier import KNN
from knn.car_evaluation.car import Car



class TestStringMethods(unittest.TestCase):


    def setUp(self):
        self.predictions = []
        self.data_train = []
        self.data_test = []
        self.labels_test = []
        with open("./data/car.data", "r") as dataset, open("./data/car.csv", "w+") as car_csv:
            lines = list(csv.reader(dataset, delimiter=","))
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
                # Save converted data to a file for checking Correlation
                car_csv.write(",".join([str(row.buying),
                                        str(row.maint),
                                        str(row.doors),
                                        str(row.persons),
                                        str(row.lugboot),
                                        str(row.safety),
                                        str(row.klass)]) + "\n")
            # Shuffle dataset
            data = sample(data, k=len(data))
            # Find the break point to split 80% train - 20% test:
            break_point = round(len(lines) * 0.2)
            # Split data
            self.data_train = data[:break_point]
            for o in data[break_point:]:
                self.labels_test.append(o.klass)
                o.klass = None
                self.data_test.append(o)


    def tearDown(self):
        # Validate accuracy, make Confusion Matrix
        predicted = pd.Series(self.predictions, name="Predicted")
        actual = pd.Series(self.labels_test, name="Actual")
        cf_matrix = pd.crosstab(predicted, actual)
        print(cf_matrix)

        match = 0
        not_match = 0
        for o in zip(self.predictions, self.labels_test):
            if o[0] == o[1]:
                match += 1
            else:
                not_match += 1

        accuracy = float(round((match / len(self.predictions)), 2))
        f_negative = cf_matrix[0][1] + cf_matrix[0][2] + cf_matrix[0][3] + cf_matrix[1][2] + cf_matrix[1][3] + cf_matrix[2][3]
        f_positive = cf_matrix[1][0] + cf_matrix[2][0] + cf_matrix[2][1] + cf_matrix[3][0] + cf_matrix[3][1] + cf_matrix[3][2]
        print("[TRAIN] %s vs [TEST]: %s" % (len(self.data_train), len(self.data_test)))
        print("[MATCH] %s vs [NOT-MATCH] %s" % (match, not_match))
        print("[ACCURACY] %s" % accuracy)
        print("[FALSE-NEGATIVE] %s vs [FALSE-POSITIVE] %s" % (f_negative, f_positive))
        print("\n")

    def test_k_3_10_dataset_8train_2test(self):
        stacked_k = []
        stacked_acc = []
        stacked_fn = []
        stacked_fp = []
        for k in range(3, 11):
            print("[K] = %s" % k)
            start_time = time.time()
            knn = KNN(k=k)
            knn.training(self.data_train)
            self.predictions = knn.predict_many(self.data_test)
            self.predictions = [o.klass for o in self.predictions]
            print("[DONE]:> %s\n" % (time.time() - start_time))

            # Validate accuracy, make Confusion Matrix
            predicted = pd.Series(self.predictions, name="Predicted")
            actual = pd.Series(self.labels_test, name="Actual")
            cf_matrix = pd.crosstab(predicted, actual)
            print(cf_matrix)

            match = 0
            not_match = 0
            for o in zip(self.predictions, self.labels_test):
                if o[0] == o[1]:
                    match += 1
                else:
                    not_match += 1

            accuracy = float(round((match / len(self.predictions)), 2)) * 100
            f_negative = float(round(((cf_matrix[0][1] + cf_matrix[0][2] + cf_matrix[0][3] + cf_matrix[1][2] + cf_matrix[1][3] + \
                         cf_matrix[2][3]) / len(self.predictions)), 2)) * 100
            f_positive = float(round(((cf_matrix[1][0] + cf_matrix[2][0] + cf_matrix[2][1] + cf_matrix[3][0] + cf_matrix[3][1] + \
                         cf_matrix[3][2]) / len(self.predictions)), 2)) * 100
            stacked_k.append(k)
            stacked_acc.append(accuracy)
            stacked_fn.append(f_negative)
            stacked_fp.append(f_positive)

            print("[ACCURACY] %s%%" % accuracy)
            print("[FALSE-NEGATIVE] %s%% vs [FALSE-POSITIVE] %s%%" % (f_negative, f_positive))
            print("\n")

        x = np.arange(len(stacked_k))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects_1 = ax.bar(x - width / 2, stacked_acc, width, label="Accuracy")
        rects_2 = ax.bar(x + width / 2, stacked_fn, width, label='False Negative')
        rects_3 = ax.bar(x + width / 2, stacked_fp, width, label='False Positive')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        # ax.set_ylabel('')
        ax.set_title('Result')
        ax.set_xticks(x)
        ax.set_xticklabels(stacked_k)
        ax.legend()
        fig.tight_layout()
        # plt.show()


if __name__ == '__main__':
    unittest.main()