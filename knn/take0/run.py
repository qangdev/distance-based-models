import csv
import time
import pandas as pd

from random import sample, randrange
from knn.take0.knn_classifier import KNN
from knn.take0.car import Car

# Step 1: Loading and prepare data
print("[Started]")
data_train, data_test = [], []
with open("./data/car.data", "r") as car_data, open("./data/car.csv", "w+") as car_csv:
    lines = list(csv.reader(car_data, delimiter=","))
    percent = round(len(lines) * 0.2)
    rows = []
    labels = []
    for line in lines:
        row = Car()  # Make Row instance and clear data
        row.buying = line[0]
        row.maint = line[1]
        row.doors = line[2]
        row.persons = line[3]
        row.lugboot = line[4]
        row.safety = line[5]
        row.klass = line[6]
        rows.append(row)
        car_csv.write(",".join([str(row.buying),
                                str(row.maint),
                                str(row.doors),
                                str(row.persons),
                                str(row.lugboot),
                                str(row.safety)])+"\n")
    # TODO: How to get smaple better???
    data_train = sample(rows.copy(), percent)  # rows.copy()[:percent]
    lables_test = []
    # t1 = [L.pop(random.randrange(len(L))) for _ in xrange(2)]
    rows_2 = rows.copy()
    for o in [rows_2.pop(randrange(len(rows_2))) for _ in range(0, len(rows) - len(data_train))]:
        row = o.clone()
        lables_test.append(o.klass)
        data_test.append(row)

# Step 2: Training KNN and Labling data
knn = KNN(k=5)
knn.training(data_train)
start_time = time.time()
result = knn.predict_many(data_test)
result = [o.klass for o in result]
print(":> %s - [DONE]" % (time.time() - start_time))

predicted = pd.Series(result, name="Predicted")
actual = pd.Series(lables_test, name="Actual")
df_confusion = pd.crosstab(predicted, actual)
print(df_confusion)

# Step 3: Validate accuracy
match = 0
not_match = 0
for o in zip(result, lables_test):
    if o[0] == o[1]:
        match += 1
    else:
        not_match += 1

print("[TRAIN] %s vs [TEST]: %s" %(len(data_train), len(data_test)))
print("[MATCH] %s" % match)
print("[NOT-MATCH] %s" % not_match)
print("[TOTAL] %s - %s" % (match/len(result), len(data_train) + len(data_test)))