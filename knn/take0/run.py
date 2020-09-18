import csv
import time
import pandas as pd

from random import sample
from knn.take0.knn_classifier import KNN
from knn.take0.car import Car

# Step 1: Loading and prepare data
print("[STARTED]")
data_train, data_test = [], []
with open("./data/car.data", "r") as car_data, open("./data/car.csv", "w+") as car_csv:
    lines = list(csv.reader(car_data, delimiter=","))
    break_point = round(len(lines) * 0.2)

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
        car_csv.write(",".join([str(row.buying),
                                str(row.maint),
                                str(row.doors),
                                str(row.persons),
                                str(row.lugboot),
                                str(row.safety),
                                str(row.klass)])+"\n")
    
    data = sample(data, k=len(data))
    data_train = data[:break_point]
    labels_test = []
    for o in data[break_point:]:
        labels_test.append(o.klass)
        o.klass = None
        data_test.append(o)

# Step 2: Training KNN and Labling data
knn = KNN(k=5)
knn.training(data_train)
start_time = time.time()
result = knn.predict_many(data_test)
result = [o.klass for o in result]
print(":> %s - [DONE]" % (time.time() - start_time))

predicted = pd.Series(result, name="Predicted")
actual = pd.Series(labels_test, name="Actual")
df_confusion = pd.crosstab(predicted, actual)

print(df_confusion)

# Step 3: Validate accuracy
match = 0
not_match = 0
for o in zip(result, labels_test):
    if o[0] == o[1]:
        match += 1
    else:
        not_match += 1

print("[TRAIN] %s vs [TEST]: %s" %(len(data_train), len(data_test)))
print("[MATCH] %s vs [NOT-MATCH] %s" % (match, not_match))
print("[TOTAL] %s - %s" % (match/len(result), len(data_train) + len(data_test)))