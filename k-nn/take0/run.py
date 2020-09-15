import csv
import time

from random import sample
from take0.knn import KNN
from take0.car import Car

# Step 1: Loading and prepare data
data_train, data_test = [], []
with open("./data/car.data", "r") as car_data:
    lines = list(csv.reader(car_data, delimiter=","))
    # lines = sample(lines, k=len(lines))
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
    data_train = sample(rows.copy(), percent) #rows.copy()[:percent]
    lables_test = []
    for o in rows.copy():
        row = o.clone()
        lables_test.append(o.klass)
        data_test.append(row)

# Step 2: Training KNN and Labling data
knn = KNN(k=5)
knn.training(data_train)
start_time = time.time()
result = knn.predict_many(data_test)
print(":> %s - Done" % (time.time() - start_time))
print("%s vs %s vs %s" % (len(result), len(data_test), len(lables_test)))
# Step 3: Validate accuracy
match = 0
not_match = 0
for o in zip(result, lables_test):
    if o[0].klass == o[1]:
        match += 1
    else:
        not_match += 1
print("[MATCH] %s" % match)
print("[NOT-MATCH] %s" % not_match)
print("[TOTAL] %s - %s" % (match/len(result), len(result)))