from knn.car_evaluation.utils import yieldloop
from math import sqrt


class KNN:
    K = 3


    def __init__(self, k):
        self.k = k
        self.labeled_data = list()


    def training(self, data):
        self.labeled_data = data


    def euclidean_distance(self, point_x, point_y):
        # point x: e.g [0,1,2,3,4,5]
        # point y: e.g [0,1,2,3,4,5]
        # How `point_x - point_y` is defined in Car class (`__sub__` magic method)
        return sqrt(sum(point_x - point_y))


    def voting(self, records, column):
        counting = {}
        for o in records:
            value = getattr(o, column) # Access to the given column
            if value not in counting:
                counting[value] = 0
            counting[value] += 1
        counting = sorted(counting.items(), key=lambda item: item[1])
        winner = counting.pop()
        return winner[0]


    def find_k_nearest(self, point):
        records = {}
        for i, o in enumerate(yieldloop(self.labeled_data)):
            # Keep the index of train data set to get the label later
            records[i] = self.euclidean_distance(point, o)

        # Sort the distances ascending
        records = sorted(records.items(), key=lambda o: o[1])
        records = records[:self.k]
        records = [self.labeled_data[o[0]] for o in records]
        return records


    def predict_one(self, record):
        # Calculate the test data distance foreach data in each class using the Euclidean distance
        k_near_points = self.find_k_nearest(record)
        # Sort the distance between data from the smallest to the largest according to number of k
        winning_value = self.voting(k_near_points, "klass")
        record.klass = winning_value
        return record


    def predict_many(self, data):
        return [self.predict_one(o) for o in yieldloop(data)]
