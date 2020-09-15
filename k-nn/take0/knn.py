from .utils import yieldloop


class KNN:
    K = 3

    def __init__(self, k):
        self.k = k
        self.labeled_data = list()


    def training(self, data):
        self.labeled_data = data


    def euclidean_distance(self, point_x, point_y):
        from math import sqrt
        distances = sum([o**2 for o in point_x - point_y])
        return sqrt(distances)


    def voting(self, records, column):
        counting = {}
        for o in records:
            value = getattr(o, column)
            if value not in counting:
                counting[value] = 0
            counting[value] += 1
        counting = sorted(counting.items(), key=lambda o: o[1])
        return next(iter(counting))[0]


    def find_k_nearest(self, point):
        records = {}
        for i, o in enumerate(yieldloop(self.labeled_data)):
            distance = self.euclidean_distance(point, o)
            records[i] = distance
        # Sorted
        records = sorted(records.items(), key=lambda o: o[1])
        records = records[:self.k]
        records = [self.labeled_data[o[0]] for o in records]
        return records


    def predict_one(self, record):
        k_near_points = self.find_k_nearest(record)
        winning_value = self.voting(k_near_points, "klass")
        record.klass = winning_value
        self.labeled_data.append(record)
        return record


    def predict_many(self, data):
        return [self.predict_one(o) for o in yieldloop(data)]
