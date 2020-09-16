from .utils import yieldloop

# Convert text-based value to number-based value
#   buying: v-high, high, med, low
#   maint: v-high, high, med, low
#   doors: 2, 3, 4, 5-more
#   persons: 2, 4, more
#   lug_boot: small, med, big
#   safety: low, med, high
#   class: unacc, acc, good, vgood

class Car:

    def __init__(self,
                 buying: int = None,
                 maint: int = None,
                 doors: int = None,
                 persons: int = None,
                 lugboot: int = None,
                 safety: int = None,
                 klass: int = None):
        self.__buying = buying
        self.__maint = maint
        self.__doors = doors
        self.__persons = persons
        self.__lugboot = lugboot
        self.__safety = safety
        self.__klass = klass


    def clone(self):
        return Car(self.buying,
                   self.maint,
                   self.doors,
                   self.persons,
                   self.lugboot,
                   self.safety,
                   self.klass)

    def __sub__(self, other):
        return (
            (self.buying - other.buying),
            (self.maint - other.maint),
            (self.doors - other.doors),
            (self.persons - other.persons),
            (self.lugboot - other.lugboot),
            (self.safety - other.safety)
        )


    @property
    def buying(self):
        return self.__buying

    @buying.setter
    def buying(self, value):
        self.__buying = ["low", "med", "high", "vhigh"].index(value) + 1

    @property
    def maint(self):
        return self.__maint

    @maint.setter
    def maint(self, value):
        self.__maint = ["low", "med", "high", "vhigh"].index(value) + 1

    @property
    def doors(self):
        return self.__doors

    @doors.setter
    def doors(self, value):
        self.__doors = ["2", "3", "4", "5more"].index(value) + 1

    @property
    def persons(self):
        return self.__persons

    @persons.setter
    def persons(self, value):
        self.__persons = ["2", "4", "6", "more"].index(value) + 1

    @property
    def lugboot(self):
        return self.__lugboot

    @lugboot.setter
    def lugboot(self, value):
        self.__lugboot = ["tiny", "small", "med", "big"].index(value) + 1

    @property
    def safety(self):
        return self.__safety

    @safety.setter
    def safety(self, value):
        self.__safety = ["vlow", "low", "med", "high"].index(value) + 1

    @property
    def klass(self):
        return self.__klass

    @klass.setter
    def klass(self, value):
        if type(value) is str:
            self.__klass = ["unacc", "acc", "good", "vgood"].index(value) + 1
        else:
            self.__klass = value