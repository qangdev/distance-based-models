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
                 buying: str = None,
                 maint: str = None,
                 doors: str = None,
                 persons: str = None,
                 lugboot: str = None,
                 safety: str = None,
                 klass: str = None):
        self.buying = ["low", "med", "high", "vhigh"].index(buying)
        self.maint = ["low", "med", "high", "vhigh"].index(maint)
        self.doors = ["2", "3", "4", "5more"].index(doors)
        self.persons = ["2", "4", "6", "more"].index(persons)
        self.lugboot = ["tiny", "small", "med", "big"].index(lugboot)
        self.safety = ["vlow", "low", "med", "high"].index(safety)
        self.klass = ["unacc", "acc", "good", "vgood"].index(klass)


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
            ((self.buying - other.buying)**2),
            ((self.maint - other.maint)**2),
            ((self.doors - other.doors)**2),
            ((self.persons - other.persons)**2),
            ((self.lugboot - other.lugboot)**2),
            ((self.safety - other.safety)**2)
        )
