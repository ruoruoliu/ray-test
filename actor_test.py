import ray

@ray.remote
class Counter(object):

    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1

    def get_value(self):
        return self.value


counter = Counter.remote()

[counter.increment.remote() for _ in range(5)]

assert ray.get(counter.get_value.remote()) == 5
