import ray


@ray.remote
def square(x):
    return x * x

obj1 = square.remote(2)
obj2 = square.remote(obj1)
assert ray.get(obj2) == 16


@ray.remote
class Counter(object):

    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1

    def get_value(self):
        return self.value

counter = Counter.remote()

@ray.remote
def call_actor_in_worker(counter):
    counter.increment.remote()

ray.get([call_actor_in_worker.remote(counter) for _ in range(5)])
assert ray.get(counter.get_value.remote()) == 5
