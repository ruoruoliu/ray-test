import ray

@ray.remote
def square(x):
    return x * x

obj_refs = []
for i in range(5):
    obj_refs.append(square.remote(i))

assert ray.get(obj_refs) == [0, 1, 4, 9, 16]
