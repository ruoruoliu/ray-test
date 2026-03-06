import numpy as np
import os
import ray
import time

import gymnasium as gym

H = 200
gamma = 0.99
decay_rate = 0.99
D = 80 * 80
learning_rate = 1e-4

def preprocess(img):
    img = img[35:195]
    img = img[::2, ::2, 0]
    img[img == 144] = 0
    img[img == 109] = 0
    img[img != 0] = 1
    return img.astype(float).ravel()

def process_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def rollout(model, env):
    observation, info = env.reset()
    prev_x = None
    xs, hs, dlogps, drs = [], [], [], []
    terminated = truncated = False
    while not terminated and not truncated:
        cur_x = preprocess(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(D)
        prev_x = cur_x

        aprob, h = model.policy_forward(x)
        action = 2 if np.random.uniform() < aprob else 3

        xs.append(x)
        hs.append(h)

        y = 1 if action == 2 else 0

        dlogps.append(y - aprob)

        observation, reward, terminated, truncated, info = env.step(action)

        drs.append(reward)

    return xs, hs, dlogps, drs


class Model(object):
    
    def __init__(self):
        self.weights = {}
        self.weights["W1"] = np.random.randn(H, D) / np.sqrt(D)
        self.weights["W2"] = np.random.randn(H) / np.sqrt(H)

    def policy_forward(self, x):
        h = np.dot(self.weights["W1"], x)
        h[h < 0] = 0
        logp = np.dot(self.weights["W2"], h)
        p = 1.0 / (1.0 + np.exp(-logp))
        return p, h

    def policy_backward(self, eph, epx, epdlogp):
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.weights["W2"])
        dh[eph <= 0] = 0
        dW1 = np.dot(dh.T, epx)
        return {"W1": dW1, "W2": dW2}


    def update(self, grad_buffer, rmsprop_cache, lr, decay):
        for k, v in self.weights.items():
            g = grad_buffer[k]
            rmsprop_cache[k] = decay * rmsprop_cache[k] + (1 - decay) * g ** 2
            self.weights[k] += lr * g / (np.sqrt(rmsprop_cache[k] + 1e-5))

def zero_grads(grad_buffer):
    for k, v in grad_buffer.items():
        grad_buffer[k] = np.zeros_like(v)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

ray.init()


@ray.remote
class RolloutWorker(object):
    
    def __init__(self):
        self.env = gym.make("ale_py:ALE/Pong-v5")

    def compute_gradient(self, model):
        xs, hs, dlogps, drs = rollout(model, self.env)
        reward_sum = sum(drs)
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)

        discounted_epr = process_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr

        return model.policy_backward(eph, epx, epdlogp), reward_sum

iterations = 20
batch_size = 4
model = Model()
actors = [RolloutWorker.remote() for _ in range(batch_size)]

running_reward = None
grad_buffer = {k: np.zeros_like(v) for k, v in model.weights.items()}
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.weights.items()}

for i in range(1, 1 + iterations):
    model_id = ray.put(model)
    gradient_ids = []

    start_time = time.time()
    gradient_ids = [actor.compute_gradient.remote(model_id) for actor in actors]
    for batch in range(batch_size):
        [grad_id], gradient_ids = ray.wait(gradient_ids)
        grad, reward_sum = ray.get(grad_id)
        for k in model.weights:
            grad_buffer[k] += grad[k]
        running_reward = (
            reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        )
    end_time = time.time()
    print(
        f"Batch {i} computed {batch_size} rollouts in {end_time - start_time} sec, running_mean is {running_reward}"
    )

    model.update(grad_buffer, rmsprop_cache, learning_rate, decay_rate)
    zero_grads(grad_buffer)

