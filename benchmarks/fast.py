import numpy as np
from pyearth import Earth
from timeit import Timer

# The robot arm example, as defined in:
# Fast MARS, Jerome H.Friedman, Technical Report No.110, May 1993, section 6.2.

np.random.seed(2)
nb_examples = 400
theta1 = np.random.uniform(0, 2 * np.pi, size=nb_examples)
theta2 = np.random.uniform(0, 2 * np.pi, size=nb_examples)
phi = np.random.uniform(-np.pi/2, np.pi/2, size=nb_examples)
l1 = np.random.uniform(0, 1, size=nb_examples)
l2 = np.random.uniform(0, 1, size=nb_examples)
x = l1 * np.cos(theta1) - l2 * np.cos(theta1 + theta2) * np.cos(phi)
y = l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2) * np.cos(phi)
z = l2 * np.sin(theta2) * np.sin(phi)
d = np.sqrt(x**2 + y**2 + z**2)


inputs = np.concatenate([theta1[:, np.newaxis],
                         theta2[:, np.newaxis],
                         phi[:, np.newaxis],
                         l1[:, np.newaxis],
                         l2[:, np.newaxis]], axis=1)
outputs = d

hp = dict(
        max_degree=5,
        minspan=1,
        endspan=1,
        max_terms=100,
        allow_linear=False,
)
model_normal = Earth(**hp)
t = Timer(lambda: model_normal.fit(inputs, outputs))
duration_normal = t.timeit(number=1)
print("Normal : MSE={0:.5f}, duration={1:.2f}s".
      format(model_normal.mse_, duration_normal))
model_fast = Earth(use_fast=True,
                   fast_K=5,
                   fast_h=1,
                   **hp)

t = Timer(lambda: model_fast.fit(inputs, outputs))
duration_fast = t.timeit(number=1)
print("Fast: MSE={0:.5f}, duration={1:.2f}s".
      format(model_fast.mse_, duration_fast))
speedup = duration_normal / duration_fast
print("diagnostic : MSE goes from {0:.5f} to {1:.5f} but it "
      "is {2:.2f}x faster".
      format(model_normal.mse_, model_fast.mse_, speedup))
