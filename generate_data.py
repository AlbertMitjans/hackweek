"""Script to generate, load and plot data."""
import matplotlib.pyplot as plt
import numpy as np


def generate_data(inputs: np.ndarray, flip_prob: float = 0):
    """Function used to generate noisy data."""
    data = np.greater(np.sin((2 * np.pi / 10) * inputs - np.pi), 0)
    # flip some labels randomly
    noisy_data = np.array([not x if np.random.random() < flip_prob else x for x in data])
    # center points around 0
    data = (data * 2) - 1
    noisy_data = (noisy_data * 2) - 1
    return data, noisy_data


X = np.arange(0, 100, 1)
labels, Y = generate_data(X, flip_prob=0.2)

# Save data
np.savetxt("data/custom_dataset.txt", np.array([X, Y, labels]).T, fmt="%2d")

# Load data
dataset = np.loadtxt("data/custom_dataset.txt")

X = np.array(dataset[:, :-2])
Y = np.array(dataset[:, -2])
labels = np.array(dataset[:, -1])

# Plot data
x_points = np.linspace(0, 100, 1000)

for xc in X[Y == labels]:
    plt.axvline(x=xc, color="green", linestyle="--", linewidth=0.5)

for xc in X[Y != labels]:
    plt.axvline(x=xc, color="red", linestyle="--", linewidth=0.5)

plt.scatter(X[Y == labels], Y[Y == labels], color="green")
plt.scatter(X[Y != labels], Y[Y != labels], color="red")
plt.plot(x_points, np.sin((2 * np.pi / 10) * x_points - np.pi))
plt.grid()
plt.show()
