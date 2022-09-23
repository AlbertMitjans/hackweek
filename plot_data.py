"""Script used to load and plot the used data."""
import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("data/custom_dataset.txt")

X = np.array(data[:, :-2])
Y = np.array(data[:, -2])
labels = np.array(data[:, -1])


def target_function(x):
    """Target function where our data is sampled from."""
    return np.sin((2 * np.pi / 10) * x - np.pi)


target_x = np.linspace(0, 50, 1000)

for xc in X[Y == labels]:
    plt.axvline(x=xc, color="green", linestyle="--", linewidth=0.5)

for xc in X[Y != labels]:
    plt.axvline(x=xc, color="red", linestyle="--", linewidth=0.5)

plt.scatter(X[Y == labels], Y[Y == labels], color="green")
plt.scatter(X[Y != labels], Y[Y != labels], color="red")
plt.plot(target_x, target_function(target_x))
plt.grid()
plt.show()
