"""1D quantum classifier"""
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import NesterovMomentumOptimizer

dev = qml.device("default.qubit", wires=4)


def layer(weights):
    """Quantum layer."""
    qml.Rot(weights[0, 0], weights[0, 1], weights[0, 2], wires=0)
    qml.Rot(weights[1, 0], weights[1, 1], weights[1, 2], wires=1)
    qml.Rot(weights[2, 0], weights[2, 1], weights[2, 2], wires=2)
    qml.Rot(weights[3, 0], weights[3, 1], weights[3, 2], wires=3)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])


@qml.qnode(dev)
def circuit(weights, input_data):
    """Quantum circuit."""
    qml.BasisState(input_data, wires=[0, 1, 2, 3])

    for weight in weights:
        layer(weight)

    return qml.expval(qml.PauliZ(0))


def variational_classifier(weights, bias, input_data):
    """Add classical bias."""
    return circuit(weights, input_data) + bias


def square_loss(labels, predictions):
    """Square loss."""
    loss = sum((label - prediction) ** 2 for label, prediction in zip(labels, predictions))

    loss /= len(labels)
    return loss


def accuracy(labels, predictions):
    """Accuracy function."""
    loss = 0
    for label, prediction in zip(labels, predictions):
        if abs(label - prediction) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss


def cost(weights, bias, input_data, labels):
    """Cost function."""
    predictions = [variational_classifier(weights, bias, x) for x in input_data]
    return square_loss(labels, predictions)


data = np.loadtxt("data/parity.txt")
X = pnp.array(data[:, :-1], requires_grad=False)
Y = pnp.array(data[:, -1], requires_grad=False)
Y = Y * 2 - np.ones(len(Y))  # shift label from {0, 1} to {-1, 1}

for i in range(5):
    print(f"X = {X[i]}, Y = {Y[i]}")

print("...")

np.random.seed(0)
NUM_QUBITS = 4
NUM_LAYERS = 2
weights_init = 0.01 * pnp.random.randn(NUM_LAYERS, NUM_QUBITS, 3, requires_grad=True)
bias_init = pnp.array(0.0, requires_grad=True)

print(weights_init, bias_init)

opt = NesterovMomentumOptimizer(0.5)
BATCH_SIZE = 5

weights = weights_init
bias = bias_init
for it in range(25):

    # Update the weights by one optimizer step
    batch_index = np.random.randint(0, len(X), (BATCH_SIZE,))
    X_batch = X[batch_index]
    Y_batch = Y[batch_index]
    weights, bias, _, _ = opt.step(cost, weights, bias, X_batch, Y_batch)

    # Compute accuracy
    predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X]
    acc = accuracy(Y, predictions)

    print(f"Iter: {it + 1:5d} | Cost: {cost(weights, bias, X, Y):0.7f} | Accuracy: {acc:0.7f} ")
