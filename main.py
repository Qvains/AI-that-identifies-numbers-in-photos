import numpy as np
import matplotlib.pyplot as plt
size_of_matrix = 10
def generate_digit_matrix(digit, size=10):
    img = np.zeros((size, size))
    if digit == 0:
        img[1:-1, 1] = 1; img[1:-1, -2] = 1
        img[1, 1:-1] = 1; img[-2, 1:-1] = 1
    elif digit == 1:
        img[:, size//2] = 1
    elif digit == 2:
        img[1, 1:-1] = img[-2, 1:-1] = 1
        img[size//2, 1:-1] = 1
        img[1:size//2, -2] = 1; img[size//2:-2, 1] = 1
    elif digit == 3:
        img[1, 1:-1] = img[-2, 1:-1] = 1
        img[size//2, 1:-1] = 1; img[1:-1, -2] = 1
    elif digit == 4:
        img[1:size//2, 1] = 1; img[size//2, 1:-1] = 1; img[:, -2] = 1
    elif digit == 5:
        img[1, 1:-1] = img[-2, 1:-1] = 1
        img[size//2, 1:-1] = 1
        img[1:size//2, 1] = 1; img[size//2:-1, -2] = 1
    elif digit == 6:
        img[1, 1:-1] = img[-2, 1:-1] = 1
        img[size//2, 1:-1] = 1
        img[1:-1, 1] = 1; img[size//2:-1, -2] = 1
    elif digit == 7:
        img[1, 1:-1] = 1; img[:, -2] = 1
    elif digit == 8:
        img[1, 1:-1] = img[-2, 1:-1] = 1
        img[size//2, 1:-1] = 1
        img[1:-1, 1] = img[1:-1, -2] = 1
    elif digit == 9:
        img[1, 1:-1] = img[-2, 1:-1] = 1
        img[size//2, 1:-1] = 1
        img[1:size//2, 1] = 1; img[1:-1, -2] = 1
    return img

def create_dataset(samples_per_digit=200, noise_level=0.2):
    X, y = [], []
    for digit in range(10):
        base = generate_digit_matrix(digit,size_of_matrix)
        for _ in range(samples_per_digit):
            noisy = base + np.random.normal(0, noise_level, base.shape)
            noisy = np.clip(noisy, 0, 1)
            X.append(noisy.flatten())
            y.append(digit)
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(samples_per_digit=300, noise_level=0.25)
X_val, y_val = create_dataset(samples_per_digit=80, noise_level=0.25)

input_size = 100 
hidden_size = 32
output_size = 10
learning_rate = 0.1
batch_size = 64
epochs = 30
np.random.seed(46)

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

def relu(x): return np.maximum(0, x)
def relu_derivative(x): return (x > 0).astype(float)

def stable_softmax(z):
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(probs, targets):
    m = targets.shape[0]
    p_correct = probs[np.arange(m), targets]
    eps = 1e-12
    return np.mean(-np.log(p_correct + eps))

def accuracy(probs, targets):
    preds = np.argmax(probs, axis=1)
    return np.mean(preds == targets)

train_loss_hist, val_loss_hist = [], []
train_acc_hist, val_acc_hist = [], []

def train(X_train, y_train, X_val=None, y_val=None):
    global W1, b1, W2, b2
    n_samples = X_train.shape[0]
    steps_per_epoch = int(np.ceil(n_samples / batch_size))

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(n_samples)
        X_shuffled, y_shuffled = X_train[perm], y_train[perm]

        epoch_loss = 0.0
        epoch_acc = 0.0

        for step in range(steps_per_epoch):
            start, end = step * batch_size, min((step + 1) * batch_size, n_samples)
            X_batch, y_batch = X_shuffled[start:end], y_shuffled[start:end]
            m = X_batch.shape[0]
 
            # ---------- FORWARD ----------
            Z1 = np.dot(X_batch, W1) + b1
            A1 = relu(Z1)
            Z2 = np.dot(A1, W2) + b2
            A2 = stable_softmax(Z2)

            # ---------- LOSS + ACC ----------
            loss = cross_entropy_loss(A2, y_batch)
            acc = accuracy(A2, y_batch)
            epoch_loss += loss * m
            epoch_acc += acc * m

            # ---------- BACKPROP ----------
            dZ2 = A2.copy()
            dZ2[np.arange(m), y_batch] -= 1
            dZ2 /= m
            dW2 = np.dot(A1.T, dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)

            dA1 = np.dot(dZ2, W2.T)
            dZ1 = dA1 * relu_derivative(Z1)
            dW1 = np.dot(X_batch.T, dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)

            # ---------- UPDATE ----------
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        # ---------- METRICS ----------
        epoch_loss /= n_samples
        epoch_acc /= n_samples

        Z1_val = np.dot(X_val, W1) + b1
        A1_val = relu(Z1_val)
        Z2_val = np.dot(A1_val, W2) + b2
        A2_val = stable_softmax(Z2_val)
        val_loss = cross_entropy_loss(A2_val, y_val)
        val_acc = accuracy(A2_val, y_val)

        train_loss_hist.append(epoch_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(epoch_acc)
        val_acc_hist.append(val_acc)

        print(f"Epoch {epoch}/{epochs}: "
              f"train_loss={epoch_loss:.4f}, train_acc={epoch_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

train(X_train, y_train, X_val, y_val)

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_loss_hist, label="Train Loss")
plt.plot(val_loss_hist, label="Val Loss")
plt.legend(); plt.title("Функция потерь")

plt.subplot(1,2,2)
plt.plot(train_acc_hist, label="Train Accuracy")
plt.plot(val_acc_hist, label="Val Accuracy")
plt.legend(); plt.title("Точность")
plt.show()

def predict(X):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    return np.argmax(stable_softmax(Z2), axis=1)

def test_network(X_val, y_val, num_tests=3, samples=10):
    for t in range(num_tests):
        indices = np.random.choice(len(X_val), samples)
        X_sample, y_true = X_val[indices], y_val[indices]
        y_pred = predict(X_sample)

        fig, axes = plt.subplots(1, samples, figsize=(15, 3))
        fig.suptitle(f"🔹 Проверка #{t+1}", fontsize=14)
        for i, ax in enumerate(axes):
            ax.imshow(X_sample[i].reshape(size_of_matrix, size_of_matrix), cmap="gray_r")
            ax.set_title(f"T:{y_true[i]} / P:{y_pred[i]}")
            ax.axis("off")
        plt.show()

test_network(X_val, y_val, num_tests=3, samples=10)

