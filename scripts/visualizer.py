import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np, argparse


def load_file(filepath):
    with open(filepath, 'r') as stream:
        lines = stream.readlines()
    N, xDim, yDim = [int(x) for x in lines[0].strip().split()]
    X = np.zeros((N, xDim))
    Y = np.zeros((N, yDim))
    for i in range(1, N + 1):
        row = [float(x) for x in lines[i].strip().split()]
        X[i - 1, :] = np.array(row[:xDim])
        Y[i - 1] = np.array(row[xDim:])
    return X, Y


def plot_classification_dataset(X_true, Y_true, X_eval, Y_eval):
    fig = plt.figure(figsize=(15, 6))
    ax_true = fig.add_subplot(121)
    ax_eval = fig.add_subplot(122)
    ax_true.scatter(X_true[:, 0], X_true[:, 1], c=Y_true, cmap='gray')
    ax_eval.scatter(X_eval[:, 0], X_eval[:, 1], c=Y_eval, cmap='gray')
    plt.show()


def plot_regression_dataset(X_true, Y_true, X_eval, Y_eval):
    fig = plt.figure(figsize=(15, 6))
    ax_true = fig.add_subplot(121, projection='3d')
    ax_eval = fig.add_subplot(122, projection='3d')
    ax_true.scatter(X_true[:, 0], X_true[:, 1], Y_true[:, 0])
    ax_eval.scatter(X_eval[:, 0], X_eval[:, 1], Y_eval[:, 0])
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualizer')
    parser.add_argument('--true', required=True, type=str)
    parser.add_argument('--eval', required=True, type=str)
    parser.add_argument('--type',
                        required=True,
                        type=str,
                        choices=['classification', 'regression'])
    args = parser.parse_args()
    X_true, Y_true = load_file(args.true)
    X_eval, Y_eval = load_file(args.eval)
    if (args.type == 'classification'):
        plot_classification_dataset(X_true, Y_true, X_eval, Y_eval)
    else:
        plot_regression_dataset(X_true, Y_true, X_eval, Y_eval)


if __name__ == "__main__":
    main()