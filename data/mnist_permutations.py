from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--o', default='./mnist_permutations.pkl', help='output file')
parser.add_argument('--n_tasks', default=10, type=int, help='number of tasks')
parser.add_argument('--seed', default=100, type=int, help='random seed')
args = parser.parse_args()
np.random.seed(args.seed)
data = input_data.read_data_sets('MNIST_data', one_hot=True)
x_tr = data.train.images
y_tr = data.train.labels

x_val = data.validation.images
y_val = data.validation.labels

x_te = data.test.images
y_te = data.test.labels

permutations = []
for i in range(args.n_tasks):
    indices = np.random.permutation(784)
    permutations.append((x_tr[:, indices], y_tr, x_val[:, indices], y_val, x_te[:, indices], y_te))
f = open(args.o, "wb")
pickle.dump(permutations, f)
f.close()
