import torch
import pandas as pd
import numpy as np
from sklearn import preprocessing

import matplotlib.pyplot as plt
import seaborn as sns

students = pd.read_csv('data_random_1.csv')

input_cols = ['student_id', 'derivated_nacin_studiranja_beautified', 'derivated_srednja_skola', 'derivated_srednja_skola', 'opstina_rodjenja', 'opstina_prebivalista', 'spol']
output_cols = ['ukupno_godina_do_diplomiranja']

target_map = {
    val: index for index, val in enumerate(students[output_cols[0]].unique())
}

del students[input_cols[0]]
input_cols.pop(0)

le = preprocessing.LabelEncoder()

for i in range(0, 6):
    le.fit(students[input_cols[i]])
    students[input_cols[i]] = le.transform(students[input_cols[i]])

input_np_array = students[input_cols].to_numpy()
target_np_array = students[output_cols].to_numpy()

plt.figure(figsize=(7, 7))
plt.title('Correlation HeatMap', y = 1.05)
sns.heatmap(students.corr(), linewidths=0.1, vmax=1.0, square=True, linecolor='white', annot=True)
plt.show()

X = torch.tensor(input_np_array, dtype=torch.float32)
y = torch.tensor(students[output_cols[0]].map(target_map).values)


def one_hot_encode(vector):
    n_classes = len(vector.unique())

    one_hot = torch.zeros((vector.shape[0], n_classes)).type(torch.LongTensor)

    return one_hot.scatter(1, vector.type(torch.LongTensor).unsqueeze(1), 1)

y_one_hot = one_hot_encode(y)


random_indices = torch.randperm(X.shape[0])

n_train = int(0.8 * X.shape[0])

X_train = X[random_indices[:n_train]]
y_train = y[random_indices[:n_train]]
y_train_one_hot = y_one_hot[random_indices[:n_train]]

X_test = X[random_indices[n_train:]]
y_test = y[random_indices[n_train:]]
y_test_one_hot = y_one_hot[random_indices[n_train:]]


w = torch.rand((6, 3))
b = torch.rand(3)


def softmax_activation(z):
    exponentials = torch.exp(z)
    exponentials_row_sums = torch.sum(exponentials, axis=1).unsqueeze(1)

    return exponentials / exponentials_row_sums


def cross_entropy_loss(y_one_hot, activations):
    return -torch.mean(
        torch.sum(
            y_one_hot * torch.log(activations), axis=1
        )
    )


lambda_param = 0.01
learning_rate = 0.1
n_iterations = 300

for i in range(1, n_iterations + 1):
    Z = torch.mm(X_train, w) + b
    A = softmax_activation(Z)
    l2_regularization = torch.sum(w ** 2)

    loss = cross_entropy_loss(y_train_one_hot, A) + lambda_param * l2_regularization

    w_gradients = -torch.mm(X_train.transpose(0, 1), y_train_one_hot - A) / n_train + (2 * lambda_param * w)
    b_gradients = -torch.mean(y_train_one_hot - A, axis=0)

    w -= learning_rate * w_gradients
    b -= learning_rate * b_gradients

    if i == 1 or i % 25 == 0:
        print("Loss at iteration {}: {}".format(i, loss))


test_predictions = torch.argmax(
    softmax_activation(torch.mm(X_test, w) + b), axis=1
)

test_accuracy = float(sum(test_predictions == y_test)) / y_test.shape[0]
print("Final Test Accuracy: {}".format(test_accuracy))
