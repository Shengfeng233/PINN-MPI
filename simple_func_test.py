import numpy as np

from pinn_model import *
import pandas as pd
import matplotlib.pyplot as plt
from pyDOE import lhs

filename_data = './stack_Re100.mat'

whole_field = np.array([[0.8, 10.8], [-2.5, 2.5]])
small_hole = np.array([[2.5, 3.5], [-0.5, 0.5]])
large_hole = np.array([[2.0, 4.0], [-1, 1]])
small_truncation = np.array([[2.5, 3.5], [-2.5, 2.5]])
large_truncation = np.array([[2.0, 4.0], [-2.5, 2.5]])
enormous_truncation = np.array([[1.01, 7.99], [-2.5, 2.5]])
hole = enormous_truncation
x, y, u, v, feature_mat = read_2D_data_surround_without_p(filename_data)
x_mat = x.numpy()
x_unique = np.unique(x)
y_unique = np.unique(y)
plt.scatter(x.numpy(), y.numpy())
plt.xlim((-0.1, 1.1))
plt.ylim((-0.1, 1.1))
plt.show()

print("ok")
