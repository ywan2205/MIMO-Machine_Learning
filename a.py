import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')

#import numpy as np
import scipy.io as spio

file1 = spio.loadmat('train_and_test_data1.mat')
train_data_real = file1['y6_save_real_image'][:]
train_data_real = np.array(train_data_real).astype(np.float32)

file3 = spio.loadmat('label1.mat')
Label = file3['parameter_tt'][:]
Label = np.array(Label).astype(np.float32)

train_data1 = train_data_real[:15000, :, : ]
test_data1 = train_data_real[15000:, :, :]

train_label1 =Label[:,:15000]
train_label1 = np.transpose(train_label1)
test_label1 = Label[:,15000:]
test_label1 = np.transpose(test_label1)

#train_data1 = train_data1 .reshape(-1, 32, 32, 1)
#test_data1 = test_data1 .reshape(-1, 32, 32, 1)

train_data1 = train_data1.reshape(-1,1024)
train_label1 = train_label1.reshape(-1, 1)
#model = MLPRegressor(hidden_layer_sizes=(200), activation="relu",
#                 solver='lbfgs', alpha=0.0001,
#                 batch_size='auto', learning_rate="adaptive",
#                 learning_rate_init=0.002,
#                 power_t=0.5, max_iter=200,tol=1e-4)

model = MLPRegressor(hidden_layer_sizes=(1000), activation="logistic",
                  solver='lbfgs', alpha=0.0001,
                  batch_size=1, learning_rate="adaptive",
                  learning_rate_init=0.001,
                  power_t=0.5, max_iter=200,tol=1e-4)


model.fit(train_data1 , train_label1)
#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

test_data1 = test_data1.reshape(-1,1024)
test_label1 = test_label1.reshape(-1, 1)


y_test_pred = model.predict(test_data1) #预测
print(y_test_pred)

Y_test1 = test_label1.reshape(5000)
my_array = np.empty(5000)
for x, each in enumerate(y_test_pred):
    my_array[x] = each


SE = (my_array - Y_test1)
#print("SE: ", SE)

MSE = sum((my_array - Y_test1)**2)/5000
print("MSE: ", MSE)

print(test_label1)

fig, ax = plt.subplots()
ax.scatter(test_label1, y_test_pred)
ax.plot([test_label1.min(), test_label1.max()], [test_label1.min(), test_label1.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()