
# coding: utf-8

# <img src='logo.jpg'>

# ## RNN Exercise
# ### Time Series Perediction
# ### Neural Network Course (Dr.Azadeh Mansouri)
# ### Kharazmi University 
# ### Department of computer and electrical engineering
# 
# 
# ### ---------------------------------------------------------------------
# 
# ### Ali Forghani Elah Abadi
# ### Student Number : 973082512 

# In[ ]:


#import packages

import tensorflow as tf     
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import sklearn as sk
from sklearn import preprocessing as ppr


# In[ ]:


#connect google drive to google colabratory

from google.colab import drive
drive.mount('/content/drive/')


# In[ ]:


#generate data from dataset csv file 

dataFrame = pd.read_csv('/content/drive/My Drive/Colab Notebooks/RNN/airline-passengers.csv', usecols = [1], header = 0)[:-1]
dataset_orig = dataFrame.values.astype(np.float32)


# In[ ]:


#plot raw data

plt.plot(dataset_orig, label = "Original Data", c = 'g')
plt.legend()
plt.xlabel('TimeStamps')
plt.ylabel("Total Passengers")
plt.show()


# In[ ]:


# normalize the dataset
scaler = ppr.MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset_orig)


# In[ ]:


#function definition to splitting data to train and test

def train_test_set(dataset, percentage_train = 0.75):
    l = len(dataset)
    train_set_end_index = int(l * 0.67)
    test_set_start_index = train_set_end_index + 1
    train_set = dataset[:test_set_start_index]
    test_set = dataset[test_set_start_index:]
    return train_set, test_set

def supervised_learning_form(train_dataset, test_dataset):
    train_x = train_dataset[:-1]
    train_y = train_dataset[1:]
    test_x = test_dataset[:-1]
    test_y = test_dataset[1:]
    return train_x, train_y, test_x, test_y


# In[ ]:


#splitting data to train and test
train_set, test_set = train_test_set(dataset)
train_x, train_y, test_x, test_y = supervised_learning_form(train_set, test_set)


# In[ ]:


# network parameters

net_size = 20
n_epochs = 2000
n_x = 1
n_y = 1
n_timestamp = n_x
n_x_vars = 1
n_y_vars = 1
learning_rate = 0.08


# In[ ]:


tf.reset_default_graph()


# In[ ]:


# input and output placeholders
X_p = tf.placeholder(tf.float32, [None, n_timestamp, n_x_vars], name = "X_input")
Y_p = tf.placeholder(tf.float32, [None, n_timestamp, n_y_vars], name = "Y_output")

rnn_inputs = tf.unstack(X_p, axis = 1)

# setup the RNN
with tf.name_scope("RNN"):
    rnnCell = tf.nn.rnn_cell.LSTMCell(net_size, name = "rnn_cell")
    rnn_outputs, final_state = tf.nn.static_rnn(rnnCell, rnn_inputs, dtype=tf.float32)

    # weights and biases

    W = tf.get_variable('W', [net_size, n_y_vars])
    b = tf.get_variable('b', [n_y_vars], initializer=tf.constant_initializer(0.0))


# In[ ]:


# run predictions
predictions = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]


# In[ ]:


y_as_list = tf.unstack(Y_p, num = n_timestamp, axis = 1)


# In[ ]:


#set mean square loss

losses = [tf.losses.mean_squared_error(labels = label, predictions = prediction) 
          for prediction, label in zip(predictions, y_as_list)]
total_loss = tf.reduce_mean(losses)
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)


# In[ ]:


#training with run session in loop

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epoch_loss = 0.0
    
    # tensorboard
    writer = tf.summary.FileWriter('./log', sess.graph)
    # training
    
    for epoch in range(n_epochs):
        
        # first loss, then predict, then run optimizer till n_epochs
        
        
        
        epoch_loss, y_train_prediction,_ = sess.run([total_loss, predictions, optimizer], 
                 feed_dict = { X_p : train_x.reshape(-1, n_timestamp, n_x_vars), 
                              Y_p : train_y.reshape(-1, n_timestamp, n_y_vars) })
        
        if epoch % 100 == 0:
            print("Loss after epoch " + str(epoch) + " : " + str(epoch_loss))
        
    print("Training mse :" + str(epoch_loss))
    
    #testing
    
    test_loss, y_test_prediction, _ = sess.run([total_loss, predictions, optimizer],
                  feed_dict = { X_p : test_x.reshape(-1, n_timestamp, n_x_vars), 
                                Y_p : test_y.reshape(-1, n_timestamp, n_y_vars) })
    print("Testing mse :" + str(test_loss))


# In[ ]:



y_train_prediction = scaler.inverse_transform(y_train_prediction[0])
y_test_prediction = scaler.inverse_transform(y_test_prediction[0])


# In[ ]:



y_train_orig = scaler.inverse_transform(train_y)


# In[ ]:



y_test_orig = scaler.inverse_transform(test_y)


# In[ ]:


y_test_prediction_shifted = np.empty_like(dataset_orig)
y_test_prediction_shifted[:len(y_train_orig) + 1] = np.nan
y_test_prediction_shifted[len(y_train_orig) + 1:-1] = y_test_prediction
y_test_prediction_shifted[-1] = np.nan


# In[ ]:


plt.plot(dataset_orig, label = "original dataset", c = 'b')
plt.plot(y_train_prediction, label = "Training set", c = 'g')
plt.plot(y_test_prediction_shifted, label = "Prediction in test set", c = 'r')
plt.legend()
plt.show()


# ## Report 

# <p dir = 'rtl' style="text-align:right;">  
#    به عنوان گزارش خلاصه ای از روند اجرا شده در بالا باید گفت پس از دریافت فایل دیتا ست از گوگل درایو آن را لود نموده و داده ها را طوری به دو مجموعه آموزش و تست تجزیه میکنیم که هر برچسب به ازای هر داده آموزشی نشانه تعداد مسافرین در ماه بعدی آن داده است . سپس با ست کردن هایپر پارامتر های شبکه و پس از آن ایجاد پلیس هولدر هایی به عنوان مکان خالی برای متغیر های هر بچ داده که قرار است به طور موازی محاسبه شوند تعریف میشود. بعد از آن با تعریف یک سلول شبکه ریکارنت و پیکر بندی آن و همچنین تنظیم تابع خطا بر روی میانگین مربعات خطا و دستور مینیمایز کردن آن در حلقه ای هر بار یک سشن ران میکنیم و در طی آن فرآیند آموزش را به انجام میرسانیم . در نهایت خروجی مشاهده شده بیانگر پیش بینی انجام شده ما روی داده های تست خواهد بود. 
# </p>
