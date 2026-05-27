import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split

%matplotlib inline


sns.set(style=&quot;whitegrid&quot;, palette=&quot;muted&quot;, font_scale=1.5)
RANDOM_SEED = 42

from google.colab import drive
drive.mount('/content/drive')

from google.colab import files
uploaded = files.upload()

#transforming shape
reshaped_segments = np.asarray(
    segments, dtype = np.float32).reshape(
    -1 , N_time_steps, N_features)

X_train, X_test, Y_train, Y_test = train_test_split(
    reshaped_segments, labels, test_size = 0.2, 
    random_state = RANDOM_SEED)

def create_LSTM_model(inputs):
    W = {
        'hidden': tf.Variable(tf.random_normal([N_features, N_hidden_units])),
        'output': tf.Variable(tf.random_normal([N_hidden_units, N_classes]))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([N_hidden_units], mean = 0.1)),
        'output': tf.Variable(tf.random_normal([N_classes]))
    }
    X = tf.transpose(inputs, [1, 0, 2])
    X = tf.reshape(X, [-1, N_features])
    hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
    hidden = tf.split(hidden, N_time_steps, 0)

    lstm_layers = [tf.contrib.rnn.BasicLSTMCell(
        N_hidden_units, forget_bias = 1.0) for _ in range(2)]
    lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)

    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, 
                                           hidden, dtype = tf.float32)

    lstm_last_output = outputs[-1]
    return tf.matmul(lstm_last_output, W['output']) + biases['output']
  L2_LOSS = 0.0015
l2 = L2_LOSS * \
  sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits = pred_y, labels = Y)) + l2

Learning_rate = 0.0025
optimizer = tf.train.AdamOptimizer(learning_rate = Learning_rate).minimize(loss)
correct_pred = tf.equal(tf.argmax(pred_softmax , 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype = tf.float32))



plt.figure(figsize=(12,8))

plt.plot(np.array(history['train_loss']), &quot;r--&quot;, label=&quot;Train loss&quot;)
plt.plot(np.array(history['train_acc']), &quot;g--&quot;, label=&quot;Train accuracy&quot;)

plt.plot(np.array(history['test_loss']), &quot;r--&quot;, label=&quot;Test loss&quot;)
plt.plot(np.array(history['test_acc']), &quot;g--&quot;, label=&quot;Test accuracy&quot;)

plt.title(&quot;Training session's progress over iteration&quot;)
plt.legend(loc = 'upper right', shadow = True)
plt.ylabel('Training Progress(Loss or Accuracy values)')
plt.xlabel('Training Epoch')
plt.ylim(0)

plt.show()


max_test = np.argmax(Y_test, axis=1)
max_predictions = np.argmax(predictions, axis = 1)
confusion_matrix = metrics.confusion_matrix(max_test, max_predictions)

plt.figure(figsize=(16,14))
sns.heatmap(confusion_matrix, xticklabels = LABELS, yticklabels = LABELS, annot =True, fmt = &quot;d&quot;)
plt.title(&quot;Confusion Matrix&quot;)
plt.xlabel('Predicted_label')
plt.ylabel('True Label')
plt.show()

reshaped_segments.shape

