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
