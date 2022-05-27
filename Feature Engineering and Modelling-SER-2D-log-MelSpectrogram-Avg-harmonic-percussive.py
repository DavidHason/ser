#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering- Speech Emotion Recognition (SER)
# DavidH

# !pip install keras --upgrade

# !pip uninstall jupyterthemes

# !pip install imagenet_utils

# !pip install tf-nightly

# !pip install librosa

# In[1]:


import glob
import os
import librosa
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import soundfile as sf
#from sklearn.externals import joblib
from tensorflow import keras 
from tensorflow.keras.preprocessing import image
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
#from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image


#from sklearn.externals import joblib
import sklearn
#from sklearn.externals import joblib
import joblib
import tensorflow.keras as tf
from tensorflow.keras import models
from tensorflow.keras import layers
#from python_utils import model_evaluation_utils as meu
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import h5py
from tensorflow.keras.models import load_model
import h5py

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())

from platform import python_version

print(python_version())

import tensorflow

from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import roc_curve, auc 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
import numpy as np
import sys
from tensorflow.keras.models import load_model
#import model_evaluation_utils
#import model_evaluation_utils as meu

import sklearn
#from sklearn.externals import joblib
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras
import tensorflow.keras as tf
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import scipy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam # I believe this is better optimizer for our case
from tensorflow.keras.preprocessing.image import ImageDataGenerator # to augmenting our images for increasing accuracy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
get_ipython().run_line_magic('matplotlib', 'inline')
import sys
from tensorflow.keras.models import load_model
#import model_evaluation_utils
#import model_evaluation_utils as meu

import json
import warnings

import numpy as np

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.util.tf_export import keras_export

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils


from tensorflow import keras 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image

def process_sound_data(data):
    data = np.expand_dims(data, axis=0)
    data = preprocess_input(data)
    return data

#from tensorflow.keras_preprocessing.image import ImageDataGenerator#from tensorflow.keras.preprocessing import image_dataset_from_directory
#from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


# In[40]:


ROOT_DIR = 'fold'

  
files = glob.glob(ROOT_DIR+'/**/*')

len(files)


# %%'../fold/fold2/'  

# In[3]:


file_name=files[41]
file_name = file_name.split('\\')[-1] #Return File Name
print (file_name)
class_label = file_name.split('-')[1] #Return Class Id
print (class_label)
class_label = class_label.split('.')[0]
class_label


# In[4]:


def get_sound_data(path, sr=44100):
    data, fsr = sf.read(path) #read sound file
    data_resample = librosa.resample(data.T, fsr, sr)
    if len(data_resample.shape) > 1:
        data_resample = np.average(data_resample, axis=0)
    return data_resample, sr

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)


# In[5]:


d, sr = get_sound_data(files[0])
len(d), sr


# In[6]:


list(windows(d, window_size=512*63))


# In[7]:


file_name = file_name.split('\\')[-1] #Return File Name
class_label = file_name.split('-')[-1] #Return Class Id
class_label = class_label.split('.')[0]
class_label


# In[8]:


from tqdm import tqdm
import librosa as lb
def feature_extractor_harmonic_percussive(path):
    data, simple_rate = lb.load(path)
    D = lb.stft(data)
    #melspec_harmonic, melspec_percussive = lb.decompose.hpss(D)
    #melspec_harmonic = np.mean(melspec_harmonic)
    #melspec_percussive = np.mean(melspec_percussive)
    #logspec_hp = np.average([melspec_harmonic, melspec_percussive])
    harmonic = lb.effects.harmonic(data,margin=5.0)
    percussive = librosa.effects.percussive(data, margin=5.0)
    #harmonic = np.mean(harmonic)
    #percussive = np.mean(percussive)
    logspec_hp = np.average([harmonic, percussive])
    return logspec_hp

def feature_extractor_logmelspectrogram(path):
    data, simple_rate = lb.load(path)
    data = lb.feature.melspectrogram(data)
    data = lb.power_to_db(data)
    data = np.mean(data,axis=1)
    return data



x_harmonic_percussive, x_logmelspectrogram, y = [], [], []
for path in tqdm(files):
    file_name = path.split('\\')[-1] #Return File Name
    class_label = path.split('-')[-1] #Return Class Id
    class_label = class_label.split('.')[0]
        
    x_harmonic_percussive.append(feature_extractor_harmonic_percussive(path))
    x_logmelspectrogram.append(feature_extractor_logmelspectrogram(path))
    y.append(class_label)

x_harmonic_percussive = np.array(x_harmonic_percussive)
x_logmelspectrogram = np.array(x_logmelspectrogram)
y = np.array(y)


# In[9]:


from collections import Counter

Counter(y)


# In[10]:


x_harmonic_percussive.shape, x_logmelspectrogram.shape, y.shape


# In[11]:


features = np.hstack((x_harmonic_percussive.reshape(len(x_harmonic_percussive), 1), x_logmelspectrogram))
labels = y
features.shape, labels.shape


# In[12]:


from collections import Counter

Counter(labels)


# In[13]:


class_map = {'0' : 'anger', '1' : 'boredom', '2' : 'disgust', '3' : 'fear', '4' : 'happiness', 
                 '5' : 'neutral', '6' : 'sadness'}

categories = list(set(labels))
sample_idxs = [np.where(labels == label_id)[0][0] for label_id in categories]
feature_samples = features[sample_idxs]
feature_samples.shape


# In[14]:


features.shape, labels.shape


# In[15]:


features = np.hstack((x_harmonic_percussive.reshape(len(x_harmonic_percussive), 1), x_logmelspectrogram))
labels = y
features.shape, labels.shape


# In[16]:


data = np.array(list(zip(features, labels)))
data.shape


# In[17]:


train, validate, test = np.split(data, [int(.6*len(data)), int(.8*len(data))])


# In[18]:


train.shape, validate.shape, test.shape


# In[19]:


print('Train:', Counter(item[1] for item in train), 
      '\nValidate:', Counter(item[1] for item in validate), 
      '\nTest:',Counter(item[1] for item in test))


# ### Train test Split

# In[20]:


from sklearn.model_selection import train_test_split
train_features, validation_features, train_labels, validation_labels =                                                         train_test_split(features, labels, test_size = 0.4, shuffle = True)

test_features, validation_features, test_labels, validation_labels  =                                                         train_test_split(validation_features, validation_labels, test_size = 0.5, shuffle = True)


# In[21]:


train_features.shape, validation_features.shape, test_features.shape
#train_features.shape, test_features.shape


# In[22]:


train_labels.shape, validation_labels.shape, test_labels.shape
#train_labels.shape, test_labels.shape


# In[23]:


from tensorflow.keras.utils import to_categorical

train_labels_ohe = to_categorical(train_labels)
validation_labels_ohe = to_categorical(validation_labels)
test_labels_ohe = to_categorical(test_labels)


# In[24]:


train_labels_ohe.shape, validation_labels_ohe.shape, test_labels_ohe.shape

#train_labels_ohe.shape, test_labels_ohe.shape


# In[25]:


optimizer = tensorflow.keras.optimizers.RMSprop(learning_rate=0.0001, decay = 1e-6)

model = models.Sequential()
model.add(layers.Dense(1024, activation='relu', input_shape=(train_features.shape[1],)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(512, activation='tanh'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(train_labels_ohe.shape[1], activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer=optimizer,                 
                      metrics=['accuracy'])

model.summary()


# In[41]:


history = model.fit(train_features,
                    train_labels_ohe,
                    epochs=500,
                    batch_size=32,
                    validation_data=(validation_features, validation_labels_ohe), 

                    shuffle=True,
                    verbose=1)


# In[42]:


predictions = model.predict(test_features)


# In[44]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Deep Neural Net Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.2)

epochs = list(range(1,501))
ax1.plot(epochs, history.history['accuracy'], label='Train Accuracy')
ax1.plot(epochs, history.history['val_accuracy'], label='Test Accuracy')
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epochs, history.history['loss'], label='Train Loss')
ax2.plot(epochs, history.history['val_loss'], label='Validation Loss')
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")


# In[45]:


from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import roc_curve, auc 

from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model
from tensorflow import keras

#vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
#                                     input_shape=(64, 64, 3))


#output = vgg.layers[-1].output
#output = keras.layers.Flatten()(output)

#model = Model(vgg.input, output)
#model.trainable = False

model.summary()

def extract_tl_features(model, base_feature_data):
    dataset_tl_features = []
    for index, feature_data in enumerate(base_feature_data):
        if (index+1) % 1000 == 0:
            print('Finished processing', index+1, 'sound feature maps')
        pr_data = process_sound_data(feature_data)
        tl_features = model.predict(pr_data)
        tl_features = np.reshape(tl_features, tl_features.shape[1])
        dataset_tl_features.append(tl_features)
    return np.array(dataset_tl_features)


def get_metrics(true_labels, predicted_labels):
    
    print('Accuracy:', np.round(
                        metrics.accuracy_score(true_labels, 
                                               predicted_labels),
                        4))
    print('Precision:', np.round(
                        metrics.precision_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        4))
    print('Recall:', np.round(
                        metrics.recall_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        4))
    print('F1 Score:', np.round(
                        metrics.f1_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        4))
                        

def train_predict_model(classifier, 
                        train_features, train_labels, 
                        test_features, test_labels):
    # build model    
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features) 
    return predictions    


def display_confusion_matrix(true_labels, predicted_labels, classes=[1,0]):
    
    total_classes = len(classes)
    level_labels = [total_classes*[0], list(range(total_classes))]

    cm = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels, 
                                  labels=classes)
    cm_frame = pd.DataFrame(data=cm, 
                            columns=pd.MultiIndex(levels=[['Predicted:'], classes], 
                                                  labels=level_labels), 
                            index=pd.MultiIndex(levels=[['Actual:'], classes], 
                                                labels=level_labels)) 
    print(cm_frame) 
    
def display_classification_report(true_labels, predicted_labels, classes=[1,0]):

    report = metrics.classification_report(y_true=true_labels, 
                                           y_pred=predicted_labels, 
                                           labels=classes) 
    print(report)
    
    
    
def display_model_performance_metrics(true_labels, predicted_labels, classes=[1,0]):
    print('Model Performance metrics:')
    print('-'*30)
    get_metrics(true_labels=true_labels, predicted_labels=predicted_labels)
    print('\nModel Classification report:')
    print('-'*30)
    display_classification_report(true_labels=true_labels, predicted_labels=predicted_labels, 
                                  classes=classes)
    print('\nPrediction Confusion Matrix:')
    print('-'*30)
    display_confusion_matrix(true_labels=true_labels, predicted_labels=predicted_labels, 
                             classes=classes)


def plot_model_decision_surface(clf, train_features, train_labels,
                                plot_step=0.02, cmap=plt.cm.RdYlBu,
                                markers=None, alphas=None, colors=None):
    
    if train_features.shape[1] != 2:
        raise ValueError("X_train should have exactly 2 columnns!")
    
    x_min, x_max = train_features[:, 0].min() - plot_step, train_features[:, 0].max() + plot_step
    y_min, y_max = train_features[:, 1].min() - plot_step, train_features[:, 1].max() + plot_step
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    clf_est = clone(clf)
    clf_est.fit(train_features,train_labels)
    if hasattr(clf_est, 'predict_proba'):
        Z = clf_est.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
    else:
        Z = clf_est.predict(np.c_[xx.ravel(), yy.ravel()])    
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=cmap)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(train_labels)
    n_classes = len(le.classes_)
    plot_colors = ''.join(colors) if colors else [None] * n_classes
    label_names = le.classes_
    markers = markers if markers else [None] * n_classes
    alphas = alphas if alphas else [None] * n_classes
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y_enc == i)
        plt.scatter(train_features[idx, 0], train_features[idx, 1], c=color,
                    label=label_names[i], cmap=cmap, edgecolors='black', 
                    marker=markers[i], alpha=alphas[i])
    plt.legend()
    plt.show()


def plot_model_roc_curve(clf, features, true_labels, label_encoder=None, class_names=None):
    
    ## Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    if hasattr(clf, 'classes_'):
        class_labels = clf.classes_
    elif label_encoder:
        class_labels = label_encoder.classes_
    elif class_names:
        class_labels = class_names
    else:
        raise ValueError('Unable to derive prediction classes, please specify class_names!')
    n_classes = len(class_labels)
    y_test = label_binarize(true_labels, classes=class_labels)
    if n_classes == 2:
        if hasattr(clf, 'predict_proba'):
            prob = clf.predict_proba(features)
            y_score = prob[:, prob.shape[1]-1] 
        elif hasattr(clf, 'decision_function'):
            prob = clf.decision_function(features)
            y_score = prob[:, prob.shape[1]-1]
        else:
            raise AttributeError("Estimator doesn't have a probability or confidence scoring system!")
        
        fpr, tpr, _ = roc_curve(y_test, y_score)      
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC curve (area = {0:0.2f})'
                                 ''.format(roc_auc),
                 linewidth=2.5)
        
    elif n_classes > 2:
        if hasattr(clf, 'predict_proba'):
            y_score = clf.predict_proba(features)
        elif hasattr(clf, 'decision_function'):
            y_score = clf.decision_function(features)
        else:
            raise AttributeError("Estimator doesn't have a probability or confidence scoring system!")

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        ## Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        ## Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        ## Plot ROC curves
        plt.figure(figsize=(6, 4))
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]), linewidth=3)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]), linewidth=3)

        for i, label in enumerate(class_labels):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                           ''.format(label, roc_auc[i]), 
                     linewidth=2, linestyle=':')
    else:
        raise ValueError('Number of classes should be atleast 2 or more')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


# In[46]:


train_features.shape


# In[47]:


test_features.shape


# In[48]:


predictions = np.argmax(model.predict(test_features), axis=-1)

class_map = {'0' : 'anger', '1' : 'boredom', '2' : 'disgust', '3' : 'fear', '4' : 'happiness', 
                 '5' : 'neutral', '6' : 'sadness'}

test_labels_categories = [class_map[str(label)] for label in test_labels]
prediction_labels_categories = [class_map[str(label)] for label in predictions]
category_names = list(class_map.values())


# In[49]:


get_metrics(true_labels=test_labels_categories, 
               predicted_labels=prediction_labels_categories)


# In[50]:


display_classification_report(true_labels=test_labels_categories, 
                                        predicted_labels=prediction_labels_categories, 
                                        classes=category_names)


# In[51]:


from sklearn.metrics import confusion_matrix
C = confusion_matrix(test_labels_categories, prediction_labels_categories)
pd.options.display.float_format = "{:,.2f}".format

CF = pd.DataFrame(C / C.astype(np.float).sum(axis=1), columns=['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness'])*100
CF.index = ['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness']
CF


# In[52]:


def classwise_accuracy():
    a = pd.crosstab(np.array(test_labels_categories) , np.array(prediction_labels_categories))
    return a.max(axis=1)/a.sum(axis=1)
accuracy_per_class = classwise_accuracy()
accuracy_per_class


# In[53]:


dfs = pd.DataFrame([accuracy_per_class.values], columns=['Anger', 'Boredom', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness'], 
                   index = ['2D-log-MelSpectrogram-Avg-harmonic-percussive'])
dfs.to_csv('accuracy_per_class.csv', mode='a', header=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




