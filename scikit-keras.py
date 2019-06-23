import os
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.utils import np_utils

import sklearn.metrics as metrics
from keras.models import Sequential
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from skmultilearn.problem_transform import LabelPowerset


from itertools import chain
from glob import iglob, glob


from keras.models import Sequential
from keras.layers import GaussianNoise
from keras.utils import multi_gpu_model
from keras.applications.xception import Xception
from keras.layers import Dropout, GlobalAveragePooling2D, Dense, Dropout, Flatten

all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('../nih_sample/','images', '*.png'))}

# lables 
dataframe = pd.read_csv('../nih_sample/sample_labels.csv')

dataframe['path'] = dataframe['Image Index'].map(all_image_paths.get)
dataframe['Patient Age'] = dataframe['Patient Age'].map(lambda x: int(x[:-1]))
dataframe = dataframe[dataframe['Finding Labels'] != 'No Finding']
all_labels = np.unique(list(chain(*dataframe['Finding Labels'].map(lambda x: x.split('|')).tolist())))
pathology_list = all_labels
dataframe['path'] = dataframe['Image Index'].map(all_image_paths.get)
dataframe = dataframe.drop(['Patient Age', 'Patient Gender', 'Follow-up #', 'Patient ID', 'View Position', 
        'OriginalImageWidth', 'OriginalImageHeight', 'OriginalImagePixelSpacing_x','OriginalImagePixelSpacing_y'], axis=1)


for pathology in pathology_list:
    dataframe[pathology] = dataframe['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)
dataframe = dataframe.drop(['Image Index', 'Finding Labels'], axis=1)

dataframe['disease_vec'] = dataframe.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])
## split the data

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(dataframe, 
                                   test_size = 0.30, 
                                   random_state = 2018)

X_train = train_df['path'].values.tolist()
y_train = np.asarray(train_df['disease_vec'].values.tolist())
X_test = test_df['path'].values.tolist()
y_test = np.asarray(test_df['disease_vec'].values.tolist())

# print(X_train)
from skimage.io import imread, imshow

print(imread(X_train[0]).shape)
images_train = np.zeros([len(X_train),128,128])
for i, x in enumerate(X_train):
    image = imread(x, as_gray=True)[::8,::8]
    images_train[i] = (image - image.min())/(image.max() - image.min())
images_test = np.zeros([len(X_test),128,128])
for i, x in enumerate(X_test):
    image = imread(x, as_gray=True)[::8,::8]
    images_test[i] = (image - image.min())/(image.max() - image.min())

X_train = images_train.reshape(len(X_train), 128, 128, 1)
X_test = images_test.reshape(len(X_test), 128, 128, 1)
X_train.astype('float32')

def create_model_multiclass():
    base_model = Xception(input_shape = (128, 128, 1), include_top = False, weights = None)
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.3))
    model.add(Dense(512))
    model.add(Dropout(0.3))
    model.add(Dense(len(all_labels), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

clf = LabelPowerset(KerasClassifier(create_model_multiclass, epochs=200, batch_size=5, verbose=1))
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred, normalize=False))


