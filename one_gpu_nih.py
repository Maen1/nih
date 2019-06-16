import os
import random
import matplotlib
import numpy as np
import pandas as pd
from itertools import chain
from glob import iglob, glob


# %matplotlib inline
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# path
# PATH = os.path.abspath(os.path.join('../','nih_sample/'))
# SOURCE_IMAGES = os.path.join(PATH,'images')
# images = glob(os.path.join(SOURCE_IMAGES,'*.png'))
# another way to get files
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

deasises = ['Hernia', 'Pneumonia', 'Fibrosis', 'Edema', 'Emphysema', 'Cardiomegaly',
        'Pleural_Thickening','Consolidation', 'Pneumothorax', 'Mass', 'Nodule', 
        'Atelectasis', 'Effusion', 'Infiltration']
df_sample = dataframe.sample(frac = 0.1)

dataframe = dataframe.drop(df_sample.index)
print(len(dataframe))
for i in df_sample:
        df_sample.at[i, 'Finding Labels'] = random.choice(deasises)
dataframe = dataframe.append(df_sample, ignore_index= True)
dataframe.drop(dataframe.tail(3).index, inplace=True)

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
print(X_train[:3])

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

# model training
from keras.models import Sequential
from keras.layers import Input, GaussianNoise, Conv2D
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet201
from keras.layers import Dropout, GlobalAveragePooling2D, Dense, Dropout, Flatten
#base_model = Xception(input_shape = (128, 128, 1), include_top = False, weights = None)
base_model = Xception(include_top = False, weights=None)
model = Sequential()
model.add(Conv2D(3,(5, 5),input_shape = (128, 128, 1)))
model.add(GaussianNoise(0.7))
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Dropout(0.3))
model.add(GaussianNoise(0.7))
model.add(Dense(len(all_labels), activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


history = model.fit(X_train, y_train, epochs = 30, verbose=1, validation_data=(X_test, y_test))

model.save('../nih_sample/d121_nih_model.h5')
def history_plot(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# history_plot(history)

predictions = model.predict(X_test, batch_size = 32, verbose = True)

from sklearn.metrics import roc_curve, auc
fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(all_labels):
    fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), predictions[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('barely_trained_net.png')


sickest_idx = np.argsort(np.sum(y_test, 1)<1)
fig, m_axs = plt.subplots(4, 4, figsize = (16, 32))
for (idx, c_ax) in zip(sickest_idx, m_axs.flatten()):
    c_ax.imshow(X_test[idx, :,:,0], cmap = 'bone')
    stat_str = [n_class[:6] for n_class, n_score in zip(all_labels, 
                                                                  y_test[idx]) 
                             if n_score>0.5]
    pred_str = ['%s:%2.0f%%' % (n_class[:4], p_score*100)  for n_class, n_score, p_score in zip(all_labels, 
                                                                  y_test[idx], predictions[idx]) 
                             if (n_score>0.5) or (p_score>0.5)]
    c_ax.set_title('Dx: '+', '.join(stat_str)+'\nPDx: '+', '.join(pred_str))
    c_ax.axis('off')
fig.savefig('trained_img_predictions.png')