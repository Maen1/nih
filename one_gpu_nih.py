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
dataframe = pd.read_csv('./sample_labels.csv')

dataframe['path'] = dataframe['Image Index'].map(all_image_paths.get)
dataframe['Patient Age'] = dataframe['Patient Age'].map(lambda x: int(x[:-1]))
# dataframe = dataframe[dataframe['Finding Labels'] != 'No Finding']
all_labels = np.unique(list(chain(*dataframe['Finding Labels'].map(lambda x: x.split('|')).tolist())))
pathology_list = all_labels
print(pathology_list)

dataframe['path'] = dataframe['Image Index'].map(all_image_paths.get)
dataframe = dataframe.drop(['Patient Age', 'Patient Gender', 'Follow-up #', 'Patient ID', 'View Position', 
        'OriginalImageWidth', 'OriginalImageHeight', 'OriginalImagePixelSpacing_x','OriginalImagePixelSpacing_y'], axis=1)

# deasises = ['Hernia', 'Pneumonia', 'Fibrosis', 'Edema', 'Emphysema', 'Cardiomegaly',
#         'Pleural_Thickening','Consolidation', 'Pneumothorax', 'Mass', 'Nodule', 
#         'Atelectasis', 'Effusion', 'Infiltration']

# use half of the dataset 
df_half = dataframe.sample(frac = 1.0, random_state = 1)

# work on 70 percent of the dataset
df_sample = df_half.sample(frac = 0.3, random_state = 1)
deasises = list(df_sample["Finding Labels"].unique())

#train data set
df_sample_train = df_sample.sample(frac = 0.35, random_state = 1)
# isolated for the test
df_sample_test = df_half.drop(df_sample.index)


df_sample = df_sample.drop(df_sample_train.index)
df_sample_train.reset_index()
for i, row in df_sample_train.iterrows():
        row['Finding Labels'] = random.choice(deasises)

df_sample = df_sample.append(df_sample_train, ignore_index= True)
df_sample.drop(df_sample.tail(5).index, inplace=True)


for pathology in pathology_list:
    df_sample[pathology] = df_sample['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)
df_sample = df_sample.drop(['Image Index', 'Finding Labels'], axis=1)

df_sample['disease_vec'] = df_sample.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])

for pathology in pathology_list:
    df_sample_test[pathology] = df_sample_test['Finding Labels'].apply(lambda x: 1 if pathology in x else 0)
df_sample_test = df_sample_test.drop(['Image Index', 'Finding Labels'], axis=1)

df_sample_test['disease_vec'] = df_sample_test.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])


## split the data
from sklearn.model_selection import train_test_split

# train_df, test_df = train_test_split(dataframe, 
#                                    test_size = 0.20, 
#                                    random_state = 5)


X_train = df_sample['path'].values.tolist()
y_train = np.asarray(df_sample['disease_vec'].values.tolist())
X_test = df_sample_test['path'].values.tolist()
y_test = np.asarray(df_sample_test['disease_vec'].values.tolist())


print(len(df_sample))
print(len(df_sample_test))
print(len(df_sample_train))
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
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.layers import Dropout, GlobalAveragePooling2D, Dense, Dropout, Flatten
import keras.backend as K

def multitask_loss(y_true, y_pred):
    # Avoid divide by 0
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # Multi-task loss
    return K.mean(K.sum(- y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred), axis=1))


base_model = MobileNet(input_shape = (128, 128, 1), include_top = False, weights=None)
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Dropout(0.3))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(len(all_labels), activation='softmax'))
model.compile(loss='mean_squared_logarithmic_error', optimizer='adamax', metrics=['top_k_categorical_accuracy'])
model.summary()
#mean_squared_logarithmic_error

history = model.fit(X_train, y_train, epochs = 50, batch_size=64, verbose=1, validation_split=0.2 , shuffle=True)

model.save('../nih_sample/mobilenet_nih_model_all_30_35.h5')
def history_plot(history):
    plt.plot(history.history['top_k_categorical_accuracy'])
    plt.plot(history.history['val_top_k_categorical_accuracy'])
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
score, acc = model.evaluate(X_test, y_test, batch_size=64)
predictions = model.predict(X_test, batch_size = 64, verbose = True)
print('Score', score)
print('Accuracy', acc)

from sklearn.metrics import roc_curve, auc

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['top_k_categorical_accuracy'])
plt.plot(history.history['val_top_k_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./images/mobilenet_accuracy_all_30_35.png')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_top_k_categorical_accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./images/mobilenet_loss_all_30_35.png')


fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
for (idx, c_label) in enumerate(all_labels):
    fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), predictions[:,idx])
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('./images/mobilenet_trained_net_all_30_35.png')


sickest_idx = np.argsort(np.sum(y_test, 1)<1)
fig, m_axs = plt.subplots(4, 4, figsize = (16, 32))
for (idx, c_ax) in zip(sickest_idx, m_axs.flatten()):
    c_ax.imshow(X_test[idx, :,:,0], cmap = 'bone')
    stat_str = [n_class[:6] for n_class, n_score in zip(all_labels, y_test[idx]) if n_score>0.5]
    pred_str = ['%s:%2.0f%%' % (n_class[:4], p_score*100)  
    for n_class, n_score, p_score in zip(all_labels, y_test[idx], predictions[idx]) if (n_score>0.5) or (p_score>0.5)]
    c_ax.set_title('Dx: '+', '.join(stat_str)+'\nPDx: '+', '.join(pred_str))
    c_ax.axis('off')
fig.savefig('./images/mobilenet_trained_img_predictions_all_30_35.png')

sickest_idx = np.argsort(np.sum(y_test, 1)<.2)
y_true = []
y_pred = []
y_pred_y_true = pd.DataFrame(columns=['y_true', 'y_pred'] )
for (idx) in zip(sickest_idx):
    # c_ax.imshow(X_test[idx, :,:,0], cmap = 'bone')
    stat_str = [n_class[:] for n_class, n_score in zip(all_labels, y_test[idx]) if (n_score>0.5)]
    pred_str = ['%s ' % (n_class[:]) for n_class, n_score, p_score in zip(all_labels, y_test[idx], predictions[idx]) if (n_score>0.5) or (p_score>0.5)]
    strA = ' '.join(stat_str)
    strP = ' '.join(pred_str)
    y_true.append(strA)
    y_pred.append(strP)
#     print('y_true: '+', '.join(stat_str)+'\tPredected: '+', '.join(pred_str))


# print(y_true[0:10])
# print(Pred[0:10])
y_pred_y_true['y_true'] = y_true
y_pred_y_true['y_pred'] = y_pred

class_names = all_labels
print(y_pred_y_true.tail(10))
y_pred_y_true.to_csv("./csv/mobilenet_nih_predictions_all_30_35.csv", header=True, index=True)

 
