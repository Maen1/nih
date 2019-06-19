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

# deasises = ['Hernia', 'Pneumonia', 'Fibrosis', 'Edema', 'Emphysema', 'Cardiomegaly',
#         'Pleural_Thickening','Consolidation', 'Pneumothorax', 'Mass', 'Nodule', 
#         'Atelectasis', 'Effusion', 'Infiltration']

# work on 70 percent of the dataset
df_sample = dataframe.sample(frac = 0.70)
deasises = list(dataframe["Finding Labels"].unique())

#train data set
df_sample_train = df_sample.sample(frac = 0.10)
# isolated for the test
df_sample_test = dataframe.drop(df_sample.index)


df_sample = df_sample.drop(df_sample_train.index)
for i in df_sample_train:
        df_sample_train.at[i, 'Finding Labels'] = random.choice(deasises)
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

from keras.models import load_model

model = load_model('my_model.h5')

socre, acc = model.evalute(X_test, y_test, batch_size=64)

print('Score', socre)
print('Accuracy', acc)