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

# work on 50 percent of the dataset
df_sample = dataframe.sample(frac = 0.50, random_state = 1)
deasises = list(df_sample["Finding Labels"].unique())

#train data set
df_sample_train = df_sample.sample(frac = 0.55, random_state = 1)
# isolated for the test
df_sample_test = dataframe.drop(df_sample.index)


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


# print(len(df_sample))
# print(len(df_sample_test))
# print(len(df_sample_train))
# print(X_train)
from skimage.io import imread, imshow

# print(imread(X_train[0]).shape)
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

model = load_model('./models/nih_model_50_01.h5')

score, acc = model.evaluate(X_test, y_test, batch_size=64)
predictions = model.predict(X_test, batch_size = 64, verbose = True)
#sub_df = pd.DataFrame()
#sub_df["ImageId"] = list(range(1, num_testing + 1))
#sub_df["Label"] = pred
#sub_df.to_csv("nih_predictions.csv", header=True, index=False)
# print(len(predictions))

print('Score', score)
print('Accuracy', acc)

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

from sklearn.metrics import hamming_loss
print('Hamming Loss:', hamming_loss(y_true, y_pred))
class_names = all_labels
print(y_pred_y_true.tail(10))
#y_pred_y_true.to_csv("test_nih_predictions_50_55.csv", header=True, index=True)


