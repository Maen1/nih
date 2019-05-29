
import os
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
                   glob(os.path.join('../input/*','images*', '*.png'))}

# lables 
dataframe = pd.read_csv('../nih_sample/sample_labels.csv')
print(dataframe.shape)
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
