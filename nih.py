import os
from glob import *
import pandas as pd

# path
PATH = os.path.abspath(os.path.join('../','nih_sample/'))
SOURCE_IMAGES = os.path.join(PATH,'images')
images = glob(os.path.join(SOURCE_IMAGES,'*.png'))
print(images[0:3])

# lables 

lables = pd.read_csv('../nih_sample/sample_labels.csv')
print(lables.head(3))

