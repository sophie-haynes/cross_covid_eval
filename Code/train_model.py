import argparse
import numpy as np
np.random.seed(3)
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={"figure.figsize": (6, 4), "figure.dpi": 150})
sns.set_style("white")
import csv
import os
import shutil
import sys
import tarfile
import zipfile
import datetime

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

from preprocessing import *

# ===== TO-DO ============================
# 
# > TF model saving
#   - Saving dir
#   - verify drive saving
#   - save predictions/labels (pickle?)
#   - save structure (models, results, experiment details )
# > Torch code
#   - data loading
#       - flatten for compatibility
#       > minaee with augmentor
#           - install augmentor
#           
#           - aug as specified
#       > darkcovidnet
#           - install fastai, torch, torchvision (specific versions)
#           - 
#   > model loading
#       > minaee
#           - clone git
#       > darkcovid
#           - define model functions (models.py)
#           - define sequential model (train_model.py)
# > Evaluation code 
#   - eval_model.py
#   > params:
#       - model_name: model to load
#       - 
# ========================================================================================================

# helper function to get data_path
def set_train_data_path(train_dataset, flatten=False):
    if flatten:
        if train_dataset == "1":
            train_data_path = "/content/drive/MyDrive/Paper3Logging/Datasets/dataset_1_flattened.zip"
        elif train_dataset == "2":
            train_data_path = "/content/drive/MyDrive/Paper3Logging/Datasets/dataset_2_flattened.zip"
        elif train_dataset == "3":
            train_data_path = "/content/drive/MyDrive/Paper3Logging/Datasets/dataset_3_flattened.zip"
        elif train_dataset == "4":
            train_data_path = "/content/drive/MyDrive/Paper3Logging/Datasets/dataset_4.zip"
        else:
            raise ValueError("Invalid train dataset value")
        return train_data_path
    else:
        if train_dataset == "1":
            train_data_path = "/content/drive/MyDrive/Paper3Logging/Datasets/dataset_1.zip"
        elif train_dataset == "2":
            train_data_path = "/content/drive/MyDrive/Paper3Logging/Datasets/dataset_2.zip"
        elif train_dataset == "3":
            train_data_path = "/content/drive/MyDrive/Paper3Logging/Datasets/dataset_3.zip"
        elif train_dataset == "4":
            train_data_path = "/content/drive/MyDrive/Paper3Logging/Datasets/dataset_4.zip"
        else:
            raise ValueError("Invalid train dataset value")
        return train_data_path

def fetch_data(train_dataset,train_data_path):
    # name for local dir
    filename = "dataset_{}".format(train_dataset)
    # Create directory for extracted files
    extract_dir = os.path.join(os.getcwd(), filename)
    os.makedirs(extract_dir, exist_ok=True)
    # uncompress files
    if train_data_path.endswith('.zip'):
        with zipfile.ZipFile(train_data_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif train_data_path.endswith('.tar'):
        with tarfile.open(train_data_path, 'r') as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        raise ValueError('Train data file type not recognized. Please use .zip or .tar files.')
    print('Data fetched and extracted successfully.')

def get_neg_pos_class_paths(path_to_data):
    labels = [d for d in next(os.walk(train_path))[1]]
    neg_path =  os.path.join(path_to_data,os.next(lab for lab in labels if lab.lower().startswith('n') == True))
    pos_path = os.path.join(path_to_data,next(lab for lab in labels if lab.lower().startswith('n') == False))
    return neg_path, pos_path


def write_model_meta_to_csv():
    try:
        with open("test.csv","a") as f:
            writer = csv.writer(f,delimiter=',')
            writer.writerow(['architecture','train_dataset','img_res','lr', 'momentum','epochs','batch_size','weights',\
                'class_weight','oversample','hist_eq'])

# ========================================================================================================

# Set up base parameters
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-a', '--architecture', help='Architecture: \\Must be of type "densenet", "conv4", "darkcovidnet" or "minaee_resnet".', required=True)
parser.add_argument('-res', '--img_res', help='Image resolution', default=224, type=int)
parser.add_argument('-lr', '--learning_rate', help='Learning rate', default=0.001, type=float)
parser.add_argument('-m', '--momentum', help='Momentum', default=0.9, type=float)
parser.add_argument('-e', '--epochs', help='Epochs', default=50, type=int)
parser.add_argument('-b', '--batch_size', help='Batch size', default=32, type=int)
parser.add_argument('-w', '--weights', help='Weights: \\Only available for densenet models. Must be of type "imagenet", "cxr8" or "wehbe"', default=None)
parser.add_argument('-ds1', '--dataset1', help='Use Dataset 1 - Minaee Covid', action='store_true', default=False)
parser.add_argument('-ds2', '--dataset2', help='Use Dataset 2 - Cohen + Brax', action='store_true', default=False)
parser.add_argument('-ds3', '--dataset3', help='Use Dataset 3 - SIIM', action='store_true', default=False)
parser.add_argument('-ds4', '--dataset4', help='Use Dataset 4 - COVIDGR', action='store_true', default=False)
parser.add_argument('-cw','--class_weight', help='Use class weighting', action='store_true', default=False)
parser.add_argument('-ov','--oversample', help='Use oversampling', action='store_true', default=False)
parser.add_argument('-he','--hist_eq', help='Use histogram equalization', action='store_true', default=False)
args = parser.parse_args()

# ========================================================================================================
# parse data param

train_dataset = None
if args.dataset1:
    train_dataset = "1"
elif args.dataset2:
    train_dataset = "2"
elif args.dataset3:
    train_dataset = "3"
elif args.dataset4:
    train_dataset = "4"
else:
    raise ValueError("No train dataset selected")

# verify valid weights are passed in
if args.weights:
    if args.architecture != "densenet":
        raise ValueError("Invalid architecture for weights! Only densenet has weight support. Please remove flag and try again.")
    else:
        if args.weights == "imagenet" or args.weights == "cxr8" or args.weights == "wehbe":
            # valid weights
            pass
        else:
            raise ValueError("Invalid weights! '{}' is not a valid value for weights. Please use 'imagenet', 'cxr8' or 'wehbe'.".format(args.weights))

# set env params from argparse
architecture = args.architecture
img_res = args.img_res
learning_rate = args.learning_rate
momentum = args.momentum
epochs = args.epochs
batch_size = args.batch_size
weights = args.weights
class_weight = args.class_weight
oversample = args.oversample
hist_eq = args.hist_eq

augmented = False


model_name = f"{architecture}_{epochs}_{train_dataset}"

# create model name
if weights is not None:
    model_name += f'_{weights}'
if class_weight:
    model_name += '_CW'
if oversample:
    model_name += '_OV'
if hist_eq:
    model_name += '_HE'

model_name = model_name.replace(" ", "_").replace("-", "_")

# ========================================================================================================
# TEMP
temp_flatten = True
# fetch training data into environment
fetch_data(train_dataset,set_train_data_path(train_dataset,temp_flatten))

# only internal data has train/test split
# ====== UPDATE! DATASETS HAVE BEEN PREPROCESSED IN NOTEBOOK IN PAPER3LOGGING NOTEBOOK====
# data_path = find_root_folder(train_dataset)

# if not check_if_train_test_split:
#     train_path,test_path,train_label,test_label = split_into_train_test(data_path)
# else:
#     train_path,test_path,train_label,test_label = get_train_test_paths_labels(data_path)

# if oversample and architecture != "minaee_resnet":
#     # only run oversampling if not minaee ~ uses augmentor package
#     train_non_path, train_covid_path = get_neg_pos_class_paths(train_path)
#     oversample(path_to_train_covid,path_to_train_non, sample_ratio=1)
data_path = "dataset_{}".format(train_dataset)
# ========================================================================================================
# modelling 

if architecture == "darkcovidnet":
    # Use PyTorch code
    import warnings
    # install packages
    warnings.warn("This architecture installs specific versions of torch, torchvision and fastai. \
        This may impact other Torch-based architectures if they are run after this command.")
    print("Checking dependencies for this architecture are installed...")
    packages_to_install = ["Augmentor", "torch"]
    for package_i in packages_to_install:
        packages = subprocess.run(["pip","list"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,)
        if package_i not in packages.stdout.decode("utf-8"):
            print("{} not installed! Attempting install...".format(package_i))
            os.system("pip install {}".format(package_i))
        else:
            print("{} already installed :)".format(package_i))
    

    augmented = True
    # raise Exception("not implemented yet...")
#     import torch
#     from torch import nn, optim
#     from torch.optim import lr_scheduler
#     import torchvision
#     from torchvision import datasets, transforms, models
    
#     # Add PyTorch-specific code here
#     # ...
elif architecture == "minaee_resnet":
    # check packages are installed
    print("Checking dependencies for this architecture are installed...")
    packages_to_install = ["Augmentor", "torch"]
    for package_i in packages_to_install:
        packages = subprocess.run(["pip","list"],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,)
        if package_i not in packages.stdout.decode("utf-8"):
            print("{} not installed! Attempting install...".format(package_i))
            os.system("pip install {}".format(package_i))
        else:
            print("{} already installed :)".format(package_i))
    print("Cloning repo...")
    os.system("pip install https://github.com/shervinmin/DeepCovid.git")
    augmented = True
    # prepare data for torch model

    for i in range(3):
        os.system("python DeepCovid/ResNet18_train.py --dataset_path")
        # !python DeepCovid/ResNet18_train.py --dataset_path /content/flattened_internal_dataset/ --batch_size 20 --epoch 50 --num_workers 4 --learning_rate 0.0001

else:
    # Use TensorFlow code
    import tensorflow as tf
    ## GLOBAL SEED ##                                                   
    tf.random.set_seed(3)
    from tensorflow import keras
    from keras import layers
    from keras import models
    from keras.callbacks import EarlyStopping
    from keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.preprocessing import image_dataset_from_directory

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # load data into env
    if oversample:
        # create oversampled data
    if class_weight:
        # calculate weights

    train_x, train_y, train_paths = load_data(os.path.join(data_path,"train"),img_res,hist_eq)
    test_x,test_y, test_paths = load_data(os.path.join(data_path,"test"),img_res,hist_eq)
    
    # Create folder for model/result storage
    os.makedirs("/content/drive/MyDrive/Paper3Logging/Models/{}".format(model_name), exist_ok=True)
    # create preds folder
    # os.makedirs("/content/drive/MyDrive/Paper3Logging/Models/{}/Predictions".format(model_name), exist_ok=True)

    for i in range(3):
        if architecture == "conv4":
            from models import conv4_tf
            model, es = conv4_tf(img_res,learning_rate, momentum)
        elif architecture == "densenet":
            from models import densenet_tf
            model, es = densenet_tf(img_res,learning_rate, momentum,weights)        
        else:
            raise ValueError("Invalid architecture! {} is not recognized as a valid model.".format(architecture))

        if class_weight_value:
            model.fit(train_x,train_y, validation_data=(test_x,test_y), epochs=epochs, batch_size=batch_size, callbacks=[es],class_weight=class_weight_value)
        else:
            model.fit(train_x,train_y, validation_data=(test_x,test_y), epochs=epochs, batch_size=batch_size, callbacks=[es])
        model.save("drive/MyDrive/Paper3Logging/Models/{}/{}_{}.h5".format(model_name,model_name, i+1))
        print("drive/MyDrive/Paper3Logging/Models/{}/{}_{}.h5".format(model_name,model_name, i+1))

        # export metadata
        if os.path.isfile("/content/drive/MyDrive/Paper3Logging/Models/models_meta.csv"):
          with open('/content/drive/MyDrive/Paper3Logging/Models/models_meta.csv', 'a') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(["{}_{}.h5".format(model_name, i+1),img_res, learning_rate,\
                                  momentum, epochs, batch_size,  \
                                  weights, data_path, class_weight,\
                                  oversample, augmented, hist_eq])
            
        else:
          # create new csv for storing model meta data
          with open('/content/drive/MyDrive/Paper3Logging/Models/models_meta.csv', 'a') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['Model_Name','Img_Res', 'Learning_Rate',\
                                 'Momentum', 'Epochs', 'Batch_Size',  \
                                 'Pre-training', 'Training_Data', 'Class_Weighted',\
                                 'Oversampled', 'Data_Aug', 'Hist_Eq'])
            filewriter.writerow(["{}_{}.h5".format(model_name, i+1),img_res, learning_rate,\
                                  momentum, epochs, batch_size,  \
                                  weights, data_path, class_weight,\
                                  oversample, augmented, hist_eq])

# print(f'Model name: {model_name}')
