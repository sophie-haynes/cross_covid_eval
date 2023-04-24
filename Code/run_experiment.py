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

from preprocessing import calculate_weights,load_data, oversample

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


def run_experiment(architecture, img_res=224,learning_rate=0.001,momentum=0.9,epochs=50,batch_size=32,weights=None,ds1=False, dataset1=False, dataset2=False, dataset3=False, dataset4=False,class_weight=False,oversample=False,hist_eq=False):
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

    if weights:
        if architecture != "densenet":
            raise ValueError("Invalid architecture for weights! Only densenet has weight support. Please remove flag and try again.")
        else:
            if weights == "imagenet" or weights == "cxr8" or weights == "wehbe":
                # valid weights
                pass
            else:
                raise ValueError("Invalid weights! '{}' is not a valid value for weights. Please use 'imagenet', 'cxr8' or 'wehbe'.".format(weights))

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
    # TEMP
    temp_flatten = True
    # fetch training data into environment
    fetch_data(train_dataset,set_train_data_path(train_dataset,temp_flatten))

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
            # create oversampled data - a sample ratio of 1 => 1(minority):1(majority), sample ratio of 2 => 1(minority):2(majority)
            oversample(os.path.join(data_path,"train","covid"),os.path.join(data_path,"train","non"), sample_ratio=1)
        if class_weight:
            # calculate weights
            class_weight_value = calculate_weights(path_to_train_covid,path_to_train_non)
        else:
            class_weight_value=None

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
