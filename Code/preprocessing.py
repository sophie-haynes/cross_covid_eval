import cv2
import glob
import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np


def get_train_test_image_folders(root_path):
    """Recursively search for the first folder beginning with 'N' that contains image files in 'jp*' or 'png' format.
    If a folder named 'train' or 'test' is encountered, the search is performed in both.
    Returns the path to the folder containing the images.
    """
    image_folder_path = None

    for dirpath, dirnames, filenames in os.walk(root_path):
        # Check if 'train' or 'test' folders are present and search inside them as well
        if 'train' in dirnames:
            image_folder_path = get_train_test_image_folders(os.path.join(dirpath, 'train'))

            if image_folder_path:
                break
        # Assume that there is no train/test split
        else:
            # Check if any of the subdirectories start with 'N'
            for dirname in dirnames:
                if dirname.lower().startswith('n'):
                    # Check if the folder contains image files
                    for filename in filenames:
                        if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg') or filename.lower().endswith('.png'):
                            image_folder_path = os.path.join(dirpath, dirname)
                            break

                    if image_folder_path:
                        break

            if image_folder_path:
                break
    return image_folder_path

# def get_test_image_folder(root_path):
#     """Recursively search for the first folder beginning with 'N' that contains image files in 'jp*' or 'png' format.
#     If a folder named 'train' or 'test' is encountered, the search is performed in both.
#     Returns the path to the folder containing the images.
#     """
#     image_folder_path = None

#     for dirpath, dirnames, filenames in os.walk(root_path):
#         # Check if 'train' or 'test' folders are present and search inside them as well
#         if 'test' in dirnames:
#             image_folder_path = find_image_folder(os.path.join(dirpath, 'test'))

#             if image_folder_path:
#                 break
#         # Assume that there is no train/test split
#         else:
#             # Check if any of the subdirectories start with 'N'
#             for dirname in dirnames:
#                 if dirname.lower().startswith('n'):
#                     # Check if the folder contains image files
#                     for filename in filenames:
#                         if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg') or filename.lower().endswith('.png'):
#                             image_folder_path = os.path.join(dirpath, dirname)
#                             break

#                     if image_folder_path:
#                         break

#             if image_folder_path:
#                 break
#     return image_folder_path

def histogram_equalise(img_arr):
    b,g,r = cv2.split(img_arr)
    bg = cv2.equalizeHist(b)
    gg = cv2.equalizeHist(g)
    rg = cv2.equalizeHist(r)
    img_arr = cv2.merge((bg,gg,rg))
    return img_arr

def resize(img_arr, img_res):
    img_arr = cv2.resize(img_arr, (img_res, img_res))
    return img_arr

def calculate_weights(path_to_train_covid,path_to_train_non):
    len_covid = len(glob.glob(path_to_train_covid+'*.*'))
    len_non = len(glob.glob(path_to_train_non+'*.*'))

    print("Number of COVID samples: {}".format(len_covid))
    print("Number of Non-COVID samples : {}".format(len_non))

    num_normal = len_non
    num_positive = len_covid

    weight_normal = 1/num_normal * ((len_non+len_covid)/2.0)
    weight_positive = 1/num_positive * ((len_non+len_covid)/2.0)

    class_weights = {0:weight_normal, 1: weight_positive}
    return class_weights 

def oversample(path_to_train_covid,path_to_train_non, sample_ratio=1):
    len_non = len(glob.glob(path_to_train_non+'*.[jp][pn]g'))
    len_covid = len(glob.glob(path_to_train_covid+'*.[jp][pn]g'))
    

    # check if negative class contains multiple negative types
    if(len_non)==0:
        # handle multi neg
        # datasets only have one layer of children if class has sub-types, therefore hardcoding wildcard for one layer will work
        path_to_train_non+="*/"
        len_non = len(glob.glob(path_to_train_non+'*.[jp][pn]g'))
    # check if positive class contains multiple positive types
    if(len_covid)==0:
        # handle multi pos
        # datasets only have one layer of children if class has sub-types, therefore hardcoding wildcard for one layer will work
        path_to_train_covid+="*/"
        len_covid = len(glob.glob(path_to_train_covid+'*.[jp][pn]g'))


    print("Number of COVID samples before oversample: {}".format(len_covid))
    print("Number of Non-COVID samples before oversample: {}".format(len_non))

    minority_class_path, majority_class_path = (path_to_train_covid,path_to_train_non) if len_covid < len_non else (path_to_train_non,path_to_train_covid)

    i = 0

    while (len(glob.glob(os.path.join(minority_class_path, "*.[jp][pn]g")))/len(glob.glob(os.path.join(majority_class_path, "*.[jp][pn]g"))))<1:
        i+=1
        for file in glob.glob(os.path.join(minority_class_path, "*.[jp][pn]g")):
            shutil.copy(file, os.path.join(minority_class_path, "dupl-{}_".format(i) + os.path.basename(file)))

    print("Number of COVID samples after oversample: {}".format(len(glob.glob(path_to_train_covid+'*.[jp][pn]g'))))
    print("Number of Non-COVID samples after oversample: {}".format(len(glob.glob(path_to_train_non+'*.[jp][pn]g'))))


def split_into_train_test(path_to_data, train_sample_size=0.6,random_state=3):
    from sklearn.model_selection import train_test_split

    labels = [d for d in next(os.walk(path_to_data))][1]
    neg_folder = next(lab for lab in labels if lab.lower().startswith('n') == True)
    pos_folder = next(lab for lab in labels if lab.lower().startswith('n') == False)

    neg_path = path_to_data+"{}/".format(neg_folder)
    pos_path = path_to_data+"{}/".format(pos_folder)

    len_non = len(glob.glob(neg_path+'*.[jp][pn]g'))
    len_covid = len(glob.glob(pos_path+'*.[jp][pn]g'))

    # check if negative class contains multiple negative types
    if(len_non)==0:
        # handle multi neg
        # datasets only have one layer of children if class has sub-types, therefore hardcoding wildcard for one layer will work
        neg_path+="*/"
        len_non = len(glob.glob(neg_path+'*.[jp][pn]g'))
    # check if positive class contains multiple positive types
    if(len_covid)==0:
        # handle multi pos
        # datasets only have one layer of children if class has sub-types, therefore hardcoding wildcard for one layer will work
        pos_path+="*/"
        len_covid = len(glob.glob(pos_path+'*.[jp][pn]g'))

    neg_labels = [0]*len_non
    pos_labels = [1]*len_covid
    all_labels =[*neg_labels,*pos_labels]
    all_paths = [*glob.glob(neg_path+'*.[jp][pn]g'),*glob.glob(pos_path+'*.[jp][pn]g')]
    train_paths, test_paths, train_labels, test_labels = train_test_split(all_paths, all_labels, 
                                                                        test_size=1-train_sample_size, 
                                                                        stratify=all_labels, 
                                                                        random_state=random_state)
    return train_paths, test_paths, train_labels, test_labels

def find_root_folder(passpath):
  # check if train/test
  if any(s.lower() == "train" for s in os.listdir(passpath)):
    passpath+="/"
    return passpath
  else:
    for subdir in os.listdir(passpath):
      if any(s.lower() == "train" for s in os.path.join(passpath, subdir)):
        passpath+="/"
        return passpath
    # found_negative = False
    for folder in os.listdir(passpath):
      if folder.lower() == "non" or folder.lower()=="n":
        passpath+="/"
        return passpath

        # found_negative = True
    # if not found_negative:
    for folder in os.listdir(passpath):
      for subfolder in os.listdir(os.path.join(passpath,folder)):
        if subfolder.lower() == "non" or subfolder.lower()=="n":
          # found_negative = True
          return os.path.join(passpath,folder)+"/"
        elif subfolder.lower() == "train":
          return os.path.join(passpath,folder)+"/"
    return None

def check_if_train_test_split(path_to_data):
    return any(s.lower() == "train" for s in os.listdir(path_to_data))



def get_train_test_paths_labels(path_to_data):
    train_test_folders = [d for d in next(os.walk(path_to_data))][1]
    train_folder = next(fold for fold in train_test_folders if fold.lower() == "train")
    test_folder = next(fold for fold in train_test_folders if fold.lower() == "test")

    train_labels = [d for d in next(os.walk(path_to_data+"{}/".format(train_folder)))][1]
    train_neg_folder = next(lab for lab in train_labels if lab.lower().startswith('n') == True)
    train_pos_folder = next(lab for lab in train_labels if lab.lower().startswith('n') == False)

    train_neg_path = path_to_data+"{}/{}/".format(train_folder,train_neg_folder)
    train_pos_path = path_to_data+"{}/{}/".format(train_folder,train_pos_folder)

    train_len_non = len(glob.glob(train_neg_path+'*.[jp][pn]g'))
    train_len_covid = len(glob.glob(train_pos_path+'*.[jp][pn]g'))

    if(train_len_non)==0:
        # handle multi neg
        # datasets only have one layer of children if class has sub-types, therefore hardcoding wildcard for one layer will work
        train_neg_path+="*/"
        train_len_non = len(glob.glob(train_neg_path+'*.[jp][pn]g'))
    # check if positive class contains multiple positive types
    if(train_len_covid)==0:
        # handle multi pos
        # datasets only have one layer of children if class has sub-types, therefore hardcoding wildcard for one layer will work
        train_pos_path+="*/"
        train_len_covid = len(glob.glob(train_pos_path+'*.[jp][pn]g'))

    train_neg_labels = [0]*train_len_non
    train_pos_labels = [1]*train_len_covid
    train_all_labels =[*train_neg_labels,*train_pos_labels]
    train_all_paths = [*glob.glob(train_neg_path+'*.[jp][pn]g'),*glob.glob(train_pos_path+'*.[jp][pn]g')]    



    test_labels = [d for d in next(os.walk(path_to_data+"{}/".format(test_folder)))][1]
    test_neg_folder = next(lab for lab in test_labels if lab.lower().startswith('n') == True)
    test_pos_folder = next(lab for lab in test_labels if lab.lower().startswith('n') == False)

    test_neg_path = path_to_data+"{}/{}/".format(test_folder,test_neg_folder)
    test_pos_path = path_to_data+"{}/{}/".format(test_folder,test_pos_folder)

    test_len_non = len(glob.glob(test_neg_path+'*.[jp][pn]g'))
    test_len_covid = len(glob.glob(test_pos_path+'*.[jp][pn]g'))

    if(test_len_non)==0:
        # handle multi neg
        # datasets only have one layer of children if class has sub-types, therefore hardcoding wildcard for one layer will work
        test_neg_path+="*/"
        test_len_non = len(glob.glob(test_neg_path+'*.[jp][pn]g'))
    # check if positive class contains multiple positive types
    if(train_len_covid)==0:
        # handle multi pos
        # datasets only have one layer of children if class has sub-types, therefore hardcoding wildcard for one layer will work
        test_pos_path+="*/"
        test_len_covid = len(glob.glob(test_pos_path+'*.[jp][pn]g'))

    test_neg_labels = [0]*test_len_non
    test_pos_labels = [1]*test_len_covid
    test_all_labels =[*test_neg_labels,*test_pos_labels]
    test_all_paths = [*glob.glob(test_neg_path+'*.[jp][pn]g'),*glob.glob(test_pos_path+'*.[jp][pn]g')]

    return train_all_paths, test_all_paths, train_all_labels, test_all_labels

    def load_data(partition_data_path, img_res, histogram_equalise):
        data = []
        labels = []
        img_paths = []

        labels = [d for d in next(os.walk(partition_data_path))[1]]

        for label in labellist:
            if label == "non":
                labelint = 0
            else:
                labelint = 1
            image_list = glob.glob('{}/{}/*'.format(partition_data_path,label))

            for img in image_list:
                img_arr = cv2.imread(img)
                if histogram_equalise:
                    b,g,r = cv2.split(img_arr)
                    beq = cv2.equalizeHist(b)
                    geq = cv2.equalizeHist(g)
                    req = cv2.equalizeHist(r)
                    img_arr = cv2.merge((beq,geq,req))
                img_arr = cv2.resize(img_arr,(img_res,img_res))
                img_arr = img_arr/255.0 # normalise
                data.append(img_arr)
                labels.append(labelint)
                img_paths.append(img)
        return np.array(data),np.array(labels),np.array(img_paths)

        
