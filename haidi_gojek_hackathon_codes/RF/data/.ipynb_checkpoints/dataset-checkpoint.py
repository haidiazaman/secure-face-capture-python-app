import os
import os.path
import torch
import torch.utils.data as data
import cv2
import numpy as np
import pandas as pd

# Dataloader for vol_3
class FaceDetection(data.Dataset):
    def __init__(self, label_folder_path, label_files ,preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        print("All label files:", label_files)
        for label_file in label_files:
            self.parse_label_file(label_folder_path+label_file)
        print(f'Total images in all datasets:{len(self.imgs_path)}')

    def parse_label_file(self, txt_path):
        print("Current label file:",txt_path)
        assert os.path.exists(txt_path) == True, "Invalid label path!"
        counter = 0
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                counter += 1
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]

                # path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                if line:
                    label = [float(x) for x in line]
                else:
                    label = []
                labels.append(label)
        self.words.append(labels)
        print(f'\tTotal images in this dataset:{counter}')

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations

        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[6]    # l1_x
            annotation[0, 7] = label[7]    # l1_y
            annotation[0, 8] = label[8]   # l2_x
            annotation[0, 9] = label[9]   # l2_y
            annotation[0, 10] = label[10]  # l3_x
            annotation[0, 11] = label[11]  # l3_y
            annotation[0, 12] = label[12]  # l4_x
            annotation[0, 13] = label[13]  # l4_y
            # if (annotation[0, 4]<0 ):
            if np.any(annotation[0, 4:13] < 0):  # if 1 of the lmarks doesnot exist -> not using for llmark_loss
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)


        return torch.from_numpy(img), target

# Dataloader for vol_1/liveness/selfie_images
class FaceDetection_vol1(data.Dataset):
    def __init__(self, label_folder_path, label_files, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        print("All label files:", label_files)
        for label_file in label_files:
            self.parse_label_file(label_folder_path+label_file)
        print(f'Total images in all datasets:{len(self.imgs_path)}')

    def parse_label_file(self, csv_path):
        print("Current label file:", csv_path)
        assert os.path.exists(csv_path) == True, "Invalid label path!"

        df = pd.read_csv(csv_path)

        for idx, row in df.iterrows():
            img_path = row['image_paths']
            self.imgs_path.append(img_path)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        return torch.from_numpy(img), self.imgs_path[index]
