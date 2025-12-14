import cv2
import numpy as np
from imblearn.over_sampling import SMOTE
from PIL import Image
import pydicom

def normalize_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img

def handle_dicom(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array
    return Image.fromarray(img)

def augment_image(img):
    # Rotation, flip, etc.
    img = np.rot90(img, k=np.random.randint(4))
    if np.random.rand() > 0.5:
        img = np.fliplr(img)
    return img

def balance_classes(X, y):
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X.reshape(len(X), -1), y)
    return X_res.reshape(-1, *X.shape[1:]), y_res
