import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, CSVLogger, EarlyStopping
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose

from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D, Dense, Dropout, Activation
# from tensorflow.keras.layers.core import Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten, Lambda, ELU, LeakyReLU, GaussianDropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l2
# from tensorflow.keras.layers.noise import GaussianDropout
from tensorflow.keras.models import Model

from datetime import datetime
from tensorflow.keras.metrics import MeanIoU
import cv2
import os
from matplotlib import pyplot as plt
from tensorflow.keras.utils import normalize
from tensorflow.keras import backend as K


from PIL import Image
from Model import UNet1, UNet2, UNet3, UNetPlus1, UNetPlusPlus1
from Util import getDatarange, prepareData, showResults, getDatarange_multiclass, multi_class_preparation

output_path_single_class = "C:/Users/alike/Documents/Research_project/code/output/single_class/"
output_path_multi_class = "C:/Users/alike/Documents/Research_project/code/output/multi_class/"
preprocessing_output = "preprocessing/"
evaluation_output = "evaluation/"


hdf5File = h5py.File('C:/Users/alike/Documents/Research_project/code/data/hdf5_data_all.h5', 'r')

def train_sample():
    pass

def train(batch_size=16, start=0, end=30, n_classes=1, img_rows=256, img_cols=256):
    if n_classes == 1:
        return train_single_class(batch_size=batch_size, start=start, end=end, n_classes=1, img_rows=img_rows, img_cols=img_cols)
    else:
        return train_multiclass(batch_size=batch_size, start=start, end=end, n_classes=n_classes, img_rows=img_rows, img_cols=img_cols)
    
def train_single_class(batch_size=16, start=0, end=30, n_classes=1, img_rows=256, img_cols=256):

    model = UNet2(img_rows=img_rows, img_cols=img_cols ,num_class=1, multiclass=False)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy']) 
    model.summary()
    X, Y = getDatarange(hdf5File,start,end,aug=True, rows=img_rows, cols=img_cols)

    
    preprocessing_idxs = np.random.choice(X.shape[0], 9, replace=False)
    for i in range(preprocessing_idxs.shape[0]):
        cv2.imwrite(os.path.join(output_path_single_class+preprocessing_output, "Xpreprocessing_sample_stage_1_" + str(i) + ".jpeg"), X[preprocessing_idxs[i]])
        cv2.imwrite(os.path.join(output_path_single_class+preprocessing_output, "Ypreprocessing_sample_stage_1_" + str(i) + ".jpeg"), Y[preprocessing_idxs[i]])   

    trainX1, evalX, trainY1, evalY = train_test_split(X, Y, test_size=0.05, shuffle=True)
    trainX, valX, trainY, valY = train_test_split(trainX1, trainY1, test_size=0.2, shuffle=True)

    print("Train set size : ", len(trainX))
    print("Validation set size : ", len(valX))
    print("Total images : ", len(X))

    trainX, trainY, valX, valY = prepareData(trainX, trainY, valX, valY)

    csv_logger = CSVLogger('train'+str('')+'.log', append=True, separator=';')
    earlystopping = EarlyStopping(monitor='accuracy', verbose=1, min_delta=0.005, patience=3, mode='max')
    callbacks_list = [csv_logger, earlystopping]

    history = model.fit(trainX, trainY, epochs=10, batch_size=batch_size, verbose=1, validation_data=(valX, valY),callbacks=callbacks_list)

    model.save('model'+str('_single_class')+ str('UNet3') +'.hdf5')
    score = model.evaluate(evalX, evalY, verbose=1)

    print("Loss: {:.2f}".format(score[0]*100))
    print("Accuracy: {:.2f}".format(score[1]*100))

    evaluation_idxs = np.random.choice(valX.shape[0], 90, replace=False)
    for i in range(evaluation_idxs.shape[0]):
        cv2.imwrite(os.path.join(output_path_single_class+evaluation_output, "Xevaluation_sample_stage_1_" + str(i) + ".jpeg"), valX[evaluation_idxs[i]]*255)
        cv2.imwrite(os.path.join(output_path_single_class+evaluation_output, "Yevaluation_sample_stage_1_" + str(i) + ".jpeg"), valY[evaluation_idxs[i]]*255)
        cv2.imwrite(os.path.join(output_path_single_class+evaluation_output, "Ppreprocessing_sample_stage_1_" + str(i) + ".jpeg")
                    , (model.predict(np.expand_dims(valX[evaluation_idxs[i]], axis=0)))*255)
   

    showResults(history)
    return model

def train_multiclass(batch_size=16, start=0, end=3, n_classes=3, img_rows=256, img_cols=256):
    
    #os.mkdir(preprocessing_output)

    model = UNetPlusPlus1(num_class=n_classes, multiclass=True)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])  
    model.summary()
    
    X, Y = getDatarange_multiclass(hdf5File,start,end,aug=True)

    preprocessing_idxs = np.random.choice(X.shape[0], 9, replace=False)
    for i in range(preprocessing_idxs.shape[0]):
        cv2.imwrite(os.path.join(output_path_multi_class+preprocessing_output, "Xpreprocessing_sample_stage_1_" + str(i) + ".jpeg"), X[preprocessing_idxs[i]])
        cv2.imwrite(os.path.join(output_path_multi_class+preprocessing_output, "Ypreprocessing_sample_stage_1_" + str(i) + ".jpeg"), Y[preprocessing_idxs[i]])   

    print("X shape --> ", X.shape)
    print("Y shape --> ", Y.shape)

    trainX, valX, trainY, valY = multi_class_preparation(X, Y, test_size=0.2)


    print("Train set size : ", len(trainX))
    print("Validation set size : ", len(valX))
    print("Total images : ", len(X))
    
    csv_logger = CSVLogger('train'+str('')+'.log', append=True, separator=';')
    earlystopping = EarlyStopping(monitor='accuracy', verbose=1, min_delta=0.001, patience=3, mode='max')
    callbacks_list = [csv_logger, earlystopping]

    history = model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=1, validation_data=(valX, valY),callbacks=callbacks_list)

    model.save('model'+str('')+'.hdf5')
    score = model.evaluate(valX, valY,verbose=1)
    print("Loss: {:.2f}".format(score[0]*100))
    print("Accuracy: {:.2f}".format(score[1]*100))
    
    evaluation_idxs = np.random.choice(valX.shape[0], 90, replace=False)
    for i in range(evaluation_idxs.shape[0]):
        cv2.imwrite(os.path.join(output_path_multi_class+evaluation_output, "Xevaluation_sample_stage_1_" + str(i) + ".jpeg")
                    , valX[evaluation_idxs[i]]*255)
        cv2.imwrite(os.path.join(output_path_multi_class+evaluation_output, "Yevaluation_sample_stage_1_" + str(i) + ".jpeg")
                    , valY[evaluation_idxs[i]]*255)
        #print("Shape --> ", ((model.predict(np.expand_dims(valX[evaluation_idxs[i]], axis=0)))*255)[0,:,:,:].shape)
        cv2.imwrite(os.path.join(output_path_multi_class+evaluation_output, "Ppreprocessing_sample_stage_1_" + str(i) + ".jpeg")
                    , ((model.predict(np.expand_dims(valX[evaluation_idxs[i]], axis=0)))*255)[0,:,:,:])
   
    showResults(history)
    return model

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path,filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            imgi = cv2.resize(img,(256,256))
            images.append(imgi)
    return np.array(images)

def prediction(model=None, pre_path=None):
    folder_path = '/home/alikeshavarzi/deep_ct/Data/imgs'
    X = load_images_from_folder(folder_path)
    predict_images = np.expand_dims(X, axis=3)
    results = model.predict(predict_images)
    results.save('result'+str(i)+'.png')

def prediction_multiclass():
    pass