"""
-------------------->  Different Methods for creating fake Pathologies  <--------------------

Current Functions:
    - intersect(img, thresh)
    - add_noise(img, medval)
    - imagefinal(img,lung)
    - createCircle(img,nb)
    - createFakeImage(lung,mask,minval,maxval) --> This method is connected to method outside this file
"""

import h5py
import numpy as np
import PIL
import PIL.Image as pil
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm


from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

from keras.utils import normalize
from sklearn.utils import shuffle

from Pathology import createFakeImage
import matplotlib.pyplot as plt

gray_white_list = [0, 2, 4, 5, 6, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29]
white_gray_list = [1, 3, 7, 8, 11, 15, 25]
empty = [8]
corrupted = [15]

def getDatarange(hdf5File,startpat,endpat,aug=True, rows=128, cols=128):
	X = []
	Y = []
	patient_folders = list(hdf5File.keys())
	for pat in tqdm(range(startpat,endpat)):
		if pat == 15 or pat == 8:
			continue
		patient = patient_folders[pat]
		print(str(patient))
		
		Lung = hdf5File.get(str(patient) + '/LUNG')
		Mask = hdf5File.get(str(patient) + '/MASK')
		Lung_items = list(Lung.items())
		Mask_items = list(Mask.items())
		for i in tqdm(range(len(Mask_items)),desc='images'):
			Lung_img_name = Lung_items[i][1]
			Mask_img_name = Mask_items[i][1]
			Lung_imgs = np.array(Lung_img_name)
			Mask_imgs = np.array(Mask_img_name)

			if pat in white_gray_list:
				left_idxs = np.where(Mask_imgs==200)
				right_idxs = np.where(Mask_imgs==100)
				Mask_imgs[left_idxs] = 100
				Mask_imgs[right_idxs] = 200

			if Lung_imgs.shape != (rows,cols):
				Lung_ima = cv2.resize(Lung_imgs,(rows,cols))
			if Mask_imgs.shape != (rows,cols):
				Mask_ima = cv2.resize(Mask_imgs,(rows,cols))
			ret,thresh=cv2.threshold(Mask_ima,2,255,cv2.THRESH_BINARY)
			X.append(Lung_ima)
			Y.append(thresh)
			if aug==True:
				ret,thresh = cv2.threshold(Mask_ima,2,255,cv2.THRESH_BINARY)
				for i in range(2) :
					Fake_Lung_imgs = createFakeImage(Lung_ima, thresh,70,150, img_rows=rows, img_cols=cols)
					if i > 1:
						sig = (i -1)*0.5
						img = cv2.GaussianBlur(Fake_Lung_imgs,(5,5),sig)
						Fake_Lung_imgs = img
					X.append(Fake_Lung_imgs)
					Y.append(thresh)
				
				for i in range(2):
					Fake_Lung_imgs = createFakeImage(Lung_ima, thresh,100,200, img_rows=rows, img_cols=cols)
					if i > 1:
						sig = (i -1)*0.5
						img = cv2.GaussianBlur(Fake_Lung_imgs,(5,5),sig)
						Fake_Lung_imgs = img
					X.append(Fake_Lung_imgs)
					Y.append(thresh)
					
				for i in range(2):
					Fake_Lung_imgs = createFakeImage(Lung_ima, thresh,150,230, img_rows=rows, img_cols=cols)
					if i > 1:
						sig = (i -1)*0.5
						img = cv2.GaussianBlur(Fake_Lung_imgs,(5,5),sig)
						Fake_Lung_imgs = img
					X.append(Fake_Lung_imgs)
					Y.append(thresh)
					
				for i in range(2):
					Fake_Lung_imgs = createFakeImage(Lung_ima, thresh,200,240, img_rows=rows, img_cols=cols)
					if i > 1:
						sig = (i -1)*0.5
						img = cv2.GaussianBlur(Fake_Lung_imgs,(5,5),sig)
						Fake_Lung_imgs = img
					X.append(Fake_Lung_imgs)
					Y.append(thresh)
	return np.array(X), np.array(Y)


def getDatarange_multiclass(hdf5File,startpat,endpat,aug=True, rows=128, cols=128):
	X = []
	Y = []
	patient_folders = list(hdf5File.keys())
	for pat in tqdm(range(startpat,endpat)):
		patient = patient_folders[pat]
		print(str(patient))
		
		Lung = hdf5File.get(str(patient) + '/LUNG')
		Mask = hdf5File.get(str(patient) + '/MASK')
		Lung_items = list(Lung.items())
		Mask_items = list(Mask.items())
		for i in tqdm(range(len(Mask_items)),desc='images'):
			Lung_img_name = Lung_items[i][1]
			Mask_img_name = Mask_items[i][1]
			Lung_imgs = np.array(Lung_img_name)
			Mask_imgs = np.array(Mask_img_name)

			if pat in white_gray_list:
				left_idxs = np.where(Mask_imgs==200)
				right_idxs = np.where(Mask_imgs==100)
				Mask_imgs[left_idxs] = 100
				Mask_imgs[right_idxs] = 200
				
			if Lung_imgs.shape != (rows,cols):
				Lung_ima = cv2.resize(Lung_imgs,(rows,cols))
			if Mask_imgs.shape != (rows,cols):
				Mask_ima = cv2.resize(Mask_imgs,(rows,cols))
			ret,thresh=cv2.threshold(Mask_ima,2,255,cv2.THRESH_BINARY)
			X.append(Lung_ima)
			Y.append(Mask_ima)
			if aug==True:
				ret,thresh = cv2.threshold(Mask_ima,2,255,cv2.THRESH_BINARY)
				for i in range(2) :
					Fake_Lung_imgs = createFakeImage(Lung_ima, thresh,70,150, img_rows=rows, img_cols=cols)
					if i > 1:
						sig = (i -1)*0.5
						img = cv2.GaussianBlur(Fake_Lung_imgs,(5,5),sig)
						Fake_Lung_imgs = img
					X.append(Fake_Lung_imgs)
					Y.append(Mask_ima)
				
				for i in range(2):
					Fake_Lung_imgs = createFakeImage(Lung_ima, thresh,100,200, img_rows=rows, img_cols=cols)
					if i > 1:
						sig = (i -1)*0.5
						img = cv2.GaussianBlur(Fake_Lung_imgs,(5,5),sig)
						Fake_Lung_imgs = img
					X.append(Fake_Lung_imgs)
					Y.append(Mask_ima)
					
				for i in range(2):
					Fake_Lung_imgs = createFakeImage(Lung_ima, thresh,150,230, img_rows=rows, img_cols=cols)
					if i > 1:
						sig = (i -1)*0.5
						img = cv2.GaussianBlur(Fake_Lung_imgs,(5,5),sig)
						Fake_Lung_imgs = img
					X.append(Fake_Lung_imgs)
					Y.append(Mask_ima)
					
				for i in range(2):
					Fake_Lung_imgs = createFakeImage(Lung_ima, thresh,200,240, img_rows=rows, img_cols=cols)
					if i > 1:
						sig = (i -1)*0.5
						img = cv2.GaussianBlur(Fake_Lung_imgs,(5,5),sig)
						Fake_Lung_imgs = img
					X.append(Fake_Lung_imgs)
					Y.append(Mask_ima)
	return np.array(X), np.array(Y)

def prepareData(trainX, trainY, valX, valY):

	#reshape the data to be of size [samples][width][height][channels]
	trainX = trainX.reshape(trainX.shape[0],trainX.shape[1],trainX.shape[2],1).astype('float32')
	valX = valX.reshape(valX.shape[0], valX.shape[1], valX.shape[2], 1).astype('float32')

	trainY = trainY.reshape(trainY.shape[0], trainY.shape[1], trainY.shape[2], 1).astype('float32')
	valY = valY.reshape(valY.shape[0], valY.shape[1], valY.shape[2], 1).astype('float32')
	
    #normalize the input values
	trainX = (trainX / 255.0).astype(np.uint8)
	valX = (valX / 255.0).astype(np.uint8)

	trainY = (trainY / 255.0).astype(np.uint8)
	valY = (valY / 255.0).astype(np.uint8)

	return trainX, trainY, valX, valY

def showResults(history):
	fig, axs = plt.subplots(1, 2, figsize=(15, 4))

	training_loss = history.history['loss']
	validation_loss = history.history['val_loss']

	training_accuracy = history.history['accuracy']
	validation_accuracy = history.history['val_accuracy']

	epoch_count = range(1, len(training_loss) + 1)

	axs[0].plot(epoch_count, training_loss, 'r--')
	axs[0].plot(epoch_count, validation_loss, 'b-')
	axs[0].legend(['Training Loss', 'Validation Loss'])

	axs[1].plot(epoch_count, training_accuracy, 'r--')
	axs[1].plot(epoch_count, validation_accuracy, 'b-')
	axs[1].legend(['Training Accuracy', 'Validation Accuracy'])
 
	plt.savefig('C:\\Users\\Mdsp\\Desktop\\Catalin project\\LungImageSegmentation-main\\results\\'+str('result')+'.jpg')
	plt.show()

def show_image(array):
	im = pil.fromarray(array)
	im.show()
	return

def display(image, mask, prediction):
	img = cv2.imshow("image", image)
	mask = cv2.imshow("mask", mask)
	pred_mask = cv2.imshow("pred_mask", prediction.astype("float32"))
	cv2.waitKey(0)

#This function keeps the learning rate at 0.001 for the first ten epochs and decreases it exponentially after that.
def scheduler(epoch):
	if epoch < 10:
		return 0.001
	else:
		return 0.001 * tf.math.exp(0.1 * (10 - epoch))

def multi_class_preparation(X, Y, test_size=0.2, n_classes=3):

	for i in range(len(Y)):
		temp_right = np.where(Y[i] == 200, 255, 0)
		temp_left = np.where(Y[i] == 100, 100, 0)
		Y[i] = np.add(temp_left, temp_right)
        
	#Encode labels... but multi dim array so need to flatten, encode and reshape
	labelencoder = LabelEncoder()
	n, h, w = Y.shape
	train_masks_reshaped = Y.reshape(-1,1)
	train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
	train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
	print("train_masks_encoded_original_shape: ", np.unique(train_masks_encoded_original_shape))

	X = normalize(X)

	trainX, valX, trainY, valY = train_test_split(X, train_masks_encoded_original_shape, test_size=0.2, shuffle=True)

	print("Class values in the dataset are ... ", np.unique(trainY))  # 0 is the background/few unlabeled 

	train_masks_cat = to_categorical(trainY, num_classes=n_classes)
	y_train_cat = train_masks_cat.reshape((trainY.shape[0], trainY.shape[1], trainY.shape[2], n_classes))

	test_masks_cat = to_categorical(valY, num_classes=n_classes)
	y_test_cat = test_masks_cat.reshape((valY.shape[0], valY.shape[1], valY.shape[2], n_classes))

	NO_OF_TRAINING_IMAGES = len(trainX)
	NO_OF_VAL_IMAGES = len(valX)

	trainX, trainY = shuffle(trainX, y_train_cat)
	valX, valY = shuffle(valX, y_test_cat)

	train_images = np.expand_dims(trainX, axis=3)

	valX_images = np.expand_dims(valX, axis=3)

	return train_images, valX_images, trainY, valY


# https://www.kaggle.com/cpmpml/fast-iou-metric-in-numpy-and-tensorflow
# https://www.kaggle.com/meaninglesslives/nested-unet-with-efficientnet-encoder
def get_iou_vector(A, B):
    # Numpy version    
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)
        
        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue
        
        # non empty mask case.  Union is never empty 
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union
        
        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45)*20)) / 10
        
        metric += iou
        
    # teake the average over all images in batch
    metric /= batch_size
    return metric


def my_iou_metric(label, pred):
    # Tensorflow version
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)
