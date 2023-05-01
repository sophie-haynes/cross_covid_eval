# "densenet", "conv4", "darkcovidnet" or "minaee_resnet"

def conv4_tf(img_res,learning_rate, momentum):
	import tensorflow as tf
	from tensorflow import keras
	from keras import layers
	from keras import models
	from keras.callbacks import EarlyStopping
	from tensorflow.keras.preprocessing import image_dataset_from_directory

	model = models.Sequential()
	model.add(layers.Conv2D(32, (3,3),input_shape=(img_res,img_res,3)))
	model.add(layers.MaxPooling2D((2,2)))
	model.add(layers.Conv2D(64, (3,3),activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
	model.add(layers.MaxPooling2D((2,2)))
	model.add(layers.Conv2D(128, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
	model.add(layers.MaxPooling2D((2,2)))
	model.add(layers.Conv2D(128, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
	model.add(layers.Flatten())
	model.add(layers.Dense(512,activation=tf.keras.layers.LeakyReLU(alpha=0.2)))
	model.add(layers.Dense(1))

	optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
	model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=[keras.metrics.BinaryAccuracy(), ])
	es = EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)
	return model,es

def densenet_tf(img_res,learning_rate, momentum,weights):
	from tensorflow import keras
	from keras import layers
	from keras import models
	from keras.callbacks import EarlyStopping
	from tensorflow.keras.preprocessing import image_dataset_from_directory
	import shutil

	if weights=="imagenet":
		base_model = keras.applications.DenseNet121(weights='imagenet', input_shape=(img_res,img_res,3), include_top=False)
		base_model.trainable = False
		inputs = keras.Input(shape=(img_res, img_res, 3))
		x = base_model(inputs, training=False)
		x = keras.layers.GlobalAveragePooling2D()(x)
		outputs = keras.layers.Dense(1)(x)
		model = keras.Model(inputs, outputs)
		optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
		model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=[keras.metrics.BinaryAccuracy(), ])
		es = EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)
		return model,es

	elif weights=="cxr8":
		import os

		# check if this file is already loaded
		if not os.path.isfile("cxr-8_dense_weights.h5"):
			# create weights file 
			# shutil.copy("/content/drive/MyDrive/Datasets/cxr-8_dense_weights.h5","cxr-8_dense_weights.h5")
			shutil.copy("/content/drive/MyDrive/Datasets/cxr-8/ultra_finetune_Binary_CXR8_Transfer_CW-Hist_50_2.h5","cxr-8_dense_weights.h5")

		base_model = keras.applications.DenseNet121(weights=None, input_shape=(img_res,img_res,3), include_top=False)
		# load manual weights from pretraining
		base_model.load_weights("cxr-8_dense_weights.h5", by_name=True)
		base_model.trainable = False
		inputs = keras.Input(shape=(img_res, img_res, 3))
		x = base_model(inputs, training=False)
		x = keras.layers.GlobalAveragePooling2D()(x)
		outputs = keras.layers.Dense(1)(x)
		model = keras.Model(inputs, outputs)

		optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
		model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=[keras.metrics.BinaryAccuracy(), ])
		es = EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)
		return model,es

	elif weights=="wehbe":
		base_model = keras.applications.DenseNet121(weights=None, input_shape=(img_res,img_res,3), include_top=False)
		base_model.load_weights("/content/drive/MyDrive/Datasets/Weights_Wehbe/DenseNet_224_up_uncrop.h5", by_name=True)
		base_model.trainable = False
		inputs = keras.Input(shape=(img_res, img_res, 3))
		x = base_model(inputs, training=False)
		x = keras.layers.GlobalAveragePooling2D()(x)
		outputs = keras.layers.Dense(1)(x)
		model = keras.Model(inputs, outputs)
		optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
		model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=[keras.metrics.BinaryAccuracy(), ])
		es = EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)
		return model,es

	else:
		raise ValueError("Invalid weights passed to densenet. {} was not recognised.".format(weights))

def densenet_tf_finetune(model,learning_rate, momentum,weights):
	from tensorflow import keras
	from keras import layers
	from keras import models
	from keras.callbacks import EarlyStopping
	from tensorflow.keras.preprocessing import image_dataset_from_directory
	import shutil

	model.trainable=True
	es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
	optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
	model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=[\
		keras.metrics.BinaryAccuracy(), \
		keras.metrics.SensitivityAtSpecificity(0.98) \
		])
	return model,es
	
def darkcovidnet_torch():
	pass

def minaee_resnet_torch():
	pass
