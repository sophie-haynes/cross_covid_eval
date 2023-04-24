# "densenet", "conv4", "darkcovidnet" or "minaee_resnet"

def conv4_tf(img_res,learning_rate, momentum):
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

	if weights=="imagenet":
		base_model = keras.applications.DenseNet121(weights='imagenet',
                                                input_shape=(img_res,img_res,3),
                                                include_top=False)
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
		if not os.path.isfile("h5_weights.h5"):
			# create weights file 
			transfer_model = keras.models.load_model("/content/drive/MyDrive/Paper2Logging/models/ultra_finetune_Binary_CXR8_Transfer_Hist_50_1")
			transfer_model.save("h5_weights.h5")

		base_model = keras.applications.DenseNet121(weights=None,
                                                    input_shape=(img_res,img_res,3),
                                                    include_top=False)
		# load manual weights from pretraining
	    base_model.load_weights("h5_weights.h5", by_name=True)
	    base_model.trainable = False
	    inputs = keras.Input(shape=(img_res, img_res, 3))
	    x = base_model(inputs, training=False)
	    x = keras.layers.GlobalAveragePooling2D()(x)
	    outputs = keras.layers.Dense(1)(x)
	    model = keras.Model(inputs, outputs)

	    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

	    model.compile(optimizer=optimizer,
	                       loss=keras.losses.BinaryCrossentropy(from_logits=True),
	                       metrics=[keras.metrics.BinaryAccuracy(), ])
	    es = EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)
	    return model,es

	elif weights=="wehbe":
		base_model = keras.applications.DenseNet121(weights=None,
                                                input_shape=(img_res,img_res,3),
                                                include_top=False)
		base_model.load_weights("/content/drive/MyDrive/Datasets/Weights_Wehbe/DenseNet_224_up_uncrop.h5", by_name=True)
		base_model.trainable = False
	    inputs = keras.Input(shape=(img_res, img_res, 3))
	    x = base_model(inputs, training=False)
	    x = keras.layers.GlobalAveragePooling2D()(x)
	    outputs = keras.layers.Dense(1)(x)
	    model = keras.Model(inputs, outputs)

	    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

	    model.compile(optimizer=optimizer,
	                       loss=keras.losses.BinaryCrossentropy(from_logits=True),
	                       metrics=[keras.metrics.BinaryAccuracy(), ])
        es = EarlyStopping(monitor='loss', patience=30, restore_best_weights=True)
        return model,es

	else:
		raise ValueError("Invalid weights passed to densenet. {} was not recognised.".format(weights))
def darkcovidnet_torch():
	pass

def minaee_resnet_torch():
	pass