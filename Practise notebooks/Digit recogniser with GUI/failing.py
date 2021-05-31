import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as bk


(xtrain, ytrain), (xtest,ytest)= mnist.load_data()
#print(xtrain.shape)
xtrain = xtrain.reshape(xtrain.shape[0],28,28,1)
xtest =xtest.reshape(xtest.shape[0],28,28,1)
input_shape=(28,28,1)

ytrain = keras.utils.to_categorical(ytrain,num_classes=10)
ytest = keras.utils.to_categorical(ytest,num_classes=10)

xtrain=xtrain.astype('float32')
xtest=xtest.astype('float32')

xtrain/=255.0
xtest/=255.0

print(xtrain.shape)
print(xtrain.shape[0],'train samples')
print(xtest.shape[0],'test samples')


#model using cnn architecture

batch_size=128
num_classes=10
epochs =10

model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


hist=model.fit(xtrain,ytrain,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(xtest,ytest))
print("Training done")
model.save('mnist.h5')
print("saved as mnist.h5")

score = model.evaluate(xtest, ytest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename,color_mode="grayscale",target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

# load an image and predict the class

def run_example():
	# load the image
	img = load_image('sample_image.png')
	# load modeli
	model = load_model('mnist.h5')
	# predict the class
	digit = model.predict_classes(img)
	print(digit[0])

# entry point, run the example
run_example()








