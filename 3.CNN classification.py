import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from random import shuffle




os.chdir('E:/Xiang Guo/JIGSAWS data/code/1_out')

TRAINDATADIR = "C:/Xiang/JIGSAWS data/Experimental_setup/1_out_test/train/"
TESTDATADIR = "C:/Xiang/JIGSAWS data/Experimental_setup/1_out_test/test/"

CATEGORIES = os.listdir(TRAINDATADIR)

num_classes = len(CATEGORIES)
IMG_SIZE = 150
data_augmentation = False
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_1_out_itr1_trained_model.h5'


training_data = []

def create_training_data():
    for category in tqdm(CATEGORIES):  
        path = os.path.join(TRAINDATADIR,category)  
        class_num = CATEGORIES.index(category)  # get the classification  
        for img in os.listdir(path):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num, img, category])  # add this to our training_data
            except Exception as e:  
                pass
create_training_data()
print(len(training_data))
shuffle(training_data)



X_train = []
y_train = []
imgs_train = []
gestures_train = []

for features,label,img,category in training_data:
    X_train.append(features)
    y_train.append(label)
    imgs_train.append(img)
    gestures_train(category)
    


X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)



#save this data
import pickle

pickle_out = open("X_train.pickle","wb")
pickle.dump(X_train, pickle_out, protocol=4)
pickle_out.close()

pickle_out = open("y_train.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()


#load data
pickle_in = open("X_train.pickle","rb")
X_train = pickle.load(pickle_in)

pickle_in = open("y_train.pickle","rb")
y_train = pickle.load(pickle_in)





import time
startime = time.time()
X_train = X_train/255.0
endtime = time.time()
endtime- startime

NAME = "1-out"








model = Sequential()
model.add(Conv2D(256, (3, 3), input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_split=0.3,shuffle=True)


endtime = time.time()
endtime- startime






# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)



#create test data
test_data = []

def create_test_data():
    for category in tqdm(os.listdir(TESTDATADIR)):  
        path = os.path.join(TESTDATADIR,category)  
        class_num = CATEGORIES.index(category)  # get the classification  
        for img in os.listdir(path):  # iterate over each image
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                test_data.append([new_array, class_num, img, category])  # add this to our training_data
            except Exception as e:  
                pass

create_test_data()

print(len(test_data))

X_test = []
y_test = []

for features,label,img,category in test_data:
    X_test.append(features)
    y_test.append(label)

print(X_test[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_train = X_train/255.0
# Convert class vectors to binary class matrices.
y_test = keras.utils.to_categorical(y_test, num_classes)





#write the result
with open('result/result.csv','w') as f:
    f.write('label,prediction\n')
            
with open('result/result.csv','a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
        model_out = (np.argmax(model.predict([data])[0]))
        f.write('{},{}\n'.format(img_num,model_out))

'''
result = pd.read_csv('result/result.csv')
np.mean(result['label']==result['prediction'])
'''




# Score trained model.
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])



#loss and accuracy curves.
# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.savefig('Loss.png') 
# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['acc'],'r',linewidth=3.0)
plt.plot(history.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.savefig('Accuracy.png') 




'''
#https://www.learnopencv.com/image-classification-using-convolutional-neural-networks-in-keras/
#Using Data Augmentation
from keras.preprocessing.image import ImageDataGenerator
 
ImageDataGenerator(
    rotation_range=10.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.,
    zoom_range=.1.,
    horizontal_flip=True,
    vertical_flip=True)

#Training with Data Augmentation
from keras.preprocessing.image import ImageDataGenerator
 
model2 = createModel()
 
model2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
 
batch_size = 256
epochs = 100
datagen = ImageDataGenerator(
#         zoom_range=0.2, # randomly zoom into images
#         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
 
# Fit the model on the batches generated by datagen.flow().
history2 = model2.fit_generator(datagen.flow(train_data, train_labels_one_hot, batch_size=batch_size),
                              steps_per_epoch=int(np.ceil(train_data.shape[0] / float(batch_size))),
                              epochs=epochs,
                              validation_data=(test_data, test_labels_one_hot),
                              workers=4)
 
model2.evaluate(test_data, test_labels_one_hot)
'''
