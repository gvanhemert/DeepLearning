import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, Input, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle5 as pickle
from tensorflow.keras import backend as K
import locale
import math

def load_dataset(path="/content/drive/MyDrive/DeepLearning/DataEmoda.npy"):
    with open(path, "rb") as fh:
      df = pickle.load(fh)
    return df

def create_input_output(df, input_labels, output_labels):
    inputImage = []
    outputImage = []
    for i in df.index:
        inputImage.append(df[input_labels][i].reshape(256,256,1))
        outputImage.append(df[output_labels][i].reshape(256,256,1))
    return np.array(inputImage), np.array(outputImage)

def create_input(df, input_labels):
    return df[input_labels].values

def create_mlp(dim):
    model = Sequential()
    #model.add(Dense(16, input_dim=dim, activation="relu"))
    model.add(Dense(64, input_dim=dim, activation='relu'))
    #model.add(Dense(256, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    #model.add(Dense(4096, activation='relu'))
    #model.add(Dense(16384, activation='relu'))
    model.add(Dense(256*256, activation='relu'))
    model.build((None, 256*256))

    return model

def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    inputShape = (height, width, depth)
    chanDim = -1
    
    inputs = Input(shape=inputShape)
    x = Conv2D(64, (3,3), padding="same")(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    ax = MaxPooling2D(pool_size=(2,2))(x)

    #Branch 1
    x = Conv2D(32, (3,3), padding="same")(ax)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    bx = MaxPooling2D(pool_size=(2,2))(x)

    #Branch 2
    x = Conv2D(32, (3,3), padding="same")(bx)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    cx = MaxPooling2D(pool_size=(2,2))(x)

    #Branch 3
    x = Conv2D(32, (3,3), padding="same")(cx)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(32, (3,3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2DTranspose(32, (3,3), padding="same")(x)

    #Branch 2
    x = Concatenate()([x, cx])
    x = Conv2D(32, (3,3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2DTranspose(32, (3,3), padding="same")(x)

    #Branch 1
    x = Concatenate()([x, bx])
    x = Conv2D(32, (3,3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2DTranspose(32, (3,3), padding="same")(x)
    
    #Main Branch
    x = Concatenate()([x,ax])
    x = Conv2D(16, (3,3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    
    model = Model(inputs, x)
    
    return model 
    
def full_model(cnn_model, mlp_model):
    
    x = cnn_model.output
    cx = mlp_model.output
    
    conv_shape = K.int_shape(x)
    
    cx = Reshape((conv_shape[1],conv_shape[2],int(conv_shape[3]/4)))(cx)
    
    x = Concatenate()([x,cx])
    
    x = Conv2D(16, (3,3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(16, (3,3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(16, (3,3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(16, (3,3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(16, (3,3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(16, (3,3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2,2))(x)
    x = Conv2DTranspose(1, (3,3), padding="same", activation="linear")(x)
    
    model = Model(inputs=[cnn_model.input, mlp_model.input], outputs = x)
    
    return model   


df = load_dataset()
(inputImages,outputImages) = create_input_output(df, 'bathy', 'hs')
inStd = np.nanstd(inputImages)
inMean = np.nanmean(inputImages)
inputImages = (inputImages - np.nanmean(inputImages))/np.nanstd(inputImages)
#(inputImages, outputImages) = ((inputImages - np.nanmean(inputImages))/np.nanstd(inputImages), (outputImages - np.nanmean(outputImages)/np.nanstd(outputImages)))

inputAttr = create_input(df, ['$\eta$', '$\zeta$','$\theta_{wave}$'])
inputAttr[:,0] = (inputAttr[:,0] - np.mean(inputAttr[:,0])) / np.std(inputAttr[:,0])
inputAttr[:,1] = (inputAttr[:,1] - np.mean(inputAttr[:,1])) / np.std(inputAttr[:,1])
#inputAttr[:,0] = inputAttr[:,0] / 50.
#inputAttr[:,1] = inputAttr[:,1] / 20.
inputAttr[:,2] = inputAttr[:,2] / (2*np.pi)


#inputImages = inputImages / 1950. #min bathy
#outputImages = outputImages / 20. #max hs

(inputImages, outputImages) = (np.nan_to_num(inputImages,nan=-1.), np.nan_to_num(outputImages, nan=-1.))

split = train_test_split(inputImages, inputAttr, outputImages, test_size=0.25, random_state=42)
(trainImgX, testImgX, trainAttrX, testAttrX, trainY, testY) = split

cnn_model = create_cnn(256, 256, 1, regress=True)
mlp_model = create_mlp(trainAttrX.shape[1])

model = full_model(cnn_model, mlp_model)
opt = Adam(learning_rate=1e-4, decay=1e-4/200)
model.compile(loss="mean_squared_error", optimizer=opt)

initial_learning_rate = 0.001
def lr_step_decay(epoch, lr):
    drop_rate = 0.5
    epochs_drop = 20.0
    return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))

print("[INFO] training model...")
model.fit(x=[trainImgX, trainAttrX], y=trainY,
          validation_data=([testImgX, testAttrX], testY),
          epochs=200, batch_size=8)
          #callbacks=[LearningRateScheduler(lr_step_decay, verbose=1)])

model.save('/content/drive/MyDrive/DeepLearning/ModelV6')



#%% Plotting
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
model = load_model('/content/drive/MyDrive/DeepLearning/ModelV6', compile = True)

Prediction = model.predict([testImgX[0:7], testAttrX[0:7]])[6][:,:,0]
Truehs = testY[6][:,:,0]

fig = plt.figure(figsize=(6,3))

ax = fig.add_subplot(1,2,1)
ax.set_title('colorMap')
plt.imshow(Prediction)

qx = fig.add_subplot(1,2,2)
plt.imshow(Truehs)

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()


