import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, Input, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle5 as pickle
from tensorflow.keras import backend as K
import locale

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


def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    inputShape = (height, width, depth)
    chanDim = -1
    
    inputs = Input(shape=inputShape)
    x = Conv2D(16, (5,5), padding="same")(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    
    x = Conv2D(32, (5,5), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64, (3,3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128, (3,3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128, (1,1), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64, (1,1), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Conv2D(32, (3,3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Conv2D(16, (3,3), padding="same")(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2DTranspose(1, (5,5), padding="same", activation="linear")(x)
    
    model = Model(inputs, x) 

    return model  


df = load_dataset()
(inputImages,outputImages) = create_input_output(df, 'bathy', 'hs')
(inputImages, outputImages) = ((inputImages - np.nanmean(inputImages))/np.nanstd(inputImages), (outputImages - np.nanmean(outputImages)/np.nanstd(outputImages)))

(inputImages, outputImages) = (np.nan_to_num(inputImages,nan=-10.), np.nan_to_num(outputImages, nan=-10.))

split = train_test_split(inputImages, outputImages, test_size=0.25, random_state=42)
(trainX, testX, trainY, testY) = split

model = create_cnn(256, 256, 1, regress=True)
opt = Adam(learning_rate=1e-2, decay=1e-3/200)
model.compile(loss="mean_squared_error", optimizer=opt)

print("[INFO] training model...")
model.fit(x=trainX, y=trainY,
          validation_data=(testX, testY),
          epochs=200, batch_size=8)

model.save('/content/drive/MyDrive/DeepLearning/ModelV3')


#%% Plotting
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
model = load_model('/content/drive/MyDrive/DeepLearning/ModelV3', compile = True)

Prediction = model.predict(testX[0:5])[3][:,:,0]
Truehs = testY[3][:,:,0]

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










