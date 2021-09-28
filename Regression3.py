import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Conv2DTranspose, Reshape
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, Input, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import backend as K
import locale

def load_dataset(path="C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Analysis Results/DataNoNaN2.npy"):
    df = pd.read_pickle(path)
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
    
    for (i,f) in enumerate(filters):
        if i == 0:
            x = inputs
            
        x = Conv2D(f, (3,3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        
    conv_shape = K.int_shape(x) 
    
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    
    x = Dense(4)(x)
    x = Activation("relu")(x)
    
    x = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation='relu')(x)
    x = BatchNormalization()(x)
    x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
    
    x = Conv2DTranspose(64, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2,2))(x)
    
    x = Conv2DTranspose(32, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2,2))(x)
    
    x = Conv2DTranspose(16, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2,2))(x)
    
    x = Conv2DTranspose(1, (3,3), padding="same", activation="linear")(x)
    
    model = Model(inputs, x)
    
    return model     


df = load_dataset()
(inputImages,outputImages) = create_input_output(df, 'bathy', 'hs')

inputImages = inputImages / 120. #min bathy
outputImages = outputImages / 20. #max hs

split = train_test_split(inputImages, outputImages, test_size=0.25, random_state=42)
(trainX, testX, trainY, testY) = split

model = create_cnn(256, 256, 1, regress=True)
opt = Adam(lr=1e-3, decay=1e-3/200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

print("[INFO] training model...")
model.fit(x=trainX, y=trainY,
          validation_data=(testX, testY),
          epochs=200, batch_size=8)

print("[INFO] predicting house prices...")
#preds = model.predict(testX)

diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price : {}".format(
    locale.currency(df["price"].mean(), grouping=True),
    locale.currency(df["price"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))


#%% Plotting
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
model = load_model(r'C:\Users\hemert\OneDrive - Stichting Deltares\Programmas\Regression Test Small\saved_model', compile = True)

Prediction = model.predict(testX[0:4])[3]
Truehs = testY[3]

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










