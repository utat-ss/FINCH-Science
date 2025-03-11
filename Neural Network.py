# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 00:35:41 2025

The purpose of this code is to create a Neural Network to predict how the spectral signature of crop residue
(the non-photosynthetic vegetation) changes over time as it decomposes into soil.

Theory of Transformation
1. Physical Analysis:
As crop residue (non-photosynthetic vegetation, NPV) decompose, their spectral properties change over time
into soil. This transformation is governed by physical and chemical processes such as organic matter decay,
moisture loss, and mineral exposure.

To represent this transformation take the observed spectrum as a total linear or non-linear mixture of spectral
signatures (NPV endmember and soil endmember). The transformation factor/fraction (Decay function of time) α(t)
represents the degree of decay/rate of decay at time t.
    
Imagine you have a smoothie made of two fruits. One being an unripe bitter banana representing
non-photosynthetic vegetation (NPV) and the other being soil a super over ripe sweet and mushy banana.
At the beginning, the smoothie is mostly NPV but over time left out in the environment the smoothie starts
ripening and now more of the smoothie is soil. Now the smoothie has a different taste and colour.

In a linear model, a mixture with NPV and soil would simply be those two as strict two amounts. Like 20% soil and 80% NPV.
In a non-linear model it is more complex. Now we observe and learn how the spectral signature of NPV changes into soil
(how the colour of a material's wavelength changes). 
Linear is a predictable way of the soil and npv changing at a constant rate and non-linear is not constant rate. In real
world scenerio its almost always non-linear
 
So FINCH sees the ground as a spectrum, the *smoothie*. This mixed smoothie (spectrum) is a blend of NPV
and soil signal.

Recipe:

NPV Signal: This is the initial ingredient
Soil Signal: This is like the ingredient that increases slowly with time (banana ripening)
Mixing Ratio α(t): This function tells us how much soil signal is in the mix at any time t.
Meaning if t is small (early stages), α(t) is low because there's little soil early on. With
time going pass, α(t) increases as more soil appears in the mix, more bananas ripening in the
smoothie. Can be written as: 

                        Spectrum(t) = α(t) * Soil + (1 - α(t)) * NPV

Where α(t) increases with time t.

NPV: Spectral Signature
Soil: Spectral Signature
α(t): Decay function of time. If t = 0 no decay has happened, meaning α(0) = thus the observed spectrum is 100%
      As time increases α(t) increases toward 1. If α(t) = 1 then the observed spectrum is 100% soil.
      Can change in a linear or non-linear way depending on the decay processes.
α(t) * Soil: This tells you how much of the soil's spectral signature is present int he overall observed spectrum,
             α(t) > 0 The more soil
(1 - α(t)) * NPV: The remaining fraction. So it tells you how much of the NPV's spectral signature is still there. 1 is
            a ratio to 100%
Spectrum(t): FINAL OBSERVED SPECTRUM!! At time t :)
             It is the combination of the two defined proportions α(t) * Soil and (1 - α(t)) * NPV, respectively


2. Incorporating Physics into Data Augmentation:
Spectral Preprocessing is for when we receive raw hyperspectral data. Raw data is noisy as there are many unwanted
materials FINCH collects. it contains too much information! The important details such as the decay process
(transformation of NPV to soil over time) is hard to see.

To fix this we use techniques like
Continuum Removal: A process where it helps clear/eliminate noise in the spectrum so subtle features become more vivid.
Normalization: This scales the data in a way so that differences between the overall brightness do not mask or cover the
               important details we need to know.
Derivative Analysis: Creates changes in the spectral curves, revealing small changes in the material's properties as decay
                     progresses.
They give us a cleaner data!

@author: Tomi Wang :)
"""
# write a function of alpha t alone, need to know processing
# can have an exponential function as an exmaple but we dont know the rate it decays so this is just an example

# the following is a practice shell


import tensorflow as tf # For building and training the neural network.
from tensorflow import keras # For building and training the neural network.
from tensorflow.keras import layers # For building and training the neural network.
import numpy as np # To create and manipulate arrays (our data).
import matplotlib.pyplot as plt # For plotting the training and validation loss over epochs.

# Generating Synthetic Data
num_samples = 1725
num_features = 210
X = np.random.rand(num_samples, num_features)
time_feature = X[:, -1]
y = time_feature + 0.1 * np.random.rand(num_samples)
y = np.clip(y, 0, 1)

split = int(0.8 * num_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(num_features,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=1)
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
print(test_loss, test_mae)

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()
