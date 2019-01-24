This folder contains the main functions of my SA.
The databank file and parameters' files are too large and not included 
****************************************************************************
purepy.py:
A 3-layer MLP based on python (without any framework)
Acceleration algorithm: Adam, mini_batch

****************************************************************************
DNN-BN-tensorflow, DNN_tensorflow:
contains all functions to build a MLP model based on tensorflow Framework
DNN-BN-tensorflow: a MLP model with batch-normalization
DNN-tensorflow: a mlp mpdel without batch-normalization

****************************************************************************
DNN-Keras, improved_DNN_keras:
all MLP models based on Keras framework
1, a 3 or 4 layers MLP
2, a 3 layers MLP with weighted_loss
3, a 3 layers MLP with random_under_sampling
4, a 3 layers MLP with over_sampling
5, a 3 layers MLP used to verify model's generalization
6, a 3 layers MLP used to verify model's specificity
notice:
When using the A method, you need to add the following code to the losses.py file in the keras library.
-------------------------------------------------------------------------------------------------------
def weighted_categorical_crossentropy(y_true, y_pred):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    weights = np.array([15, 5, 25, 10, 1])  #this is array you want to multiply with loss
    weights = K.variable(weights)
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    # calc
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    return loss
-------------------------------------------------------------------------------------------------------

****************************************************************************
CNN:
contains all CNN models based on Keras
a 2D and a 1D CNN model (The actual calculation structure are the same)
a CNN model using signals in time and frequency domain
a CNN model used for verify model's generalization
a CNN model used for verify model's specificity

collector.py:
used to save data from different File 
save all patients' data into a H5 file

****************************************************************************
dataloader.py:
load data from EDF.file
used to transform data format

****************************************************************************
tools.py:
all used help functions
mainly:
function used for feature extraction:
compute_derivation, compute_Hjorth, compute_dominant_fre, compute_EDF, compute_spectral_moments,
compute_bandpower, compute_power_ratios, feature_extraction

function used to save data as H5 file:
Save_Data, Save_Data_1, Save_Data_2, Save_features

function used to convert string to bumber:
Convert_StageToNumber

function used to add butterworth_filter:
butter_bandpass, butter_bandpass_filter

function used to plot Hypnogram:
plot_hypnogram_1, show_hypnogram
****************************************************************************
