import os
import sys
import numpy as np
if sys.version_info[0] < 3:
    import cPickle
else:
    import pickle
import gzip

# import pdb

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Dropout
from keras import backend as K
from keras.layers import Input
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
# from keras.optimizers import SGD
from keras import callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

K.set_image_data_format('channels_first')

def read_dataset(filename='alldata.data'):
    if filename[-3:] == '.gz':
        f = gzip.open(filename, 'rb')
    else:
        f = open(filename, 'rb')
    if sys.version_info[0] < 3:
        dataxy = cPickle.load(f)
    else:
        dataxy = pickle.load(f, encoding='latin1')
    f.close()
    return dataxy

def save_dataset(filename='alldata.data.gz', data=None):
    if filename[-3:] == '.gz':
        f = gzip.open(filename, 'wb')
    else:
        f = open(filename, 'wb')
    if sys.version_info[0] < 3:
        cPickle.dump(data, f, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        pickle.dump(data, f, protocol=2)
    f.close()

def preprocessing_img(x):
    x = np.asarray(x).astype(np.float32)
    x /= 255.0
    
    return x
    
def add_random_noise(image):
    if np.random.rand() > 0.4:
        if np.random.rand() > 0.5:
            return add_noise(image, 'gauss')
        else:
            return add_noise(image, 's&p')
    else:
        return image
    
'''
Parameters
----------
image : ndarray
    Input image data. Will be converted to float.
mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
'''

def add_noise(image, noise_typ='gauss'):
    maxv = max(1, np.max(image))
    
    if noise_typ == "gauss":
        mean = 0
        var = 0.1 * maxv
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,image.shape)
        # gauss = gauss.reshape(image.shape)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = maxv
        
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        gauss = np.random.randn(image.shape)
        # gauss = gauss.reshape(image.shape)        
        noisy = image + image * gauss
        return noisy

###########################################################
## Fine-tune InceptionV3 on a new set of classes
def build_inceptionV3(input_shape):
    # this could also be the output a different Keras model or layer
    input_tensor = Input(shape=input_shape)  # this assumes K.image_data_format() == 'channels_first'

    # create the base pre-trained model, original input was (3, 299, 299)
    base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)

    ## let's visualize layer names and layer indices to see how many layers we should freeze:
    # for i, layer in enumerate(base_model.layers):
       # print(i, layer.name)
       
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(2, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=input_tensor, outputs=predictions)
    
    return base_model, model



def main(argv = ['ds1', 'None', '0']):
    ds = argv[0]
    resume_model = None
    if len(argv) > 1:
        resume_model = argv[1] if argv[1]!='None' else None
    im_noise = False
    if len(argv) > 2:
        im_noise = int(argv[2])==1
    
    
    nb_epoch2 = 70

    if not os.path.exists('models'):
        os.mkdir('models')
    
    # load data
    tx, ty = read_dataset(ds+'/train.data')
    vx, vy = read_dataset(ds+'/valid.data')
    
    nTrainSample = len(ty)
    
    tx = preprocessing_img(tx)
    vx = preprocessing_img(vx)
    
    tx = np.asarray(tx)[:, np.newaxis, :, :]
    vx = np.asarray(vx)[:, np.newaxis, :, :]
    tx = np.repeat(tx, 3, axis=1)
    vx = np.repeat(vx, 3, axis=1)
    
        
    ty = np_utils.to_categorical(ty, 2)
    vy = np_utils.to_categorical(vy, 2)
    
    ## build model
    base_model, model = build_inceptionV3(input_shape=(3,200,200))

    datagen = ImageDataGenerator(
        preprocessing_function=add_random_noise if im_noise else None,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.02,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.02,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images
        
    
    if resume_model is not None:
        print('Resume model: ', resume_model)
        model.load_weights(resume_model)
    


    # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    
    filepath2="models/"+ds+"_incep3-{epoch:02d}-{val_acc:.3f}.hdf5"
    checkpoint2 = callbacks.ModelCheckpoint(filepath2, monitor='val_acc', verbose=0, save_best_only=True, mode='max', save_weights_only=True)
    
    early_stopper   = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=40)
    
    callbacks_list2 = [early_stopper, checkpoint2]

    # we train our model 
    model.fit_generator(datagen.flow(tx, ty, batch_size=40, shuffle=True),
                        steps_per_epoch=nTrainSample/40,
                        epochs=nb_epoch2,
                        validation_data=(vx, vy),
                        callbacks=callbacks_list2)

            
    # serialize last weights to HDF5
    model.save_weights(ds+"_incep3-%d.hdf5"%(nb_epoch2))
    
    print("Saved last weights to disk")


if __name__ == "__main__":
    main(sys.argv[1:])
    