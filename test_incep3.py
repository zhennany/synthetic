import os
import sys
import numpy as np
if sys.version_info[0] < 3:
    import cPickle
else:
    import pickle
import gzip

import shutil
from scipy import misc
from sklearn import metrics
import matplotlib.pyplot as plt
# import pdb

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import callbacks

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
    
def evaluate(gt, p):
    gt = np.asarray(gt)
    p = np.asarray(p)
    tmp = np.array([np.array([p, np.ones(len(gt),)*(1 - 1e-15)]).min(axis=0), np.ones(len(gt),)*1e-15]).max(axis=0)
    logloss = -np.sum(gt*np.log(tmp)+(1-gt)*np.log(1-tmp))/float(len(gt))
    return logloss
    
def get_error(gt, p):
    gt = np.asarray(gt)
    p = np.asarray(p).astype(np.int8)
    error = np.sum(np.not_equal(gt, p))
    return float(error)/len(gt)
    
def get_confusion_matrix(gt, p):
    gt = np.asarray(gt)
    p = np.asarray(p).astype(np.int8)
    tp = np.sum(np.logical_and(gt==1, p==1))
    tn = np.sum(np.logical_and(gt==0, p==0))
    fp = np.sum(np.logical_and(gt==0, p==1))
    fn = np.sum(np.logical_and(gt==1, p==0))
    return ((tn, fp), (fn, tp))
    
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


def main(argv = ['ds1', 'test.data', 'None']):
    debug = True
    ds = argv[0]
    testfile = argv[1]
    modelfile = None
    if len(argv) > 2:
        if argv[2] != 'None':
            modelfile = argv[2]
    
    
    nb_epoch2 = 70
    
    base_model, model = build_inceptionV3(input_shape=(3,200,200))

    if modelfile is None:
        model_weights = ds+"_incep3-%d.hdf5"%(nb_epoch2)
    else:
        model_weights = modelfile
    
    print('load weight: ',model_weights)
    model.load_weights(model_weights)
    
    # load testing data
    tx, ty = read_dataset(os.path.join(ds,testfile))
    tx = preprocessing_img(tx)
    tx = np.asarray(tx)[:, np.newaxis, :, :]
    tx = np.repeat(tx, 3, axis=1)
    
    print('test on %d samples'%len(ty))
    
    preds = model.predict(tx, verbose=0)
    print("error:",get_error(ty, preds[:,1]>preds[:,0]))
    print("confusion:")
    print(get_confusion_matrix(ty, preds[:,1]>preds[:,0]))
    
    fpr, tpr, thresholds = metrics.roc_curve(ty, preds[:,1], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    plt.ion()
    plt.plot(fpr, tpr)
    plt.title('auc: '+str(auc))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    
    #----------
    ## debug: dump error cases
    if debug:
        err_dump_dir = 'err_'+ds+'_'+testfile[:-len('.data')]
        if os.path.exists(err_dump_dir):
            shutil.rmtree(err_dump_dir)
        os.mkdir(err_dump_dir)
        
        os.mkdir(os.path.join(err_dump_dir, '0'))
        os.mkdir(os.path.join(err_dump_dir, '1'))
        n_0 = sum(np.array(ty)==0)
        for i in range(n_0):
            y, py = ty[i], preds[i,1]
            if y != (py>0.5):
                misc.imsave(os.path.join(err_dump_dir, str(y), str(i)+'_'+str(py)+'.png'), tx[i, 0])
        for i in range(n_0, len(ty)):
            y, py = ty[i], preds[i,1]
            if y != (py>0.5):
                misc.imsave(os.path.join(err_dump_dir, str(y), str(i-n_0)+'_'+str(py)+'.png'), tx[i, 0])
    #----------
    
    ## evaluate the model
    # ty = np_utils.to_categorical(ty, 2)
    # scores = model.evaluate(tx, ty, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    

if __name__ == "__main__":
    main(sys.argv[1:])
    