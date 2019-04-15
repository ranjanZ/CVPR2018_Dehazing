import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, concatenate, Conv2DTranspose,Concatenate
from keras.layers import Conv2D, MaxPooling2D,Dropout
from keras import backend as K
import os
import numpy as np
#import matplotlib.pyplot as plt
from keras.utils.generic_utils import get_custom_objects
from keras import objectives
from keras import backend as K
from keras.engine.topology import Layer
from model import *

#get_custom_objects().update({"customloss": customloss})
save_dir = os.path.join(os.getcwd(),'../models')





PATCH_H =None
PATCH_W =None



def load_data():
    I=np.random.random((100,128,128,3))
    J=np.random.random((100,128,128,3)) 

    return I,J



def zero_loss(y_true, y_pred):
    return K.zeros_like(y_true)


class LossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(LossLayer, self).__init__(**kwargs)

      
    def lossfun(self, I,AT,T,J):
        T=K.tile(T,[1,1,1,3])
        
        loss3 =K.mean(K.abs(J-((I-AT)/(T+0.01))))
        loss2 =K.mean(K.abs(I-J*T-AT))
        loss=loss3+loss2

        return loss

    def call(self, inputs):
       	I = inputs[0]
        AT = inputs[1]
        T = inputs[2]
        J = inputs[3]

          
        loss = self.lossfun(I,AT,T,J)
        self.add_loss(loss, inputs=inputs)

        return I


class My_Callback(keras.callbacks.Callback):
    def __init__(self,mod,tracks=0):
        self.validation_data = None
        self.mod = model
        self.tracks = tracks
        # self.epoch_no = epoch_no

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        if(self.tracks%3==0):
            self.mod.save('../models/temp_models/{}--.h5'.format(self.tracks))
            print(self.tracks)
        self.tracks+=1

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass




 


model = get_model()

model_save = My_Callback(model)



I = Input(shape=(PATCH_H,PATCH_W,3))
J = Input(shape=(PATCH_H,PATCH_W,3))
[T_out,AT_out]=model([I])
loss_out=LossLayer()([I,AT_out,T_out,J])


model_train = Model(inputs=[I,J],outputs=[loss_out])
model.summary()
model_train.compile(optimizer='Adagrad',loss=[zero_loss])
print('Training -----')





Hazy,Real=load_data()     #have to write the function
#shape of the Hazy and Real variable  are (num of sample,128,128,3) 


model_train.fit([Hazy,Real],Hazy,epochs=9999999,shuffle=True,verbose=True,batch_size=50,callbacks=[model_save])

#model.save("../models/patch_new_loss.h5")


