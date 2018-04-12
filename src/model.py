from keras.models import Model
from keras.layers import Input, concatenate, BatchNormalization
from keras.layers import MaxPooling2D, Cropping2D
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import Concatenate


def get_model():
    '''
    Made with the assumption of input shape (2^n, 2^n, 3)
    '''
    x = Input(shape=(None, None, 3))
    #x = Input(shape=(128, 128, 3))

    # contracting path
    x_c = Conv2D(4, (3, 3), strides=(1, 1), activation='elu')(x)
    x_c = Conv2D(4, (3, 3), strides=(1, 1), activation='elu')(x_c)
    x_m1 = x_c
    
    x_c = Conv2D(8, (3, 3), strides=(2, 2), activation='elu')(x_c)
    x_c = Conv2D(8, (3, 3), strides=(1, 1), activation='elu')(x_c)
    x_c = Conv2D(8, (3, 3), strides=(1, 1), activation='elu')(x_c)
    x_m2 = x_c
    
    x_c = Conv2D(16, (3, 3), strides=(2, 2))(x_c)
    
    # fork t path from here
    x_t_ex = x_c
    
    # t expanding path
    x_t_ex = Conv2DTranspose(8, (3, 3), strides=(2, 2),
                             activation='elu')(x_t_ex)
    x_t_ex = Concatenate()([x_m2, x_t_ex])
    
    x_t_ex = Conv2DTranspose(8, (3, 3), strides=(1, 1),
                             activation='elu')(x_t_ex)
    x_t_ex = Conv2DTranspose(8, (3, 3), strides=(1, 1),
                             activation='elu')(x_t_ex)
    x_t_ex = Conv2DTranspose(4, (4, 4), strides=(2, 2),
                             activation='elu')(x_t_ex)
    x_t_ex = Concatenate()([x_m1, x_t_ex])
    
    x_t_ex = Conv2DTranspose(4, (3, 3), strides=(1, 1),
                             activation='elu')(x_t_ex)
    x_t_out = Conv2DTranspose(1, (3, 3), strides=(1, 1),
                              activation='sigmoid')(x_t_ex)

    # Go futher down for A_path
    x_c = Conv2D(16, (3, 3), strides=(1, 1), activation='elu')(x_c)
    x_c = Conv2D(16, (3, 3), strides=(1, 1), activation='elu')(x_c)
    x_c = Conv2D(16, (3, 3), strides=(1, 1), activation='elu')(x_c)
    x_m3 = x_c
    
    x_c = Conv2D(32, (3, 3), strides=(2, 2), activation='elu')(x_c)
    x_c = Conv2D(32, (3, 3), strides=(1, 1), activation='elu')(x_c)
    x_c = Conv2D(32, (3, 3), strides=(1, 1), activation='elu')(x_c)
    x_c = Conv2D(32, (3, 3), strides=(1, 1), activation='elu')(x_c)
    x_m4 = x_c
    
    x_c = Conv2D(64, (3, 3), strides=(2, 2))(x_c)
    
    x_A_ex = x_c
    # A expanding path
    x_A_ex = Conv2DTranspose(32, (4, 4), strides=(2, 2),
                             activation='elu')(x_A_ex)
    x_A_ex = Concatenate()([x_m4, x_A_ex])
    
    x_A_ex = Conv2DTranspose(32, (3, 3), strides=(1, 1),
                             activation='elu')(x_A_ex)
    x_A_ex = BatchNormalization()(x_A_ex)
    x_A_ex = Conv2DTranspose(32, (3, 3), strides=(1, 1),
                             activation='elu')(x_A_ex)
    x_A_ex = BatchNormalization()(x_A_ex)
    x_A_ex = Conv2DTranspose(32, (3, 3), strides=(1, 1),
                             activation='elu')(x_A_ex)
    x_A_ex = BatchNormalization()(x_A_ex)
    x_A_ex = Conv2DTranspose(32, (4, 4), strides=(2, 2),
                             activation='elu')(x_A_ex)
    x_A_ex = Concatenate()([x_m3, x_A_ex])
    
    x_A_ex = Conv2DTranspose(16, (3, 3), strides=(1, 1),
                             activation='elu')(x_A_ex)
    x_A_ex = BatchNormalization()(x_A_ex)
    x_A_ex = Conv2DTranspose(16, (3, 3), strides=(1, 1),
                             activation='elu')(x_A_ex)
    x_A_ex = BatchNormalization()(x_A_ex)
    x_A_ex = Conv2DTranspose(16, (3, 3), strides=(1, 1),
                             activation='elu')(x_A_ex)
    x_A_ex = BatchNormalization()(x_A_ex)
    x_A_ex = Conv2DTranspose(8, (3, 3), strides=(2, 2),
                             activation='elu')(x_A_ex)
    x_A_ex = Concatenate()([x_m2, x_A_ex])
    
    x_A_ex = Conv2DTranspose(8, (3, 3), strides=(1, 1),
                             activation='elu')(x_A_ex)
    x_A_ex = BatchNormalization()(x_A_ex)
    x_A_ex = Conv2DTranspose(8, (3, 3), strides=(1, 1),
                             activation='elu')(x_A_ex)
    x_A_ex = BatchNormalization()(x_A_ex)
    x_A_ex = Conv2DTranspose(4, (4, 4), strides=(2, 2),
                             activation='elu')(x_A_ex)
    x_A_ex = Concatenate()([x_m1, x_A_ex])
    
    x_A_ex = Conv2DTranspose(4, (3, 3), strides=(1, 1),
                             activation='elu')(x_A_ex)
    x_A_ex = BatchNormalization()(x_A_ex)
    x_tA_out = Conv2DTranspose(3, (3, 3), strides=(1, 1),
                               activation='sigmoid')(x_A_ex)
 
    model = Model(inputs=[x], outputs=[x_t_out, x_tA_out])
    #model = Model(inputs=[x], outputs=[x_t_out])

    return model


