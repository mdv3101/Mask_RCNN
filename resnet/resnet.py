import tensorflow as tf
from tesnorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add ,Input, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten, Dense

def res_identity(x,filters):
    # Identity block
    # Dimentsion does not changes
    # 3 blocks

    x_skip = x
    f1,f2 = filters

    x = Conv2D(f1,kernel_size=(1,1),stride=(1,1),padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f1,kernel_size=(3,3),stride=(1,1),padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f2,kernel_size=(1,1),stride=(1,1),padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)


    x = Add()[x,x_skip]
    x = Activation('relu')(x)

    return x

def res_conv(x,s,filters):
    # Input size changes
    x_skip =x
    f1,f2 =filters

    x = Conv2D(f1,kernel_size=(1,1),stride=(s,s),padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f1,kernel_size=(3,3),stride=(1,1),padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(f2,kernel_size=(1,1),stride=(1,1),padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    x_skip = Conv2D(f2,kernel_size=(1,1),strides=(s,s),padding='valid', kernel_regularizer=l2(0.001))(x_skip)
    x_skip = BatchNormalization()(x_skip)

    x = Add()[x,x_skip]
    x = Activation('relu')(x)

    return x

def resnet50():
    input_im = Input(shape=(train_im.shape[1], train_im.shape[2], train_im.shape[3]))
    x = ZeroPadding2D(padding=(3, 3))(input_im)

    x = Conv2D(64,kernel_size=(7,7),strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')
    x = MaxPooling2D((3,3),strides=(2,2))(x)

    # 2nd stage

    x = res_conv(x,s=1,filters=(64,256))
    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))

    # 3rd stage

    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))

    # 4th stage

    x = res_conv(x, s=2, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))

    # 5th stage

    x = res_conv(x, s=2, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))

    x =AveragePooling2D((2,2),padding='same')(x)

    x = Flatten()(x)
    x = Dense(len(class_types), activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs=input_im, outputs=x, name='Resnet50')

    return model
