import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from keras.models import Model
from keras.layers import Dense, Input, Dropout, Lambda
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import Adam, SGD
from classification_models.resnet import ResNet34

kernel_initializer = 'he_normal'


def r34(input_shape=None, channels=2, lr=1e-4, weights=None, classes=4, **kwargs):

    input_shape_resnet = (None, None, 3) if K.image_data_format() == 'channels_last' else (3, None, None)
    resnet_model = ResNet34(input_shape=input_shape_resnet, include_top=False, weights='imagenet')
    resnet_model = Model(resnet_model.input, resnet_model.get_layer('stage4_unit1_relu1').output)

    if input_shape is None:
        if K.image_data_format() == 'channels_last':
            input_shape = (None, None, channels)
        else:
            input_shape = (channels, None, None)
    main_input = Input(input_shape)
    x = Convolution2D(3, (1, 1), kernel_initializer=kernel_initializer)(main_input)
    x = resnet_model(x)
    x = GlobalAveragePooling2D(name='pool1')(x)
    main_output = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model.
    model = Model(main_input, main_output, name='r34')

    if weights is not None:
        print('Load weights from', weights)
        model.load_weights(weights)

    optimizer = SGD(lr, momentum=0.95, decay=0.0005, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def r34_256(input_shape=None, channels=2, lr=1e-4, weights=None, classes=4, **kwargs):

    input_shape_resnet = (None, None, 3) if K.image_data_format() == 'channels_last' else (3, None, None)
    resnet_model = ResNet34(input_shape=input_shape_resnet, include_top=False, weights='imagenet')
    resnet_model = Model(resnet_model.input, resnet_model.get_layer('stage4_unit1_relu1').output)

    if input_shape is None:
        if K.image_data_format() == 'channels_last':
            input_shape = (None, None, channels)
        else:
            input_shape = (channels, None, None)
    main_input = Input(input_shape)
    x = Convolution2D(3, (1, 1), kernel_initializer=kernel_initializer)(main_input)
    x = resnet_model(x)
    x = GlobalAveragePooling2D(name='pool1')(x)
    main_output = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model.
    model = Model(main_input, main_output, name='r34')

    if weights is not None:
        print('Load weights from', weights)
        model.load_weights(weights)

    model_ = Model(main_input, resnet_model.get_output_at(1))
    x = model_(main_input)
    x = GlobalAveragePooling2D()(x)
    model256 = Model(main_input, x, name='r34_256')

    optimizer = SGD(lr, momentum=0.95, decay=0.0005, nesterov=True)
    model256.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model256


if __name__ == '__main__':
    model = r34_256()
    pass
