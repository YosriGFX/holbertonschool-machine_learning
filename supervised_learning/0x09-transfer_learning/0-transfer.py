#!/usr/bin/env python3
'''A python script that trains a convolutional
neural network to classify the CIFAR 10 dataset'''
import tensorflow.keras as K


if __name__ == "__main__":
    (xtrain, ytrain), (xtest, ytest) = K.datasets.cifar10.load_data()
    img_preprocessor = K.applications.densenet.preprocess_input
    igen = K.preprocessing.image.ImageDataGenerator
    xgen = igen(
        preprocessing_function=img_preprocessor,
        zoom_range=0.2,
        rotation_range=30,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    trainFlow = xgen.flow(
        xtrain,
        K.utils.to_categorical(ytrain),
        batch_size=100
    )
    testFlow = igen(
        preprocessing_function=img_preprocessor
    )
    testFlow = testFlow.flow(
        xtest,
        K.utils.to_categorical(ytest),
        batch_size=250
    )
    if True:
        base_model = K.applications.DenseNet121(
            input_shape=xtrain.shape[1:],
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        out = K.layers.Dense(
            1000,
            activation='sigmoid',
            kernel_initializer='he_normal'
        )(base_model.output)
        out = K.layers.Dense(
            10,
            activation='softmax',
            kernel_initializer='he_normal'
        )(out)
        model = K.models.Model(
            base_model.input, out
        )
        model.compile(
            loss='categorical_crossentropy',
            metrics=['acc'],
            optimizer='sgd'
        )
    else:
        model = K.models.load_model('cifar10.h5')
    checkP = K.callbacks.ModelCheckpoint(
        'cifar10.h5', 'val_acc', 0, True
    )
    stop = K.callbacks.EarlyStopping(
        patience=100, verbose=1
    )
    model.fit_generator(
        trainFlow,
        steps_per_epoch=500,
        epochs=1000000,
        validation_data=testFlow,
        shuffle=True,
        validation_steps=200,
        verbose=1,
        callbacks=[checkP, stop]
    )


def preprocess_data(X, Y):
    '''Function that pre-processes
    the data for your model'''
    return K.applications.densenet.preprocess_input(
        X
    ), K.utils.to_categorical(Y)
