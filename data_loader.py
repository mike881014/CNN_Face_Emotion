from keras_preprocessing.image import ImageDataGenerator

def pre_process():
    TRAINING_DIR = './train/'
    training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.15
    )

    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        subset='training',
        target_size=(48, 48),
        class_mode='categorical'
    )

    validation_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        subset='validation',
        target_size=(48, 48),
        class_mode='categorical'
    )

    return train_generator, validation_generator
