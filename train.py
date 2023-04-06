import os
import data_loader
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

def train(train_data, val_data):
    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    model.summary()

    model_checkpoint = ModelCheckpoint('./model/face_classify4.h5',
                                       monitor='loss',
                                       verbose=1,
                                       mode='min',
                                       period=1,
                                       save_weights_only=False,
                                       save_best_only=True)

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])

    history = model.fit_generator(train_data, epochs=100, validation_data=val_data, verbose=1,
                                  callbacks=[model_checkpoint])

    return history

def analysis(model):
    acc = model.history['accuracy']
    val_acc = model.history['val_accuracy']
    loss = model.history['loss']
    val_loss = model.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.show()

if __name__ == "__main__":
    train_data = data_loader.pre_process()[0]
    val_data = data_loader.pre_process()[1]

    analysis(train(train_data, val_data))

