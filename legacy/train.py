import glob
import tensorflow as tf
import random
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.utils import Sequence


class CustomDataGenerator(Sequence):
    def __init__(self, video_files, audio_files, labels_files, batch_size, shuffle=True):
        self.video_files = video_files
        self.audio_files = audio_files
        self.labels_files = labels_files
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(video_files))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.video_files) / self.batch_size))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        video_batch = np.array([np.load(self.video_files[k]) for k in indexes])
        audio_batch = np.array([np.load(self.audio_files[k]) for k in indexes])
        label_batch = np.array([np.load(self.labels_files[k]) for k in indexes])
        label_batch = to_categorical(label_batch)

        audio_batch = audio_batch.reshape((audio_batch.shape[0], 192, 1))

        # print(f"Labels shape: {label_batch.shape}")
        # print(f"Videos shape: {video_batch.shape}")
        # print(f"Audio shape: {audio_batch.shape}")

        return [video_batch, audio_batch], label_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def build_model():
    video_input = tf.keras.layers.Input(shape=(192, 128 * 128))
    audio_input = tf.keras.layers.Input(shape=(192, 1))
    
    video = tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(192, 128 * 128))(video_input)
    video = tf.keras.layers.Dropout(0.2)(video)
    video = tf.keras.layers.LSTM(128, return_sequences=True)(video)
    video = tf.keras.layers.Dropout(0.2)(video)
    video = tf.keras.layers.LSTM(128, return_sequences=True)(video)
    video = tf.keras.layers.Dropout(0.2)(video)
    video = tf.keras.layers.Dense(128, activation='relu')(video)


    audio = tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(192, 1))(audio_input)
    audio = tf.keras.layers.Dropout(0.2)(audio)
    audio = tf.keras.layers.LSTM(128, return_sequences=True)(audio)
    audio = tf.keras.layers.Dropout(0.2)(audio)
    audio = tf.keras.layers.LSTM(128, return_sequences=True)(audio)
    audio = tf.keras.layers.Dropout(0.2)(audio)
    audio = tf.keras.layers.Dense(128, activation='relu')(audio)
    
    merged = tf.keras.layers.concatenate([video, audio])
    merged = tf.keras.layers.Dense(128, activation='relu')(merged)
    merged = tf.keras.layers.Dropout(0.2)(merged)
    output = tf.keras.layers.Dense(2, activation='softmax')(merged)
    
    model = tf.keras.Model(inputs=[video_input, audio_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

def lr_scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        lr = lr * 0.1
    return lr

def train_model(video_data, audio_data, labels, batch_size):
    # Split the data into train and validation sets
    train_video_files, val_video_files, train_audio_files, val_audio_files, train_labels_files, val_labels_files = train_test_split(video_data, audio_data, labels, test_size=0.2)
    
    train_generator = CustomDataGenerator(train_video_files, train_audio_files, train_labels_files, batch_size)
    val_generator = CustomDataGenerator(val_video_files, val_audio_files, val_labels_files, batch_size)

    # Build the model
    model = build_model()

    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    lr_callback = LearningRateScheduler(lr_scheduler)
    
    # Train the model on the train data
    model.fit(
        train_generator,
        epochs=50,
        steps_per_epoch=len(train_video_files) // batch_size,
        validation_data=val_generator,
        validation_steps=len(val_video_files) // batch_size,
        callbacks=[early_stopping, lr_callback]
    )

    model.save('trained_model.h5')
    
    return model

video_features_files = glob.glob('training_dir/processed_video*.npy')
audio_features_files = glob.glob('training_dir/processed_audio*.npy')
labels_files = glob.glob('training_dir/labels*.npy')

# Example usage

model = train_model(video_features_files, audio_features_files, labels_files, 32)
