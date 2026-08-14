import glob
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def build_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(32, input_shape=input_shape, return_sequences=True))
    model.add(tf.keras.layers.LSTM(32))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

# Search the training directory for the required files
video_features_files = glob.glob('training_dir/processed_video*.npy')
# audio_features_files = glob.glob('training_dir/audio_features*.npy')
labels_files = glob.glob('training_dir/labels*.npy')

# Load the video and audio features into arrays
video_features = np.concatenate([np.load(f) for f in video_features_files])
# audio_features = np.concatenate([np.load(f) for f in audio_features_files])

# Load the labels into an array
labels = np.concatenate([np.load(f) for f in labels_files])

def train_model(video_data, labels):
    # Split the data into train and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(video_data, labels, test_size=0.2)
    
    # Get the input shape for the first video
    input_shape = train_data[0].shape
    
    # Build the model
    model = build_model(input_shape)
    
    # Train the model on the train data
    history = model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))
    
    # Save the trained model
    model.save('trained_model.h5')
    
    return model

# Example usage
labels = to_categorical(labels)

print(f"Labels shape: {labels.shape}")
print(f"Videos shape: {video_features.shape}")

model = train_model(video_features, labels)
