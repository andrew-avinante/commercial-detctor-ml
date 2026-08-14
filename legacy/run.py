import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def predict_video(video_file, audio_file, model_file):
    # Load video data from .npy file
    video = np.array([np.load(video_file)])
    audio = np.array([np.load(audio_file)])

    # audio = audio.reshape(-1, 1)
    # audio = np.hstack((audio, audio))

    audio = audio.reshape((1, 192, 1))
    print(f"Videos shape: {video.shape}")
    print(f"Audio shape: {audio.shape}")

    # Load the model from .h5 file
    model = tf.keras.models.load_model(model_file, custom_objects={'KerasLayer':hub.KerasLayer})
    
    # Predict the label for each frame in the video
    predictions = model.predict([video, audio], batch_size=1)
    
    return predictions

if __name__ == '__main__':
    # Replace with the actual file paths
    prefix = 'test'
    num = '004'

    video_file = f'{prefix}_dir/processed_video_{num}.npy'
    audio_file = f'{prefix}_dir/processed_audio_{num}.npy'
    model_file = 'trained_model.h5'
    predictions = predict_video(video_file, audio_file, model_file)

    isFade = False
    count = 0
    result = np.argmax(predictions, axis=2)
    print(result)
    for i in result[0]:
        count += 1
        if i == 1 and isFade == False:
            start = count / 24
            print(f"Start: {start}, Time: {count}")
            isFade = True
        elif i == 0 and isFade == True:
            print(f"End: {count / 24}, Time: {count}")
            isFade = False
