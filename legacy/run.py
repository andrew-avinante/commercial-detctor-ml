import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def predict_video(video_file, model_file):
    # Load video data from .npy file
    video = np.load(video_file)
    
    # Load the model from .h5 file
    model = tf.keras.models.load_model(model_file, custom_objects={'KerasLayer':hub.KerasLayer})
    
    # Predict the label for each frame in the video
    predictions = model.predict(video)
    
    return predictions

if __name__ == '__main__':
    # Replace with the actual file paths
    video_file = 'test_dir/processed_video_0.npy'
    model_file = 'trained_model.h5'
    video_features = np.load(video_file)
    predictions = predict_video(video_file, model_file)

    isFade = False
    count = 0
    result = np.array(np.argmax(predictions, axis=1))
    print(result)
    for i in result:
        count += 1
        if i == 1 and isFade == False:
            start = count / 24
            print(f"Start: {start}, Time: {count}")
            isFade = True
        elif i == 0 and isFade == True:
            print(f"End: {count / 24}, Time: {count}")
            isFade = False
