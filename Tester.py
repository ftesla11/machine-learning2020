# imports
import os, glob, pickle,random
import librosa
import librosa.display
import soundfile
import numpy as np  
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import Recorder as recorder

#DataFlair - Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

#DataFlair - Emotions to observe
observed_emotions=['neutral', 'calm', 'happy', 'sad','angry','fearful', 'disgust','surprised']
print('Welcome to the Emotion Analyser!')
#DataFlair - Load the data and extract features for each sound file
def load_data(glob_pattern="/dataset/Actor_*/*.wav"):
    X,x,y=[],[],[]
    for file in glob.glob(glob_pattern):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions: 
            continue
        Xo, feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        X.append(Xo)
        x.append(feature)
        y.append(emotion)
    return X,x,y
#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        flatness = np.mean(librosa.feature.spectral_flatness(y=X))
        contrast = np.mean(librosa.feature.spectral_contrast(y=X))
        result = np.hstack((result, flatness, contrast))
    return (X, result)

print('Press m for multiplayer Perceptron or k for K-nearest neighbour')
inp = input()
if inp == 'm':
    f = open('model.pickle','rb')
    model = pickle.load(f)    
    f.close()
else:
    f = open('knn.pickle','rb')
    model = pickle.load(f)    
    f.close()

while(True):
    print('Read: The kids are talking by the door')
    recorder.record_to_file('test/Actor_01/02-01-03-02-01-01-08.wav')

    Z,z,w = load_data(os.getcwd() + "/test/Actor_*/*.wav")
    new_test = np.array(z)
    new_pred=model.predict(new_test)
    #os.remove('test/Actor_01/02-01-03-02-01-01-08.wav')
    print(new_pred) 
    print('Press q to quit or anything else to keep going')
    inp = input()
    if inp == 'q':
        exit()