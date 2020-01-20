import pickle
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from matplotlib import cm


train_pickle_in = open(os.getcwd() + "/sample_data/DEMO/DEMO.data/train.pkl", "rb")
test_pickle_in = open(os.getcwd() +"/sample_data/DEMO/DEMO.data/test.pkl", "rb")

train = pickle.load(train_pickle_in)
test = pickle.load(test_pickle_in)


def extract_mfcc(data,sr=16000):
    results = []
    counter = 0
    for d in data:
        counter += 1
        r = librosa.feature.mfcc(d,sr=16000,n_mfcc=13)
        r = r.transpose()
        print(r.shape)
        if counter % 20 == 0:
            plt.plot(d)

            mfcc_data = np.swapaxes(r, 0, 1)
            fig, ax = plt.subplots()
            cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')
            ax.set_title('MFCC')
            ax.set_xlabel("Features")
            ax.set_ylabel("Time")
            #Showing mfcc_data
            plt.show()
            plt.plot(r)
            plt.show()
        results.append(r)
    return results

results = extract_mfcc(test)
