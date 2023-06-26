import numpy as np
from PIL import Image
from scipy.io.wavfile import read
from os import path

def ImgToNpConverter()->np.ndarray:
    if path.exists("./Data/Img_np.npy"):
        ret_np_img = np.load("./Data/Img_np.npy")
        print("Image to Numpy Load Completed")
        return ret_np_img
    
    ret_np_img = np.empty(shape=(0, 128, 128, 3), dtype=np.uint8)
    for i in range(300):
        route      = "./Data/Images/" + str(i) + ".jpg"
        img        = Image.open(route)
        np_img     = np.asarray(img, dtype=np.uint8)
        np_img     = np_img.reshape(1, 128, 128, 3)
        ret_np_img = np.append(ret_np_img, np_img, axis=0)

    np.save("./Data/Img_np.npy", ret_np_img)
    print("Image to Numpy Converter Completed")
    return ret_np_img

def WavToNpConverter()->np.ndarray:
    if path.exists("./Data/Wav_np.npy"):
        ret_np_wav = np.load("./Data/Wav_np.npy")
        print(".wav to Numpy Load Completed")
        return ret_np_wav
    
    ret_np_wav = np.empty(shape=(0, 1, 32768), dtype=np.float32)
    for i in range(300):
        route      = "./Data/Sounds/" + str(i) + ".wav"
        sound   = read(route)[1]
        np_wav     = np.fft.fft(sound)
        np_wav     = np.fft.fftshift(np_wav)
        np_wav     = np_wav.reshape(1, 1, 32768)
        ret_np_wav = np.append(ret_np_wav, np_wav, axis=0)
    
    np.save("./Data/Wav_np.npy", ret_np_wav)
    print(".wav to Numpy Converter Completed")
    return ret_np_wav
