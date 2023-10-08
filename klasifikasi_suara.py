import streamlit as st
import pandas as pd
import pickle
import librosa
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from streamlit_option_menu import option_menu

#navigasi sidebar
# horizontal menu
selected2 = option_menu(None, ["Data", "Preprocessing data", "Modelling", 'Implementasi'], 
    icons=['house', 'cloud-upload', "list-task", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

#halaman Data
if (selected2 == 'Data') :
    st.title('Menampilkan Data yang Sudah di Normalisasi')

    st.write("Ini adalah data audio yang diambil dari kaggle")
    st.write("Data ini berisi 10 kolom klasifikasi")

    audio = st.file_uploader('Upload Audio',['mp3','wav'])
    if audio:
        st.audio(audio)
        y, sr=librosa.load(audio)
        # UNTUK MENGHITUNG NILAI ZCR
        zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y=y))
        zcr_median = np.median(librosa.feature.zero_crossing_rate(y=y))
        zcr_std_dev = np.std(librosa.feature.zero_crossing_rate(y=y))
        zcr_kurtosis = kurtosis(librosa.feature.zero_crossing_rate(y=y)[0])
        zcr_skew = skew(librosa.feature.zero_crossing_rate(y=y)[0])

        # UNTUK MENGHITUNG NILAI RMSE
        rmse = np.sum(y**2) / len(y)
        rmse_median = np.median(y**2)
        rmse_std_dev = np.std(y**2)
        rmse_kurtosis = kurtosis(y**2)
        rmse_skew = skew(y**2)

        fitur={
            'ZCR Mean':zcr_mean, 
            'ZCR Median':zcr_median, 
            'ZCR Std Dev':zcr_std_dev, 
            'ZCR Kurtosis':zcr_kurtosis, 
            'ZCR Skew':zcr_skew, 
            'RMSE':rmse, 
            'RMSE Median':rmse_median, 
            'RMSE Std Dev':rmse_std_dev, 
            'RMSE Kurtosis':rmse_kurtosis, 
            'RMSE Skew':rmse_skew
        }
        data = pd.DataFrame(fitur,index=[0])
        st.write('Data Audio')
        st.write(data)
        with open('scaler.pkl','rb') as prepro:
            skala = pickle.load(prepro)
        data_norm = skala.transform(data)
        st.write('Data hasil Normalisasi')
        st.write(data_norm)
        with open('knn_audio.pkl','rb') as model:
            knn = pickle.load(model)
        predict = knn.predict(data_norm)
        for i in predict:
            st.write('Emosinya adalah',i)




