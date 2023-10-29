import streamlit as st
import pandas as pd
import pickle
import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu

#navigasi sidebar
# horizontal menu
selected2 = option_menu(None, ["Data", "Modelling", 'Implementasi'], 
    icons=['cloud-upload', "list-task", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")


#halaman Data
if (selected2 == 'Data') :
    st.title('deskripsi data')
    st.write("Data ini berisi tentang pengklasifikasian molekul bernama sirtuin6 dimana energi yang terkandung didalamnya termasuk rendah atau tinggi")
    df = pd.read_csv('SIRTUIN6_excel.csv')
    st.write(df)
    data = pd.read_csv('sirtuin6.csv')
    st.write(data)



#halaman modelling
if (selected2 == 'Modelling'):
    st.title('Modelling')

    pilih = st.radio('Pilih', ('KNN', 'Random Forest', 'ANN'))

    if (pilih == 'KNN'):
        st.title(' Nilai Akurasi 0,90%')
    elif (pilih == 'Random Forest'):
        st.title(' Nilai Akurasi 0,95%')
    elif (pilih == 'ANN'):
        st.title(' Nilai Akurasi 0,95%')
         
#halaman Implementasi
if (selected2 == 'Implementasi'):
    st.title('Implementasi')

    # membaca model
    #loadknn = pickle.load(open('knn_sirtuin6.pkl', 'rb'))

    # judul web
    st.title('Aplikasi Prediksi Sirtuin6')

    #membagi kolom
    col1, col2, col3 = st.columns(3)

    with col1:
        SC_5 = st.number_input('SC_5', format="%.6f")
        SP_6 = st.number_input('SP_6', format="%.6f")
    with col2: 
        SHBd = st.number_input('SHBd', format="%.6f")	
        minHaaCH = st.number_input('minHaaCH', format="%.6f")	
    with col3:
        maxwHBa = st.number_input('maxwHBa', format="%.6f")	
        FMF = st.number_input('FMF', format="%.6f")	


    button = st.button('Klasifikasi sirtuin6')
    if button:
        #if SC_5 !=0 and SP_6 !=0 and SHBd !=0 and minHaaCH !=0 and maxwHBa !=0 and FMF !=0 :
        fitur = {
                "SC_5":SC_5,
                "SP_6":SP_6,
                "SHBd":SHBd,
                "minHaaCH":minHaaCH,
                "maxwHBa":maxwHBa,
                "FMF":FMF,
            }
        data = pd.DataFrame(fitur,index=[0])
        import pickle
        with open('scaler_sirtuin6.pkl','rb') as prepro:
            skala = pickle.load(prepro)
            data_norm = skala.transform(data)
        st.write("data hasil normalisasi")
        st.write(data_norm)
        with open('rf_sirtuin6.pkl','rb') as model_RF :
            loadknn = pickle.load(model_RF)
            predict_RF = loadknn.predict(data_norm)
        if predict_RF == '1':
            st.write('Low_BFE')
        else:
            st.write('High_BFE')
            


        st.write("SC_5:", SC_5)
        st.write("SP_6:", SP_6)
        st.write("SHBd:", SHBd)
        st.write("minHaaCH:", minHaaCH)
        st.write("maxwHBa:", maxwHBa)
        st.write("FMF:", FMF)

        #else:
           # st.write('kolom belum terisi')




    # code untuk prediksi
    #sirtuin6 = ''

    # membuat tombol untuk prediksi
    #if st.button('Test Prediksi Sirtuin6'):
        #sirtuin6_predict = loadknn.predict([[SC_5,SP_6,SHBd,minHaaCH,maxwHBa,FMF]])
        #st.write(SC_5,SP_6,SHBd,minHaaCH,maxwHBa,FMF)
        #if(sirtuin6_predict == 1):
           # sirtuin6 = 'High_BFE'
       # else:
           # sirtuin6 = 'Low_BFE'
        
        #st.success(sirtuin6)
        