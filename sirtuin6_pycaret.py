import streamlit as st      
import pandas as pd         
import pickle
import numpy as np
from streamlit_option_menu import option_menu
from pycaret.classification import *
from PIL import Image

#navigasi sidebar
# horizontal menu
selected2 = option_menu(None, ["Data", "Implementasi", "Profil"], 
    icons=['cloud-upload', "list-task", 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")


#halaman Data
if (selected2 == 'Data') :
    st.title('Deskripsi Data')
    st.write("Data ini berisi tentang pengklasifikasian molekul bernama sirtuin6 dimana energi yang terkandung didalamnya termasuk rendah atau tinggi")
    df = pd.read_csv('SIRTUIN6_excel.csv')
    st.write(df)


         
#halaman Implementasi
if (selected2 == 'Implementasi'):
    st.title('Implementasi')

    # judul web
    st.title('Aplikasi Prediksi Sirtuin6')

    #membagi kolom
    SC_5 = st.number_input('SC_5', format="%.6f")
    on = st.checkbox('Penjelasan SC_5')
    if on:
        st.write('SC_5 merupakan parameter sequential dari molekul sirtuin6')

    SP_6 = st.number_input('SP_6', format="%.6f") 
    on = st.checkbox('Penjelasan SP_6')
    if on:
        st.write('SP_6 merupakan karakteristik posisi ke-6 dalam molekul sirtuin6')

    SHBd = st.number_input('SHBd', format="%.6f")
    on = st.checkbox('Penjelasan SHBd')
    if on:
        st.write('SHBd merupakan jumlah atom hidrogen yang terkandung dalam molekul')

    minHaaCH = st.number_input('minHaaCH', format="%.6f")
    on = st.checkbox('Penjelasan minHaaCH')
    if on:
        st.write('minHaaCH merupakan jumlah minimum atom yang terkandung dalam molekul')

    maxwHBa = st.number_input('maxwHBa', format="%.6f")
    on = st.checkbox('Penjelasan maxwHBa')
    if on:
        st.write('maxwHBa merupakan berat maksimum atom yang terkandung dalam molekul')

    FMF = st.number_input('FMF', format="%.6f")
    on = st.checkbox('Penjelasan FMF')
    if on:
        st.write('FMF merupakan ukuran molekul / atom')


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
        dt = pd.DataFrame(fitur,index=[0])
        import pickle
        loaded_model = load_model('dt_sirtuin6')
        prediction = predict_model(loaded_model,data=dt)
        hasil_prediksi = prediction['prediction_label']
        st.write("data hasil prediksi")
        st.write(hasil_prediksi)
        for i in hasil_prediksi:
            if i == 'Low_BFE':
                st.write('Low_BFE')
            elif i == 'High_BFE':
                st.write('High_BFE')
            else:   
                st.write('Tidak ada prediksi yang sesuai')


#halaman profil
if (selected2 == 'Profil'):
    st.title('Biodata')

    st.write('Nama : Indah Kharisma')
    st.write('NIM : 210411100147')
    st.write('Kelas : PSD A')

