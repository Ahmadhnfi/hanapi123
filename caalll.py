import pickle
import numpy as np
import streamlit as st


#load save model
RFC=pickle.load(open('rf_call.pkl','rb'))
scale=pickle.load(open('sc_call.pkl','rb'))

#judul web
st.title("prediksi klaster covid")
primaryColor="#F63366"
backgroundColor="#FFFFFF"

#untuk input data
col1, col2=st.columns(2)
with col1:
    Case_Type=st.text_input("Case_Type")
    if Case_Type != '':
        Case_Type = float(Case_Type)  # Konversi ke float
with col2:
    Cases=st.text_input("Cases")
    if Cases != '':
        Cases = float(Cases)  # Konversi ke float
with col1:
    Difference=st.text_input("Difference")
    if Difference != '':
        Difference = float(Difference)  # Konversi ke float
with col2:
    Country_Region=st.text_input("Country_Region")
    if Country_Region != '':
        Country_Region = float(Country_Region)  # Konversi ke float


#kode untuk predikisi
Prediksi_klaster_covid =''
if st.button("Prediksi Klaster covid SEKARANG"):
    # Mengubah argumen menjadi array numpy dua dimensi
    sc=scale.transform([[Cases,Difference]])
    # Melakukan prediksi dengan XGBoost
    Prediksi_klaster = RFC.predict([[Case_Type,sc[0][0],sc[0][1],Difference]])
    
    if Prediksi_klaster[0]==0:
        Prediksi_klaster_covid ="0"
    elif Prediksi_klaster[0] == 3:
        Prediksi_klaster_covid = "3"
    elif Prediksi_klaster[0] == 4:
        Prediksi_klaster_covid = "4"
    elif Prediksi_klaster[0] == 2:
        Prediksi_klaster_covid = "2"
    elif Prediksi_klaster[0] == 1:
        Prediksi_klaster_covid = "1"
    else:
        Prediksi_klaster_covid = "tidak ditemukan klaster"

st.success(Prediksi_klaster_covid)