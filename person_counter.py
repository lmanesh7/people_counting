import streamlit as st
import pandas as pd
import pyrebase

config = {
    "apiKey": "AIzaSyBz9K9JuS4R6xIJrkVCA6kJ6BcuO_kx9aI",
    "authDomain": "test-a9823.firebaseapp.com",
    "databaseURL": "https://test-a9823-default-rtdb.firebaseio.com",
    "projectId": "test-a9823",
    "storageBucket": "test-a9823.appspot.com",
    "messagingSenderId": "1097818977650",
    "appId": "1:1097818977650:web:85b5d03d168fe3db7259eb",
    "measurementId": "G-KN10KC82JL"

}

firebase = pyrebase.initialize_app(config)
db = firebase.database()

user = db.child("person counter").get()
us = user.val()
keys_list = []
vals_list = []
for i in us.values():
    # print(i)
    for j in i:
        keys_list.append(j)
        t = i[j]
        vals_list.append(t)
l = []
for i in keys_list:
    if i != "date":
        l.append(int(i))
opc = []
dat = []
for i in range(len(vals_list)):
    if i%2==0:
        opc.append(vals_list[i])
    else:
        dat.append(vals_list[i])


df = pd.DataFrame({"lpc": l, "Time": dat,"Total":opc})

st.title("people counter")
st.write("No of persons in the present frame: %d" %(l[-1]))
st.write("No of persons in the total frames: %d" %(opc[-1]))
if st.checkbox("Show total db"):
    st.write(df)
