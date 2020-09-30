import streamlit as st 
import numpy as np
import pickle

import pandas as pd

pickle_in = open("classifier.pkl" , 'rb')

classifier = pickle.load(pickle_in)


def welcome():
    
    return "welcome all"


def predict_note_authentication(variance,skewness,curtosis,entropy):
    
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    
    return prediction

def main():

    st.title("Bank Authenticator")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """

    st.markdown(html_temp , unsafe_allow_html = True)

    variance = st.text_input("variance", "Type Here")
    skewness = st.text_input("skewness", "Type Here")
    curtosis = st.text_input("curtosis", "Type Here")
    entropy = st.text_input("entropy", "Type Here")

    result =  ""

    if st.button("PREDICT"):

        result  = predict_note_authentication(variance,skewness,curtosis,entropy)

    st.success("The Output is {}".format(result))

    if st.button("ABOUT"):

        st.text("Lets Learn")
        st.text("Built a Project Using Streamlit")

 


if __name__ == "__main__":
    main()
