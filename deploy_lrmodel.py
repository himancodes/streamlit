import streamlit as st
import pickle

# Load the pickled model
with open('lrmodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Create the Streamlit web app
st.header("Diamond Price Model")

st.sidebar.header("This is a web app")

depth = float(st.number_input("Enter depth"))
table = float(st.number_input("Enter table"))

st.write("Depth is:", depth)
st.write("Table is:", table)

yhat_test = model.predict([[depth,table]])

st.write("b0 is", round(model.intercept_, 3))
st.write("b1 is", round(model.coef_[0], 3))
st.write("yhat test is", yhat_test)
