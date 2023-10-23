import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

import streamlit as st
import inputs

import pickle
import lightgbm as lgb

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Bank Customer Churn Prediction')
st.image('Churn.png')
st.write('This app will predict if a customer is going to churn or not.')

cat_vars = ['Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
num_vars = ['Customer_Age', 'Gender', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
model_vars = cat_vars + num_vars

df_input = pd.DataFrame(columns=model_vars)

choice = st.selectbox('How do you prefer input customer data?', ['Type manually parameters', 'Upload .csv'])

if choice == 'Type manually parameters':
    Education_Level = st.selectbox('Education Level', ('Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate', 'Unknown'))
    Marital_Status = st.selectbox('Marital Status', ('Single', 'Married', 'Divorced', 'Unknown'))
    Income_Category = st.selectbox('Income Category', ('Less than $40K', '$40K - $60K', '$60K - $80K', '$80K - $120K', '$120K +', 'Unknown'))
    Card_Category = st.selectbox('Card Category' , ('Blue', 'Silver', 'Gold', 'Platinum'))
    Gender = st.selectbox('Gender', ('M', 'F'))   
    Customer_Age = st.number_input('Customer Age', step=1, format='%d')
    Dependent_count = st.number_input('Dependents', step=1, format='%d')
    Months_on_book = st.number_input('Months on book', step=1, format='%d')
    Total_Relationship_Count = st.number_input('Number of products with the client', step=1, format='%d')
    Months_Inactive_12_mon = st.slider('Number of inactive months in the last 12 months', 0,12,1, step=1)
    Contacts_Count_12_mon = st.number_input('Number of contacts in the last 12 months', step=1, format='%d')
    Credit_Limit = st.number_input('Credit Card Limit')
    Total_Revolving_Bal = st.number_input('Total Revolving Balance')
    Avg_Open_To_Buy = st.number_input('Open to buy - Average last 12 months')
    Total_Amt_Chng_Q4_Q1 = st.number_input('Total Amount Changing Q4-Q1')
    Total_Trans_Amt = st.number_input('Total Transations Amount last 12 months')
    Total_Trans_Ct = st.number_input('Number of transations last 12 months', step=1, format='%d')
    Total_Ct_Chng_Q4_Q1 = st.number_input('Total count changing Q4-Q1')
    Avg_Utilization_Ratio = st.number_input('Credit Card utilization ratio')

    df_input.loc[0] = [Education_Level, Marital_Status, Income_Category, Card_Category, Customer_Age, Gender, Dependent_count, Months_on_book, Total_Relationship_Count, Months_Inactive_12_mon, Contacts_Count_12_mon, Credit_Limit, Total_Revolving_Bal, Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio]
    df1_input = inputs.prepare(df_input)

    pred = pd.DataFrame(model.predict(df1_input))
    pred.rename(columns={0:'Churn'}, inplace=True)
    pred['Churn'] = pred['Churn'].replace({0:'No', 1:'Yes'})

    st.markdown('Client is going to churn?')
    st.title(str(pred['Churn'][0]))

else:
    st.write('Please upload a CSV file with data in the correct order:')
    st.write('Education_Level, Marital_Status, Income_Category, Card_Category, Customer_Age, Gender, Dependent_count, Months_on_book, Total_Relationship_Count, Months_Inactive_12_mon, Contacts_Count_12_mon, Credit_Limit, Total_Revolving_Bal, Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio')
    upload_file = st.file_uploader('Choose a file')
    if upload_file is not None:
        df_input = pd.read_csv(upload_file)

        df1_input = inputs.prepare(df_input)

        pred = pd.DataFrame(model.predict(df1_input))
        pred.rename(columns={0:'Churn'}, inplace=True)
        pred['Churn'] = pred['Churn'].replace({0:'No', 1:'Yes'})

        st.dataframe(pred, use_container_width=True)
        st.download_button(label='Download CSV', data=pred.to_csv(index=False).encode('utf-8'), mime='text/csv', file_name='predictions.csv')














