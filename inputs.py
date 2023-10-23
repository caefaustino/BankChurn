import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def prepare(x_in):
    # Uploading csv file
    df = pd.read_csv('BankChurners.csv')
    df.head()

    # Changing values in column "Gender" 0 = M // 1 = F
    df['Gender'] = df['Gender'].replace({'M':0, 'F':1})
    x_in['Gender'] = x_in['Gender'].replace({'M':0, 'F':1})

    # Dropping column CLIENTNUM
    df = df.drop(columns='CLIENTNUM')

    cat_vars = ['Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
    num_vars = ['Customer_Age', 'Gender', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
    model_vars = cat_vars + num_vars

    x = df.filter(model_vars)

    x1 = pd.concat([x, x_in])

    x1 = pd.get_dummies(x1)

    scaler = MinMaxScaler()

    x1 = pd.DataFrame(scaler.fit_transform(x1), index=x1.index, columns=x1.columns)

    x_in = x1.tail(len(x_in))

    return(x_in)