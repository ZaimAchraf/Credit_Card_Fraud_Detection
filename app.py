import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

header = st.container()
dataset = st.container()
analyse = st.container()
analyse1 = st.container()
analyse2 = st.container()
analyse3 = st.container()
features = st.container()
training = st.container()
training1 = st.container()
training2 = st.container()
training3 = st.container()

st.sidebar.write('''
# Credit Card Fraud Detection
''')
page = st.sidebar.selectbox("", options=["Models Description", "Predict New Data"], index = 0)

with header:
    st.title('Welcome to my interface')

    if(page == "Models Description"):
        

        with dataset:
            st.header("Transactions dataset")
            data = pd.read_csv("../creditcard.csv")
            st.write(data.head())
            classes = data['Class'].value_counts()

        with analyse:
            st.subheader("#Data Description")
            st.header("Data Analysis")

            st.write(data.describe().transpose())

            with analyse1:
                st.subheader("#Deal with Null Values")
                col1, col2, col3 = st.columns(3)
                infos = data.isnull().sum()
                infos.columns = ["Column", "Sum"]
                
                col1.write(" ")
                col2.write(infos)
                col3.write(" ")

                st.markdown("**We can esily see that there is no \nnull_values in our dataset \nso we don't have to deal with that**")

            with analyse2 :
                st.subheader("#Count values of each Class")
                col1, col2 = st.columns(2)
                classes = classes.reset_index()
                classes.columns = ["Class", "Count"]
                col1.write(" ")
                col1.write(" ")
                col1.write(" ")
                col1.write(" ")
                col1.write(" ")
                col1.write(classes)
                fig = px.pie(classes, values='Count', names='Class')
                fig.update_layout(width=300,height=300)
                col2.write(fig)
                st.markdown("**We can see that data is equaly \ndistributed between classes so\nin this case we need to extract just\na part of the major class**")
                
                st.subheader("Means befor extraction data")
                st.write(data.groupby('Class').mean())

                legal_trans = data[data['Class'] == 0]
                fraud       = data[data['Class'] == 1]
                legal_sample = data.sample(n=fraud.shape[0])
                new_data = pd.concat([legal_sample, fraud], axis=0)
                
                st.subheader("Means after extraction data")
                st.write(new_data.groupby('Class').mean())

                col1, col2 = st.columns(2)
                classes = new_data['Class'].value_counts()
                classes = classes.reset_index()
                classes.columns = ["Class", "Count"]
                col1.write(" ")
                col1.write(" ")
                col1.write(" ")
                col1.write(" ")
                col1.write(" ")
                col1.write(classes)
                fig = px.pie(classes, values='Count', names='Class')
                fig.update_layout(width=300,height=300)
                col2.write(fig)
                st.markdown("**Now the data is more suitable for our model**")
                







        with features:
            st.header('Correlation Annalysis')
            fig, ax = plt.subplots()
            sns.heatmap(data.corr(), ax=ax)
            st.write(fig)

        with training:
            st.header('Model Training')

            with training1 :
                st.text(" ")
                st.text(" ")

                st.subheader("#Spliting data")
                st.markdown("**What should be the size of test dataset (%) ?**")
                test_size = st.slider("", min_value = 10, max_value = 60, value = 30, step = 10)
                
                st.subheader("#Logistique Regression Model")
                col1, col2 = st.columns(2)

                col1.markdown("**Choose the max_iter parameter**")
                n = col1.slider("  ", min_value = 1, max_value = 10, value = 5, step = 1)

                X = new_data.drop(columns='Class', axis=1)
                Y = new_data.Class 
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size/100)

                
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.fit_transform(X_test)
                st.write(X_train)

                model = LogisticRegression()
                model.fit(X_train, Y_train)
                train_prediction = model.predict(X_train)
                train_accuracy = accuracy_score(train_prediction, Y_train)
                test_prediction = model.predict(X_test)
                test_accuracy = accuracy_score(test_prediction, Y_test)
                col2.write('''
                #### Accuracy obtained in train
                ''')
                col2.write(train_accuracy)
                col2.write('''
                #### Accuracy obtained in test
                ''')
                col2.write(test_accuracy)

                
            with training2 :
                st.text(" ")
                st.text(" ")
                st.text(" ")
                st.text(" ")
                st.subheader("#Support Vector Machine Model")
                col1, col2 = st.columns(2)
                # col1.markdown("**What should be the size of test dataset (%) ?**")
                # test_size = col1.slider(" ", min_value = 10, max_value = 60, value = 30, step = 10)
                
                col1.markdown("**Choose a degree to use with SVM**")
                Degree = col1.slider("  ", min_value = 1, max_value = 9, value = 3, step = 1)

                # Scores = []
                # degrees = [i for i in range(1, 10)] 

                # for i in range(1,10):
                #     model = SVC(kernel="poly",degree=i)
                #     model.fit(X_train, Y_train);
                #     Scores.append(1 - model.score(X_test, Y_test)) 
                
                # chart_data = pd.DataFrame(
                # Scores,
                # columns=['Scores'])


                # st.line_chart(chart_data)

                model = SVC(degree=Degree)
                model.fit(X_train, Y_train)
                train_prediction = model.predict(X_train)
                train_accuracy = accuracy_score(train_prediction, Y_train)
                test_prediction = model.predict(X_test)
                test_accuracy = accuracy_score(test_prediction, Y_test)
                col2.write('''
                #### Accuracy obtained in train
                ''')
                col2.write(train_accuracy)
                col2.write('''
                #### Accuracy obtained in test
                ''')
                col2.write(test_accuracy)

            with training3 :
                st.text(" ")
                st.text(" ")
                st.text(" ")
                st.text(" ")
                st.subheader("#K-nearest neighbors")
                col1, col2 = st.columns(2)
                # col1.markdown("**What should be the size of test dataset (%) ?**")
                # test_size = col1.slider(" ", min_value = 10, max_value = 60, value = 30, step = 10)
                
                col1.markdown("**Choose the number of neighbors**")
                n = col1.slider("   ", min_value = 1, max_value = 10, value = 5, step = 1)

                # Scores = []
                # degrees = [i for i in range(1, 10)] 

                # for i in range(1,10):
                #     model = SVC(kernel="poly",degree=i)
                #     model.fit(X_train, Y_train);
                #     Scores.append(1 - model.score(X_test, Y_test)) 
                
                # chart_data = pd.DataFrame(
                # Scores,
                # columns=['Scores'])


                # st.line_chart(chart_data)

                model = KNeighborsClassifier(n_neighbors=n)
                model.fit(X_train, Y_train)
                train_prediction = model.predict(X_train)
                train_accuracy = accuracy_score(train_prediction, Y_train)
                test_prediction = model.predict(X_test)
                test_accuracy = accuracy_score(test_prediction, Y_test)
                col2.write('''
                #### Accuracy obtained in train
                ''')
                col2.write(train_accuracy)
                col2.write('''
                #### Accuracy obtained in test
                ''')
                col2.write(test_accuracy)    
            

    if (page == "Predict New Data"):

        data_file = st.file_uploader("Upload CSV File:", type=['csv'])
        model = st.sidebar.selectbox("Select Model To use", options=["Logistic Regression", "Support Vector Machine", "K-nearest neighbors"], index = 0)

        if data_file is not None :

            data_to_predict = pd.read_csv(data_file)
            st.subheader("#Data Before Prediction")
            st.write(data_to_predict)
            
            if model == "Logistic Regression" :
                model = LogisticRegression()
            elif model == "K-nearest neighbors" :
                model = model = SVC(kernel="poly")
            else :
                model = KNeighborsClassifier(5)

            data = pd.read_csv("../creditcard.csv")

            legal_trans = data[data['Class'] == 0]
            fraud       = data[data['Class'] == 1]
            legal_sample = data.sample(n=fraud.shape[0])
            new_data = pd.concat([legal_sample, fraud], axis=0)

            X = new_data.drop(columns='Class', axis=1)
            Y = new_data.Class 
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2)

            model.fit(X_train, Y_train)
            prediction = model.predict(data_to_predict)

            predicted_data = data_to_predict
            predicted_data['Class'] = prediction

            st.subheader("#Prediction")
            col1, col2, col3 = st.columns(3)
            col1.text("")
            
            col2.write(prediction)
            col3.text("")
            
            st.subheader("#Data after Prediction")
            st.write(predicted_data)

            st.download_button(label="Download", data=predicted_data.to_csv(), file_name="result.csv", mime="text/csv")

