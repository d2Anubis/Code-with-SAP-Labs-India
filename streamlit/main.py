import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
import streamlit as st

import xgboost
import shap
import sklearn
import sklearn.tree 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pickle
shap.initjs()
import streamlit.components.v1 as components
st.set_page_config(page_title="Model XAI",layout='wide')
st.set_option('deprecation.showPyplotGlobalUse', False)



#code to read datasets
@st.cache
def read_dataset():
    X_train = pd.read_csv("./dataset/X_train.csv")
    X_test = pd.read_csv("./dataset/X_test.csv")
    Y_train = pd.read_csv("./dataset/Y_train.csv")
    Y_test = pd.read_csv("./dataset/Y_test.csv")
    
    return X_train, X_test, Y_train, Y_test

#code to read save models from disk
@st.cache(allow_output_mutation=True)
def get_models():
    knn = pickle.load(open('./models/knn.sav', 'rb'))
    svc_linear = pickle.load(open('./models/svc_linear.sav', 'rb'))
    linear_lr = pickle.load(open('./models/linear_lr.sav', 'rb'))
    dtree = pickle.load(open('./models/dtree.sav', 'rb'))
    rforest = pickle.load(open('./models/rforest.sav', 'rb'))
    nn = pickle.load(open('./models/nn.sav', 'rb'))

    return [knn, svc_linear, linear_lr, dtree, rforest, nn]

#template to render shap plots
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


model_names = ['KNN', 'SVM', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Neural Network']

#left side sidebar tab
usage = st.sidebar.selectbox('Select Usages', options=['Numerical Data', 'Image Data', 'Text Data'], index=1, key='usage')
sidebar_button = st.sidebar.button('Render', key='render')
st.sidebar.header("Team : d2Anubis")
st.sidebar.markdown("""
- Niharika 
- Yashwant""")


if usage == 'Numerical Data':
    st.header("**Model Explainers for Numerical Data**")

    st.markdown("<hr>", unsafe_allow_html=True)

    selected_model = st.selectbox('Select models', options=model_names, index=3, key='model_names')
    models = get_models()
    X_train, X_test, Y_train, Y_test = read_dataset()

    use_model = models[model_names.index(selected_model)]
    explainer = shap.KernelExplainer(use_model.predict_proba, X_train)


    #code to use our own datasets to test the explainer model
    with st.beta_expander("Enter your own dataset to test"):
        user_input = st.text_input("Enter your own dataset", value="1,1,0,0,0,9083,0,228,360,1,1")
        _user_input = pd.DataFrame([list(map(int, user_input.strip().split(",")))])
        _shap_values = explainer.shap_values(_user_input)
        st_shap(shap.force_plot(explainer.expected_value[0], _shap_values[0], _user_input))

    
    #code to select rows from test dataset for explanation
    with st.beta_expander("Local Explanation"):
        select_row = st.select_slider(label="Select row from Test dataset", options=range(len(X_test)), key='row')
        st.markdown("Row from dataset <hr>",unsafe_allow_html=True)
        st.dataframe(data=[X_test.iloc[select_row,:]])

        st.markdown("<hr>", unsafe_allow_html=True)
        if st.button("Run Local Explanation"):
            shap_values = explainer.shap_values(X_test.iloc[select_row,:])
            st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0], X_test.iloc[select_row,:]))

    
    #code for Global Explanation for overall dataset (time consuming)
    with st.beta_expander("Global Explanation"):
        st.text("Note: Global Explanations take time to compute")
        
        if st.button("Run Global Explanation"):
            
            shap_values = explainer.shap_values(X_test)
            st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0], X_test), 500)

elif usage == "Image Data":
    st.header("**Model Explainers for Image Data**")
    st.markdown("<hr>", unsafe_allow_html=True)

    shap_values = pickle.load(open('./models/image_shap_values.pkl', 'rb'))
    x_test = pickle.load(open('./dataset/image_x_test.pkl', 'rb'))

    image = st.select_slider(label="Select image number", options=range(len(x_test)), key='image')

    st.pyplot(shap.image_plot(shap_values, x_test[image:image+1]))




