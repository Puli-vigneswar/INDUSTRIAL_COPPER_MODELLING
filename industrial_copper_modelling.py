import streamlit as st 
import numpy as np 
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelBinarizer

#def function for predicting the selling price based on regreesion model

def get_sp():
    with open('D:/copperman/decisiontreemodel.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
        
    with open('D:/copperman/standardscaler.pkl', 'rb') as file:
        scaler_loaded = pickle.load(file)
        
    with open('D:/copperman/onehotencoder.pkl', 'rb') as file:
        ohe_loaded1 = pickle.load(file)
        
    with open('D:/copperman/onehotencoder2.pkl', 'rb') as file:
        ohe_loaded2 = pickle.load(file)
        #tons_quantity_log','application','thickness_log', 'width','country','customer','product_ref',"item_type","status"
    new_sample= np.array([[np.log(float(quantity_tons1)),application1,np.log(float(thickness1)),
                           float(width1),country1,float(customer1),
                           int(prref1),item_type1,status1]])
    new_sample_ohe = ohe_loaded1.transform(new_sample[:, [7]]).toarray()
    new_sample_be = ohe_loaded2.transform(new_sample[:, [8]]).toarray()
    new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
    new_sample1 = scaler_loaded.transform(new_sample)
    new_pred = loaded_model.predict(new_sample1)[0]
    return new_pred
#def function for status prediction
def pred_stat():
    
    with open('D:/copperman/classificationmodel.pkl', 'rb') as file:
        loaded_modelc1 = pickle.load(file)
    with open('D:/copperman/classificationscaler.pkl', 'rb') as file:
        scaler_loadedc2 = pickle.load(file)

    with open('D:/copperman/classification_encoder_t.pkl', 'rb') as file:
        encode_loadedc2 = pickle.load(file)

    # 'tons_quantity', 'selling_price_log','application', 'thickness_log', 'width','country','customer','product_ref'
    sample = np.array([[np.log(float(quantity_tons2)), np.log(float(sellingp2)), application2,
                            np.log(float(thickness2)),float(width2),country2,
                            int(customer2),int(prref2),item_type2]])
    sample_ohe = encode_loadedc2.transform(sample[:, [8]]).toarray()
    new_sample = np.concatenate((sample[:, [0,1,2, 3, 4, 5, 6,7]], sample_ohe), axis=1)
    new_sample = scaler_loadedc2.transform(new_sample)
    new_pred1 = loaded_modelc1.predict(new_sample)
    return new_pred1


df=pd.read_csv("D:/copperman/dataframe_coppermodel.csv")

st.header(":green[INDUSTRIAL COPPER MODELLING]")

tab1,tab2,tab3=st.tabs([":orange[Predict selling price]",":orange[Predict status]",":orange[Learning outcomes]"]) 

with tab1:
    st.subheader(":violet[predict the selling price of the copper products]")
    col1,col2=st.columns(2)
    with col1:
        st.caption("Alter the default details and add your product details")
        #value inputs of predection
        
        apopt=df["application"].unique()
        application1=st.selectbox("select value of :green[application]",sorted(apopt))

        contopt=df["country"].unique()
        country1=st.selectbox("select value of :green[country]",sorted(contopt))
        staopt=df["status"].unique()
        status1=st.selectbox("select :green[status]",staopt)
        prefopt=df["product_ref"].unique()
        prref1=st.selectbox("select :green[product ref]",sorted(prefopt))
        itopt=df["item_type"].unique()
        item_type1=st.selectbox("select :green[item type]",itopt)
    with col2:
        quantity_tons1 = st.text_input("Enter Quantity Tons :red[(Min:611728 & Max:1722207579)]",1818181)
        thickness1 = st.text_input("Enter thickness :red[(Min:0.18 & Max:400)]",76)
        width1 = st.text_input("Enter width :red[(Min:1, Max:2990)]",2500)
        customer1 = st.text_input("customer ID :red[(Min:12458, Max:30408185)]",9870327)
    
    try:
        if st.button(":rainbow[show predicted price]"):
            new_pred=get_sp()
            
            st.write('## :green[Predicted selling price:] ', np.exp(new_pred))
    except ValueError:
        st.warning("enter valid product details")
        
with tab2:
    
    st.subheader(":violet[Predict the status of product]")
    col1,col2=st.columns(2)
    with col1:
        apopt=df["application"].unique()
        application2=st.selectbox("select value of :green[application]",sorted(apopt),key=2)

        contopt=df["country"].unique()
        country2=st.selectbox("select value of :green[country]",sorted(contopt),key=3)
        
        prefopt=df["product_ref"].unique()
        prref2=st.selectbox("select :green[product ref]",sorted(prefopt),key=5)
        itopt=df["item_type"].unique()
        item_type2=st.selectbox("select :green[item type]",itopt,key=6)
    with col2:
        quantity_tons2 = st.text_input("Enter Quantity Tons :red[(Min:611728 & Max:1722207579)]",564789,key=10)
        thickness2 = st.text_input("Enter thickness :red[(Min:0.18 & Max:400)]",300,key=11)
        width2 = st.text_input("Enter width :red[(Min:1, Max:2990)]",54,key=12)
        customer2 = st.text_input("customer ID :red[(Min:12458, Max:30408185)]",23546,key=14)
        sellingp2=st.text_input("enter selling price of product",56,key=15)
        
    if st.button(":rainbow[check status]"):
        try:
                
            new_pred1=pred_stat()
            if new_pred1==1:
                st.write(" Product status :green[ WON ]")
            elif  new_pred1==0:
                st.write("Product Status   :red[   LOST ]")
        except ValueError:
            st.warning("Enter valid Product Details ")
with tab3:
    st.subheader(":blue[Learning Outcomes of Industrial Copper Modelling project--]")
    st.markdown("")
    st.write("   >          Exploratory Data Analysis ")
    st.write("   >          Streamlit")
    st.write("   >          Model building")
    st.write("   >          Sklearn")
    st.write("   >          python scripting  ")
    st.write("   >          Data visualisation")
    
    
    
    st.markdown("")
    st.caption("Thankyou for the Support and Assistance  Team guvi")
    
    
    