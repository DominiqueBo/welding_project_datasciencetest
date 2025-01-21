
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report


    
dataset = pd.read_csv('/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/_new_test_curves_pivoted_3_test_train.csv', sep=";")
#/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/
artificial_label = pd.read_csv('/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/_artificial_labels.csv', sep=",")
print(dataset.shape)

dataset_labeled=pd.merge(dataset, artificial_label, on='POINT_ID', how='right')

dataset_labeled=dataset_labeled.rename(columns={'POINT_ID':'id'})

dataset_labeled=dataset_labeled.fillna(0)

#Remove rows with 0
dataset_labeled=dataset_labeled[dataset_labeled['CURRENT'] != 0]

#keep positive
dataset_labeled_p=dataset_labeled[~(dataset_labeled.iloc[:, 7:647] < 0).any(axis=1)]
dataset_labeled_p = dataset_labeled_p[~(dataset_labeled.iloc[:, 7:647] > 300).any(axis=1)]
#Save id
dataset_labeled_id_p=dataset_labeled_p.id

#data save
data_sav_p=dataset_labeled_p

dataset_THO_p=dataset_labeled_p.TH_ORDER
dataset_labeled_p['TH_ORDER'] = pd.Categorical(dataset_labeled_p['TH_ORDER']).codes
dataset_THO_id_p=dataset_labeled_p.TH_ORDER

dataset_labeled_p.drop(columns=['TH_ORDER','assesment_txt','id'],inplace=True)

from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

target_p=dataset_labeled_p.iloc[:,646]

target_p=target_p*10

data_p=dataset_labeled_p.drop(columns=['assesment'])

#removing Amper TH force so as to not integrate them in the PCA
data_p_ft=data_p.iloc[:,6:]
X_train,X_test,y_train,y_test=train_test_split(data_p_ft,target_p,test_size=0.2,random_state=12)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



#page3 load model
data_2D=joblib.load('/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/data_2D_pca_model.joblib')
data_2D_s=joblib.load('/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/data_2D_s_pca_model.joblib')
data_2D_M = joblib.load('/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/data_2D_s_pca_model_explo.joblib')

dbscan_labels_l = joblib.load('/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/dbscan_labels.joblib')
Agglo_labels_l = joblib.load('/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/Agglo_labels.joblib')
kmeans_labels_l = joblib.load('/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/kmeans_labels.joblib')
lof_labels_l = joblib.load('/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/lof_labels.joblib')

clf_rdf_l=joblib.load('/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/clf_rdf.joblib')
clf_svm_l=joblib.load('/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/clf_svm.joblib')
knn_l=joblib.load('/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/knn.joblib')
dt_clf_l=joblib.load('/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/dt_clf.joblib')


def page1():#Project presentation
    st.set_page_config(page_title="Spot Welding resistance curve prediction", layout="wide")
    st.title("Spot Welding resistance curve prediction")
    st.image("/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/welding_shop.jpg")
    st.subheader("Presentation of the project", divider=True)
    st.write("**Context:** In Stellantis welding workshops, we produce from 300 to 600 cars a day. There are around 5000 welding points per car, and each minute a car is produced. Quality is carmaker main concern: Each point must be a good welding point. But in the real production life, quality drift happen. If a major quality issue is detected, new welding operation could be necessary or worst case, body car must be destroyed, in all case it raise car value production. Our project will help to identify bad welding spot among the millions of each day so as to avoid    major quality issues.")
    st.subheader("Objectives", divider=True)
    st.write('What are the main objectives to be achieved? Describe in a few lines.We have a dataset where welding points are not classified/labelled as good or bad.')
    st.write('1 – First objective, try several unsupervised clusterisation so as to identify the bad welding points. This first clusterisation will be validated/corrected manually by the business in order to have a relevant dataset with labels.')
    st.write('2 – Build a classification model able to identify bad and good welding points in new labeled dataset..")')

def page2(): #DataVizualization/preprocessing
    st.title("Data Presentation")
    st.subheader("Dataset describe", divider=True)
    st.dataframe(dataset.describe())
    st.write('**POINT_ID:** point id')
    st.write('**CURRENT:** Current apply for the welding point')
    st.write('**FORCE:** Force apply for the welding point')
    st.write('**TH_ORDER:** concatenation of TH1 & TH2 & TH3')
    st.write('**TH1,TH2,TH3:** The welding point could have 2 or 3 iron sheets, Thx give the thickness of each iron sheet')
    st.write('**0 to 640:** electric resistance (ohm) during the welding, from 0 ms to 640 ms')


    #Before cleaning
    st.subheader("Before cleaning", divider=True)
    df_max=dataset_labeled.iloc[:,7:640].max()
    df_min=dataset_labeled.iloc[:,7:640].min()
    df_Q1=quartiles = dataset_labeled.iloc[:,7:640].quantile(0.25)
    df_Q2=quartiles = dataset_labeled.iloc[:,7:640].quantile(0.50)
    df_Q3=quartiles = dataset_labeled.iloc[:,7:640].quantile(0.75)
    df_max_min=pd.DataFrame()
    df_max_min['max']=df_max
    df_max_min['Q1']=df_Q1
    df_max_min['Q2']=df_Q2
    df_max_min['Q3']=df_Q3
    df_max_min['min']=df_min

    
    plt.figure(figsize=(16, 4))
    plt.plot(df_max_min)
    plt.xticks([])
    plt.ylim(-50, 1000)
    plt.xlabel('Ms [0 to 640]')
    plt.ylabel('Resistance')
    plt.legend(['Max','Q1','Q2','Q3','Min'])
    plt.title('Resistance distribution before cleaning')
    st.pyplot(plt)
    plt.close()
    st.write('')



    st.subheader("after cleaning", divider=True)
    #after cleaning
    st.write('CURRENT=0 removed')
    st.write('Negative value removed')
    st.write('N/A value filled with 0')
    df_max_p=dataset_labeled_p.iloc[:,7:640].max()
    df_min_p=dataset_labeled_p.iloc[:,7:640].min()
    df_Q1_p=quartiles = dataset_labeled_p.iloc[:,7:640].quantile(0.25)
    df_Q2_p=quartiles = dataset_labeled_p.iloc[:,7:640].quantile(0.50)
    df_Q3_p=quartiles = dataset_labeled_p.iloc[:,7:640].quantile(0.75)
    df_max_min_p=pd.DataFrame()
    df_max_min_p['max']=df_max_p
    df_max_min_p['Q1']=df_Q1_p
    df_max_min_p['Q2']=df_Q2_p
    df_max_min_p['Q3']=df_Q3_p
    df_max_min_p['min']=df_min_p

    plt.figure(figsize=(16, 4))
    plt.plot(df_max_min_p)
    plt.xticks([])
    plt.ylim(-50, 1000)
    plt.xlabel('Ms [0 to 640]')
    plt.ylabel('Resistance')
    plt.legend(['Max','Q1','Q2','Q3','Min'])
    plt.title('Resistance distribution before cleaning')
    st.pyplot(plt)
    plt.close()

    #zoom on specific data
    st.subheader("zoom on sample data", divider=True)
    row_values = dataset_labeled_p.iloc[3470,7:640].values.tolist()
    row_values1 = dataset_labeled_p.iloc[300,7:640].values.tolist()
    row_values2 = dataset_labeled_p.iloc[200,7:640].values.tolist()
 
    plt.ylim(0, 300)
    plt.plot(range(1,634),row_values)
    plt.plot(range(1,634),row_values1)
    plt.plot(range(1,634),row_values2)
    plt.xlabel('Ms')
    plt.ylabel('Resistance')
    st.pyplot(plt)
    plt.close()
#######
def page3(): #Data Clusterisation exploration
    st.title('Clusterisation exploration')
    st.subheader("Comparing non Scaled data vs Scaled data", divider=True)
    col1, col2 = st.columns(2)
    with col1:
        #data_2D=joblib.load('/Workspace/Users/j552405@inetpsa.com/Streamlit_welding/data_2D_pca_model.joblib')
        fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.scatter(data_2D[:, 0], data_2D[:, 1], c = target_p, cmap=plt.cm.Spectral,label='Thickness Order')

        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')

        ax.set_title("Data (not standardized) projected onto the 2 axes of PCA")
        legend1 = ax.legend( title="Legend")
        st.pyplot(plt)
        plt.close()
        st.write('The scaled data is more compact, so the potential clusters are maybe not well separated.')
        st.write('So we will use non scaled data for our clusterisation attempt.')

    with col2:
        #data_2D_s=joblib.load('/Workspace/Users/j552405@inetpsa.com/Streamlit_welding/data_2D_s_pca_model.joblib')
        fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.scatter(data_2D[:, 0], data_2D_s[:, 1], c = target_p, cmap=plt.cm.Spectral,label='Thickness Order')

        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')

        ax.set_title("Standardized Data projected onto the 2 axes of PCA")
        legend1 = ax.legend( title="Legend")
        st.pyplot(plt)
        plt.close()

    st.subheader("PCA axis data exploration", divider=True)
    # Chargement des données
    #data_2D_M = joblib.load('/Workspace/Users/j552405@inetpsa.com/Streamlit_welding/data_2D_s_pca_model_explo.joblib')
    pca_x = st.slider("PCA X", 0, 5, 0)
    pca_y = st.slider("PCA Y", 0, 5, 5)
    # Supposons que target_p et dataset_THO_id_p sont déjà définis
    # Si ce n'est pas le cas, ajoutez leur chargement ici
    mask = target_p != 10
    lgd = dataset_THO_id_p
    lgd_label="Iron Sheet thickness order"
    lgd2 = target_p

    # Création de la figure et des axes
    fig, ax = plt.subplots(figsize=(6, 6))  # Utilisation d'une taille plus raisonnable

    # Titre
    col3, col4 = st.columns([1,3])
    
    with col3:
        
        DD_projection = st.radio(
        "Selection of the 2D projections",
        ["Thickness Order", "DBSCAN", "Local Outlier", "KMeans", "AgglomerativeClustering"])

        if DD_projection == "Thickness Order]":
            lgd = dataset_THO_id_p
            lgd_label="Iron Sheet thickness order"
        elif DD_projection == "DBSCAN":
            lgd = dbscan_labels_l
            lgd_label="DBSCAN Clusters"
        elif DD_projection == "Local Outlier":
            lgd = lof_labels_l
            lgd_label="Local Outlier Clusters"
        elif DD_projection == "KMeans":
            lgd = kmeans_labels_l
            lgd_label="KMeans Clusters"
        elif DD_projection == "AgglomerativeClustering":
            lgd = Agglo_labels_l
            lgd_label="AgglomerativeClustering Clusters"

    with col4:
        plt.title('PCA 2D projection of the data')
        # Scatter plot avec deux couches
        scatter1 = ax.scatter(data_2D_M[:, pca_x], data_2D_M[:, pca_y], c=lgd, cmap=plt.cm.Spectral, label=lgd_label)
        scatter2 = ax.scatter(data_2D_M[:, pca_x][mask], data_2D_M[:, pca_y][mask], c=lgd2[mask], cmap='viridis', marker='x', label='Bad welding points')

        # Légende
        legend1 = ax.legend( title="Legend")
        
        

        # Axes labels
        ax.set_xlabel('PCA'+(str(pca_x)), labelpad=10)
        ax.set_ylabel('PCA'+(str(pca_y)), labelpad=10)

        # Affichage avec Streamlit
        st.pyplot(fig)
        plt.close()
#####
def page4():#title="Classification Models exploration"
    
    st.title("Classification Models exploration")
    st.subheader("Random Forest Classifier", divider=True)
    score_clf_rdf_l=(clf_rdf_l.score(X_test_scaled, y_test))
    st.metric("Model Score", f"{score_clf_rdf_l:.2%}")
    y_pred=clf_rdf_l.predict(X_test_scaled)

    col1, col2 = st.columns([2,3])
    with col1:
        st.write('0=Anomaly Defect','3=Shunt Defect')
        st.write('7=Gap Defect','10=Good')
        st.write(pd.crosstab(y_test,y_pred,rownames=['test'],colnames=['pred']))

    with col2:
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, target_names=['Anomaly Defect', 'Shunt Defect', 'Gap Defect', 'Good'], output_dict=True)).transpose())




    st.subheader("SVM Classifier", divider=True)
    score_clf_svm_l=(clf_svm_l.score(X_test_scaled, y_test))
    st.metric("Model Score", f"{score_clf_svm_l:.2%}")
    y_pred_svm=clf_svm_l.predict(X_test_scaled)

    col3, col4 = st.columns([2,3])
    with col3:
        st.write('0=Anomaly Defect','3=Shunt Defect')
        st.write('7=Gap Defect','10=Good')
        st.write(pd.crosstab(y_test,y_pred_svm,rownames=['test'],colnames=['pred']))
    with col4:
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred_svm, target_names=['Anomaly Defect', 'Shunt Defect', 'Gap Defect', 'Good'], output_dict=True)).transpose())

    st.subheader("KNN Classifier", divider=True)
    score_clf_knn_l=(knn_l.score(X_test_scaled, y_test))
    st.metric("Model Score", f"{score_clf_knn_l:.2%}")
    y_pred_knn=knn_l.predict(X_test_scaled)
    
    col5, col6 = st.columns([2,3])
    with col5:
        st.write('0=Anomaly Defect','3=Shunt Defect')
        st.write('7=Gap Defect','10=Good')
        st.write(pd.crosstab(y_test,y_pred_knn,rownames=['test'],colnames=['pred']))
    with col6:
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred_knn, target_names=['Anomaly Defect', 'Shunt Defect', 'Gap Defect', 'Good'], output_dict=True)).transpose())

    st.subheader("Decision Tree Classifier", divider=True)
    score_dt_clf_l=(dt_clf_l.score(X_test_scaled, y_test))
    st.metric("Model Score", f"{score_dt_clf_l:.2%}")
    y_pred_dt_clf=dt_clf_l.predict(X_test_scaled)

    col7, col8 = st.columns([2,3])
    with col7:
        st.write('0=Anomaly Defect','3=Shunt Defect')
        st.write('7=Gap Defect','10=Good')
        st.write(pd.crosstab(y_test,y_pred_dt_clf,rownames=['test'],colnames=['pred']))
    with col8:
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred_dt_clf, target_names=['Anomaly Defect', 'Shunt Defect', 'Gap Defect', 'Good'], output_dict=True)).transpose())

def page5():#DataFrame filtering
    
    st.title("DataFrame filtering")
        #zoom on specific data
    st.subheader("zoom on sample data", divider=True)
    col1, col2 = st.columns([3,3])
    # Widgets de filtrage
    with col1:
        dataset_labeled_g=dataset_labeled_p
        dataset_labeled_g['assesment']=dataset_labeled_g['assesment']*10
        ass_select = ["OK", "GAP", "SHUNT", "ANOMALI"]
        selection = st.segmented_control("Welding point status", ass_select,default="OK", selection_mode="multi")
        # Dictionnaire de correspondance
        status_to_value = {
            "OK": 10,
            "GAP": 7,
            "SHUNT": 3,
            "ANOMALI": 0
        }

        # Construire la liste des valeurs correspondantes
        selected_values = [status_to_value[status] for status in selection]

        current_sel = st.multiselect("CURRENT",options=sorted(dataset_labeled_g['CURRENT'].unique()),default=sorted(dataset_labeled_g['CURRENT'].unique()))
        force_sel = st.multiselect("FORCE", options=sorted(dataset_labeled_g['FORCE'].unique()),default=sorted(dataset_labeled_g['FORCE'].unique()))
        TH1_sel = st.multiselect("TH1",options=sorted(dataset_labeled_g['TH1'].unique()),default=sorted(dataset_labeled_g['TH1'].unique()))
        TH2_sel = st.multiselect("TH2",options=sorted(dataset_labeled_g['TH2'].unique()),default=sorted(dataset_labeled_g['TH2'].unique()))
        TH3_sel = st.multiselect("TH3",options=sorted(dataset_labeled_g['TH3'].unique()),default=sorted(dataset_labeled_g['TH3'].unique()))

        dataset_labeled_p_filtered=dataset_labeled_p[
                                                    (dataset_labeled_g['assesment'].isin(selected_values))&
                                                    (dataset_labeled_g['CURRENT'].isin(current_sel))&
                                                    (dataset_labeled_g['FORCE'].isin(force_sel))&
                                                    (dataset_labeled_g['TH1'].isin(TH1_sel))&
                                                    (dataset_labeled_g['TH2'].isin(TH2_sel))&
                                                    (dataset_labeled_g['TH3'] .isin(TH3_sel))
                                                    ]
        
        

    with col2:

        st.dataframe(dataset_labeled_p_filtered)

        for i in range(len(dataset_labeled_p_filtered)):
            if dataset_labeled_p_filtered.iloc[i,646] == 0:
                row_values = dataset_labeled_p_filtered.iloc[i,7:640].values.tolist()
                plt.plot(range(1,634),row_values,color='red',zorder=3)
            if dataset_labeled_p_filtered.iloc[i,646] == 7:
                row_values = dataset_labeled_p_filtered.iloc[i,7:640].values.tolist()
                plt.plot(range(1,634),row_values,color='orange',zorder=2)
            if dataset_labeled_p_filtered.iloc[i,646] == 3:
                row_values = dataset_labeled_p_filtered.iloc[i,7:640].values.tolist()
                plt.plot(range(1,634),row_values,color='yellow',zorder=1)
            if dataset_labeled_p_filtered.iloc[i,646] == 10:
                row_values = dataset_labeled_p_filtered.iloc[i,7:640].values.tolist()
                plt.plot(range(1,634),row_values,color='green',zorder=0)

        plt.xlabel('Ms')
        plt.ylabel('Resistance')
        plt.ylim(0, 300)
        st.pyplot(plt)
        plt.close()


    


#####################################################
#st Menu
page=st.navigation({"Spot Welding resistance curve prediction":[st.Page(page1,title="Project presentation"), st.Page(page2,title="DataVizualization/preprocessing"),
st.Page(page3,title="Clusterisation exploration"),
st.Page(page4,title="Classification Models exploration"),
st.Page(page5,title="Test area")
]})
page.run()