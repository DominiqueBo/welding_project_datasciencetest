
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

rado_df = pd.read_csv('/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/rado_df.csv', sep=",")
rado_df['assesment_txt'] = rado_df['assesment_txt'].replace({'OK': 10, 'gap': 7, 'anomaly': 0, 'shunt': 3})
rado_df.drop(columns=['Unnamed: 0','Unnamed: 0.1','POINT_ID','TH_ORDER','OK','gap','anomaly','shunt'],inplace=True)
rado_df_display=rado_df
target_rd=rado_df['assesment_txt']
rado_df.drop(columns=['assesment_txt'],inplace=True)

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

#train set rado
X_train_rado, X_test_rado, y_train_rado, y_test_rado = train_test_split(rado_df, target_rd, test_size=0.2, random_state=12, stratify=target_rd)
X_train_scaled_rado = scaler.fit_transform(X_train_rado)
X_test_scaled_rado = scaler.transform(X_test_rado)



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

dt_clf_rado = joblib.load('/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/dt_clf_rado.joblib')
xgb_clf_rado = joblib.load('/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/xgb_clf_rado.joblib')


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
    
    st.subheader("Anomaly/good welding curves examples", divider=True)
    st.write("All these curves share the same parameters: Current=8.4A, Force=2.2N, TH1=0.8mm, TH2=1.5mm, TH3=0.")
    st.write("Among all these curves, one has not the same shape and is classified as bad welding point by the business.")
    dataset_labeled_gb=dataset_labeled_p
    dataset_labeled_gb['assesment']=dataset_labeled_gb['assesment']*10
		
    dataset_labeled_p_filtered_gb=dataset_labeled_gb[
                                        (dataset_labeled_gb['assesment'].isin([0,10])&
                                        (dataset_labeled_gb['CURRENT'].isin([8.4]))&
                                        (dataset_labeled_gb['FORCE'].isin([2.2]))&
                                        (dataset_labeled_gb['TH1'].isin([0.8]))&
                                        (dataset_labeled_gb['TH2'].isin([1.5]))&
                                        (dataset_labeled_gb['TH3'] .isin([0]))
                                        )]
    plt.figure(figsize=(4, 4))
    for i in range(len(dataset_labeled_p_filtered_gb)):
        if dataset_labeled_p_filtered_gb.iloc[i,646] == 0:
            row_values = dataset_labeled_p_filtered_gb.iloc[i,7:640].values.tolist()
            plt.plot(range(1,634),row_values,color='red',zorder=3,label='Anomaly')
        if dataset_labeled_p_filtered_gb.iloc[i,646] == 10:
            row_values = dataset_labeled_p_filtered_gb.iloc[i,7:640].values.tolist()
            plt.plot(range(1,634),row_values,color='green',zorder=0,label='Goods')

    plt.legend(['Anomaly','Goods'])
    plt.xlabel('Ms')
    plt.ylabel('Resistance')
    plt.ylim(0, 300)
    st.pyplot(plt)
    plt.close()

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
    st.write('The goal is to compare the 2D projection of the data before and after scaling, and identify the most relevant technic before clusterization.')
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
    st.subheader("Artificial Labeling phase", divider=True)
    st.subheader('Welding points dataset has been analysed by the business so as to label the data as anomalies or not.')

    st.image("/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/Rado_pic0.png")
    st.image("/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/Rado_pic1.png")
    st.image("/Workspace/Users/j552405@inetpsa.com/streamlit_project_mar24_cds_int_stellantis_2/Rado_pic2.png")

    
    st.subheader('the next project steps will use thess new labels.')
    st.subheader(' ')
    st.subheader("2D PCA Axis projection according clusterisation models results", divider=True)
    st.write('The goal is to compare the 2D projection models and see if clusters could isolate the bad welding points.') 
    st.write('The models try to identify 4 clusters for these 4 categories OK (good welding points) GAP,SHUNT,ANOMALY(Bad welding points)')
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
    fig, ax = plt.subplots(figsize=(6, 6))  

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
    st.subheader("Clusterisation exploration, conclusion", divider=True)
    st.write('Result: no clear spacial separation detected, bad points are melted with good points in all cases')
    st.write('Next phase will be to use classification models to detect the anomalies')
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

        
    st.subheader("Welding Business propose to test classification with a new reduced and enhanced dataset.")
    st.write("The dataset only include 2 measures point (1&10) and 4 new columns r_max, r_max_time, slope & intercept, giving another aspect of the welding curve", divider=True)
    st.dataframe(rado_df_display.head(5))
    st.subheader("Decision Tree Classifier with new dataset after GridSearchCSV optimization", divider=True)
    
    score_dt_clf_rado=(dt_clf_rado.score(X_test_scaled_rado, y_test_rado))
    st.metric("Model Score", f"{score_dt_clf_rado:.2%}")
    y_pred_dt_clf_rado=dt_clf_rado.predict(X_test_scaled_rado)
    col9, col10 = st.columns([2,3])
    with col9:
        st.write('0=Anomaly Defect','3=Shunt Defect')
        st.write('7=Gap Defect','10=Good')
        st.write(pd.crosstab(y_test_rado,y_pred_dt_clf_rado,rownames=['test'],colnames=['pred']))
    with col10:
        st.dataframe(pd.DataFrame(classification_report(y_test_rado, y_pred_dt_clf_rado, target_names=['Anomaly Defect', 'Shunt Defect', 'Gap Defect', 'Good'], output_dict=True)).transpose())

    st.subheader("XGBOOST Classifier with new dataset after GridSearchCSV optimization", divider=True)
    # Création d'un mapping pour les classes
    unique_classes = sorted(set(y_train_rado))
    class_mapping = {cls: i for i, cls in enumerate(unique_classes)}

    # Remappage des classes en entiers consécutifs
    y_train_map = y_train_rado.map(class_mapping)
    y_test_map = y_test_rado.map(class_mapping)

    score_xgb_clf_rado=(xgb_clf_rado.score(X_test_scaled_rado, y_test_map))
    st.metric("Model Score", f"{score_xgb_clf_rado:.2%}")
    y_pred_xgb_clf_rado=xgb_clf_rado.predict(X_test_scaled_rado)
    col11, col12 = st.columns([2,3])
    with col11:
        st.write('0=Anomaly Defect','1=Shunt Defect')
        st.write('2=Gap Defect','3=Good')
        st.write(pd.crosstab(y_test_map,y_pred_xgb_clf_rado,rownames=['test'],colnames=['pred']))
    with col12:
        st.dataframe(pd.DataFrame(classification_report(y_test_map, y_pred_xgb_clf_rado, target_names=['Anomaly Defect', 'Shunt Defect', 'Gap Defect', 'Good'], output_dict=True)).transpose())


    
    st.subheader("Classification Models exploration, conclusion", divider=True)
    st.write('DecisionTree and XGBoost  are the best models to identify most bad welding points, less good points predicted as bad.')
    st.write('Lots of bad points are not detected as bad, but the model remains relevant for the business.')
    
def page5():#Curves Display
    
    st.title("Curves Display")

    col1, col2 = st.columns([3,3])
    # Widgets de filtrage
    with col1:
        dataset_labeled_g=dataset_labeled_p
        dataset_labeled_g['assesment']=dataset_labeled_g['assesment']*10
        ass_select = ["OK", "GAP", "SHUNT", "ANOMALY"]
        selection = st.segmented_control("Welding point status", ass_select,default=["OK","ANOMALY"], selection_mode="multi")
        # Dictionnaire de correspondance
        status_to_value = {
            "OK": 10,
            "GAP": 7,
            "SHUNT": 3,
            "ANOMALY": 0
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
        
        styles = {
            0: {'color': 'red', 'zorder': 3, 'label': 'Anomaly'},
            7: {'color': 'orange', 'zorder': 2, 'label': 'Gap'},
            3: {'color': 'yellow', 'zorder': 1, 'label': 'Shunt'},
            10: {'color': 'green', 'zorder': 0, 'label': 'Good'}
        }

        plt.figure(figsize=(10, 6))  # Ajuster la taille du graphe

        # Variable pour suivre les labels déjà affichés
        labels_shown = set()

        for i in range(len(dataset_labeled_p_filtered)):
            label_value = dataset_labeled_p_filtered.iloc[i, 646]
            if label_value in styles:
                row_values = dataset_labeled_p_filtered.iloc[i, 7:640].values.tolist()
                style = styles[label_value]
                label = style['label'] if style['label'] not in labels_shown else None
                plt.plot(range(1, 634), row_values, color=style['color'], zorder=style['zorder'], label=label)
                labels_shown.add(style['label'])

        plt.legend()
        plt.xlabel('Ms')
        plt.ylabel('Resistance')
        plt.ylim(0, 300)
        st.pyplot(plt)
        plt.close()



def page6():#Conclusion
    st.title("Project Conclusion: Welding Quality Control with Machine Learning")

    st.markdown("""
    ### Overview
    This project explored advanced data science techniques to detect spot welding quality issues at Stellantis Body Shops. The primary objective was to prevent costly production errors by detecting defective welding points. Several machine learning algorithms were tested for both **unsupervised clustering** and **supervised classification**, providing valuable insights into their performance.

    ---

    ### Algorithm Testing and Achievements

    #### Unsupervised Clustering
    - **Models Tested**: DBSCAN, Local Outlier Factor, K-Means, Agglomerative Clustering, and Autoencoders.
    - **Best Performer**: The **Autoencoder** stood out, generating reliable artificial labels for subsequent supervised classification.

    #### Supervised Classification
    - **Algorithms Tested**: Random Forest, SVM, KNN, Decision Tree, and XGBoost.
    - **Decision Tree Classifier** offered the best balance between **accuracy and interpretability**.
    - **XGBoost** showed **robust performance**, particularly with reduced datasets.
    - **Grid Search** optimization for the Decision Tree and XGBOOST gived some improvements.

    ---

    ### Feature Engineering
    Feature engineering was critical for handling the high-dimensional dataset (640 features). Key steps included:

    1. **Normalization** to ensure comparable data ranges.
    2. **Cleaning**: Removal of negative values, handling missing data, and excluding irrelevant zero-resistance pauses.
    3. **Dimensionality Reduction**: 
    - Averaging features in groups of 10 columns to reduce dimensionality. However, this approach masked critical details, negatively impacting model performance.
    - **Principal Component Analysis (PCA)** was applied, but it did not enhance results due to domain-specific pattern loss.

    #### Why PCA Did Not Improve Performance
    - **Loss of Domain-Specific Patterns**: Welding resistance curves contain subtle, domain-specific signals essential for defect identification. PCA’s unsupervised nature discards these.
    - **High Feature Correlation**: Resistance measurements are highly correlated. PCA collapses these into fewer components, potentially losing important variations.
    - **Nonlinear Relationships**: PCA assumes linear relationships, which are inadequate for capturing the complexity of welding data.

    ---

    ### Why Engineered Features Delivered Superior Results
    1. **Critical Time Window**: The initial milliseconds of the welding process are crucial for revealing quality issues, as contact imperfections manifest early.
    2. **Focused Signal Analysis**: Limiting analysis to the initial resistance curve enhances the signal-to-noise ratio by excluding irrelevant variations.
    3. **Simplified Patterns**: Engineering features from this early segment condenses data into a more informative subset, allowing models like **XGBoost** and **Decision Trees** to focus on the most relevant details.

    ---

    ### Key Outcomes
    - The project successfully automated defect detection, distinguishing good from bad welding points with high accuracy.
    - **High-quality, well-labeled data** was a key factor. When labels accurately reflect true classes, most supervised learning algorithms perform effectively.
    - Domain-specific variables, such as force, current, thickness, and **engineered features like contact resistance**, significantly enhanced anomaly detection.

    ### Conclusion
    This project emphasizes the importance of tailored feature engineering based on expert knowledge of the welding process. Focusing on the contact resistance in the initial phase of the welding cycle led to superior model performance, demonstrating how domain expertise and machine learning can effectively combine for robust quality control.
    """)
    


#####################################################
#st Menu
page=st.navigation({"Spot Welding resistance curve prediction":[st.Page(page1,title="Project presentation"), st.Page(page2,title="DataVizualization/preprocessing"),
st.Page(page3,title="Clusterisation exploration"),
st.Page(page4,title="Classification Models exploration"),
st.Page(page5,title="Curves Display"),
st.Page(page6,title="Conclusion")
]})
page.run()