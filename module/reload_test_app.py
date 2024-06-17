'''Module de relecture des données d\'apprentissage'''
__author__ = "Quentin Jéré"
__maintainer__ =  "Quentin Jéré"
__email__ = "quentin.jere@univ-tlse3.fr"
__status__ = "Development"

import os
import sys
import pandas as pd
# import modin.pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras_tuner
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
# from keras.utils import np_utils
from keras.utils import to_categorical
import glob
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn import preprocessing
import matplotlib.gridspec as gridspec
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pydot
import graphviz
from IPython.display import SVG,display,Image,Markdown,HTML
import json
import pickle
import ipywidgets as widgets
import time

class reload_app:
    def __init__(self,
                 path="../modeles/",
                 path_decal='../spectres_csv/decalage/',
                 path_mel='../spectres_csv/melange/',
                 modeles=[],
                 metrique=True,
                 cm=True,
                 cm_melange=True,
                 cm_app=False,
                 verbose=False,
                 disp_df=False):
        self.path=path
        self.path_decal=path_decal
        self.path_mel=path_mel
        self.modeles=modeles
        self.metrique=metrique
        self.cm=cm
        self.cm_melange=cm_melange
        self.cm_app=cm_app
        self.verbose=verbose
        self.disp_df=disp_df
        self.model_name=None

        #self.reload()

    def read_csv(self):
        name_decal=self.model_name[11:]
        mul=int(name_decal[11:])*2+1
        test_data_intensite=pd.read_csv(self.path_decal+'spectres_ng/'+name_decal+'.csv', sep=";",index_col=0)
        #test_data_x=pd.read_csv('../spectres_csv/spectres_ng/data_x.csv', sep=";",index_col=0)
        test_2_data_intensite=pd.read_csv(self.path_decal+'spectres_proc/'+name_decal+'.csv', sep=";",index_col=0)
        test_data_proba=pd.read_csv(self.path_decal+'data_proba_'+name_decal+'.csv', sep=";",index_col=0)
        test_2_data_intensite=test_2_data_intensite.add_suffix('_proc')
        #test_2_data_x=pd.read_csv('../spectres_csv/spectres_proc/data_x.csv', sep=";",index_col=0)
        self.test_data_proba_sauv=test_data_proba
        self.test_data_R_intensite=pd.concat([test_data_intensite.T,test_2_data_intensite.T])#.reset_index(drop=True)
        test_data_R_proba=pd.concat([test_data_proba.drop(['composes'],axis=1).T,test_data_proba.drop(['composes'],axis=1).T]).set_index(self.test_data_R_intensite.index,drop=True)#.reset_index(drop=True)
        self.test_data_R_proba_sauv=test_data_R_proba

        test_data_R_proba=self.test_data_R_proba_sauv.loc[self.index_train_proba]
        test_test_proba=self.test_data_R_proba_sauv.loc[self.index_test_proba]
        df=self.test_data_R_intensite.loc[self.index_test_proba]
        self.test_data_R_intensite=self.test_data_R_intensite.loc[self.index_train_proba]
        ts=df.shape[0]
        ts_2=self.test_data_R_intensite.shape[0]

        test_train=test_data_intensite.T
        test_train=tf.constant(test_train)
        test_train=tf.reshape(test_train,[(29*mul),21800,1])
        self.test_test=df
        self.test_test=tf.constant(self.test_test)
        self.test_test=tf.reshape(self.test_test,[ts,21800,1])
        self.test_train_R=self.test_data_R_intensite
        self.test_train_R=tf.constant(self.test_train_R)
        self.test_train_R=tf.reshape(self.test_train_R,[ts_2,21800,1])
        self.test_data_proba=np.array(test_data_proba.drop(['composes'],axis=1).T)
        self.test_data_R_proba=np.array(test_data_R_proba)
        self.test_data_test_proba=np.array(test_test_proba)

        if self.cm_melange == True:
            test_intensite_melange=pd.read_csv(self.path_mel+'spectres_proc/data_intensite.csv', sep=";",index_col=0)
            self.test_intensite_melange=test_intensite_melange.rename(columns={'intensité':'intensité_EtOH_EtOH'})
            test_proba_melange=pd.read_csv(self.path_mel+'data_proba.csv', sep=";",index_col=0)
            self.test_proba_melange_save=test_proba_melange
            self.test_proba_data=pd.read_csv('../spectres_csv/data_proba.csv', sep=";",index_col=0)

            self.test_mel=self.test_intensite_melange.T
            self.test_mel=tf.constant(self.test_mel)
            self.test_mel=tf.reshape(self.test_mel,[435,21800,1])
            #test_train=np.array(test_train).reshape(3,21800,1)
            self.test_proba_melange=np.array(test_proba_melange.drop(['mélange'],axis=1))
        if self.verbose == True:
            print(f"\033[38;2;255;255;128m\033[1m Dossier charger : \033[0m")
            print(f"\t {self.path_decal}")
            if self.cm_melange == True:
                print(f"\t {self.path_mel}")
            print(f"\033[38;2;0;255;0m\033[1m Chargement des fichiers CSV : \033[0m")
            print(f"\tOK")
        if self.disp_df == True:
            print("Intensités du jeu d'entrainement")
            display(self.test_data_R_intensite)
            print("Probabilités du jeu d'entrainement")
            display(self.test_data_R_proba_sauv)
            print("Intensités du jeu de test")
            display(df)
            print("Probabilités du jeu de test")
            display(self.test_data_test_proba)
            if self.cm_melange == True:
                print("Intensités du jeu de mélanges")
                display(test_intensite_melange)
                print("Probabilités du jeu de mélanges")
                display(self.test_proba_melange_save)

    def load_var(self):
        self.index_train_proba=[]
        self.index_test_proba=[]
        self.h=dict()
        self.test_app=dict()
        with open(self.path+self.model_name+"/index_train_proba.pkl", "rb") as fp:
            self.index_train_proba = pickle.load(fp)
        with open(self.path+self.model_name+"/index_test_proba.pkl", "rb") as fp:
            self.index_test_proba = pickle.load(fp)
        if self.metrique == True:
            with open(self.path+self.model_name+"/history.pkl", "rb") as fp:
                self.h = pickle.load(fp)
        if self.cm_app == True:
            with open(self.path+self.model_name+"/test_app.pkl", "rb") as fp:
                self.test_app = pickle.load(fp)
        if self.verbose == True:
            print(f"\033[38;2;255;255;128m\033[1m Dossier charger : \033[0m")
            print(f"\t {self.path}")
            print(f"\033[38;2;0;255;0m\033[1m Chargement des variables : \033[0m")
            print(f"\tOK")
        if self.disp_df == True:
            print("Indexes du jeu d'entrainement")
            print(self.index_train_proba)
            print("Indexes du jeu de test")
            print(self.index_test_proba)
            if self.metrique == True:
                print("Historique d'apprentissage")
                print(self.h)
            if self.cm_app == True:
                print("Test réalisés au cour de l'apprentissage")
                print(self.test_app)

    def load_model(self):
        self.model=keras.saving.load_model(self.path+self.model_name+'/model.keras', custom_objects=None, compile=True, safe_mode=True)
        if self.verbose == True:
            print(f"\033[38;2;255;255;128m\033[1m Dossier charger : \033[0m")
            print(f"\t {self.path}")
            print(f"\033[38;2;0;255;0m\033[1m Chargement du modèle : \033[0m")
            print(f"\tOK")

    def pred(self):
        t_start=time.time()
        self.ytrain_hat=self.model.predict(self.test_train_R,verbose = 1)#prédiction des valeurs d'entrainement par le modèle dans l'état
        self.ytest_hat=self.model.predict(self.test_test,verbose = 1)#prédiction des valeurs de test par le modèle dans l'état 

        if self.cm_melange == True:
            self.li=[]
            for i in range(len(self.model.layers)):
                if self.model.layers[i].name[:4]=='conv':
                    self.li.append(i)
            ksi=''
            for i in self.li:
                ksi+='_'+str(self.model.layers[i].get_config()['kernel_size'][0])
            self.test_pred_mel=self.model.predict(self.test_mel)
            self.vis_conv_test={}
            for i in self.li:
                model_vis_conv = keras.Model(inputs=self.model.inputs, outputs=self.model.layers[i].output)
                self.vis_conv_test.update({'c_'+str(i):model_vis_conv.predict(self.test_mel)})
        t_stop=time.time()
        print(f"\033[38;2;128;0;255mDuration : {t_stop-t_start}s")

    def trace_metrique(self):
        fig, graph = plt.subplots(figsize=(1500/72,8),layout="constrained")
        graph.plot(self.h['loss'],lw=0.75, color='blue',zorder=10,alpha=1,label='loss')
        graph.plot(self.h['val_loss'],lw=0.75, color='green',zorder=10,alpha=1,label='val_loss')
        graph.set_xlabel("Nombre d'epochs")
        graph.set_ylabel("Fonction de pertes")
        graph.set_title("Graphique représentant les fonctions de pertes au cour de l'apprentissage du modèle"+self.model_name)
        # graph.set_ylim(0,1e0)
        graph.axhline(0,-1e4,1e8,linestyle=':',color='black',zorder=-10)
        graph.legend()
        graph.grid()
        plt.show()

        fig, graph = plt.subplots(figsize=(1500/72,8),layout="constrained")
        graph.plot(self.h['accuracy'],lw=0.75, color='blue',zorder=10,alpha=1,label='accuracy')
        graph.plot(self.h['val_accuracy'],lw=0.75, color='green',zorder=10,alpha=1,label='val_accuracy')
        graph.set_xlabel("Nombre d'epochs")
        graph.set_ylabel("Fonction de métrique")
        graph.set_title("Graphique représentant les fonctions de métriques au cour de l'apprentissage du modèle"+self.model_name)
        # graph.set_ylim(0,1e0)
        graph.axhline(0,-1e4,1e8,linestyle=':',color='black',zorder=-10)
        graph.legend()
        graph.grid()
        plt.show()

        fig, graph = plt.subplots(figsize=(1500/72,8),layout="constrained")
        graph.plot(self.h['top_k_categorical_accuracy'],lw=0.75, color='blue',zorder=10,alpha=1,label='top k accuracy')
        graph.plot(self.h['val_top_k_categorical_accuracy'],lw=0.75, color='green',zorder=10,alpha=1,label='val_top_k_accuracy')
        graph.set_xlabel("Nombre d'epochs")
        graph.set_ylabel("Fonction de métrique")
        graph.set_title("Graphique représentant les fonctions de métriques au cour de l'apprentissage du modèle"+self.model_name)
        # graph.set_ylim(0,1e0)
        graph.axhline(0,-1e4,1e8,linestyle=':',color='black',zorder=-10)
        graph.legend()
        graph.grid()
        plt.show()

        fig, graph = plt.subplots(figsize=(1500/72,8),layout="constrained")
        graph.plot(self.h['auc'],lw=0.75, color='blue',zorder=10,alpha=1,label='auc')
        graph.plot(self.h['val_auc'],lw=0.75, color='green',zorder=10,alpha=1,label='val_auc')
        graph.set_xlabel("Nombre d'epochs")
        graph.set_ylabel("Fonction de métrique")
        graph.set_title("Graphique représentant les fonctions de métriques au cour de l'apprentissage du modèle"+self.model_name)
        # graph.set_ylim(0,1e0)
        graph.axhline(0,-1e4,1e8,linestyle=':',color='black',zorder=-10)
        graph.legend()
        graph.grid()
        plt.show()

        fig, graph = plt.subplots(figsize=(1500/72,8),layout="constrained")
        graph.plot(self.h['precision'],lw=0.75, color='blue',zorder=10,alpha=1,label='precision')
        graph.plot(self.h['val_precision'],lw=0.75, color='green',zorder=10,alpha=1,label='val_precision')
        graph.set_xlabel("Nombre d'epochs")
        graph.set_ylabel("Fonction de métrique")
        graph.set_title("Graphique représentant les fonctions de métriques au cour de l'apprentissage du modèle"+self.model_name)
        # graph.set_ylim(0,1e0)
        graph.axhline(0,-1e4,1e8,linestyle=':',color='black',zorder=-10)
        graph.legend()
        graph.grid()
        plt.show()

        fig, graph = plt.subplots(figsize=(1500/72,8),layout="constrained")
        graph.plot(self.h['recall'],lw=0.75, color='blue',zorder=10,alpha=1,label='recall')
        graph.plot(self.h['val_recall'],lw=0.75, color='green',zorder=10,alpha=1,label='val_recall')
        graph.set_xlabel("Nombre d'epochs")
        graph.set_ylabel("Fonction de métrique")
        graph.set_title("Graphique représentant les fonctions de métriques au cour de l'apprentissage du modèle"+self.model_name)
        # graph.set_ylim(0,1e0)
        graph.axhline(0,-1e4,1e8,linestyle=':',color='black',zorder=-10)
        graph.legend()
        graph.grid()
        plt.show()

        fig, graph = plt.subplots(figsize=(1500/72,8),layout="constrained")
        graph.plot(self.h['lr'],lw=0.75, color='blue',zorder=10,alpha=1,label='learning rate')
        graph.set_xlabel("Nombre d'epochs")
        graph.set_ylabel("Learning rate")
        graph.set_title("Graphique représentant le taux d'apprentissage au cour de l'apprentissage du modèle"+self.model_name)
        # graph.set_ylim(0,1e0)
        graph.axhline(0,-1e4,1e8,linestyle=':',color='black',zorder=-10)
        graph.legend()
        graph.grid()
        plt.show()

    def trace_cm(self):
        colors = ["black", "blue", "cyan","lightcyan","white"]#
        self.cmap1 = mp.colors.LinearSegmentedColormap.from_list("mycmap", colors)
        cm_tr = confusion_matrix(np.array(self.test_data_R_proba).argmax(axis=1), np.array(self.ytrain_hat).argmax(axis=1))#Création de la matrice de confusion sur les données d'entrainement
        cm_tt = confusion_matrix(np.array(self.test_data_test_proba).argmax(axis=1), np.array(self.ytest_hat).argmax(axis=1))#Création de la matrice de confusion sur les données de test
        label=self.test_data_proba_sauv['composes']
        fig, (graph_1,graph_2) = plt.subplots(nrows=1,ncols=2,figsize=(21,10),layout="constrained")
        sns.heatmap(pd.DataFrame(cm_tr, columns=self.test_data_proba_sauv['composes'], index=self.test_data_R_intensite.index[:29]), ax=graph_1, cmap=self.cmap1, annot=True, linewidth=.25,linecolor="grey",xticklabels=label,yticklabels=label,cbar_kws={'label':'Nombre de spectres'})
        sns.heatmap(pd.DataFrame(cm_tt, columns=self.test_data_proba_sauv['composes'], index=self.test_data_proba_sauv['composes']), ax=graph_2, cmap=self.cmap1, annot=True, linewidth=.25,linecolor="grey",xticklabels=label,yticklabels=label,cbar_kws={'label':'Nombre de spectres'})
        graph_1.set_xlabel("Composé prédit")#Ajout d'un titre à l'axe x
        graph_1.set_ylabel("Composé réel")#Ajout d'un titre à l'axe y
        graph_1.tick_params(axis='x',top=True, labeltop=True, bottom=True, labelbottom=True,labelrotation=90)
        graph_1.tick_params(axis='y',left=True, labelleft=True, right=True, labelright=True,labelrotation=0)
        graph_1.plot([29,0],[29,0],linestyle=':',color='lawngreen',zorder=10)
        graph_2.set_xlabel("Composé prédit")#Ajout d'un titre à l'axe x
        graph_2.set_ylabel("Composé réel")#Ajout d'un titre à l'axe y
        graph_2.tick_params(axis='x',top=True, labeltop=True, bottom=True, labelbottom=True,labelrotation=90)
        graph_2.tick_params(axis='y',left=True, labelleft=True, right=True, labelright=True,labelrotation=0)
        graph_2.plot([29,0],[29,0],linestyle=':',color='lawngreen',zorder=10)
        plt.show()

    def trace_cm_app(self):
        for i in self.test_app:
            cm_tr = confusion_matrix(np.array(self.test_data_R_proba).argmax(axis=1), np.array(self.test_app[i]['y_train_pred']).argmax(axis=1))#Création de la matrice de confusion sur les données d'entrainement
            cm_tt = confusion_matrix(np.array(self.test_data_test_proba).argmax(axis=1), np.array(self.test_app[i]['y_test_pred']).argmax(axis=1))#Création de la matrice de confusion sur les données de test
            label=self.test_data_proba_sauv['composes']
            fig, (graph_1,graph_2) = plt.subplots(nrows=1,ncols=2,figsize=(21,10),layout="constrained")
            sns.heatmap(pd.DataFrame(cm_tr, columns=self.test_data_proba_sauv['composes'], index=self.test_data_R_intensite.index[:29]), ax=graph_1, cmap=self.cmap1, annot=True, linewidth=.25,linecolor="grey",xticklabels=label,yticklabels=label,cbar_kws={'label':'Nombre de spectres'})
            sns.heatmap(pd.DataFrame(cm_tt, columns=self.test_data_proba_sauv['composes'], index=self.test_data_proba_sauv['composes']), ax=graph_2, cmap=self.cmap1, annot=True, linewidth=.25,linecolor="grey",xticklabels=label,yticklabels=label,cbar_kws={'label':'Nombre de spectres'})
            graph_1.set_xlabel(f"Composé prédit à l'epoch {i}")#Ajout d'un titre à l'axe x
            graph_1.set_ylabel("Composé réel")#Ajout d'un titre à l'axe y
            graph_1.tick_params(axis='x',top=True, labeltop=True, bottom=True, labelbottom=True,labelrotation=90)
            graph_1.tick_params(axis='y',left=True, labelleft=True, right=True, labelright=True,labelrotation=0)
            graph_1.plot([29,0],[29,0],linestyle=':',color='lawngreen',zorder=10)
            graph_2.set_xlabel(f"Composé prédit à l'epoch {i}")#Ajout d'un titre à l'axe x
            graph_2.set_ylabel("Composé réel")#Ajout d'un titre à l'axe y
            graph_2.tick_params(axis='x',top=True, labeltop=True, bottom=True, labelbottom=True,labelrotation=90)
            graph_2.tick_params(axis='y',left=True, labelleft=True, right=True, labelright=True,labelrotation=0)
            graph_2.plot([29,0],[29,0],linestyle=':',color='lawngreen',zorder=10)
            plt.show()

    def trace_cm_melange(self):
        colors = ["black", "blue", "cyan"]#"lightcyan"
        cmap1 = mp.colors.LinearSegmentedColormap.from_list("mycmap", colors)
        # --------------------------------------------------------------------------------------------------------------------------

        mel=0.5

        test_mat=np.empty((29,29))
        test_mat[:]=0.0
        test_mat=pd.DataFrame(test_mat)

        for k in range(self.test_pred_mel.shape[0]):
            x=np.array(self.test_proba_melange[k:k+1])
            x_p=np.array(self.test_pred_mel[k:k+1])
            if np.argwhere(x>=mel).shape[0]>=2:
                # print("bloucle 1")
                index=np.argwhere(x==mel)
                # print(index)
                test=np.empty((x.shape[1],x.shape[1]))
                test[:]=np.nan
                # print(test.shape)
                for i in range(x.shape[1]):
                    for j in range(x.shape[1]):
                        # print(i,j)
                        if i==index[0][1] and j==index[1][1]:
                            test[i][j]=x_p[0][index[0][1]]-x[0][index[0][1]]
                        elif i==index[1][1] and j==index[0][1]:
                            test[i][j]=x_p[0][index[1][1]]-x[0][index[1][1]]
                        else:
                            test[i][j]=0
                        # print(test)
            elif np.argwhere(x>=mel).shape[0]==1:
                # print("bloucle 2")
                index=np.argwhere(x>=mel)
                # print(index)
                test=np.empty((x.shape[1],x.shape[1]))
                test[:]=np.nan
                # print(test.shape)
                for i in range(x.shape[1]):
                    for j in range(x.shape[1]):
                        # print(i,j)
                        if i==index[0][1] and j==index[0][1]:
                            test[i][j]=x_p[0][index[0][1]]-x[0][index[0][1]]
                        else:
                            test[i][j]=0
                        # print(test)
            # print(f"\n\n{test}\n\n")
            test=pd.DataFrame(test)
            test_mat=test_mat.add(test)
            # display(test,test_mat)
        # display(test_mat)

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        colors = ["red", "magenta","lawngreen","cyan","blue"]#"lightcyan"
        nodes = [-0.5, -0.25,-0.15,  0.15,0.25, 0.5]
        cmap2 = mp.colors.ListedColormap(colors,"mycmap")
        # cmap2=mp.colors.from_levels_and_colors(nodes,colors)
        norm = mp.colors.BoundaryNorm([-0.5, -0.30,-0.15,  0.15,0.30, 0.5], cmap2.N)
        tick=[-0.50,-0.40,-0.30,-0.20,-0.15,-0.10,0.00,0.10,0.15,0.20,0.30,0.40,0.50]
        label=self.test_proba_data['composes']

        fig,graph = plt.subplots(figsize=(18.5,14.5),facecolor='white',layout="constrained")#Positionnement et ajout d'un titre à la première matrice de confusion ,layout="constrained"
        sns.heatmap(test_mat, ax=graph, annot=False, fmt=".0f",cmap=cmap2,norm=norm,vmin=-0.5,vmax=0.5, linewidth=.5,linecolor="grey",xticklabels=label,yticklabels=label,cbar_kws={'label':'écart de la prédiction avec la valeur réelle','ticks':tick,'spacing':'proportional'})#Tracé de la matrice de confusionplt.cm.Blues

        graph.set_xlabel("Elément du mélange", fontsize = 12)#Ajout d'un titre à l'axe x
        graph.set_ylabel("Elément du mélange", fontsize = 12)#Ajout d'un titre à l'axe y
        graph.tick_params(axis='x',top=True, labeltop=True, bottom=True, labelbottom=True,labelrotation=90)
        graph.tick_params(axis='y',left=True, labelleft=True, right=True, labelright=True,labelrotation=0)
        graph.plot([29,0],[29,0],linestyle=':',color='black',zorder=10)
        graph.grid(linestyle=':',color='grey',zorder=-10)

        plt.show()#Affichage du graphique

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        mel=0.5

        test_mat=np.empty((29,29))
        test_mat[:]=0.0
        test_mat=pd.DataFrame(test_mat)
        count_pred=0

        for k in range(self.test_pred_mel.shape[0]):
            x=np.array(self.test_proba_melange[k:k+1])
            x_p=np.array(self.test_pred_mel[k:k+1])
            # print(x[0],x_p.shape)
            max_x=x[0].argsort()
            max_x_p=x_p[0].argsort()
            max_x=np.sort(max_x[-2:])
            max_x_p=np.sort(max_x_p[-2:])
            # print(max_x,max_x_p)
            if np.argwhere(x>=mel).shape[0]>=2:
                # print("bloucle 1")
                index=np.argwhere(x==mel)
                # print(index)
                test=np.empty((x.shape[1],x.shape[1]))
                test[:]=np.nan
                # print(test.shape)
                for i in range(x.shape[1]):
                    for j in range(x.shape[1]):
                        # print(i,j)
                        if i==index[0][1] and j==index[1][1]:
                            if max_x[0]==max_x_p[0] or max_x[0]==max_x_p[1]:
                                test[i][j]=1.0
                                count_pred+=1
                            else:
                                test[i][j]=0.0
                        elif i==index[1][1] and j==index[0][1]:
                            if max_x[1]==max_x_p[1] or max_x[1]==max_x_p[0]:
                                test[i][j]=1.0
                                count_pred+=1
                            else:
                                test[i][j]=0.0
                        else:
                            test[i][j]=0
                        # print(test)
            elif np.argwhere(x>=mel).shape[0]==1:
                # print("bloucle 2")
                index=np.argwhere(x>=mel)
                # print(index)
                test=np.empty((x.shape[1],x.shape[1]))
                test[:]=np.nan
                # print(test.shape)
                for i in range(x.shape[1]):
                    for j in range(x.shape[1]):
                        # print(i,j)
                        if i==index[0][1] and j==index[0][1]:
                            if max_x[0]==max_x_p[0] or max_x[0]==max_x_p[1] or max_x[1]==max_x_p[0] or max_x[1]==max_x_p[1]:
                                test[i][j]=1.0
                                count_pred+=1
                            else:
                                test[i][j]=0.0
                        else:
                            test[i][j]=0
                        # print(test)
            # print(f"\n\n{test}\n\n")
            test=pd.DataFrame(test)
            test_mat=test_mat.add(test)
            # display(test,test_mat)
        # display(test_mat)
        print(f"\n\n\033[1m\033[38;2;255;255;0mNombres de bonnes prédictions : \033[38;2;0;255;0m{count_pred}\033[38;2;255;255;255m/\033[38;2;255;0;0m{29*29}")

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        colors = ["red","lawngreen"]#"lightcyan"
        nodes = [-0.5, -0.25,-0.15,  0.15,0.25, 0.5]
        cmap2 = mp.colors.ListedColormap(colors,"mycmap")
        # cmap2=mp.colors.from_levels_and_colors(nodes,colors)
        norm = mp.colors.BoundaryNorm([0.0, 0.5,1.0], cmap2.N)
        tick=[0.0,1]
        label=self.test_proba_data['composes']

        fig,graph = plt.subplots(figsize=(18.5,14.5),facecolor='white',layout="constrained")#Positionnement et ajout d'un titre à la première matrice de confusion ,layout="constrained"
        sns.heatmap(test_mat, ax=graph, annot=False, fmt=".0f",cmap=cmap2,norm=norm,vmin=-0.5,vmax=0.5, linewidth=.5,linecolor="grey",xticklabels=label,yticklabels=label,cbar_kws={'label':'écart de la prédiction avec la valeur réelle','ticks':tick,'spacing':'proportional'})#Tracé de la matrice de confusionplt.cm.Blues.ax.set_yticklabels(['< -1', '0', '> 1'])
        graph.collections[0].colorbar.set_ticklabels(['non prédit','prédit'])
        graph.set_xlabel("Elément du mélange", fontsize = 12)#Ajout d'un titre à l'axe x
        graph.set_ylabel("Elément du mélange", fontsize = 12)#Ajout d'un titre à l'axe y
        graph.tick_params(axis='x',top=True, labeltop=True, bottom=True, labelbottom=True,labelrotation=90)
        graph.tick_params(axis='y',left=True, labelleft=True, right=True, labelright=True,labelrotation=0)
        graph.plot([29,0],[29,0],linestyle=':',color='black',zorder=10)
        graph.grid(linestyle=':',color='grey',zorder=-10)

        plt.show()#Affichage du graphique
        #plt.savefig('../Graphiques/mat_proba_4conv_TD_LSTM50_kd'+ksi+'.png', dpi=300, format='png', metadata=None, bbox_inches=None, pad_inches=0.1, facecolor='auto', edgecolor='auto', backend=None)




        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        mel=0.5
        mel_tol=0.10

        test_mat=np.empty((29,29))
        test_mat[:]=0.0
        test_mat=pd.DataFrame(test_mat)
        count_pred=0

        for k in range(self.test_pred_mel.shape[0]):
            x=np.array(self.test_proba_melange[k:k+1])
            x_p=np.array(self.test_pred_mel[k:k+1])
            # print(x[0],x_p.shape)
            max_x=x[0].argsort()
            max_x_p=x_p[0].argsort()
            max_x=np.sort(max_x[-2:])
            max_x_p=np.sort(max_x_p[-2:])
            # print(max_x,max_x_p)
            if np.argwhere(x>=mel).shape[0]>=2:
                # print("bloucle 1")
                index=np.argwhere(x==mel)
                # print(index)
                test=np.empty((x.shape[1],x.shape[1]))
                test[:]=np.nan
                # print(test.shape)
                for i in range(x.shape[1]):
                    for j in range(x.shape[1]):
                        # print(i,j)
                        if i==index[0][1] and j==index[1][1]:
                            if x_p[0][index[0][1]]>=mel_tol:
                                test[i][j]=1.0
                                count_pred+=1
                            else:
                                test[i][j]=0.0
                        elif i==index[1][1] and j==index[0][1]:
                            if x_p[0][index[1][1]]>=mel_tol:
                                test[i][j]=1.0
                                count_pred+=1
                            else:
                                test[i][j]=0.0
                        else:
                            test[i][j]=0
                        # print(test)
            elif np.argwhere(x>=mel).shape[0]==1:
                # print("bloucle 2")
                index=np.argwhere(x>=mel)
                # print(index)
                test=np.empty((x.shape[1],x.shape[1]))
                test[:]=np.nan
                # print(test.shape)
                for i in range(x.shape[1]):
                    for j in range(x.shape[1]):
                        # print(i,j)
                        if i==index[0][1] and j==index[0][1]:
                            if x_p[0][index[0][1]]>=mel_tol:
                                test[i][j]=1.0
                                count_pred+=1
                            else:
                                test[i][j]=0.0
                        else:
                            test[i][j]=0
                        # print(test)
            # print(f"\n\n{test}\n\n")
            test=pd.DataFrame(test)
            test_mat=test_mat.add(test)
            # display(test,test_mat)
        # display(test_mat)
        print(f"\n\n\033[1m\033[38;2;255;255;0mNombres de bonnes prédictions : \033[38;2;0;255;0m{count_pred}\033[38;2;255;255;255m/\033[38;2;255;0;0m{29*29}")

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        colors = ["red","lawngreen"]#"lightcyan"
        nodes = [-0.5, -0.25,-0.15,  0.15,0.25, 0.5]
        cmap2 = mp.colors.ListedColormap(colors,"mycmap")
        # cmap2=mp.colors.from_levels_and_colors(nodes,colors)
        norm = mp.colors.BoundaryNorm([0.0, 0.5,1.0], cmap2.N)
        tick=[0.0,1]
        label=self.test_proba_data['composes']

        fig,graph = plt.subplots(figsize=(18.5,14.5),facecolor='white',layout="constrained")#Positionnement et ajout d'un titre à la première matrice de confusion ,layout="constrained"
        sns.heatmap(test_mat, ax=graph, annot=False, fmt=".0f",cmap=cmap2,norm=norm,vmin=-0.5,vmax=0.5, linewidth=.5,linecolor="grey",xticklabels=label,yticklabels=label,cbar_kws={'label':'écart de la prédiction avec la valeur réelle','ticks':tick,'spacing':'proportional'})#Tracé de la matrice de confusionplt.cm.Blues.ax.set_yticklabels(['< -1', '0', '> 1'])
        graph.collections[0].colorbar.set_ticklabels(['non prédit','prédit'])
        graph.set_xlabel("Elément du mélange", fontsize = 12)#Ajout d'un titre à l'axe x
        graph.set_ylabel("Elément du mélange", fontsize = 12)#Ajout d'un titre à l'axe y
        graph.tick_params(axis='x',top=True, labeltop=True, bottom=True, labelbottom=True,labelrotation=90)
        graph.tick_params(axis='y',left=True, labelleft=True, right=True, labelright=True,labelrotation=0)
        graph.plot([29,0],[29,0],linestyle=':',color='black',zorder=10)
        graph.grid(linestyle=':',color='grey',zorder=-10)

        plt.show()#Affichage du graphique
        #plt.savefig('../Graphiques/mat_'+str(mel_tol*100)+'_4conv_TD_LSTM50_kd'+ksi+'.png', dpi=300, format='png', metadata=None,bbox_inches=None, pad_inches=0.1,facecolor='auto', edgecolor='auto',backend=None)







        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


        test_pred_mel2=pd.DataFrame(self.test_pred_mel,index=self.test_proba_melange_save['mélange'],columns=self.test_proba_melange_save.drop(['mélange'],axis=1).columns)
        def f(p,q):
            print(p+'_'+q)
            pl=(p+'_'+q)
            prob_test=test_pred_mel2.loc[test_pred_mel2.index==pl].T
            display(prob_test.style.background_gradient(cmap=cmap1))
            fig, graph = plt.subplots(figsize=(1500/72,8),layout="constrained")
            graph.plot(np.linspace(10.0,0.0,21800),self.test_intensite_melange['intensité_'+pl],lw=0.75,color='black',zorder=10,alpha=1,label='spectre melange theorique')#
            graph.set_xlabel("Chemical Shift / ppm")
            graph.set_ylabel("Signal (a.u.)")
            graph.set_ylim(0,1e0)
            graph.axhline(0,-1e4,1e8,linestyle=':',color='black',zorder=-10)
            graph.axvline(5.5,-1e4,1e6,linestyle=':',color='red',zorder=-10)
            graph.axvline(4.5,-1e4,1e6,linestyle=':',color='red',zorder=-10)
            graph.invert_xaxis()
            graph.legend()
            graph.grid()
            plt.show()
            index=self.test_proba_melange_save.loc[self.test_proba_melange_save['mélange']==pl].index
            for i in self.li:
                fig, graph = plt.subplots(figsize=(1500/72,4),layout="constrained")
                for j in range((self.vis_conv_test['c_'+str(i)].shape[2])):
                    graph.plot(np.linspace(10.0,0.0,self.vis_conv_test['c_'+str(i)].shape[1]),self.vis_conv_test['c_'+str(i)][index[0],:,j],lw=0.5,zorder=0,label=('plan de convolution '+str(j)))
                graph.set_xlabel("Chemical Shift / ppm")
                graph.set_ylabel("Signal (a.u.)")
                graph.set_title("Graphique représentant les cartes de caratéristique de la couche "+str(i)+" du modél")
                # graph.set_ylim(-0.5,0.5e0)
                graph.axhline(0,-1e4,1e8,linestyle=':',color='black',zorder=-10)
                graph.grid()
                graph.invert_xaxis()
                plt.show()
            lt=list(np.array(prob_test))
            max_1=max(lt)
            lt.remove(max_1)
            max_2=max(lt)
            print(max_1,max_2)
        w=widgets.interactive(f,p=self.test_proba_data['composes'],q=self.test_proba_data['composes'])
        display(w)

    def reload(self):
        for self.model_name in self.modeles:
            display(Markdown(f'<h4 style="background:black;color:cyan"><b> {self.model_name} </b></h4>'))
            self.load_var()
            self.read_csv()
            self.load_model()
            self.pred()
            if self.metrique == True:
                self.trace_metrique()
            if self.cm == True:
                self.trace_cm()
            if self.cm_app == True:
                self.trace_cm_app()
            if self.cm_melange == True:
                self.trace_cm_melange()
            print(f"\n\n\n\n")
        print(f"\t\t\t\t\033[38;2;255;0;0m\033[1m--------------------Fin--------------------\033[0m")