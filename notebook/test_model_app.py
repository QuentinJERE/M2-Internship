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
import json
import pickle
import time

#------------------------------------------------------------------------------------------------------------------------------------------

print(f"\033[1m------------------------------------------------------------------------------------------------------------------------------------------\033[0m")
print(f"\t\t\t\t\033[38;2;0;255;0m\033[1m--------------------Début--------------------\033[0m")
print(f"\033[1m------------------------------------------------------------------------------------------------------------------------------------------\033[0m")

#------------------------------------------------------------------------------------------------------------------------------------------

dec=['0.05_0.000_4','0.05_0.001_4','0.05_0.010_4','0.05_0.000_20','0.05_0.001_20','0.05_0.010_20']
mod=['model_1_40','model_1_80','model_1_80','model_1_80','model_1_80','model_1_80']
fra=[0.60,0.20,0.20,0.20,0.20,0.20]

#------------------------------------------------------------------------------------------------------------------------------------------

for u,v,k in zip(dec,mod,fra):
    print(f"\033[1m------------------------------------------------------------------------------------------------------------------------------------------\033[0m")
    print(f"\t\t\033[38;2;0;255;0m\033[1m--------------------Début-{v}_{u}-avec-un-tirage-de-{k}--------------------\033[0m")
    print(f"\033[1m------------------------------------------------------------------------------------------------------------------------------------------\033[0m")

    name_decal=u
    name_model=v

    try:
        os.mkdir('../modeles/'+name_model+'_'+name_decal+'/')
        print(f"\033[38;2;255;255;0m\033[1mDossier créé\033[0m")
    except OSError as error:
        print(f"\033[38;2;255;0;0m\033[1m{error}\033[0m")

    #------------------------------------------------------------------------------------------------------------------------------------------
    
    print(f"\033[38;2;255;128;0m\033[1mChargement des données\033[0m")
    mul=int(name_decal[11:])*2+1
    test_data_intensite=pd.read_csv('../spectres_csv/decalage/spectres_ng/'+name_decal+'.csv', sep=";",index_col=0)
    test_data_x=pd.read_csv('../spectres_csv/spectres_ng/data_x.csv', sep=";",index_col=0)
    test_data_proba=pd.read_csv('../spectres_csv/decalage/data_proba_'+name_decal+'.csv', sep=";",index_col=0)
    test_2_data_intensite=pd.read_csv('../spectres_csv/decalage/spectres_proc/'+name_decal+'.csv', sep=";",index_col=0)
    test_2_data_intensite=test_2_data_intensite.add_suffix('_proc')
    test_2_data_x=pd.read_csv('../spectres_csv/spectres_proc/data_x.csv', sep=";",index_col=0)
    test_data_proba_sauv=test_data_proba
    test_data_R_intensite=pd.concat([test_data_intensite.T,test_2_data_intensite.T])#.reset_index(drop=True)
    test_data_R_proba=pd.concat([test_data_proba.drop(['composes'],axis=1).T,test_data_proba.drop(['composes'],axis=1).T]).set_index(test_data_R_intensite.index,drop=True)#.reset_index(drop=True)
    test_data_R_proba_sauv=test_data_R_proba
    df=test_data_R_intensite[0:0]
    for i in test_data_R_intensite[:29].index:
        df=pd.concat([df,test_data_R_intensite.filter(regex=i,axis=0).drop([i,i+'_proc'],axis=0).sample(frac=k,axis=0)])
    ts=df.shape[0]
    test_test_proba=test_data_R_proba_sauv.loc[df.index]
    test_data_R_intensite=test_data_R_intensite.drop(df.index,axis=0)
    test_data_R_proba=test_data_R_proba.drop(test_test_proba.index,axis=0)
    ts_2=test_data_R_intensite.shape[0]

    test_train=test_data_intensite.T
    test_train=tf.constant(test_train)
    test_train=tf.reshape(test_train,[(29*mul),21800,1])
    test_test=df
    test_test=tf.constant(test_test)
    test_test=tf.reshape(test_test,[ts,21800,1])
    test_train_R=test_data_R_intensite
    test_train_R=tf.constant(test_train_R)
    test_train_R=tf.reshape(test_train_R,[ts_2,21800,1])
    #test_train=np.array(test_train).reshape(3,21800,1)
    test_data_proba=np.array(test_data_proba.drop(['composes'],axis=1).T)
    test_data_R_proba=np.array(test_data_R_proba)
    test_data_test_proba=np.array(test_test_proba)

    #------------------------------------------------------------------------------------------------------------------------------------------

    with open('../modeles/'+name_model+'_'+name_decal+'/index_train_intensite.pkl', "wb") as fp:
        pickle.dump(test_data_R_intensite.index,fp)
    print(f"\033[38;2;0;255;255m\033[1mIndexs sauvegardés\033[0m")

    with open('../modeles/'+name_model+'_'+name_decal+'/index_test_intensite.pkl', "wb") as fp:
        pickle.dump(df.index,fp)
    print(f"\033[38;2;0;255;255m\033[1mIndexs sauvegardés\033[0m")

    with open('../modeles/'+name_model+'_'+name_decal+'/index_train_proba.pkl', "wb") as fp:
        pickle.dump(test_data_R_intensite.index,fp)
    print(f"\033[38;2;0;255;255m\033[1mIndexs sauvegardés\033[0m")

    with open('../modeles/'+name_model+'_'+name_decal+'/index_test_proba.pkl', "wb") as fp:
        pickle.dump(test_test_proba.index,fp)
    print(f"\033[38;2;0;255;255m\033[1mIndexs sauvegardés\033[0m")

    #------------------------------------------------------------------------------------------------------------------------------------------
        
    class PredictionCallback(tf.keras.callbacks.Callback):
        def __init__(self, model, x_train, x_test,dict_test_app):
            self.model = model
            self.x_train = x_train
            self.x_test = x_test
            self.dict_test_app=dict_test_app
        def on_epoch_end(self, epoch, logs={}):
            print(f"\n\t\033[38;2;128;0;255m\033[1mPrédictions : \033[0m\033[38;2;0;255;255m")
            y_train_pred = self.model.predict(self.x_train)
            y_test_pred = self.model.predict(self.x_test)
            self.dict_test_app.update({epoch:{'y_train_pred':y_train_pred,'y_test_pred':y_test_pred}})
            print(f"\n\t\033[38;2;255;0;0m\033[1m----------Fin-prédictions----------\033[0m")

    #------------------------------------------------------------------------------------------------------------------------------------------
    

    print(f"\033[38;2;255;0;255m\033[1mGénération du modèle\033[0m")

    if 'model' in globals():
        del model
    if 'history' in globals():
        del history

    initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)

    input=keras.Input(shape=(21800,1), name="in_layer")

    x_conv2=keras.layers.Conv1D(filters=12, kernel_size=3, strides=1, padding='same', name="conv1D_1_1")(input)
    x2=keras.layers.BatchNormalization(name='batchnorm_1_1')(x_conv2)
    x2=keras.layers.ReLU(name='activ_1_1')(x2)
    x2=keras.layers.Flatten(name='flatten_1_1')(x2)
    x2=keras.layers.Dense(1000, activation="relu", name="dense_1_1")(x2)

    x_conv=keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', name="conv1D_1")(input)
    x=keras.layers.MaxPooling1D(pool_size=7, strides=None, padding="same" ,name="maxpool1D_1")(x_conv)
    x=keras.layers.BatchNormalization(name='batchnorm_1')(x)
    x=keras.layers.ReLU(name='activ_1')(x)
    x_conv=keras.layers.Conv1D(64, 9, strides=4, padding="same", name="conv1D_2", activation="relu")(x)
    x=keras.layers.MaxPooling1D(pool_size=9, strides=None, padding="same" ,name="maxpool1D_2")(x_conv)
    x=keras.layers.BatchNormalization(name='batchnorm_2')(x)
    x=keras.layers.ReLU(name='activ_2')(x)
    x=keras.layers.Flatten(name='flatten_1_2')(x)
    x=keras.layers.Dense(1000, activation="relu", name="dense_1_2")(x)

    x=keras.layers.Concatenate(axis=-1,name='concat_1')([x2,x])
    x=keras.layers.Dense(1000, activation="relu", name="dense_1")(x)
    x=keras.layers.Dropout(0.5,name="dropout_2")(x)
    x=keras.layers.Dense(500, activation="relu", name="dense_2")(x)
    x_out=keras.layers.Dense(29, activation="softmax", name="proba_end")(x)


    model = keras.Model(inputs=input,outputs=x_out)
    model.summary()

    #------------------------------------------------------------------------------------------------------------------------------------------


    dict_test_app=dict()
    t_start=time.time()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy',keras.metrics.TopKCategoricalAccuracy(k=1,name="top_k_categorical_accuracy"),keras.metrics.AUC(name="auc"),keras.metrics.Precision(name="precision"),keras.metrics.Recall(name="recall")])

    print(f"\033[38;2;0;0;255m\033[1mDébut apprentissage\033[0m")

    history=model.fit(
        test_train_R,
        test_data_R_proba,
        validation_data = (test_test,test_data_test_proba),
        epochs=400,
        shuffle=True,
        verbose = 1,
        callbacks= [keras.callbacks.EarlyStopping(monitor='accuracy', mode='auto', verbose=1, patience=30,restore_best_weights=False),
                    keras.callbacks.TensorBoard("./test/tb_log_1",write_graph=False,write_images=True,update_freq="epoch",histogram_freq=0),
                    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, min_lr=0.00000001, verbose=1, mode="auto", min_delta=1e-4),
                    keras.callbacks.ModelCheckpoint('../model_checkpoint/'+name_model+'_'+name_decal,monitor="val_loss",verbose=1,save_best_only=False,save_weights_only=False,mode="min",save_freq="epoch",initial_value_threshold=None,),
                    keras.callbacks.BackupAndRestore('../model_backup/'+name_model+'_'+name_decal+'/', save_freq="epoch", delete_checkpoint=True),
                    PredictionCallback(model,test_train_R,test_test,dict_test_app)])#
    t_stop=time.time()
    print(f"\033[38;2;255;0;0m\033[1mFin apprentissage\033[0m")
    print(f"\033[38;2;128;0;255mDuration : {t_stop-t_start}s")

    #------------------------------------------------------------------------------------------------------------------------------------------

    h=history.history
    with open('../modeles/'+name_model+'_'+name_decal+'/history.pkl', "wb") as fp:
        pickle.dump(h,fp)
    print(f"\033[38;2;0;255;255m\033[1mHystorique d'aprentissage sauvegardés\033[0m")

    with open('../modeles/'+name_model+'_'+name_decal+'/test_app.pkl', "wb") as fp:
        pickle.dump(dict_test_app,fp)
    print(f"\033[38;2;0;255;255m\033[1mTest d'apprentissage sauvegardés\033[0m")

    model.save('../modeles/'+name_model+'_'+name_decal+'/model.keras', overwrite=True)
    print(f"\033[38;2;0;255;0m\033[1mModèle sauvegardés\033[0m")

    print(f"\033[1m------------------------------------------------------------------------------------------------------------------------------------------\033[0m")
    print(f"\t\t\033[38;2;255;0;0m\033[1m--------------------Fin-{v}_{u}-avec-un-tirage-de-{k}--------------------\033[0m")
    print(f"\033[1m------------------------------------------------------------------------------------------------------------------------------------------\033[0m")

#------------------------------------------------------------------------------------------------------------------------------------------
print(f"\033[1m------------------------------------------------------------------------------------------------------------------------------------------\033[0m")
print(f"\t\t\t\t\033[38;2;255;0;0m\033[1m--------------------Fin--------------------\033[0m")
print(f"\033[1m------------------------------------------------------------------------------------------------------------------------------------------\033[0m")
