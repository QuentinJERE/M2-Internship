'''Module de traitement des spectres RMN à partir des fichiers Bruker brute'''
__author__ = "Quentin Jéré"
__maintainer__ =  "Quentin Jéré"
__email__ = "quentin.jere@univ-tlse3.fr"
__status__ = "Development"

import pandas as pd
import nmrglue as ng
from matplotlib import pyplot as plt
import numpy as np
import os
from keras.utils import to_categorical
import scipy
from IPython.display import display,Markdown,display_html
class Traitement:
    def __init__(self,
                 path="../SPECTRES_RMN",
                 path_save="../spectres_csv/",
                 sol_mode='low',
                 sol_fl=5,
                 ps_p0=None,
                 ps_p1=None,
                 med_nw=24,
                 med_sf=24,
                 med_sigma=5,
                 column_1='x',
                 column_2='intensité',
                 val_cut_0=0.2,
                 val_cut_10=10.0,
                 val_cut_sol_min=4.5,
                 val_cut_sol_max=5.5,
                 nb_max=21800,
                 val_min_ppm=0.0,
                 val_max_ppm=10.0,
                 trace=False,
                 verbose=False,
                 disp_df=False):
        
        self.path_main=path
        self.path=None
        self.path_proc=None
        self.path_save=path_save
        self.sol_mode=sol_mode
        self.sol_fl=sol_fl
        self.ps_p0=ps_p0
        self.ps_p1=ps_p1
        self.med_nw=med_nw
        self.med_sf=med_sf
        self.med_sigma=med_sigma
        self.column_1=column_1
        self.column_2=column_2
        self.val_cut_0=val_cut_0
        self.val_cut_10=val_cut_10
        self.val_cut_sol_min=val_cut_sol_min
        self.val_cut_sol_max=val_cut_sol_max
        self.nb_max=nb_max
        self.val_min_ppm=val_min_ppm
        self.val_max_ppm=val_max_ppm
        self.dic=None
        self.dic_save=None
        self.data=None
        self.pdic=None
        self.pdic_save=None
        self.pdata=None
        self.C=None
        self.uc=None
        self.uc_proc=None
        self.df1=None
        self.df2=None
        self.df3=None
        self.df4=None
        self.df5=None
        self.df6=None
        self.cut_0=None
        self.cut_10=None
        self.cut_45_55=None
        self.ecart=None
        self.truc=None
        self.truc_2=None
        self.val_max=None
        self.val_min=None
        self.point_max=None
        self.point_min=None
        self.file_path=None
        self.time=None
        self.p=None
        self.p_2=None
        self.val=None
        self.trace=trace
        self.verbose=verbose
        self.disp_df=disp_df

        self.traite()

    def charge_dossier(self):
        '''charge le chemin d'accés de chaque dossier contenant les fichiers Bruker pour chaque composés(file_path)'''
        self.path=self.path_main+self.file_path
        dir=os.listdir(self.path)
        self.path=self.path+'/'+dir[0]+'/'
        self.path_proc=self.path+'pdata/1/'
        if self.verbose == True:
            print(f"\033[38;2;255;255;128m\033[1m Dossier charger : \033[0m")
            print(f"\t {self.path_main}")
            print(f"\033[38;2;255;255;0m\033[1m Fichier chargés : \033[0m")
            print(f"\t\t {self.path} \n\t\t {self.path_proc}")

    def read(self):
        '''Lecture des fichiers Bruker'''
        self.dic,self.data = ng.bruker.read(self.path)
        self.dic_save=self.dic
        self.pdic,self.pdata = ng.bruker.read_pdata(self.path_proc)
        self.pdic_save=self.pdic
        if self.verbose == True:
            print(f"\033[38;2;255;128;128m\033[1m Lecture des fichiers binaire : \033[0m")
            print(f"\tOK\n\tOK")

    def convert(self):
        '''Convertion des fichiers Bruker en données nmrpipe afin d'utiliser les fonctions de traitement les plus abouties de nmrglue'''
        self.data = ng.bruker.remove_digital_filter(self.dic, self.data)
        self.C = ng.convert.converter()
        self.C.from_bruker(self.dic,self.data)
        self.dic,self.data = self.C.to_pipe()
        self.uc = ng.pipe.make_uc(self.dic,self.data)
        self.C.from_bruker(self.pdic,self.pdata)
        self.pdic,self.pdata = self.C.to_pipe()
        self.uc_proc = ng.pipe.make_uc(self.pdic,self.pdata)
        if self.verbose == True:
            print(f"\033[38;2;200;255;128m\033[1m Conversion au format nmrpipe : \033[0m")
            print(f"\tOK\n\tOK")

    def solvant(self):
        '''Traitement du pic du solvant avec fl correspondant à la fenêtre'''
        self.dic,self.data = ng.process.pipe_proc.sol(self.dic, self.data, mode=self.sol_mode, fl=self.sol_fl)
        if self.verbose == True:
            print(f"\033[38;2;255;0;128m\033[1m Elimination du solvant : \033[0m")
            print(f"\tOK")

    def filtre(self):
        """Filtrage des bases fréquences correspondant au solvant"""
        self.time=(self.dic_save['acqus']['TD'])/(2*self.dic_save['acqus']['SW_h'])
        b,a=scipy.signal.bessel(1, [60], 'highpass', output='ba', norm='phase',fs=self.data.shape[0]/self.time)
        self.data=scipy.signal.filtfilt(b, a, self.data ,method='gust')
        self.data=scipy.signal.filtfilt(b, a, self.data ,method='gust')
        if self.verbose == True:
            print(f"\033[38;2;255;0;128m\033[1m Filtrage des bases fréquences du solvant : \033[0m")
            print(f"\tOK")

    def apodisation(self):
        self.dic,self.data = ng.process.pipe_proc.em(self.dic, self.data, lb=0.3)
        if self.verbose == True:
            print(f"\033[38;2;128;0;128m\033[1m Apodisation de la FID : \033[0m")
            print(f"\tOK")

    def trans_fourier(self):
        '''Transformées de Fourier'''
        self.dic,self.data = ng.process.pipe_proc.ft(self.dic,self.data,auto=False)
        if self.verbose == True:
            print(f"\033[38;2;0;255;255m\033[1m Transformée de Fourrier : \033[0m")
            print(f"\tOK")

    def make_df(self):
        '''Construction de tableau'''
        self.pdic,self.pdata = ng.process.pipe_proc.di(self.pdic,self.pdata)
        self.df1=pd.DataFrame(self.uc.ppm_scale(),columns=[self.column_1])
        self.df2=pd.DataFrame(self.data,columns=[self.column_2])
        self.df1[self.column_2] = self.df2[self.column_2]
        self.df1_proc=pd.DataFrame(self.uc_proc.ppm_scale(),columns=[self.column_1])
        self.df2_proc=pd.DataFrame(self.pdata,columns=[self.column_2])
        self.df1_proc[self.column_2] = self.df2_proc[self.column_2]
        if self.verbose == True:
            print(f"\033[38;2;128;0;255m\033[1m Génération des tableaux : \033[0m")
            print(f"\tOK\t\tOK")
        if self.disp_df == True:
            df1_styler = self.df1.style.set_table_attributes("style='display:inline'").set_caption('Nmrglue spectrum').set_properties(color="white", align="center",**{"border": "1px solid white"}).set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'blue')]},{"selector": "th:not(.index_name)","props": "background-color: green; color: white; text-align: center"}])
            df2_styler = self.df1_proc.style.set_table_attributes("style='display:inline'").set_caption('TopSpin spectrum').set_properties(color="white", align="center",**{"border": "1px solid white"}).set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'blue')]},{"selector": "th:not(.index_name)","props": "background-color: green; color: white; text-align: center"}])
            display_html("<div style='height:400px;overflow:auto;width:fit-content'>"+df1_styler._repr_html_()+df2_styler._repr_html_()+"</div>", raw=True)

    def est_bruit(self):
        truc=self.df1.loc[(self.df1[self.column_1]>0.1)&(self.df1[self.column_1]<0.5)]
        self.p=np.polynomial.polynomial.polyfit(truc[self.column_1],truc[self.column_2],deg=2)
        truc=self.df1.loc[(self.df1[self.column_1]>9.5)&(self.df1[self.column_1]<10.0)]
        self.p_2=np.polynomial.polynomial.polyfit(truc[self.column_1],truc[self.column_2],deg=2)
        self.m=0.0
        self.e=np.std(self.data[0:100].real)/8
        self.m_i=0.0
        self.e_i=np.std(self.data[0:100].imag)/8
        test=self.df1_proc.loc[(self.df1_proc[self.column_1]<11.0)&(self.df1_proc[self.column_1]>10.0)]
        test=test[self.column_2]
        self.m_proc=0.0
        self.e_proc=np.std(np.array(test).real)
        if self.verbose == True:
            print(f"\033[38;2;128;64;64m\033[1m Meusure du bruit : \033[0m")
            print(f"\tOK\t\tOK")
            print(f"\tEcart-type :\t\033[32mBruit réelle spectre nmrglue : {self.e}\t\033[31mBruit imaginnaire spectre nmrglue : {self.e_i}\t\033[32mBruit réelle spectre TopSpin : {self.e_proc}\033[0m")

    def resample(self):
        dim=self.df1_proc.shape
        self.df1_proc=scipy.signal.resample(self.df1_proc, self.nb_max, domain='time')
        self.df1_proc=pd.DataFrame(self.df1_proc,columns=[self.column_1,self.column_2])
        self.df1_proc=self.df1_proc.sort_values(by=self.column_1, ascending=False)
        if self.verbose == True:
            print(f"\033[38;2;255;128;0m\033[1m rééchantillonnage du spectre TopSpin : \033[0m")
            print(f"\t{dim}-->rééchantillonnés en-->{self.df1_proc.shape}")
            print(f"\tOK")

    def cut(self):
        '''Troncature de la plage de données'''
        self.cut_0=self.df1[self.df1[self.column_1]<self.val_cut_0]
        self.cut_10=self.df1[self.df1[self.column_1]>self.val_cut_10]
        self.df3=self.df1.drop(self.cut_0.index).drop(self.cut_10.index)
        self.cut_0=self.df1_proc[self.df1_proc[self.column_1]<self.val_cut_0]
        self.cut_10=self.df1_proc[self.df1_proc[self.column_1]>self.val_cut_10]
        self.df3_proc=self.df1_proc.drop(self.cut_0.index).drop(self.cut_10.index)
        if self.verbose == True:
            print(f"\033[38;2;255;0;255m\033[1m Découpe des spectres : \033[0m")
            print(f"\tOK\t\tOK")
        if self.disp_df == True:
            df1_styler = self.df3.style.set_table_attributes("style='display:inline'").set_caption('Nmrglue spectrum').set_properties(color="white", align="center",**{"border": "1px solid white"}).set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'blue')]},{"selector": "th:not(.index_name)","props": "background-color: green; color: white; text-align: center"}])
            df2_styler = self.df3_proc.style.set_table_attributes("style='display:inline'").set_caption('TopSpin spectrum').set_properties(color="white", align="center",**{"border": "1px solid white"}).set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'blue')]},{"selector": "th:not(.index_name)","props": "background-color: green; color: white; text-align: center"}])
            display_html("<div style='height:400px;overflow:auto;width:fit-content'>"+df1_styler._repr_html_()+df2_styler._repr_html_()+"</div>", raw=True)

    def complete(self):
        '''Complétion de la plage de données jusqu\'aux limites fixées incluses'''
        self.df3=self.df3.reset_index(drop=True)
        self.ecart=self.nb_max-self.df3.shape[0]
        self.truc=self.ecart//2
        self.truc_2=self.ecart-self.truc
        self.val_max=self.df3.at[0,self.column_1]+(1/(self.df3.shape[0]*100))
        self.val_min=self.df3.at[self.df3.shape[0]-1,self.column_1]-(1/(self.df3.shape[0]*100))
        x_1=np.linspace(self.val_min,0,self.truc_2)
        self.point_min=pd.DataFrame({self.column_1:np.linspace(self.val_min,self.val_min_ppm,self.truc_2),self.column_2:[self.p[0]+(self.p[1]*x_1[i])+(self.p[2]*x_1[i]**2)+complex(np.random.normal(self.m,self.e),np.random.normal(self.m_i,self.e_i)) for i in range(self.truc_2)]})
        x_2=np.linspace(10,self.val_max,self.truc)
        self.point_max=pd.DataFrame({self.column_1:np.linspace(self.val_max_ppm,self.val_max,self.truc),self.column_2:[self.p_2[0]+(self.p_2[1]*x_2[i])+(self.p_2[2]*x_2[i]**2)+complex(np.random.normal(self.m,self.e),np.random.normal(self.m_i,self.e_i)) for i in range(self.truc)]})
        self.df3=pd.concat([self.point_max,self.df3])
        self.df3=pd.concat([self.df3,self.point_min])
        self.df3=self.df3.reset_index(drop=True)
        self.df3_proc=self.df3_proc.reset_index(drop=True)
        self.ecart=self.nb_max-self.df3_proc.shape[0]
        self.truc=self.ecart//2
        self.truc_2=self.ecart-self.truc
        self.val_max=self.df3_proc.at[0,self.column_1]+(1/(self.df3_proc.shape[0]*100))
        self.val_min=self.df3_proc.at[self.df3_proc.shape[0]-1,self.column_1]-(1/(self.df3_proc.shape[0]*100))
        self.point_min=pd.DataFrame({self.column_1:np.linspace(self.val_min,self.val_min_ppm,self.truc_2),self.column_2:[np.random.normal(self.m_proc,self.e_proc) for i in range(self.truc_2)]})
        self.point_max=pd.DataFrame({self.column_1:np.linspace(self.val_max_ppm,self.val_max,self.truc),self.column_2:[np.random.normal(self.m_proc,self.e_proc) for i in range(self.truc)]})
        self.df3_proc=pd.concat([self.point_max,self.df3_proc])
        self.df3_proc=pd.concat([self.df3_proc,self.point_min])
        self.df3_proc=self.df3_proc.reset_index(drop=True)
        self.pdata=np.array(self.df3_proc[self.column_2])
        if self.verbose == True:
            print(f"\033[38;2;128;0;255m\033[1m Complétion des spectres : \033[0m")
            print(f"\tOK\t\tOK")
        if self.disp_df == True:
            df1_styler = self.df3.style.set_table_attributes("style='display:inline'").set_caption('Nmrglue spectrum').set_properties(color="white", align="center",**{"border": "1px solid white"}).set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'blue')]},{"selector": "th:not(.index_name)","props": "background-color: green; color: white; text-align: center"}])
            df2_styler = self.df3_proc.style.set_table_attributes("style='display:inline'").set_caption('TopSpin spectrum').set_properties(color="white", align="center",**{"border": "1px solid white"}).set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'blue')]},{"selector": "th:not(.index_name)","props": "background-color: green; color: white; text-align: center"}])
            display_html("<div style='height:400px;overflow:auto;width:fit-content'>"+df1_styler._repr_html_()+df2_styler._repr_html_()+"</div>", raw=True)

    def fun(self,ph0,ph1):
        dic_t,data_t = ng.process.pipe_proc.ps(self.dic,self.data,p0=ph0,p1=ph1)
        dic_t,data_t = ng.process.pipe_proc.di(dic_t,data_t)
        dic_t,data_t = ng.process.pipe_proc.med(dic_t,data_t, nw=24, sf=24, sigma=5)
        somme=0
        # test=pd.DataFrame({'X':data_p['X'],'INT':data_t})
        # test=test.loc[(test['X']<4.0)&(test['X']>3.75)]
        # test=test['INT']
        for d in data_t:
            somme+=min(0,d)
        fun=abs(somme)
        #print(fun)
        # print(ph0,ph1)
        return fun

    def fun_2(self,ph1,ph0):
        dic_t,data_t = ng.process.pipe_proc.ps(self.dic,self.data,p0=ph0,p1=ph1)
        dic_t,data_t = ng.process.pipe_proc.di(dic_t,data_t)
        dic_t,data_t = ng.process.pipe_proc.med(dic_t,data_t, nw=24, sf=24, sigma=5)
        somme=0
        # test=pd.DataFrame({'X':data_p['X'],'INT':data_t})
        # test=test.loc[(test['X']<3.0)&(test['X']>2.5)]
        # test=test['INT']
        for d in data_t:
            somme+=min(0,d)
        fun=abs(somme)
        #print(fun)
        # print(ph0,ph1)
        return fun


    def opt_phase(self):
        self.data=np.array(self.df3[self.column_2])
        self.ps_p0=90.0#self.dic_save['procs']['PHC0']
        self.ps_p1=0.0#self.dic_save['procs']['PHC1']
        self.result=scipy.optimize.minimize(self.fun, x0=self.ps_p0,method='Powell',args=self.ps_p1)
        self.result_1=scipy.optimize.minimize(self.fun_2, x0=self.ps_p1,method='Powell',args=self.result.x)
        self.result=scipy.optimize.minimize(self.fun, x0=self.result.x,method='Powell',args=self.result_1.x)
        self.result_1=scipy.optimize.minimize(self.fun_2, x0=self.result_1.x,method='Powell',args=self.result.x)
        if self.verbose == True:
            print(f"\033[38;2;0;0;255m\033[1m Optimisation des angles de correction de phase : \033[0m")
            print(f"\t{self.result.message}")
            print(f"\t{self.result_1.message}")
            print(f"\t{self.result.message}")
            print(f"\t{self.result_1.message}")
            print(f"\t\033[1;34mph0\t\tph1\033[0m")
            print(f"\t{self.result.x}\t{self.result_1.x}")

    def phase(self):
        '''Correction de phase avec p0 et p1 en degrès'''
        self.dic,self.data = ng.process.pipe_proc.ps(self.dic,self.data,p0=self.result.x,p1=self.result_1.x)
        if self.verbose == True:
            print(f"\033[38;2;0;255;255m\033[1m Correction de phase : \033[0m")
            print(f"\tOK")

    def delcomplex(self):
        '''Suppression des parties complexe'''
        self.dic,self.data = ng.process.pipe_proc.di(self.dic,self.data)
        self.pdic,self.pdata = ng.process.pipe_proc.di(self.pdic,self.pdata)
        if self.verbose == True:
            print(f"\033[38;2;255;0;64m\033[1m Suppression des parties complexes : \033[0m")
            print(f"\tOK\t\tOK")

    def baseline(self):
        '''Correction médiane de la ligne de base avec nw correspondant à la taille de la fenêtre et sf au lisage'''
        self.dic,self.data = ng.process.pipe_proc.med(self.dic,self.data, nw=self.med_nw, sf=self.med_sf, sigma=self.med_sigma)
        if self.verbose == True:
            print(f"\033[38;2;255;0;0m\033[1m Correction de la ligne de base : \033[0m")
            print(f"\tOK")

    def module(self):
        '''Calcule du module'''
        self.dic,self.data = ng.pipe_proc.mc(self.dic,self.data)

    def make_df_2(self):
        '''Construction de tableau'''
        self.df2=pd.DataFrame(self.data,columns=[self.column_2])
        self.df3[self.column_2]=self.df2[self.column_2]
        #test=self.df3.loc[(self.df3[self.column_1]<3.5)&(self.df3[self.column_1]>3.0)]
        #test=test[self.column_2]
        self.df2_proc=pd.DataFrame(self.pdata,columns=[self.column_2])
        self.df3_proc[self.column_2]=self.df2_proc[self.column_2]
        if self.verbose == True:
            print(f"\033[38;2;0;128;255m\033[1m Reconstruction des tableaux : \033[0m")
            print(f"\tOK\t\tOK")
        if self.disp_df == True:
            df1_styler = self.df3.style.set_table_attributes("style='display:inline'").set_caption('Nmrglue spectrum').set_properties(color="white", align="center",**{"border": "1px solid white"}).set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'blue')]},{"selector": "th:not(.index_name)","props": "background-color: green; color: white; text-align: center"}])
            df2_styler = self.df3_proc.style.set_table_attributes("style='display:inline'").set_caption('TopSpin spectrum').set_properties(color="white", align="center",**{"border": "1px solid white"}).set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'blue')]},{"selector": "th:not(.index_name)","props": "background-color: green; color: white; text-align: center"}])
            display_html("<div style='height:400px;overflow:auto;width:fit-content'>"+df1_styler._repr_html_()+df2_styler._repr_html_()+"</div>", raw=True)

    def extract_info_data(self):
        self.test_info=self.read_csv('../data_info.csv')
        col=[]
        ra=int((self.test_info.shape[1])/4)
        for c in range(ra):
            col.append('supprimer/conserver_'+str(c+1))
        self.cons=self.test_info[col].isna()
        if self.verbose == True:
            print(f"\033[38;2;128;128;255m\033[1m Extraction des positions des différents pics : \033[0m")
            print(f"\tOK\t\tOK")

    def elim_imp(self):
        b=np.array(self.cons.columns)
        pic=0
        for j in b:
            #print(self.cons.at[i,j])
            if self.cons.at[self.file_path,j]!=True:
                pic+=1
        #print(pic)
        #print("zone à éliminer :",pic+1)
        self.val=[self.val_max_ppm]
        for k in range(pic):
            c='min_'+str(k+1)
            d='max_'+str(k+1)
            self.val.append(self.test_info.at[self.file_path,d])
            self.val.append(self.test_info.at[self.file_path,c])
        self.val.extend([self.val_min_ppm])
        #print(val)
        for l in range(0,(pic+1)*2,2):
            #print(l)
            indexs=self.df3.loc[(self.df3[self.column_1]<self.val[l])&(self.df3[self.column_1]>self.val[l+1])]
            self.df3.loc[indexs.index,[self.column_2]]=[[np.random.normal(self.m,(self.e))] for i in range(indexs.shape[0])]
            indexs=self.df3_proc.loc[(self.df3_proc[self.column_1]<self.val[l])&(self.df3_proc[self.column_1]>self.val[l+1])]
            self.df3_proc.loc[indexs.index,[self.column_2]]=[[np.random.normal(self.m_proc,self.e_proc)] for i in range(indexs.shape[0])]
            if self.verbose == True:
                print(f"\tZone à élimininées : {l}\t\033[31m{self.val[l+1]} - {self.val[l]}\033[0m")
            #display(self.df3)        
        for m in range(1,(pic)*2,2):
            test=self.df3_proc.loc[(self.df3_proc[self.column_1]<(self.val[m+1]+0.01))&(self.df3_proc[self.column_1]>self.val[m+1])]
            test=test[self.column_2]
            # display(test)
            moy=np.mean(np.array(test).real)
            indexs=self.df3_proc.loc[(self.df3_proc[self.column_1]>self.val[m+1])&(self.df3_proc[self.column_1]<self.val[m])]
            self.df3_proc.loc[indexs.index,[self.column_2]]=self.df3_proc.loc[indexs.index,[self.column_2]]-moy
            if self.verbose == True:
                print(f"\tZone à conservées : {m}\t\033[32m{self.val[m+1]} - {self.val[m]}\033[0m")
        if self.verbose == True:
            print(f"\033[38;2;0;0;255m\033[1m Elimination des impuretés : \033[0m")
            print(f"\tOK\t\tOK")
        if self.disp_df == True:
            df1_styler = self.df3.style.set_table_attributes("style='display:inline'").set_caption('Nmrglue spectrum').set_properties(color="white", align="center",**{"border": "1px solid white"}).set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'blue')]},{"selector": "th:not(.index_name)","props": "background-color: green; color: white; text-align: center"}])
            df2_styler = self.df3_proc.style.set_table_attributes("style='display:inline'").set_caption('TopSpin spectrum').set_properties(color="white", align="center",**{"border": "1px solid white"}).set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'blue')]},{"selector": "th:not(.index_name)","props": "background-color: green; color: white; text-align: center"}])
            display_html("<div style='height:400px;overflow:auto;width:fit-content'>"+df1_styler._repr_html_()+df2_styler._repr_html_()+"</div>", raw=True)

    def interpolate(self):
        new_x=np.linspace(10,0,21800)
        interpolate=scipy.interpolate.interp1d(np.array(self.df3[self.column_1]),np.array(self.df3[self.column_2]))
        interpolate_proc=scipy.interpolate.interp1d(np.array(self.df3_proc[self.column_1]),np.array(self.df3_proc[self.column_2]))
        new_y=interpolate(new_x)
        new_y_proc=interpolate_proc(new_x)
        self.df3=pd.DataFrame({self.column_1:new_x,self.column_2:new_y})
        self.df3_proc=pd.DataFrame({self.column_1:new_x,self.column_2:new_y_proc})
        if self.verbose == True:
            print(f"\033[38;2;128;128;255m\033[1m interpolation : \033[0m")
            print(f"\tOK\t\tOK")

    def normalise(self):
        '''Normalisation des données entre 0 et 1'''
        self.df3[self.column_2] = self.df3[self.column_2]/self.df3[self.column_2].max()
        self.df3_proc[self.column_2] = self.df3_proc[self.column_2]/(self.df3_proc[self.column_2].max())
        if self.verbose == True:
            print(f"\033[38;2;0;255;255m\033[1m Normalisation : \033[0m")
            print(f"\tOK\t\tOK")

    def make_csv(self):
        '''Sauvegardes des données traitées sous forme de fichierss csv'''
        path_save2=self.path_save+'spectres_ng/'+self.file_path+'.csv'
        self.df3.to_csv(path_save2,';')
        path_save2=self.path_save+'spectres_proc/'+self.file_path+'.csv'
        self.df3_proc.to_csv(path_save2,';')
        if self.verbose == True:
            print(f"\033[38;2;0;255;0m\033[1m Sauvegarde des tableaux au formats CSV : \033[0m")
            print(f"\tOK\t\tOK")

    def read_csv(self,path):
        '''Lecture de fichier au format csv'''
        self.dfr=pd.read_csv(path, sep=";",index_col=0)
        return self.dfr

    def graph(self):
        '''Tracé de spectres à partir de fichier de données au format csv'''
        b=np.array(self.val)
        display(Markdown(f'<h4 style="background:black;color:cyan"><b> {self.file_path} </b></h4>'))
        fig, graph = plt.subplots(figsize=(1500/72,8),layout="constrained")
        graph.plot(self.df3[self.column_1],self.df3[self.column_2],lw=0.5,color='blue',zorder=1,alpha=1,label='Nmrglue spectrum')
        graph.plot(self.df3_proc[self.column_1],self.df3_proc[self.column_2],lw=0.5,color='lightgreen',zorder=2,alpha=1,label='TopSpin spectrum')
        graph.set_title(self.file_path+" NMR spectra")
        graph.set_xlabel("Chemical Shift (ppm)")
        graph.set_ylabel("Signal normalized")
        graph.set_ylim(-0.05,1e0)
        graph.axhline(0,-1e4,1e8,linestyle=':',color='black',alpha=1,zorder=-10)
        for i in b:
            graph.axvline(i,-1e2,1e2,linestyle=':',color='grey',alpha=0.5,zorder=-10)
        graph.axvline(5.5,-1e2,1e2,linestyle=':',color='red',alpha=0.5,zorder=-10)
        graph.axvline(4.5,-1e2,1e2,linestyle=':',color='red',alpha=0.5,zorder=-10)
        graph.legend()
        graph.grid()
        graph.invert_xaxis()
        plt.show()
        if self.verbose == True:
            print(f"\tOK")

    def init_data_base(self):
        '''Création des tableaux de la base de données'''
        self.df4=pd.DataFrame(index=np.linspace(0,self.nb_max-1,self.nb_max).astype(int))
        self.df5=pd.DataFrame(index=np.linspace(0,self.nb_max-1,self.nb_max).astype(int))
        self.df6=[]
        self.df4_proc=pd.DataFrame(index=np.linspace(0,self.nb_max-1,self.nb_max).astype(int))
        self.df5_proc=pd.DataFrame(index=np.linspace(0,self.nb_max-1,self.nb_max).astype(int))
        self.df6_proc=[]

    def make_data_base(self):
        '''Création des tableaux de données de la base de données'''
        rsuffix_int='_'+self.file_path
        rsuffix_x='_'+self.file_path
        path_save3=self.path_save+'spectres_ng/'+'data_intensite.csv'
        path_save4=self.path_save+'spectres_ng/'+'data_x.csv'
        #path_save5=self.path_save+'data_proba.csv'
        self.df4=self.df4.join(self.df3[self.column_2], rsuffix=rsuffix_int)
        self.df5=self.df5.join(self.df3[self.column_1], rsuffix=rsuffix_x)
        self.df6.append({'composes':self.file_path})
        self.df4.to_csv(path_save3,';')
        self.df5.to_csv(path_save4,';')
        #self.df6.to_csv(path_save5,';')
        rsuffix_int='_'+self.file_path
        rsuffix_x='_'+self.file_path
        path_save3=self.path_save+'spectres_proc/'+'data_intensite.csv'
        path_save4=self.path_save+'spectres_proc/'+'data_x.csv'
        #path_save5=self.path_save+'data_proba.csv'
        self.df4_proc=self.df4_proc.join(self.df3_proc[self.column_2], rsuffix=rsuffix_int)
        self.df5_proc=self.df5_proc.join(self.df3_proc[self.column_1], rsuffix=rsuffix_x)
        self.df6_proc.append({'composes':self.file_path})
        self.df4_proc.to_csv(path_save3,';')
        self.df5_proc.to_csv(path_save4,';')
        #self.df6_proc.to_csv(path_save5,';')
        if self.verbose == True:
            print(f"\033[38;2;0;0;255m\033[1m Création et sauvegardes des tableaux de données : \033[0m")
            print(f"\tOK\t\tOK")
        if self.disp_df == True:
            df1_styler = self.df4.style.set_table_attributes("style='display:inline'").set_caption('Nmrglue spectrum').set_properties(color="white", align="center",**{"border": "1px solid white"}).set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'blue')]},{"selector": "th:not(.index_name)","props": "background-color: green; color: white; text-align: center"}])
            df2_styler = self.df4_proc.style.set_table_attributes("style='display:inline'").set_caption('TopSpin spectrum').set_properties(color="white", align="center",**{"border": "1px solid white"}).set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'blue')]},{"selector": "th:not(.index_name)","props": "background-color: green; color: white; text-align: center"}])
            df3_styler = self.df5.style.set_table_attributes("style='display:inline'").set_caption('Nmrglue spectrum').set_properties(color="white", align="center",**{"border": "1px solid white"}).set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'blue')]},{"selector": "th:not(.index_name)","props": "background-color: green; color: white; text-align: center"}])
            df4_styler = self.df5_proc.style.set_table_attributes("style='display:inline'").set_caption('TopSpin spectrum').set_properties(color="white", align="center",**{"border": "1px solid white"}).set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'blue')]},{"selector": "th:not(.index_name)","props": "background-color: green; color: white; text-align: center"}])
            display_html("<div style='height:400px;overflow:auto;width:fit-content'>"+df1_styler._repr_html_()+df2_styler._repr_html_()+df3_styler._repr_html_()+df4_styler._repr_html_()+"</div>", raw=True)

    def make_data_proba(self):
        '''Création du tableau d'identification des composés sous forme de catégories'''
        path_save5=self.path_save+'data_proba.csv'
        self.df6=pd.DataFrame(self.df6)
        y_array = self.df6.index.copy()
        y_array = y_array.to_numpy()
        yohe = to_categorical(y_array)
        del y_array
        self.df6=self.df6.join(pd.DataFrame(yohe,columns=[self.df6['composes']],index=self.df6.index))
        self.df6.to_csv(path_save5,';')
        if self.verbose == True:
            print(f"\033[38;2;0;0;255m\033[1m Création et sauvegardes du tableaux de classification : \033[0m")
            print(f"\tOK\t\tOK")
        if self.disp_df == True:
            df1_styler = self.df6.style.set_table_attributes("style='display:inline'").set_caption('Nmrglue spectrum').set_properties(color="white", align="center",**{"border": "1px solid white"}).set_table_styles([{'selector': 'tr:hover','props': [('background-color', 'blue')]},{"selector": "th:not(.index_name)","props": "background-color: green; color: white; text-align: center"}])
            display_html("<div style='height:400px;overflow:auto;width:fit-content'>"+df1_styler._repr_html_()+"</div>", raw=True)

    def traite(self):
        '''Processuce de traitement global sur l'ensemble de la base de données'''
        self.path=self.path_main
        self.extract_info_data()
        self.init_data_base()
        for self.file_path in os.listdir(self.path):
            print("==================================================================================================================================")
            print(f"\t\t\t\t\t\t\033[1m---- {self.file_path} ----\033[0m")
            print("==================================================================================================================================")
            self.charge_dossier()
            self.read()
            self.convert()
            self.solvant()
            self.filtre()
            self.apodisation()
            self.trans_fourier()
            self.make_df()
            self.est_bruit()
            self.resample()
            self.cut()
            self.complete()
            self.opt_phase()
            self.phase()
            self.delcomplex()
            self.baseline()
            self.make_df_2()
            self.elim_imp()
            self.interpolate()
            self.normalise()
            self.make_csv()
            self.make_data_base()
            if self.trace == True:
                self.graph()
            print("==================================================================================================================================")
            print(f"\t\t\t\t\t\t\033[1m---- Traitement terminé ----\033[0m")
            print("==================================================================================================================================")
        self.make_data_proba()

#-----------------------------------------------------------------------

    
    def traite_one(self,path):
        '''Processuce de traitement global sur un spectre'''
        self.path=path
        #self.charge_dossier()
        self.read()
        self.convert()
        self.solvant()
        self.trans_fourier()
        self.phase()
        self.delcomplex()
        self.baseline()
        self.module()
        self.make_df()
        self.cut()
        self.complete()
        self.normalise()
        self.make_csv()