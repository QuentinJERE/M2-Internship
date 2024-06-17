# M2-Internship
## Automatic identification by deep learning methods of NMR signals in a complex mixture:  Application to metabolomics.

Metabolomics is an extremely important field, involving the characterization of metabolites present in a biological sample. It has numerous applications, notably in the fields of medicine, nutrition, agro-environment and the study of biochemical processes. It is based on various analytical techniques, such as liquid chromatography, mass spectroscopy and nuclear magnetic resonance (NMR). NMR applied to metabolomics enables non-destructive analysis. It can also be used to analyse complex mixtures and requires less sample preparation and separation. The recent development of less expensive low-field NMR spectrometers makes NMR analysis more accessible. However, the identification of compounds in complex mixtures remains very time-consuming. Thus, the use of machine learning methods applied to metabolomics is of great interest for the automatic identification of complex mixtures.
The aim of this project is to use deep learning methods to automatically identify specific compounds in NMR signals obtained from mixtures of several metabolites. We have chosen to develop different artificial neural networks (ANN) and analyse their performance in identifying the different compounds presents in the database.

First, this study involved processing the data to make it usable by a neural network, as well as augmenting the data to form a larger learning database. To do this, the Free Induction Decays (FID) in the database were stripped of solvent. They were then transformed into spectra to which phase and baseline corrections were applied. Finally, the impurities present in the spectra were eliminated from all the spectra in the database. The data were then augmented by shifting the spectra according to the chemical shifts, and random shifts were added to avoid any bias. Next, an ANN was built and optimised using a convolutional network based on the performance obtained. In a second step, another ANN was built with a multi-branch structure based on convolutional networks. A last ANN was also considered based on a structure of recurrent neural networks.
Finally, NMR spectra of binary mixtures in equivalent proportions were generated from the spectra of the compounds in the database, and identification tests were carried out using the previously developed neural networks to compare the effectiveness of these different neural networks in identifying compounds into mixtures. 

### Environnement : <br>

[`Intel® Distribution for Python*`](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html) (Python 3.9.18)
```
pip install tensorflow==2.15
```
```
pip install intel-extension-for-tensorflow
```
```
pip install nmrglue==0.10
pip install ipympl==0.9.4
pip install ipywidgets==8.1.2
pip install scikit-learn-intelex==2024.2.0
pip install scipy==1.10.1
pip install seaborn==0.13.2
```
