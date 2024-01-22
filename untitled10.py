# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 20:28:00 2023

@author: ecems
"""

import pandas as pd #veri analizi için kullanılır.
import numpy as np #Sayısal hesaplamalar yapmak için
import matplotlib as plt #veri görselleştirmesi için kullanılır
from sklearn.linear_model import LinearRegression, LogisticRegression #doğrusal regresyon modeli oluşturmak ve eğitmek için kullanılır
from sklearn.tree import DecisionTreeClassifier #karar ağaçları kullanmak için
from sklearn.model_selection import train_test_split, GridSearchCV 
#Bu fonksiyon, veri setini eğitim ve test kümelerine ayırmak için kullanılır.
#Bu sınıf, bir model için en iyi hiperparametreleri bulmak için kapsamlı bir parametre arama yapar
from sklearn.neighbors import KNeighborsClassifier #k-komsu algoritması 
from sklearn.preprocessing import StandardScaler, MinMaxScaler #verileri ölçeklendirmek için
from sklearn.ensemble import RandomForestClassifier #rastgele orman algoritması için
from sklearn.svm import SVC #destek vektör algoritmaları
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, confusion_matrix
import seaborn as sns #veri görselleştirmek için
import scipy as sp #bilimsel hesaplamalar için bir dizi algoritma ve araç içeren geniş bir kütüphanedir
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'C:\Users\ecems\.spyder-py3\Sarap kalite analiz 2\wine.csv')
print(df.head())

df.info()

df.columns = ["Sabit Asitlik","Uçucu Asitlik","Sitrik Asit","Kalan Şeker","Klorür","Serbest Kükürt Dioksit","Toplam Kükürt Dioksit","Yoğunluk","pH","Sülfat","Alkol","Kalite"]
print(df.describe().T)

# Önce eksik değerler kontrol edilir
df.isnull().sum()

# Eksik değerleri doldurma (gerekliyse)
df['Sitrik Asit'].fillna(df['Sitrik Asit'].mean(), inplace=True)

# Sonra tüm eksik değerleri içeren satırları kaldırma
df.dropna(inplace=True)

sns.countplot(x = df.Kalite) #adet grafiği oluşturduk
plt.grid(axis = "y", ls = ":", color = "black") #görünümü özelleştirmek için grid komutu kullanırız.
plt.title("kalite değişkeni frekans grafiği")
plt.show()
print(df.Kalite.value_counts())


plt.clf() #yeni çizim yapmak için 
sns.scatterplot(data=df, x='Yoğunluk', y='Alkol', hue='Kalite', palette="bright") #scatter plot grafiği oluşturduk
plt.show()

df['Kalite Durumu'] = df['Kalite'].astype('category').cat.codes #burada bad ve good değerleri kategorik değerlere dönüştürülür [1,0]
print(df.head())

df1 = df.drop('Kalite',axis=1) #drop kalite adındaki sütunu cıkarır ve kalite dısındakileri alır, axis=1 ise sütun üzerinde işlem yapılacağını belirtir.
df1.info()

y = df1['Kalite Durumu']
y.head()

X = df1.drop('Kalite Durumu',axis=1)
X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
print(X.shape)
print(X_train.shape) #%80 eğitim
print(X_test.shape) #%20 test gibi bir değer atadı.

kfold = KFold(n_splits=5, shuffle=True, random_state=0) 
total= np.zeros((2, 2))

# K-fold çapraz doğrulama ile modelin değerlendirilmesi
sonuc=[]
conf_mat = []
for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    '''
    model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)'''
    

    knn_model = KNeighborsClassifier(n_neighbors=4)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    
    '''
    rfc_model = RandomForestClassifier(random_state = 0,max_features = 1,min_samples_leaf = 1,max_depth = 16)
    rfc_model.fit(X_train, y_train)
    y_pred = rfc_model.predict(X_test)'''
    
    karmasiklik_matrisi = confusion_matrix(y_test, y_pred)
    conf_mat.append(karmasiklik_matrisi)
    total += karmasiklik_matrisi

print("Genel Karmaşıklık Matrisi:", total)
sonuc.append(total)

total = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(10,10))
sns.heatmap(total/np.sum(total), annot=True, fmt='.2%',annot_kws={'size': 8}, cmap='Reds')
plt.show()    

    
