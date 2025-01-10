# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 02:43:49 2025

@author: emrek
"""

"""1.AŞAMA VERİ SETİNİ YÜKLEME"""
import pandas as pd # dataframe iç,n analiz yapmamızı sağlayan kütüphane
import numpy as np# vektör ve matematik için
import seaborn as sns # görselleştirme içim 
import matplotlib.pyplot as plt# ''
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.neighbors import KNeighborsClassifier , NeighborhoodComponentsAnalysis , LocalOutlierFactor
from sklearn.decomposition import PCA


# warning ler hata değil sürümden olabiliyor bunları kapatmamız gerekiyor
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv("cancer.csv") #veri setini yükledik
data.drop(['Unnamed: 32','id'], inplace = True , axis =1) # id ve unnamed olan sütünları sildik gereksiz ve fazlalıktı

data =data.rename(columns ={"diagnosis":"target"})

sns.countplot(data["target"]) # plot ile görselleştirip kaç tane olduğunu görüyoruz
#B iyi huylu , M kötü huylu
print(data.target.value_counts())
# string verileri sayıya çevirme
data["target"]= [1 if i.strip()== "M" else 0 for i in data.target] # iyi huylular sıfır , kötü huyluları 1 e çeviriyoruz string ifadeden kurtuluyoruz i.strip diyince boşlukların sorunlar çıkarmasına engel oluyor 
print(len(data))
print(data.head()) # ilk 5 veri

print("data bilgisi")
print("Data shape",data.shape)

data.info()

describe = data.describe()
# verileri incelediğimizde descripe da çok fazla veriler arası fark var bunun ,ıcın standartize ve normalize etmeliyiz biz burada standartize edeceğiz"""
#BÜYÜK SAYILAR KÜÇÜK SAYILARA BASKIN GELEBİLİR
#STANDARTİZE ETMELİYİZ"""
#eksik veri ve aykırı değer bulmalı ve 0 olan değerlere bakılmalı burada missing value olmadığından 0 olan değerleri 0 kabul edeceğiz"""
 

"""#2.AŞAMA  DETAYLI VERİLERİN ANALİZİ"""
#EDA
#KORELASYON MATRİXİ VERİLERİN TÜM ÖZELLİKLERİNİ BİRBİRİYLE İLİŞKİSİNE BAKAR -1 İLE 1 ARASINDA BULUNUR 1 POZİTİF GÜÇLÜ İLİŞKİ , NOT DEFTERİNE BAK DAHA DETAY İÇİN"""


corr_matrix = data.corr()
sns.clustermap(corr_matrix, annot= True, fmt = ".2f")
plt.title("KORELASYON İLİŞKİSİ")
plt.show()

#
threshold =0.75 # korelasyonu 0.75  ten yüksek olanları alıcaz altaki satır o görebi görücek ,bu değeri değiştirerek ilişkileri anlayabiliriz
filtre =np.abs(corr_matrix["target"]) > threshold
corr_features = corr_matrix.columns[filtre].tolist()
sns.clustermap(data[corr_features].corr() , annot= True , fmt=".2f")
plt.title("KORELASYON İLİŞKİSİNİN 0.75 DEN BÜYÜK OLANLARI")

#BOX PLOT =Verinin merkezi eğilimini, yayılımını ve olası aykırı değerlerini (outlier) görselleştirmeye yarar.ÖRNEK ÇEYREKLER VAR Q3-Q1 = ÇEYREKLER ARASI GİBİ DAHA FAZLA BİLGİ NOT DA VAR """


#box plot
data_melted = pd.melt(data , id_vars="target",
                      var_name ="features",
                      value_name= "value")
plt.figure()
sns.boxplot(x= "features" , y = "value", hue= "target" , data=data_melted)
plt.xticks(rotation=90)
plt.show()

#box plotda bazen standartize edilmemiş verilerde çok yer kapladığından gözükmeye anlam veremeyebiliyoruz bu yüzden pair plot kullanıcaz analizde eda da çok kullanışlıdır"""


#PAİR PLOT AŞAMASI
sns.pairplot(data[corr_features], diag_kind ="kde", markers="+",hue ="target")
plt.show()

"""#3.AŞAMA OUTLİER DETECTİON : AYKIRI DEĞER ANALİZİ"""
#LOCAL OUTLİER DETECTİON YÖNTEMİNİ YAPACAĞIZ NOT DA ANLATIŞI VAR
y= data.target
x= data.drop(["target"], axis=1)# targeti çıkardıktan sonra geriye kalanlar 
columns = x.columns.tolist() # target dışındakileri columns diye yapıyoruz

#local outlier factor ile aykırı değerleri bulma
clf = LocalOutlierFactor(n_neighbors=20) # komşu sayısı veriyoruz , sormamız gereken diğer soru kendimize = ideal komşu sayısı ?
y_pred =clf.fit_predict(x)# lof algoritmasının nesnenesi clf aykırı değer için öğren diyoruz , tahmin  ettiriyoruz
X_score = clf.negative_outlier_factor_ # x score da negatifli yaptık sayıyı pozitif gibi düşün - sini es geç 1 den büyükse aykırı değer olabilir fakat , veri sayısı az ise bunları atmayabiliriz , bunun oran orantısına bakılmalı ve doğru karar verilmeli

outlier_score = pd.DataFrame()
outlier_score["score"]= X_score

threshold = -2.5 # aykırı değerin sınırı
filtre = outlier_score["score"]<threshold #-2.5 den küçükleri yani -3 gibi bunları çıkarmak için aşağıda yapacağımız işlem
outlier_index = outlier_score[filtre].index.tolist()# index çeviriyoruz dataframe değil o yüzden .index dicez

plt.figure()

plt.scatter(x.iloc[outlier_index,0] , x.iloc[outlier_index,1] , color ="blue" , s =50 ,label ="Outliers")

plt.scatter(x.iloc[:,0] , x.iloc[:,1] , color="k" , s = 3 , label = "Data points")
radius = (X_score.max() - X_score)/(X_score.max() - X_score.min())# yarı çap

outlier_score["radius"]=radius # görselleştirmek  için yapıyoruz bunu 
plt.scatter(x.iloc[:,0] , x.iloc[:,1] , s = 1000*radius , edgecolors="r" , facecolors ="none" , label="Outlier Scores") # kırmızı çaplar ne kadar geniş ise outlier yani aykırı değer olması o kadar fazla , birçok boyutlu olduğundan burdan anlıyoruz 
plt.legend()
plt.show()

x = x.drop(outlier_index)
y = y.drop(outlier_index).values



""" train test split"""
# y = data frame de ki kanser mi değil mi olanları temssil ediyor 
# x = dataframe deki target dışındakileri temsil ediyor 
#X_test = 0.3 lük kısım target yok , 
# veri setini ayırıyoruz , testlerle doğruluk ölçülüyor , trainde eğitim verisi  eğitim verisinden öğrenme sonucu testlerde doğruluğu bakılır 
test_size = 0.3
X_train ,X_test , Y_train , Y_test = train_test_split(x, y ,  test_size = test_size ,random_state =42) # random state her seferimde aynı şekilde karışmasını sağlar yani bir karışma yapıyor ve birdahakine aynı şekil karışıyor


"""STANDARTİZASYON AŞAMASI"""
#veriler arasında çok farklılık var ise standardizasyon yaparız , veri düzgün dağılıma sahip değil ise gaus dağılıma sahip değilse yani normalizasyon yapılır
# normalizasyon 0 ile 1 arasına ölçeklendirir , standartiazsyon ise Veriyi, ortalaması 0 ve standart sapması 1 olacak şekilde ölçekler.

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)# bunda fit demememizin sebebi bi üst satırda zaten öğrettik scalleri burda da aynı yaptığı işlemi x_test de de yapmasını isteyeceğiz

X_train_df = pd.DataFrame(X_train , columns = columns) # x yani target dışındakileri columnsa eşitlemiştik
X_train_df_describe = X_train_df.describe() # standartiazsyona bakıyoruz oldu mu

"""------KNN-------"""

knn =KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train , Y_train)#"eğitim verisinden öğren"
y_pred = knn.predict(X_test)#eğitim verisinden öğrendiğini test veri setinde tahmin et -- BURADA XTEST DEMEMİZİN SEBEBİ ÖĞRENDİĞİNİ GİDİYOR X_TESTDEKİ YANİ TARGET DIŞINDAKİ VERİLERDE TAHMİN EDİYOR

cm = confusion_matrix(Y_test , y_pred)
acc = accuracy_score(Y_test , y_pred)
score = knn.score(X_test , Y_test)

print("score : ",score)
print("confisun matrix : " ,cm)
print("knn accuracy :",acc)



"""---EN İYİ KNN PARAMETRESİNİ BULMA-----"""

def KNN_bEST_PARAMS(X_train , X_test , Y_train , Y_test):
    k_range = list(range(1,31))
    weight_options = ["Uniform", "distance"]
    print()
    param_grid = dict(n_neighbors = k_range , weights = weight_options)
    
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn , param_grid , cv=10 , scoring="accuracy")
    grid.fit(X_train,Y_train)
    
    print("Best Training score:{} with parameters:{}".format(grid.best_score_ , grid.best_params_))
    print()
    
    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(X_train ,Y_train)
    
    y_pred_test = knn.predict(X_test)
    y_pred_train = knn.predict(X_train)#öğrenme setinde öğrendiklerimizi test de deniyoruz
    
    cm_test = confusion_matrix(Y_test , y_pred_test)#confisun matrix ile daha önceki y ile şu an en iyi parametreyi karşılaştırmak için yapıyprız
    cm_train = confusion_matrix(Y_train ,y_pred_train)
    
    # accuracy
    acc_test = accuracy_score(Y_test , y_pred_test)
    acc_train =accuracy_score(Y_train ,y_pred_train)
    
    print("Test score : {}, Train score: {}".format(acc_test,acc_train))
    print()
    print("Confusion matrix test : ",cm_test)
    print("Confusion matrix train : ",cm_train)
    
    return grid

grid=KNN_bEST_PARAMS(X_train, X_test, Y_train, Y_test)
    


"""------PCA BOYUT İNDİRGEME-------"""
"""GÖRSELLEŞTİRME VE BOYUT İNDİRGEME SAĞLAR"""
"""TEKRARDAN SCALLER ETMEMİZ LAZIM PCA TÜM HEPSİNİ KULLANIYOR"""    

scaller = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca =PCA(n_components=2)# 30 adet özellik var 2 ye düşürmek istediğimizi söyledik
pca.fit(x_scaled)
pca.transform(x_scaled)
X_reduced_pca = pca.transform(x_scaled)
pca_data = pd.DataFrame(X_reduced_pca , columns=["p1","p2"])
pca_data["target"]=y

sns.scatterplot(x="p1" , y="p2" , hue="target" , data=pca_data)
plt.title("PCA P1 VS P2")

"""burda pca üzerine gittiğimiz için böyle yaptık"""
X_train_pca ,X_test_pca , Y_train_pca , Y_test_pca = train_test_split(X_reduced_pca, y ,  test_size = test_size ,random_state =42)
print("grid pca")
grid_pca = KNN_bEST_PARAMS(X_train_pca ,X_test_pca , Y_train_pca , Y_test_pca)
"""daha net görselleştirme"""
# visualize 
cmap_light = ListedColormap(['orange',  'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

h = .05 # step size in the mesh
X = X_reduced_pca
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = grid_pca.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class classification (k = %i, weights = '%s')"
          % (len(np.unique(y)),grid_pca.best_estimator_.n_neighbors, grid_pca.best_estimator_.weights))


"""NCA İLE YAPMAK , SADECE İKİ KISMA AYIRMAZ BU TARZ PROBLEMLERDE DAHA İYİ SINIFLANDIRMA SAĞLAR"""


nca = NeighborhoodComponentsAnalysis(n_components = 2, random_state = 42)
nca.fit(x_scaled, y)
X_reduced_nca = nca.transform(x_scaled)
nca_data = pd.DataFrame(X_reduced_nca, columns = ["p1","p2"])
nca_data["target"] = y
sns.scatterplot(x = "p1",  y = "p2", hue = "target", data = nca_data)
plt.title("NCA: p1 vs p2")

X_train_nca, X_test_nca, Y_train_nca, Y_test_nca = train_test_split(X_reduced_nca, y, test_size = test_size, random_state = 42)

grid_nca = KNN_bEST_PARAMS(X_train_nca, X_test_nca, Y_train_nca, Y_test_nca)

# visualize 
cmap_light = ListedColormap(['orange',  'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'darkblue'])

h = .2 # step size in the mesh
X = X_reduced_nca
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = grid_nca.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("%i-Class classification (k = %i, weights = '%s')"
          % (len(np.unique(y)),grid_nca.best_estimator_.n_neighbors, grid_nca.best_estimator_.weights))

# %% find wrong decision
knn = KNeighborsClassifier(**grid_nca.best_params_)
knn.fit(X_train_nca,Y_train_nca)
y_pred_nca = knn.predict(X_test_nca)
acc_test_nca = accuracy_score(y_pred_nca,Y_test_nca)
knn.score(X_test_nca,Y_test_nca)

test_data = pd.DataFrame()
test_data["X_test_nca_p1"] = X_test_nca[:,0]
test_data["X_test_nca_p2"] = X_test_nca[:,1]
test_data["y_pred_nca"] = y_pred_nca
test_data["Y_test_nca"] = Y_test_nca

plt.figure()
sns.scatterplot(x="X_test_nca_p1", y="X_test_nca_p2", hue="Y_test_nca",data=test_data)

diff = np.where(y_pred_nca!=Y_test_nca)[0]
plt.scatter(test_data.iloc[diff,0],test_data.iloc[diff,1],label = "Wrong Classified",alpha = 0.2,color = "red",s = 1000)
"""MODELİN HANGİ NOKTALARDA YANLIŞ SINIFLANDIRDIĞINI BULMAK"""
knn = KNeighborsClassifier(**grid_nca.best_params_)
knn.fit(X_train_nca,Y_train_nca)
y_pred_nca = knn.predict(X_test_nca)
acc_test_nca = accuracy_score(y_pred_nca,Y_test_nca)
knn.score(X_test_nca,Y_test_nca)

test_data = pd.DataFrame()
test_data["X_test_nca_p1"] = X_test_nca[:,0]
test_data["X_test_nca_p2"] = X_test_nca[:,1]
test_data["y_pred_nca"] = y_pred_nca
test_data["Y_test_nca"] = Y_test_nca

plt.figure()
sns.scatterplot(x="X_test_nca_p1", y="X_test_nca_p2", hue="Y_test_nca",data=test_data)

diff = np.where(y_pred_nca!=Y_test_nca)[0]
plt.scatter(test_data.iloc[diff,0],test_data.iloc[diff,1],label = "Wrong Classified",alpha = 0.2,color = "red",s = 1000)



"""TEST SONUÇLARININ DEĞERLENDİRİLMESİ"""




