import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn import svm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import shapiro
from sklearn.linear_model import LogisticRegression
from scipy import stats

#LETTURA
data=pd.read_csv('data.csv')
data=data.sample(frac=1).reset_index(drop=True)#mescola le righe

#PULIZIA DATASET: pulizia da duplicati, colonne inutili e valori NaN
data = data.drop_duplicates(subset=['song_title','artist'], keep=False)
data = data.drop(columns=['song_title'])
data = data.drop(columns=['artist'])
data = data.drop(columns=['Unnamed: 0'])
print(data.info())
print(f"{np.sum(data.isnull())}")


#i soliamo le variabili in input con la variabile che vogliamo predirre 
#successivamente e stampiamo la nostra matrice di correlazione
df=data[["acousticness","danceability", "duration_ms","energy","instrumentalness","key",
         "liveness","loudness","mode","speechiness","tempo",
         "time_signature","valence"
         ]]
output=data[["target"]]

#MATRICE DI CORRELAZIONE
matrice_correlazione=df .corr()
print(matrice_correlazione)


#FASE DI EDA:
#tramite degli istogrammi e degli scatter osserviamo la distribuzione e il comportamento
#dei vari attributi 

sns.histplot(data["danceability"],bins=40)
plt.xlabel("danceability") 
plt.ylabel("frequenza")
plt.show()



sns.histplot(data["energy"],bins=40)
plt.xlabel("energy") 
plt.ylabel("frequenza")
plt.show()


sns.histplot(data["key"],bins=12)
plt.xlabel("key") 
plt.ylabel("frequenza")
plt.show()

sns.histplot(data["loudness"],bins=40)
plt.xlabel("loudness") 
plt.ylabel("frequenza")
plt.show()

sns.scatterplot(data=data, x=np.arange(0,len(data[["time_signature"]]) ),y="time_signature")
plt.xlabel("frequenza")
plt.ylabel("time_signature")
plt.show()
#c' è un outlier con time_signature=1
data=data[data["time_signature"]!=1]
sns.scatterplot(data=data, x=np.arange(0,len(data[["time_signature"]]) ),y="time_signature")
plt.xlabel("frequenza")
plt.ylabel("time_signature")
plt.show()


sns.histplot(data=data, x="mode", bins=2, hue="mode", palette=["purple", "yellow"])
print(np.sum(data["mode"]==1))
print(np.sum(data["mode"]==0))
plt.xlabel("mode") 
plt.ylabel("frequenza")
plt.legend(labels=["major", "minor"])
plt.show()

sns.histplot(data=data, x="target", bins=2, hue="target", palette=["blue", "red"])
print(np.sum(data["target"]==1))
print(np.sum(data["target"]==0))
plt.xlabel("liked") 
plt.ylabel("frequenza")
plt.legend(labels=["liked", "not_liked"])
plt.show()

sns.histplot(data["speechiness"],bins=40)
plt.xlabel("speechiness") 
plt.ylabel("frequenza")
plt.show()
#ci sono un paio di outliers da eliminare
data=data[data["speechiness"]<0.5]
sns.histplot(data["speechiness"],bins=40)
plt.xlabel("speechiness") 
plt.ylabel("frequenza")
plt.show()

sns.histplot(data["acousticness"],bins=40)
plt.xlabel("acousticness") 
plt.ylabel("frequenza")
plt.show()

sns.histplot(data["instrumentalness"],bins=40)
plt.xlabel("instrumentalness") 
plt.ylabel("frequenza")
plt.show()


sns.histplot(data["liveness"],bins=40)
plt.xlabel("liveness") 
plt.ylabel("frequenza")
plt.show()
#anche qui eliminiamo qualche outlier
data=data[data["liveness"]<0.8]
sns.histplot(data["liveness"],bins=40)
plt.xlabel("liveness") 
plt.ylabel("frequenza")
plt.show()

sns.histplot(data["valence"],bins=40)
plt.xlabel("valence") 
plt.ylabel("frequenza")
plt.show()

sns.histplot(data["tempo"],bins=40)
plt.xlabel("tempo") 
plt.ylabel("frequenza")
plt.show()

sns.histplot(data["duration_ms"],bins=40)
plt.xlabel("duration_ms") 
plt.ylabel("frequenza")
plt.show()
#anche qui sono presentiun paio di outliers
data=data[data["duration_ms"]<600000]
data=data[data["duration_ms"]>50000]
sns.histplot(data["duration_ms"],bins=40)
plt.xlabel("duration_ms") 
plt.ylabel("frequenza")
plt.show()


db=data[data["target"]==1]
dr=data[data["target"]==0]
plt.plot(np.arange(0,len(dr["energy"]) ),dr["energy"],"ro")
plt.plot(np.arange(0,len(db["energy"]) ),db["energy"],"b+")
plt.xlabel("frequenza")
plt.ylabel("energy")
plt.title("red=liked       blue=don't like")
plt.show()

#eliminiamo la colonna loudness avente solo valori negativi
#e probabilmente corrotti
data = data.drop(columns=['loudness'])

plt.plot(np.arange(0,len(dr["tempo"]) ),dr["tempo"],"ro")
plt.plot(np.arange(0,len(db["tempo"]) ),db["tempo"],"b+")
plt.xlabel("frequenza")
plt.ylabel("tempo")
plt.title("red=liked       blue=don't like")
plt.show()

plt.plot(np.arange(0,len(dr["valence"]) ),dr["valence"],"ro")
plt.plot(np.arange(0,len(db["valence"]) ),db["valence"],"b+")
plt.xlabel("frequenza")
plt.ylabel("valence")
plt.title("red=liked       blue=don't like")
plt.show()

plt.plot(np.arange(0,len(dr["instrumentalness"]) ),dr["instrumentalness"],"ro")
plt.plot(np.arange(0,len(db["instrumentalness"]) ),db["instrumentalness"],"b+")
plt.xlabel("frequenza")
plt.ylabel("instrumentalness")
plt.title("red=liked       blue=don't like")
plt.show()

plt.plot(np.arange(0,len(dr["key"]) ),dr["key"],"ro")
plt.plot(np.arange(0,len(db["key"]) ),db["key"],"b+")
plt.xlabel("frequenza")
plt.ylabel("key")
plt.title("red=liked       blue=don't like")
plt.show()


plt.plot(np.arange(0,len(dr["danceability"]) ),dr["danceability"],"ro")
plt.plot(np.arange(0,len(db["danceability"]) ),db["danceability"],"b+")
plt.xlabel("frequenza")
plt.ylabel("danceability")
plt.title("red=liked       blue=don't like")
plt.show()

plt.plot(np.arange(0,len(dr["liveness"]) ),dr["liveness"],"ro")
plt.plot(np.arange(0,len(db["liveness"]) ),db["liveness"],"b+")
plt.xlabel("frequenza")
plt.ylabel("liveness")
plt.title("red=liked       blue=don't like")
plt.show()

plt.plot(np.arange(0,len(dr["duration_ms"]) ),dr["duration_ms"],"ro")
plt.plot(np.arange(0,len(db["duration_ms"]) ),db["duration_ms"],"b+")
plt.xlabel("frequenza")
plt.ylabel("duration_ms")
plt.title("red=liked       blue=don't like")
plt.show()

plt.plot(np.arange(0,len(dr["speechiness"]) ),dr["speechiness"],"ro")
plt.plot(np.arange(0,len(db["speechiness"]) ),db["speechiness"],"b+")
plt.xlabel("frequenza")
plt.ylabel("speechiness")
plt.title("red=liked       blue=don't like")
plt.show()

db=data[data["target"]==1]
dr=data[data["target"]==0]
plt.plot(dr["energy"],dr["valence"],"ro")
plt.plot(db["energy"],db["valence"],"b+")
plt.xlabel("energy")
plt.ylabel("valence")
plt.title("red=liked       blue=don't like")
plt.show()


sns.scatterplot(data=data, x="liveness",y="danceability", color='green')
plt.xlabel("liveness")
plt.ylabel("danceability")
plt.show()


sns.scatterplot(data=data, x="energy",y="acousticness")
plt.xlabel("energy")
plt.ylabel("acousticness")
plt.show()


db=data[data["mode"]==1]
dr=data[data["mode"]==0]
plt.plot(dr["danceability"],dr["valence"],"ro")
plt.plot(db["danceability"],db["valence"],"b+")
plt.xlabel("danceability")
plt.ylabel("valence")
plt.title("red=mode       blue=don't mode")
plt.show()
data=data.sample(frac=1).reset_index(drop=True)

#SPLITTING
#separiamo dal nostro dataFrame una parte di training e una di test(75% e 25%)
data_train,data_test=model_selection.train_test_split(data, train_size=1520)
#e facciamo la stessa cosa sul data_train sta volta trovandoci
#il data_val e facendo in modo sia grande al' incirca come il data_test.
data_train,data_val=model_selection.train_test_split(data_train, train_size=1100)


#dividiamo ulteriormente il validation set, il train set e 
#il test set così da poter distiguere le variabili in input e in output

X_train=data_train[["acousticness","danceability", "duration_ms","energy","instrumentalness","key",
         "liveness","mode","speechiness","tempo",
         "time_signature","valence"
         ]]
Y_train=data_train["target"]
y_val=data_val["target"]
x_val=data_val[["acousticness","danceability", "duration_ms","energy","instrumentalness","key",
         "liveness","mode","speechiness","tempo",
         "time_signature","valence"
         ]]
x_test=data_test[["acousticness","danceability", "duration_ms","energy","instrumentalness","key",
         "liveness","mode","speechiness","tempo",
         "time_signature","valence"
         ]]
y_test=data_test["target"]




#REGRESSIONE LINEARE
#impostiamo come modello la regressione lineare 
model = LinearRegression()
 
#REGRESSIONE LINEARE tra le variabili di danceability e valence
#cioè i valori correlati meglio positivamente 	0.44

x_train=X_train[["danceability"]]
y_train=X_train[["valence"]]
model.fit(x_train,y_train)
y_test_pred=model.predict(data_test[["danceability"]])

plt.scatter(data_test[["danceability"]], data_test[["valence"]], color='blue')
plt.plot(data_test[["danceability"]], y_test_pred, linestyle='--',color='red')
plt.xlabel("danceability")
plt.ylabel("valence")
plt.title("REGERESSIONE LINEARE: danceability e valence")
plt.show()
#essendo una regressione lineare chiaramente i valori sono approssimati ad 
#una retta

#stima dei coeficenti della retta
print('Intercept of the model:',model.intercept_)
print('Coefficient of the line:',model.coef_)

#MSE(mean squared error) e R^2(valore di r quadro)
MSE= mean_squared_error(data_test[["valence"]], y_test_pred)
r_2=r2_score(data_test[["valence"]], y_test_pred)
print(f"mean_squared_error: {MSE}")
print(f"r_quadro: {r_2}")

#calcoliamoci i nostri residui, sempre della parte test e mettiamoli su un grafico scatter 
residui=data_test[["valence"]]-y_test_pred
sns.boxplot(residui, color='green')
plt.title('BOXPLOT: RESIDUI')
plt.show()

#usiamo la nostra funzione di shapiro Wilk per trovare il p_value
#così da poter valutare se i nostri residui sono normali
shapiro_test=shapiro(residui)
p_value=shapiro_test[1]
print(f"p_value={p_value}")
#Se ho un p_value molto basso (<0.05) non ho residui normali

#traccio la retta di REGRESIONE LINEARE tra energy e acousticness -0.65
# cioè i due correlati più negativamente 
model = LinearRegression()
x_train=X_train[["energy"]]
y_train=X_train[["acousticness"]]

model.fit(x_train,y_train)
y_test_pred=model.predict(data_test[["energy"]])

plt.scatter(data_test[["energy"]], data_test[["acousticness"]], color='blue')#dati di addestramento
plt.plot(data_test[["energy"]], y_test_pred, linestyle='--',color='red')
plt.xlabel("energy")
plt.ylabel("acousticness")
plt.title("REGERESSIONE LINEARE: ballabilità e acusticità")
plt.show()
print('Intercept of the model:',model.intercept_)
print('Coefficient of the line:',model.coef_)

#MSE(mean squared error) e R^2(valore di r quadro)
MSE= mean_squared_error(data_test[["acousticness"]], y_test_pred)
r_2=r2_score(data_test[["acousticness"]], y_test_pred)
print(f"mean_squared_error: {MSE}")
print(f"r_quadro: {r_2}")

residui=data_test[["acousticness"]]-y_test_pred
sns.boxplot(residui)
plt.title('BOXPLOT: RESIDUI')
plt.show()

#test di shapiro wilk sui residui
shapiro_test=shapiro(residui)
p_value=shapiro_test[1]
print(f"p_value={p_value}")



#DECISIONE SU MODELLO SVM E HYPERMARAMETRIES
best_acc=0
for i in np.arange(1,10):
   
    model=svm.SVC(C=3,kernel="poly",degree=i)
    model.fit(X_train,Y_train)
    y_pred=model.predict(x_val)
    ME=np.sum(y_pred!=y_val)
    MR=ME/len(y_pred) 
    Acc=1-MR
    if Acc>best_acc:
        best_acc=Acc
        best_degree=i
        print(f"   {best_degree}   {best_acc}")  
        """
    model=svm.SVC(C=3,kernel="rbf",gamma=i)
    model.fit(X_train,Y_train)
    y_pred=model.predict(x_val)
    ME=np.sum(y_pred!=y_val)
    MR=ME/len(y_pred) 
    Acc=1-MR
    if Acc>best_acc:
        best_acc=Acc
        best_gamma=i
        print(f"   {best_gamma}   {best_acc}")     
          
      """  
        
#REGRESSIONE LOGISTICA
model_logistic=LogisticRegression()
data_train,data_test=model_selection.train_test_split(data, train_size=1520)
data_train,data_val=model_selection.train_test_split(data_train, train_size=1100)
       
X_train=data_train[["acousticness","danceability", "duration_ms","energy","instrumentalness","key",
         "liveness","mode","speechiness","tempo",
         "time_signature","valence"
         ]]

Y_train=data_train["target"]
y_val=data_val["target"]
x_val=data_val[["acousticness","danceability", "duration_ms","energy","instrumentalness","key",
         "liveness","mode","speechiness","tempo",
         "time_signature","valence"
         ]]
x_test=data_test[["acousticness","danceability", "duration_ms","energy","instrumentalness","key",
         "liveness","mode","speechiness","tempo",
         "time_signature","valence"
         ]]
y_test=data_test["target"]
  

print("test Logistic")

model_logistic.fit(X_train, Y_train)
y_pred_val=model_logistic.predict(x_val)
ME=np.sum(y_pred_val!=y_val)
MR=ME/len(y_val)
Acc_logistic=1-MR
print("LOGISTIC REGRESSION")
print(f"Logistic Regression - ME: {ME}, MR: {MR}, Acc: {Acc_logistic}")

#Test regressione logistica
y_pred_test=model_logistic.predict(x_test)
ME_test=np.sum(y_pred_test!=y_test)
MR_test=ME_test/len(y_test)
Acc_logistic=1-MR_test
print(f"Logistic Regression - ME: {ME_test}, MR: {MR_test}, Acc: {Acc_logistic}")

 

#PROVIAMO K-VOLTE IL NOSTRO MODELLO e raccogliamo le varie accuratezze nel vettore
#accuracies
k=15
accuracies = np.zeros(k)

for i in np.arange(0,k):
 data=data.sample(frac=1).reset_index(drop=True)
 data_train,data_test=model_selection.train_test_split(data, train_size=1520)
 data_train,data_val=model_selection.train_test_split(data_train, train_size=1100)
       
 X_train=data_train[["acousticness","danceability", "duration_ms","energy","instrumentalness","key",
         "liveness","mode","speechiness","tempo",
         "time_signature","valence"
         ]]

 Y_train=data_train["target"]
 y_val=data_val["target"]
 x_val=data_val[["acousticness","danceability", "duration_ms","energy","instrumentalness","key",
         "liveness","mode","speechiness","tempo",
         "time_signature","valence"
         ]]
 x_test=data_test[["acousticness","danceability", "duration_ms","energy","instrumentalness","key",
         "liveness","mode","speechiness","tempo",
         "time_signature","valence"
         ]]
 y_test=data_test["target"]
  
 model_best=svm.SVC(C=3, kernel="poly",degree=best_degree)
 model_best.fit(X_train,Y_train) 
 y_pred_test=model_best.predict(x_test)
 ME_test=np.sum(y_pred_test!=y_test)
 MR_test=ME_test/len(y_test)
 Acc_test=1-MR_test
 accuracies[i]=Acc_test

#VALUTAZIONE RISULTATI OTTENUTI
#una volta eseguita la nostra predizione per un certo numero di k-volte
#valutiamo la media, la mediana,la deviazione e i quartili di ogni accuratezza
print("TEST SUL MODELLO SVC")
media = np.mean(accuracies)
std = np.std(accuracies)
mediana = np.median(accuracies)
diffusione=np.ptp(accuracies)
quartili = np.percentile(accuracies, [25, 50, 75])
print(f"Accuratezza media: {media}")
print(f"Deviazione Standard: {std}")
print(f"Mediana accuratezza: {mediana}")
print(f"Diffusione: {diffusione}")
print(f"Quartili: {quartili}")
print(f"Range interquartile: {quartili[2]-quartili[0]}")

#valutiamo un INTERVALLO DI CONFIDENZA con una probabilità del 95%
confidenza = 0.95
alpha=1-confidenza
t_crit = stats.t.ppf(1-alpha/2, df=k-1)
margine_errore = t_crit * (std / np.sqrt(k))
IC_basso = media - margine_errore
IC_alto = media + margine_errore

print(f"Intervallo di confidenza al 95%: ({IC_basso}, {IC_alto})")

#vediamo il comportamento del nostro modello 
#su un grafico e con un BOXPLOT
sns.histplot(accuracies, bins=k, color='green')
plt.title('DISTRIBUZIONE DELLE ACCURATEZZE')
plt.xlabel('Accuratezze')
plt.ylabel('Frequenza')
plt.show()

sns.boxplot(accuracies,color='orange')
plt.title('BOXPLOT')
plt.xlabel('Accuratezze')
plt.axhline(y=quartili[0], color='r', linestyle='--')
plt.axhline(y=quartili[2], color='r', linestyle='--')
plt.show()




