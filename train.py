import numpy as np
import pandas as pd
import joblib as jb
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


#Đọc dữ liệu
dt = pd.read_csv("full_data.csv")

#Đổi dữ liệu kiểu chuỗi sang số để tính toán 
dt['bmi'] = dt['bmi'].fillna(dt['bmi'].mean())
np.unique(dt.gender)
dt['gender'].replace(['Female', 'Male', 'Other'],[0,1,2],inplace=True)
np.unique(dt.ever_married)
dt['ever_married'].replace(['No', 'Yes'],[0,1],inplace=True)
np.unique(dt.work_type)
dt['work_type'].replace(['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children'],[0,1,2,3,4],inplace=True)
np.unique(dt.Residence_type)
dt['Residence_type'].replace(['Rural', 'Urban'],[0,1],inplace=True)
np.unique(dt.smoking_status)
dt['smoking_status'].replace(['Unknown','formerly smoked', 'never smoked', 'smokes'],[0,1,2,3],inplace=True)




#gán X = các cột thuộc tính, y = cột nhãn
X = dt.iloc[:,0:10]
y = dt.stroke
	
	
#phân chia dữ liệu bằng nghi thức Hold-out
from sklearn.model_selection import train_test_split
print("------------------------------------------------------------------")
print("Do chinh xac cac giai thuat khi phan chia du lieu bang nghi thuc Hold-out sau 10 lan lap:")

##### Bayes-Naive
model = GaussianNB()


##### Decision Tree
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=10, max_depth=3, min_samples_leaf=5)


##### KNN
Knn = KNeighborsClassifier(n_neighbors=5)


#Huấn luyện mô hình
acc_bayes = 0.0
acc_clf = 0.0
acc_knn = 0.0

array=[10,20,30,50,100,26,56,77,99,22]
for i in range(0,10):
    print('Lan lap ',i)

    X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=3/10, random_state=array[i])
    
    model.fit(X_train, y_train)
    y_pred_bayes = model.predict(X_test)
    print ("Bayes", accuracy_score(y_test, y_pred_bayes)*100)
    acc_bayes += accuracy_score(y_test, y_pred_bayes)

    clf_gini.fit(X_train,y_train)
    y_pred_clf = clf_gini.predict(X_test)
    print ("Decision tree: ", accuracy_score(y_test, y_pred_clf)*100)
    acc_clf += accuracy_score(y_test, y_pred_clf)

    Knn.fit(X_train,y_train)
    y_pred_KNN = Knn.predict(X_test)
    print ("KNeighbors: ", accuracy_score(y_test, y_pred_KNN)*100)
    acc_knn += accuracy_score(y_test, y_pred_KNN)
    print('------------------')

print("\n")
print("Do chinh xac trung binh cua 10 lan lap:")
print("-------------------------------------------")
print("Bayes: ", acc_bayes/10*100)
print("Cay quyet dinh: ", acc_clf/10*100)
print("KNN: ", acc_knn/10*100)

#Save model
filename = 'Bayes_model.sav'
jb.dump(model, filename)


