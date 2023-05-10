import numpy as np
import pandas as pd
import joblib as jb
from tkinter import *
from tkinter.filedialog import askopenfilename

data = askopenfilename()
dt = pd.read_csv(data)

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

#load model
model = jb.load('Bayes_model.sav')

#dự báo nhãn
result = model.predict(dt)

#in kết quả
print("Ket qua chuan doan: ", result)

for i in dt.index:
    if result[i] == 1: 
        result2 = 'Có nguy cơ'
    if result[i] == 0:
        result2 = 'Không có nguy cơ'
    print("Bệnh nhân thứ ", i+1," :",result2)







