import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel


###################### Data Import
path=r'C:\Users\Anu\Downloads\datasets\breast cancer data\breast cancer data.csv'
b_data=pd.read_csv(path)

#Drop Unnecessary Feature
b_data=b_data.drop('id',1)
b_data=b_data.drop('Unnamed: 32',1)
###############################

b_data['diagnosis']=[0 if x=='M' else 1 for x in b_data['diagnosis']]


################# Outlier Treatment
def ot_detect(x,a,b):
    col_names=x.iloc[:,a:b].columns
    for a in col_names:
        q1=np.percentile(x[a],25)
        q3=np.percentile(x[a],75)
        iqr=q3-q1
        floor=q1-1.5*iqr
        ceil=q3+1.5*iqr
        outlier_indices=list(x[a].index[(x[a]<floor)|(x[a]>ceil)])
        print(a, len(outlier_indices))
ot_detect(b_data,1,31)
def ot_treatment(x,a,b):
    col_names=x.iloc[:,a:b].columns
    for a in col_names:
        q1=np.percentile(x[a],25)
        q3=np.percentile(x[a],75)
        iqr=q3-q1
        floor=q1-1.5*iqr
        ceil=q3+1.5*iqr
        x[a]=[floor if value < floor else value for value in x[a]]
        x[a]=[ceil if value>ceil else value for value in x[a]]
ot_treatment(b_data,1,31)


########### Skewness Treatment###########
def skewness_detect(x,a,b):
    col_names=x.iloc[:,a:b].columns
    for a in col_names:
        print(a,'->',stats.skew(x[a]))

skewness_detect(b_data,1,31)

def skewness_removal(x,a,b):
    col_names=x.iloc[:,a:b].columns
    for a in col_names:
        x[a]=np.sqrt(x[a])
        
skewness_removal(b_data,1,31)
#########################################

###### Feature Scaling ###############
scalar=StandardScaler()
b_data_scaled=pd.DataFrame(scalar.fit_transform(b_data.iloc[:,1:31]),columns=b_data.iloc[:,1:31].columns)

########## Data Splitting#################
x_train,x_test,y_train,y_test=train_test_split(b_data_scaled,b_data['diagnosis'],test_size=0.2,random_state=13)


############ Model Fitting
log_reg=LogisticRegression()
log_reg.fit(x_train,y_train)

################# Dimension Reduction #################
model=SelectFromModel(log_reg,prefit=True)
X_new_train=model.transform(x_train)
X_new_test=model.transform(x_test)

######### Fitting Model on new Data #######################
log_reg_1=LogisticRegression().fit(X_new_train,y_train)


############### Performance ###########################
from sklearn.model_selection import cross_val_predict
cross_val_score(log_reg_1,X_new_train,y_train,scoring='accuracy',cv=5)
y_train_pred=cross_val_predict(log_reg_1,X_new_train,y_train,cv=5)
confusion_matrix(y_train,y_train_pred)


###### Test Data #############
y_test_pred=cross_val_predict(log_reg_1,X_new_test,y_test,cv=5)
confusion_matrix(y_test,y_test_pred)

### Precision Recal Score on Test Data##
from sklearn.metrics import precision_score,recall_score
precision_score(y_test,y_test_pred)
recall_score(y_test,y_test_pred)




 