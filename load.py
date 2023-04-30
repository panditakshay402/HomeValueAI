import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import StandardScaler

#load train_data
df=pd.read_csv(r"F:\house_price_prediction\data\train.csv")
#print all train data
''' print(df)
print("shape of train_data are: ",df.shape) '''          

#load test_data
df1=pd.read_csv(r"F:\house_price_prediction\data\test.csv")
''' print(df1)
print("shape of the test_data are: ",df1.shape) '''

''' pd.set_option("display.max_columns",None)  #display maximum columns 
print(df1.head())   #print the head of dataset
print(df1.tail()) '''    #print the Tail of dataset

print("\n")
x1_train=df.iloc[:,[2,4]]
y1_test=df1.iloc[:,5]
print("iloc function of train data: ",x1_train.head())
print("iloc function of test data: ",y1_test.head())
print("\n")

#data integration(multiple table data use)
df2=pd.concat([df,df1])
print("shape of Integrated data of ",df2.shape)
print(df2.info())   #info is use to find the information

print("\n")
int_features=df2.select_dtypes(include=["int64"]).columns
print("Total number of integer features: ",int_features.shape[0])
print("Integer features name: ",int_features.to_list())

print("\n")
float_features=df2.select_dtypes(include=["float64"]).columns
print("Total number of floatting features: ",float_features.shape[0])
print("Integer features name: ",float_features.to_list())

print("\n")
object_features=df2.select_dtypes(include=["object"]).columns
print("Total number of objectical features: ",object_features.shape[0])
print("Integer features name: ",object_features.to_list())

# Get the Statistical information of numerical Features.(for use documentation format)
print("\n")
print(df2.describe())

print("\n")
# pd.set_option("display.max_rows",None)
print(df2["MSSubClass"])      #print the mssubclass features data columns

# Data Cleaning (Methods to Handle missing values,Visualise null value, Get the null value percentage for every feature, ML Algorithms work with missing values.)
print("\n")
plt.figure(figsize=(16,9))
sns.heatmap(df2.isnull())         #print the value true and false is the form of white and black.
plt.savefig("EDIT_DATA_Img/heatmap_DF_of_null_values.png") 

#give the null value for every feature.
print("\n") 
null_percent=df2.isnull().sum()/df2.shape[0]*100
print("percentage of null values: \n",null_percent)

# Drop columns/features (as per observation we will not drop any feature from dataset.)
print("\n")
miss_value_50_perc=null_percent[null_percent>50]
print("missing null values greater than 50%: \n",miss_value_50_perc)
print("\n the Alley values are:\n",df2["Alley"].value_counts())

miss_value_20_50_perc=null_percent[(null_percent>20)&(null_percent<51)]
print("\n missing null values greater than 20%: \n",miss_value_20_50_perc)

miss_value_5_20_perc=null_percent[(null_percent>5)&(null_percent<21)]
print("\n missing null values greater than 5%: \n",miss_value_5_20_perc)

plt.figure(figsize=(16,9))
print("\n",sns.heatmap(df2[miss_value_5_20_perc.keys()].isnull()))
plt.savefig("EDIT_DATA_Img/miss_value_5_20_perc.png") 

#missing value imputation
missing_value_feat=null_percent[null_percent>0]
print("\n Total missing value Feature: ",len(missing_value_feat))
print("\n missing value feature are: \n",missing_value_feat)

object_na_feat=missing_value_feat[missing_value_feat.keys().isin(object_features)]
print("\n missing of object feature data: ",len(object_na_feat))

int_na_feat=missing_value_feat[missing_value_feat.keys().isin(int_features)]
print("\n missing of integer feature data: ",len(int_na_feat))

float_na_feat=missing_value_feat[missing_value_feat.keys().isin(float_features)]
print("\n missing of floating feature data: ",len(float_na_feat))

# Handling MSZonning for reference purpose
print("\n mszoning data: \n",df2["MSZoning"].value_counts())
plt.figure(figsize=(16,9))
# sns.countplot(df["MSZoning"])
plt.savefig("EDIT_DATA_Img/MSZoning_plot_chart.png") 

#backup of original data
df_mvi=df2.copy()        #mvi(missing value imputation)
print(df_mvi.shape)

mszoning_mode=df2["MSZoning"].mode()[0]
df_mvi["MSZoning"].replace(np.nan,mszoning_mode,inplace=True)
df_mvi["MSZoning"].isnull().sum()
# sns.countplot(df_mvi["MSZoning"])
# plt.savefig("EDIT_DATA_Img/df_mvi.png")

def oldNewCountPlot(df2,df_new,feature):
    plt.subplot(121)
    sns.countplot(df2["MSZoning"]) 
    plt.title("old Data Distribution")
    plt.subplot(122)
    sns.countplot(df_new["MSZoning"])
    plt.title("New Data Distribution")
    oldNewCountPlot(df2,df_mvi,"MSZoning")
    plt.savefig("EDIT_DATA_Img/df_mvi_subplot.png")
    
def boxHistPlot(df2,figsize=(16,5)):
    plt.figure(figsize=figsize)  
    plt.subplot(121)
    plt.boxplot(df2)
    plt.subplot(122)
    sns.histplot(df2)
    boxHistPlot(df2["LotFrontage"])
    
    print("\n")
def oldNewBoxMistPlot(df2,df_new,feature): 
    plt.subplot(221)
    sns.boxplot(df2[feature])
    plt.title("old Data Distribution") 
    
    plt.subplot(222)
    sns.displot(df2[feature])
    plt.title("old Data Distribution")
    plt.subplot(223)
    sns.boxplot(df_new[feature])
    plt.title("New Data Distribution") 
oldNewBoxMistPlot(df2,df_mvi,"LotFrontage")

print("\n")
missing_value_feat=null_percent[null_percent>0]
print("Total missing vakue features: ",len(missing_value_feat))
print("missing_value_feat \n",missing_value_feat)
print("\n")
print(df2["Utilities"].value_counts())

print("\n")
utilities_mode=df2["Utilities"].mode()[0]
df_mvi["Utilities"].replace(np.nan,utilities_mode,inplace=True)
print(df_mvi["Utilities"].isnull().sum())

print("\n")
print(df2["Exterior1st"].value_counts())
print("\n")
print(df2["Exterior2nd"].value_counts())

print("\n")
exterior1st_mode=df2["Exterior1st"].mode()[0]
exterior2nd_mode=df2["Exterior2nd"].mode()[0]
df_mvi["Exterior1st"].replace(np.nan,exterior1st_mode,inplace=True)
df_mvi["Exterior2nd"].replace(np.nan,exterior2nd_mode,inplace=True)
print(df_mvi["Exterior1st"].isnull().sum())
print(df_mvi["Exterior2nd"].isnull().sum())

print("\n")
print(df2["Exterior1st"].mode()[0])

print("\n")
plt.figure(figsize=(16,9))
sns.heatmap(df2[["MasVnrType","MasVnrArea"]].isnull())         #print the value true and false is the form of white and black.
plt.savefig("EDIT_DATA_Img/masvrntype.png") 

print(df2[df2[["MasVnrType","MasVnrArea"]].isnull().any(axis=1)])

print(df2["MasVnrType"].value_counts())

print("\n")
masVnrType_mode=df2["MasVnrType"].mode()[0]
df_mvi["MasVnrType"].replace(np.nan,masVnrType_mode,inplace=True)
print(df_mvi["MasVnrType"].isnull().sum())

# print(boxHistPlot(df2["MasVnrType"]))
masVnrarea_cont=0
df_mvi["MasVnrArea"].replace(np.nan,masVnrarea_cont,inplace=True)
print(df_mvi["MasVnrArea"].isnull().sum())

print("\n")
cat_bsmt_feat=["BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2"]
num_bsmt_feat=["BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","BsmtFullBatch","BsmtHalfBath"]

plt.figure(figsize=(16,9))
sns.heatmap(df2[cat_bsmt_feat].isnull()) 
plt.savefig("EDIT_DATA_Img/cat_bsmt_feat.png")

for feat in cat_bsmt_feat:
    print(f"value count of {feat}: {df2[feat].value_counts()}")
    
bsmt_cont="NA"
for feat in cat_bsmt_feat:
    df_mvi[feat].replace(np.nan,bsmt_cont,inplace=True)

print(df_mvi[cat_bsmt_feat].isnull().sum())

# df_bsmt=df2[cat_bsmt_feat+num_bsmt_feat]
# df_bsmt[df_bsmt.isnull().any(axis=1)]
print("\n")
print(df2["Electrical"].value_counts())

print("\n")
print(df2["KitchenQual"].value_counts())

print("\n")
df_ekk=df2[["Electrical","KitchenQual","KitchenAbvGr"]]
df_ekk[df_ekk.isnull().any(axis=1)]
print("\n")
print(df2["Functional"].value_counts())
print("\n")
print(df2["SaleType"].value_counts())

other_cat_feat=["FireplaceQu","PoolQC","Fence","MiscFeature"]
for feat in other_cat_feat:
    print(f"value count of {feat}: {df2[feat].value_counts()}")
print("\n")
num_garage_feat=["GarageArea","GarageFinish","GarageQual","GarageCond"]
cat_garage_feat=["GarageYrBlt","GarageCars","GarageArea"]
df_garafe=df2[num_garage_feat+cat_garage_feat]
print(df_garafe[df_garafe.isnull().any(axis=1)])

#Feature Transformation
##convert numarical feature to categorical feature

for_num_conv=["MSSubClass","YearBuilt","YearRemodAdd","GarageYrBlt","MoSold","YrSold"]
for feat in for_num_conv:
    print(f"(feat):data type=(df_mvi[feat].dtype)")

print(df_mvi[for_num_conv].head())
print("\n")
print(df_mvi["MoSold"].unique())
print("\n")

df['MoSold'] = df['MoSold'].apply(lambda x : calendar.month_abbr[x])
df['MoSold'].unique()

quan = list(df.loc[:, df.dtypes != 'object'].columns.values)
print(quan)

print(len(quan))
obj_feat = list(df.loc[:, df.dtypes == 'object'].columns.values)
print(obj_feat)

#calendar.month_abbr

df_mvi["MoSold"]=df_mvi["MoSold"].apply(lambda x: calendar.month_abbr[x])
print(df_mvi["MoSold"].unique())

for feat in for_num_conv:
    df_mvi[feat]=df_mvi[feat].astype(str)
    print(df_mvi[feat])
for feat in for_num_conv:
    print(f"(feat):data type=(df_mvi[feat].dtype)")

##Categorical to Nimerical Convert
ordinal_end_var=["ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinSF1","BsmtFinType2","HeatingQC","KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC","Functional","GarageFinish","PavedDrive","Utilities"]
print("Total number of feature to convert ordinal numrical format:",len(ordinal_end_var))

# CategoricalDtype
print("\n",df_mvi["ExterQual"].value_counts())
print("\n")
df_mvi["ExterQual"]=df_mvi["ExterQual"].astype(CategoricalDtype(categories=["Po","Fa","TA","Gd","Ex"],ordered=True)).cat.codes
print(df_mvi["ExterQual"])
print("\n",df_mvi["ExterQual"].unique())

print("\n",df_mvi["BsmtExposure"].value_counts())
df_mvi["BsmtExposure"]=df_mvi["BsmtExposure"].astype(CategoricalDtype(categories=["NA","No","Mn","Av","Gd"],ordered=True)).cat.codes
print("\n",df_mvi["BsmtExposure"].value_counts())


df['BsmtFinType1'] = df['BsmtFinType1'].astype(CategoricalDtype(categories=['NA', 'Unf', 'LwQ', 'Rec', 'BLQ','ALQ', 'GLQ'], ordered = True)).cat.codes
df['BsmtFinType2'] = df['BsmtFinType2'].astype(CategoricalDtype(categories=['NA', 'Unf', 'LwQ', 'Rec', 'BLQ','ALQ', 'GLQ'], ordered = True)).cat.codes
df['BsmtQual'] = df['BsmtQual'].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['ExterQual'] = df['ExterQual'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['ExterCond'] = df['ExterCond'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['Functional'] = df['Functional'].astype(CategoricalDtype(categories=['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod','Min2','Min1', 'Typ'], ordered = True)).cat.codes
df['GarageCond'] = df['GarageCond'].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['GarageQual'] = df['GarageQual'].astype(CategoricalDtype(categories=['NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['GarageFinish'] = df['GarageFinish'].astype(CategoricalDtype(categories=['NA', 'Unf', 'RFn', 'Fin'], ordered = True)).cat.codes
df['HeatingQC'] = df['HeatingQC'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['KitchenQual'] = df['KitchenQual'].astype(CategoricalDtype(categories=['Po', 'Fa', 'TA', 'Gd', 'Ex'], ordered = True)).cat.codes
df['PavedDrive'] = df['PavedDrive'].astype(CategoricalDtype(categories=['N', 'P', 'Y'], ordered = True)).cat.codes
df['Utilities'] = df['Utilities'].astype(CategoricalDtype(categories=['ELO', 'NASeWa', 'NASeWr', 'AllPub'], ordered = True)).cat.codes

print(df_mvi.info())
df['Utilities'].unique()

print("\n")

##One Hot Encoding for Nominal Categorical Data
df_encod=df_mvi.copy()
object_features=df_encod.select_dtypes(include="object").columns.tolist()
print("Total object  data type features:",len(object_features))
print("Feature: \n",object_features)

print("\n",df_encod[object_features].head(2))
print("Shape of DF before encoding:",df_encod.shape)
df_encod=pd.get_dummies(df_encod,columns=object_features,prefix=object_features,drop_first=True)
# print(df_encod)
print("Shape of DF after encoding:",df_encod.shape)
print(df_encod.head(2))
# print("\n",df_encod["MSZoning"].value_counts())
# df_encod1=pd.get_dummies(df_encod["MSZoning"],columns="MSZoning",prefix="MSZoning",drop_first=True)
# print(df_encod1)

##Split data for training & Testing

print("\n shape of encoding is: ",df_encod.shape)
len_train=df.shape[0]
print("\n length of encoding data is: ",len_train)
x_train=df_encod[:len_train].drop("SalePrice",axis=1)
y_train=df_encod["SalePrice"][:len_train]

x_test=df_encod[len_train:].drop("SalePrice",axis=1)

print("\n Shape of x_train data:",x_train.shape)
print("\n Shape of x_train data:",y_train.shape)
print("\n Shape of x_test data:",x_test.shape)

##Feature Scaling 
sc=StandardScaler()
sc.fit(x_train)

#formula z=(x+u)/s
x_train=sc.transform(x_train)
x_test=sc.transform(x_test)
print("\n",x_test[:3,:])
print("\n",x_train[:3,:])
print("sc.mean value are: ",sc.mean_)
print("avriable value are: ",sc.var_)

##train ml model
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression

from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

svr=SVR()
lr=LinearRegression()
sgdr=SGDRegressor()

knr=KNeighborsRegressor()
gpr=GaussianProcessRegressor()
dtr=DecisionTreeRegressor()
gbr=GradientBoostingRegressor()

rfr=RandomForestRegressor()
xgbr=XGBRegressor()
mlpr=MLPRegressor()
ir=IsotonicRegression()

models={"a":["LinearRegression",lr],
        "b":["SVR",svr],
        "c":["SGDRegressor",sgdr],
        "d":["KNeighborsRegressor",knr],
        "e":["GaussianProcessRegressor",gpr],
        "f":["DecisionTreeRegressor",dtr],
        "g":["GradientBoostingRegressor",gbr],
        "h":["RandomForestRegressor",rfr],
        "i":["XGBRegressor",xgbr],
        "j":["MLPRegressor",mlpr],
        "k":["IsotonicRegression",ir]
        }
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer,r2_score

def test_model(model, x_train=x_train, y_train=y_train):
    cv=KFold(n_splits=7, shuffle=True, random_state=45)
    r2=make_scorer(r2_score)
    r2_val_score=cross_val_score(model, x_train, y_train, cv=cv, scoring=r2)
    score=[r2_val_score.mean()]
    return score

models_score=[]
for model in models:
    print("training model: ",models[model][0])
    score=test_model(models[model][1],x_train,y_train)
    print("Score of model are: ",score)
    models_score.append([models[model][0],score[0]])
    
    print("model score are: ",models_score)
