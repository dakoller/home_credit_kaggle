import pandas as pd
from pprint import pprint
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import Imputer
import category_encoders as ce

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras.models import Sequential
#from keras.layers import Dense, Dropout, Activation, Flatten
#from keras.layers import Conv2D, MaxPooling2D

names = ["Nearest Neighbors", "Linear SVM",
         #"RBF SVM", "Gaussian Process",
         #"Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         #"Naive Bayes",
         "QDA"
         ]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    #SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    #DecisionTreeClassifier(max_depth=5),
    #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #MLPClassifier(alpha=1),
    #AdaBoostClassifier(),
    #GaussianNB(),
    QuadraticDiscriminantAnalysis()
    ]

class PipelineFriendlyLabelBinarizer(LabelBinarizer):
    def fit_transform(self, X, y=None):
        return super(PipelineFriendlyLabelBinarizer, self).fit_transform(X)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        #print(X.info())
        return X[self.attribute_names].values

def load_training_data():
    csv_path = os.path.join('..\\..\\data\\raw\\application_train.csv', 'application_train.csv')
    return pd.read_csv(csv_path)

def split_train_test_data(data,test_size = 0.2):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def getBureauData():
    csv_path = os.path.join('..\\..\\data\\raw\\bureau.csv', 'bureau.csv')
    df = pd.read_csv(csv_path)
    #print(df.describe())

    aggregations = {
        'SK_ID_BUREAU': {
            'LOAN_COUNT': 'count',
            'AMT_ANNUITY': 'mean',
        }
    }

    loan_count = df.groupby('SK_ID_CURR')\
        .agg(aggregations)

    return loan_count


data = load_training_data()
#print(data.info())
#print(data.head())

#print(data.describe())

#print(data['CODE_GENDER'].value_counts())
#print(data['FLAG_OWN_REALTY'].value_counts())

#print(data["EXT_SOURCE_1"].value_counts())
target = data['TARGET']

b_data = getBureauData()

#exit()
data = pd.merge(data,b_data,on='SK_ID_CURR', left_index=True)
#data.rename(index=str, columns = { ('SK_ID_BUREAU', 'LOAN_COUNT'): 'LOAN_COUNT',  ('SK_ID_BUREAU', 'AMT_ANNUITY'):'ANNUITY' })
data = data.rename(columns = {data.columns[-1] : 'AMT_ANNUITY'})
data = data.rename(columns = {data.columns[-2] : 'LOAN_COUNT'})

#print(type(data))
#print(data.columns)

#print(data['CODE_GENDER'].value_counts())

num_attribs = ['AMT_INCOME_TOTAL', 'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','CNT_CHILDREN','LOAN_COUNT', 'AMT_ANNUITY','AMT_CREDIT']
cat_attribs = ['CODE_GENDER','FLAG_OWN_CAR']
flag_attribs = ['CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY']


num_pipeline = Pipeline([
    ('selector',DataFrameSelector(num_attribs)),
    ('imputer', Imputer(missing_values='NaN', strategy='mean')),
    ('std_scaler',StandardScaler()),
])

flag_pipeline = Pipeline([
    ('selector', DataFrameSelector(flag_attribs)),
    ('encoder', ce.OrdinalEncoder()),
])
'''
cat_pipeline = Pipeline([
    ('selector',DataFrameSelector(cat_attribs)),
    ('label_binarizer', PipelineFriendlyLabelBinarizer()),
    #('multi_binaizer', MultiLabelBinarizer()),
])
'''
full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline',num_pipeline),
    ('flag_pipeline', flag_pipeline),
   # ('cat_pipeline',cat_pipeline)
])

data_prepared = full_pipeline.fit_transform(data)

df2 = pd.DataFrame(data_prepared)
#, columns= num_attribs + flag_attribs)

pprint(df2)
corr = df2.corr()
pprint(corr[0].sort_values(ascending= False))

#print(data_prepared.shape)
#print(type(data_prepared))
#print(data_prepared[:10])
'''
f1=0.0

n1= ''
_clf = None

print('Evaluate classifiers:')
for name,clf in zip(names,classifiers):
    clf.fit(data_prepared,data['TARGET'])
    _clf = clf
    print('name: %s' % name)

    survived_pred = cross_val_predict(clf, data_prepared, data['TARGET'],cv=3)
    print(confusion_matrix(data['TARGET'],survived_pred))
    _f1 = f1_score(data['TARGET'],survived_pred)
    print('Precision score %0.2f' % precision_score(data['TARGET'],survived_pred))
    print('Recall score %0.2f' % recall_score(data['TARGET'], survived_pred))

    print(_f1)

    if _f1 > f1:
        f1 = _f1
        n1 = name


print('Best classifier: %s with F1 = %0.2f' % (name,f1))
print(_clf)


'''

