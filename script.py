# ------- import librairies ------- # 
import wandb
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

# ------- import csv ------- # 
df = pd.read_csv('data/clean_csv.csv')
df.drop(columns=['Unnamed: 0'],inplace=True)
df.head()

# ------ split data ------- # 

y = df['TARGET']
X = df.drop('TARGET', axis=1)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=42)

for train, test in sss.split(X, y):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]

# ------- Init wandb ------- # 

wandb.init(project="lgbm_classifier", entity="zao_data")

# ------- Pipeline ------- # 

pipe_lgbm = Pipeline(
[
    ('scaler', StandardScaler()),
    ('model', LGBMClassifier())
])

# ------- fit modele ------- # 

pipe_lgbm.fit(X_train, y_train)


# ------- log des métriques ------- # 

wandb.log({'f1_score' : f1_score,
          'accuracy' : accuracy_score,
          'recall' : recall_score,
          'precision': precision_score})


# ------- alerting ------- # 

thresh = 0.6
wandb.alert(
    title = "Low F1 Score", 
    text = f"F1 score {f1_score} is below the acceptable threshold {thresh}"
)