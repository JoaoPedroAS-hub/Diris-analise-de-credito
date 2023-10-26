
# 01 Importanto as bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from mpl_toolkits.basemap import Basemap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix


# 02. Leitura de dados e diagnóstico inicial:

# 02a.Lendo os dados
credito = pd.read_csv('http://dl.dropboxusercontent.com/s/xn2a4kzf0zer0xu/acquisition_train.csv?dl=0')
credito.head()
credito.info()

# 02b. Localizando valores faltantes
((credito.isnull().sum()/credito.shape[0])*100).sort_values(ascending = False)


# 02c. Valores unicos
credito.nunique().sort_values(ascending = True)

# 02d. Informações sobre as variáveis numéricas
credito.describe()

# 03. Limpeza dos dados

# 03a. Descatar variáveis com apenas um ou uma grande quantidade de valores únicos
drop = ['external_data_provider_credit_checks_last_2_year', 'ok_since', 'channel',
        'target_fraud','ids', 'last_amount_borrowed', 'profile_phone_number', 
        'reason', 'zip', 'user_agent', 'job_name', 'external_data_provider_first_name',
        'last_borrowed_in_months', 'state', 'shipping_zip_code']

credito_clean = credito.drop(labels=drop, axis=1) #axis 1 columns

# 03b. Lidando com valores inf de renda informada.
credito_clean = credito_clean[credito_clean['reported_income'] != np.inf]

# 03b. Lidando com valores negativos de visualização de email
credito_clean.loc[credito_clean['external_data_provider_email_seen_before'] < 0,
             'external_data_provider_email_seen_before'] = np.nan

# 03c. Transformando a coluna facebook_profile
credito_clean['facebook_profile'].fillna(value=False, inplace=True, axis=0)
credito_clean['facebook_profile'] = credito_clean['facebook_profile'].map({True: 'Sim', False: 'Não'})

# 03d. Substituindo valores na coluna e-mail:
credito_clean.loc[credito_clean['email'] == 'hotmaill.com', 'email'] = 'hotmail.com'
credito_clean.loc[credito_clean['email'] == 'gmaill.com', 'email'] = 'gmail.com'

# 03e. Deixando apenas a sigla do estado
credito_clean['shipping_state'] = credito_clean['shipping_state'].str.replace("BR-","")

# 03f. Removendo variáveis que não tem informações do target Defeault.
credito_clean.dropna(subset = ['target_default'], inplace = True)
credito_clean['target_default'] = credito_clean['target_default'].map({True: 1, False:0})

credito_clean.head()


# 03g. Substituindo NAs de variáveis categóricas com valores proporcionais e de variáveis numéricas com a mediana:

def preencher_proporcional(col):
    """ Preenche valores ausentes na mesma proporção dos valores presentes

    Recebe uma coluna e retorna a coluna com os valores ausentes preenchidos
    na proporção dos valores previamente existentes."""
    
    # Gerando o dicionário com valores únicos e sua porcentagens
    percentages = col.value_counts(normalize=True).to_dict()

    # Tranformando as chaves e valores do dicionário em listas      
    percent = [percentages[key] for key in percentages]
    labels = [key for key in percentages]

    # Utilizando as listas para prencher os valores nulos na proporção correta 
    s = pd.Series(np.random.choice(labels, p=percent, size=col.isnull().sum()))
    col = col.fillna(s)
    
    # Verificando se todos os valores ausentes foram preenchidos e
    # preenchendo os que não tiverem sido
    if len(col.isnull()) > 0:
        col.fillna(value=max(percentages, key=percentages.get), inplace=True, axis=0)
        
    return col


for col in credito_clean.iloc[:,1:].columns.tolist():
    if credito_clean[col].dtypes == 'O': #Objects
        credito_clean[col] = preencher_proporcional(credito_clean[col])
    else:
        credito_clean[col].fillna(value=credito_clean[col].median(), inplace=True, axis=0)

credito_clean.isnull().sum()

range(4)