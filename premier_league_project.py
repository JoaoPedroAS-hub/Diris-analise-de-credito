import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import datetime
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

# 01: Importing the dataset
soccer = pd.read_csv("data/soccer18-19.csv")
variables = pd.read_csv("data/variable_explanation.csv")

# 01a: Alterando o tipo de algumas variáveis
soccer["Date"] = soccer["Date"].astype("datetime64[ns]")

soccer.info()


# 02: Análise univariada

# 02a: Quantidade de jogos ao longo do tempo
len(soccer["Date"].unique())

over_time = soccer.groupby("Date")["Date"].count()

plot = sns.lineplot(over_time)

type(plot)
plot.set_xticklabels(plot.get_xticklabels(), rotation=45)
plot


# 02b Existe diferença entre jogar fora e dentro de casa?
len(soccer["HomeTeam"].unique())

home_goals = soccer.groupby("HomeTeam")["HY","AY"].sum()

sns.boxplot(home_goals)

test_ind = stats.ttest_ind(home_goals["HY"], home_goals["AY"])
test_rel = stats.ttest_rel(home_goals["HY"], home_goals["AY"])

test

# 02c Times com maior proporção de gols em relação aos chutes

eff = soccer.groupby("HomeTeam")["FTHG","HST"].sum()
eff["Aproveitamento"] = eff["FTHG"]/eff["HST"]
eff.sort_values("Aproveitamento", ascending = False)

# Quantidade de jogos por time
wins = soccer.groupby(["AwayTeam","FTR"]).size()

wins.replace("A","Winner")
