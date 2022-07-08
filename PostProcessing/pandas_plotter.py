import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_onbound = pd.read_csv('../df_onbound.csv')
df_onbound = df_onbound.transpose()

df_nearbound = pd.read_csv('../df_nearbound.csv')
df_nearbound = df_nearbound.transpose()

df_inbulk = pd.read_csv('../df_inbulk.csv')
df_inbulk = df_inbulk.transpose()

df_random1 = pd.read_csv('../df_random1.csv')
df_random1 = df_random1.transpose()

df_random2 = pd.read_csv('../df_random2.csv')
df_random2 = df_random2.transpose()

df_onbound.plot(title="ONBOUND")
df_nearbound.plot(title="NEARBOUND")
df_inbulk.plot(title="INBULK")
df_random1.plot(title="random1")
df_random2.plot(title="random2")

plt.show()