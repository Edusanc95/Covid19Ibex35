import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hac
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import datetime
import seaborn as sns
import statistics

def pca_df_creation(df):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df)
    principal_df = pd.DataFrame(data=principal_components,
                                columns=['principal component 1', 'principal component 2'])

    print(pca.explained_variance_ratio_)  # 0.85
    return principal_df


def pca_plot(df, title):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=20)

    ax.scatter(df['principal component 1'],
               df['principal component 2'],
               s=50,
               alpha=0)

    for i in range(len(df.index)):
        plt.text(df['principal component 1'][i], df['principal component 2'][i], str(i),
                 ha="center",
                 va="center")

    ax.grid()
    plt.show()


# Closed price is the one that's going to be saved

df = pd.DataFrame()
source_folder = "data"
scaler = MinMaxScaler()

start_data_recollection_date = datetime.date(2020, 2, 20)
end_data_recollection_date = datetime.date(2020, 5, 20)

end_training_date = datetime.date(2020, 5, 8)

for filename in os.listdir(source_folder):
    company_df = pd.read_csv("{0}/{1}".format(source_folder, filename))
    company_df['Date'] = pd.to_datetime(company_df['Date'], format="%Y/%m/%d")
    company_df = company_df.set_index('Date')
    company_df = company_df[start_data_recollection_date:end_data_recollection_date]
    company_df['Close'] = scaler.fit_transform(company_df['Close'].values.reshape(-1, 1))
    company_df = company_df['Close']
    df[filename.replace('.MC', '').replace('.csv', '')] = company_df

df = df.dropna()
df.index = pd.DatetimeIndex(df.index)
df.interpolate(method='linear', axis=0).ffill().bfill()
plt.plot(df.loc[:, df.columns != 'IBEX'])
plt.show()

# df_covid19 = pd.read_csv("https://covid19.isciii.es/resources/serie_historica_acumulados.csv")
df_covid19 = pd.read_csv("agregados.csv")
df_covid19['FECHA'] = pd.to_datetime(df_covid19['FECHA'], format="%d/%m/%Y")
df_covid19 = df_covid19.groupby(['FECHA']).sum()
df_covid19['CASOS'] = df_covid19['CASOS'] + df_covid19['PCR+']
df_covid19 = df_covid19.drop(columns=['PCR+', 'TestAc+'])

for feature_daily in df_covid19.columns:
    df_covid19[feature_daily] = df_covid19[feature_daily].diff().fillna(0)

df_covid19.columns = ['Cases', 'Hospitalizations', 'Critical Hospitalizations', 'Deaths']
df_covid19.plot(subplots=True, layout=(2, 2))
plt.show()

# Do the clustering
# Correlation because we care about how do they grow
# Or euclidean now that it is mximized
Z = hac.linkage(df.transpose(), method='single', metric='correlation')

# Plot dendogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
hac.dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

clusters = hac.fcluster(Z, 0.2, criterion='distance', depth=2, R=None, monocrit=None)

df_ibex_columns = df.columns
df['Cases'] = df_covid19['Cases']
df['Deaths'] = df_covid19['Deaths']
df['Hospitalizations'] = df_covid19['Hospitalizations']
df['Critical Hospitalizations'] = df_covid19['Critical Hospitalizations']

# Heatmap
plt.figure(figsize=(12, 10))
cor = df.corr()
sns.heatmap(cor[['Cases', 'Deaths', 'Hospitalizations', 'Critical Hospitalizations']], annot=True, cmap=plt.cm.Reds)
plt.show()

mean_absolute_errors = []
for company in df_ibex_columns:
    X_train = df[:end_training_date][['Cases']]
    y_train = df[:end_training_date][company]

    X_test = df[end_training_date + datetime.timedelta(days=1):][['Cases']]
    y_test = df[end_training_date + datetime.timedelta(days=1):][company]

    RF_Model = RandomForestRegressor(n_estimators=100,
                                     criterion="mae")
    labels = y_train
    features = X_train
    # Fit the RF model with features and labels.
    rgr = RF_Model.fit(features, labels)

    # Now that we've run our models and fit it, let's create
    # dataframes to look at the results
    y_predict = rgr.predict(X_test)
    mae = mean_absolute_error(y_test, y_predict)
    mean_absolute_errors.append(mae)
    print("Company: {0}".format(company))
    print("MAE: {0}".format(mae))

print("Mean of the errors: {0}".format(statistics.mean(mean_absolute_errors)))
print("Stdev of the errors: {0}".format(statistics.stdev(mean_absolute_errors)))
