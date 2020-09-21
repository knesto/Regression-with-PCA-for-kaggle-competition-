import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression



#view settings
pd.set_option('display.max_rows', 30000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



def merge_files():
    #merge files and delete column SalePrice
    input_train = pd.read_csv("train.csv",delimiter=',',index_col='Id')
    input_test = pd.read_csv("test.csv",delimiter=',',index_col='Id')
    input_train_after_drop_column = input_train.drop(columns=['SalePrice'])
    final_file= input_train_after_drop_column.append(input_test,sort=False)
    return input_train,final_file

def data_evaluation(final_file):
    final_file.head()
    final_file.shape
    #replace some important NA values
    final_file["Alley"].fillna('No_alley_access', inplace = True)
    final_file["BsmtQual"].fillna('No_Basement', inplace = True)
    final_file["BsmtCond"].fillna('No_Basement', inplace = True)
    final_file["BsmtExposure"].fillna('No_Basement', inplace = True)
    final_file["BsmtFinType1"].fillna('No_Basement', inplace = True)
    final_file["BsmtFinType2"].fillna('No_Basement', inplace = True)
    final_file["FireplaceQu"].fillna('No_Fireplace', inplace = True)
    final_file["GarageType"].fillna('No_Garage', inplace = True)
    final_file["GarageFinish"].fillna('No_Garage', inplace = True)
    final_file["GarageQual"].fillna('No_Garage', inplace = True)
    final_file["GarageCond"].fillna('No_Garage', inplace = True)
    final_file["PoolQC"].fillna('No_Pool', inplace = True)
    final_file["Fence"].fillna('No_Fence', inplace = True)
    final_file["MiscFeature"].fillna('None', inplace = True)
    
    
    # Summary Statistics of the data
    final_file.describe()
    # View the datatype of each column in the dataset
    final_file.dtypes
    # Number of  null values in the data
    final_file.isnull().sum().plot(kind="bar",figsize=(15,7), color="#61d199")
    # Number of non - null values in the data
    #final_file.notnull().sum().plot(kind="bar",figsize=(15,7), color="#61d199")
    #more info
    final_file.info()
    plt.show()
    
    

def replace_missing_values(final_file) :

    
    #Numeric Variables-Replace Null Values with median
    #String Variables-Replace Null Values with the most_frequent value
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    for i in final_file.columns:
      if isinstance(final_file[i].iloc[0], str):
          final_file[i] = imputer.fit_transform(final_file[[i]])

      else :
          final_file[i] = final_file[i].fillna(final_file[i].median())
      
    
def label_en(final_file):
    
    """final_file = pd.concat([final_file, pd.get_dummies(final_file['Alley'], prefix='Alley', drop_first=True)],axis=1)
    final_file = pd.concat([final_file, pd.get_dummies(final_file['BsmtQual'], prefix='BsmtQual', drop_first=True)],axis=1)
    final_file = pd.concat([final_file, pd.get_dummies(final_file['BsmtCond'], prefix='BsmtCond', drop_first=True)],axis=1)
    final_file = pd.concat([final_file, pd.get_dummies(final_file['BsmtExposure'], prefix='BsmtExposure', drop_first=True)],axis=1)
    final_file = pd.concat([final_file, pd.get_dummies(final_file['BsmtFinType1'], prefix='BsmtFinType1', drop_first=True)],axis=1)
    final_file = pd.concat([final_file, pd.get_dummies(final_file['BsmtFinType2'], prefix='BsmtFinType2', drop_first=True)],axis=1)
    final_file = pd.concat([final_file, pd.get_dummies(final_file['FireplaceQu'], prefix='FireplaceQu', drop_first=True)],axis=1)
    final_file = pd.concat([final_file, pd.get_dummies(final_file['GarageType'], prefix='GarageType', drop_first=True)],axis=1)
    final_file = pd.concat([final_file, pd.get_dummies(final_file['GarageFinish'], prefix='GarageFinish', drop_first=True)],axis=1)
    final_file = pd.concat([final_file, pd.get_dummies(final_file['GarageQual'], prefix='GarageQual', drop_first=True)],axis=1)
    final_file = pd.concat([final_file, pd.get_dummies(final_file['GarageCond'], prefix='GarageCond', drop_first=True)],axis=1)
    final_file = pd.concat([final_file, pd.get_dummies(final_file['PoolQC'], prefix='PoolQC', drop_first=True)],axis=1)
    final_file = pd.concat([final_file, pd.get_dummies(final_file['Fence'], prefix='Fence', drop_first=True)],axis=1)
    final_file = pd.concat([final_file, pd.get_dummies(final_file['MiscFeature'], prefix='MiscFeature', drop_first=True)],axis=1)
    final_file.drop(['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu',
                     'GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature'], axis=1, inplace=True)"""
    #replace string values with vectors
    label_encoder = LabelEncoder()
    for i in final_file.columns:
      if isinstance(final_file[i].iloc[0], str):
          final_file[i]= label_encoder.fit_transform(final_file[i])
    return final_file     
  

def variable_distribution(final_file) :
    # view the distribution of variables
    # plot
    f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
    sns.distplot( final_file['BsmtCond'] , color="skyblue", ax=axes[0, 0])
    sns.distplot( final_file['BsmtExposure'] , color="olive", ax=axes[0, 1])
    sns.distplot( final_file['FireplaceQu'] , color="gold", ax=axes[1, 0])
    sns.distplot( final_file['PoolQC'] , color="teal", ax=axes[1, 1])
    
    plt.show()


def pearsoncorr(final_file) :
    # Compute the correlation matrix
    pearsoncorr = final_file.corr(method='pearson')
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
   
    sns.heatmap(pearsoncorr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},)

    plt.show()
    #descending sort of dataframe according to correlation matrix
    pos_pearsoncorr=pearsoncorr.abs()
    sum_column = pos_pearsoncorr.sum(axis=0)
    
    result = final_file.append(sum_column,ignore_index=True)
    sorted_df = result.sort_values(result.last_valid_index(),ascending = False, axis=1)
    sorted_df.drop(sorted_df.tail(1).index,inplace=True)
    
    return sorted_df


def standard_deviation(final_file):
     #descending sort of dataframe according to standard deviation
     final_file_std= final_file.reindex(final_file.std().sort_values(ascending = False).index,axis=1)

     return(final_file_std)
    
  

def standardization_score(final_file):
    cols = list(final_file.columns)
    df_ = pd.DataFrame()
    for col in cols:
        df_[col] = (final_file[col] - final_file[col].mean())/final_file[col].std(ddof=0)
    return df_


def number_PCA_components(final_file):
    
    #Fitting the PCA algorithm with our Data to choose n_components
    pca = PCA().fit(final_file)#Plotting the Cumulative Summation of the Explained Variance
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Variance (%)') #for each component
    plt.title('SalesPrice Explained Variance')
    plt.show()
    
def PCA_model(data,n_components):
    
    # Create a PCA instance: pca
    pca = PCA(n_components)
    principalComponents= pd.DataFrame(pca.fit_transform(data), index=data.index)
   

    # Plot the explained variances
    features = range(pca.n_components_)
    plt.bar(features, pca.explained_variance_ratio_, color='black')
    plt.xlabel('PCA features')
    plt.ylabel('variance %')
    plt.xticks(features)
    plt.show()
    

    return principalComponents

def n_cluster(PCA_components) :
    
    ks = range(1, 35)
    inertias = []
    for k in ks:
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)
        
        # Fit model to samples
        model.fit(PCA_components.iloc[:,:35])
        
        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)
        
    plt.plot(ks, inertias, '-o', color='black')
    plt.xlabel('number of clusters, k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()

def kmeans_cluster(PCA_components):
    #input to k-means
    PCA_components = PCA_components.iloc[:,:35]
    #number of clusters
    kmeans = KMeans(n_clusters=2)
    pred_y= pd.DataFrame(kmeans.fit_predict(PCA_components), index=PCA_components.index)
    total_file= pd.merge(PCA_components, pred_y, left_index=True, right_index=True, how='inner')

    
    return total_file


def regression(cluster1,cluster2):
    #####cluster 1
    #train incluedes the SalePrice Value
    train_cl1 = cluster1[cluster1['SalePrice'].notnull()]
    test_cl1= cluster1[cluster1['SalePrice'].isnull()]
    
    
    #number of components
    x_train_cl1 = train_cl1.iloc[:,:35]
    y_train_cl1 = train_cl1['SalePrice']
    x_test_cl1 = test_cl1.iloc[:,:35]
    
    

    clf_cl1= LinearRegression()
    clf_cl1.fit(x_train_cl1,y_train_cl1)
  
    pred_cl1= pd.DataFrame(clf_cl1.predict(x_test_cl1), index=x_test_cl1.index)

    #####cluster 2

    train_cl2 = cluster2[cluster2['SalePrice'].notnull()]
    test_cl2= cluster2[cluster2['SalePrice'].isnull()]
    
    

    x_train_cl2 = train_cl2.iloc[:,:35]
    y_train_cl2 = train_cl2['SalePrice']
    x_test_cl2 = test_cl2.iloc[:,:35]
    
    

    clf_cl2= LinearRegression()
    clf_cl2.fit(x_train_cl2,y_train_cl2)
  
    pred_cl2= pd.DataFrame(clf_cl2.predict(x_test_cl2), index=x_test_cl2.index)

   
    frames = [pred_cl1, pred_cl2]

    result = pd.concat(frames)

    result.columns = ['SalePrice']
    result.to_csv('submission.csv', sep=',', encoding='utf-8')
    
# Main function that runs the program
def main():
    input_train,final_file = merge_files()
    data_evaluation(final_file)
    replace_missing_values(final_file)
    final_file=label_en(final_file)
    
    variable_distribution(final_file)

    #----A----#
    final_file_pearson = pearsoncorr(final_file)
    
    #----B----#
    final_file_std =standard_deviation(final_file)
    
    
    final_file_after_stand = standardization_score(final_file)

    #----C----#
    final_file_c = pearsoncorr(final_file_after_stand)

    #----D----#
    final_file_d = standard_deviation(final_file_after_stand)

    #----E----#
    number_PCA_components(final_file)
    PCA_components=PCA_model(final_file,8)
      
    #----F----#
    number_PCA_components(final_file_after_stand)
    PCA_components_stand =PCA_model(final_file_after_stand,75)
    
    n_cluster(PCA_components_stand)
    data_after_cluster=kmeans_cluster(PCA_components_stand)
    result = pd.concat([data_after_cluster, input_train['SalePrice']], axis=1)
    cluster1 = result[result['0_y'] == 0]
    cluster2 = result[result['0_y'] == 1]
   
    
    regression(cluster1,cluster2)
   
    


if __name__ == '__main__':
    main()
    
