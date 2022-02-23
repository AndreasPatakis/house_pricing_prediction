import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def df_column_to_one_hot_vector(df):
    """Returns a dataframe of each value in OneHotVector representation"""
    column_unique = list(df.drop_duplicates())
    template_vector = np.zeros(len(column_unique),dtype=int)
    dict = {}
    for i,entry in enumerate(column_unique):
        dict[entry] = template_vector.copy()
        dict[entry][i] = 1
    return dict

def normalize(df,features):
    """Using MinMaxScaler to normalize our data"""
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

def standarize(df,features):
    """Using Standard Scaler to standarize data"""
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

def pca(df,features,dimentions=3):
    """Applying Principle Component Analysis to reduce our dimentions to 3"""
    pca = PCA(n_components=dimentions)
    pca = pca.fit(df[features])
    pca_features = pca.transform(df[features])
    return pd.DataFrame(pca_features)

if __name__ == "__main__":

    DATA_PATH = './housing.csv'

    df = pd.read_csv(DATA_PATH)
    #Shuffle the data
    df = df.sample(frac=1)
    num_of_records = df.shape[0]
    pd.set_option('display.max_columns', None)
    print("\nOur dataframe currently: \n\n{}".format(df.head()))

    #pandas wraps 'string' type as 'object' type, so in this way we will find the categorical data
    numerical_attr = list(df.select_dtypes(exclude='object').columns.copy())
    numerical_attr_str = ', '.join(numerical_attr)
    categ_attr = list(df.select_dtypes(include='object').columns.copy())
    categ_attr_str = ', '.join(categ_attr)
    print("\nColumns with numerical data are: {}".format(numerical_attr_str))
    print("\nColumns with categorical data are: {}".format(categ_attr_str))


    """HANDLING CATEGORICAL DATA"""
    print("\n-----Processing categorical data-----")
    #We choose to drop all categorical entries that dont have a value
    print("\nDropping categorical records with no value...")
    df.dropna(subset=categ_attr,inplace=True)
    print('{} records have been dropped.'.format(num_of_records-df.shape[0]))

    for categ_column in categ_attr:
        categ_dict = df_column_to_one_hot_vector(df[categ_column])
        new_columns = list(categ_dict.keys())
        df[new_columns] = np.nan
        for key in categ_dict.keys():
            df.loc[df[categ_column]==key,new_columns] = categ_dict[key]
        df.drop([categ_column],axis=1,inplace=True)
        print("\n'{}' categorical attribute replaced by attributes '{}' with resprect to OneHotVector representation.".format(categ_column,', '.join(new_columns)))


    """HANDLING NUMEIRICAL DATA"""
    print("\n-----Processing numerical data-----")

    y_value = 'median_house_value'

    #replacing empty records with the attribute's median value
    for col in numerical_attr:
        na_values = df[col].isna().sum()
        print("\nAttribute '{}' has {} empty records.".format(col,na_values))
        if na_values:
            attr_median = df[col].median()
            df[col].fillna(attr_median, inplace=True)
            print('Records got replaced by the median value {}.'.format(attr_median))

    print('\n--Scaling data--')
    print('\nNormalization...',end='')
    df_norm = normalize(df,numerical_attr).copy()
    print('DONE')

    print('Standarization...',end='')
    df_stand = standarize(df,numerical_attr).copy()
    print('DONE')

    print('\n--Applying Principle Component Analysis to get an instance of our data set in 3 dimensions--')
    pca_features = df_norm.drop(columns=[y_value])
    df_pca = pca(pca_features,pca_features.columns,dimentions=3)
    df_pca.rename(columns={0:'PCA_1',1:'PCA_2',2:'PCA_3'}, inplace=True)
    df_pca['median_house_value'] = df['median_house_value']
    df_pca = normalize(df_pca,df_pca.columns).copy()
    print('PCA completed successfully!')

    print('\n--Exporting csv files--')
    df_norm.to_csv('./housing_normalized.csv',index=False)
    df_stand.to_csv('./housing_standarized.csv',index=False)
    df_pca.to_csv('./housing_PCA.csv',index=False)
    print('Exporting completed successfully!')
