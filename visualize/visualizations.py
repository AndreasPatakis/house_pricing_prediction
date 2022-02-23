import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


df = pd.read_csv('../preprocess/housing.csv')
df_norm = pd.read_csv('../preprocess/housing_normalized.csv')
df_stand = pd.read_csv('../preprocess/housing_standarized.csv')
df_pca = pd.read_csv('../preprocess/housing_PCA.csv')

FIGURES_DIR = './Figures'
def boxplot_features(df,features,title):
    fig, axs = plt.subplots(2, 5)
    fig.suptitle(title,fontweight='bold')
    i=0
    j=0
    for col in features:
        print(i,j)
        axs[i,j].boxplot(df[col])
        axs[i,j].set_title('Feature: '+col)
        j=(j+1)%5
        if j==0:
            i+=1
    plt.show()

def hist_features(df,features, plot=True,save=False,categ=''):
    """Plot the histograms of the all features"""
    print("\nCalculating histograms...")
    basic_path = FIGURES_DIR+'/Histograms'

    if save:
        if not os.path.exists(basic_path+'/'+categ):
            os.makedirs(basic_path+'/'+categ)


    for col in features:
        name = col+'_HIST'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(df[col], bins=200, edgecolor='black')
        ax.set_title('Feature: '+col+' '+categ)
        ax.set_xlabel('Range of Values')
        ax.set_ylabel('Num of Observations')
        if save:
            print("Saving histogram of feature: ",col)
            plt.savefig(basic_path+'/'+categ+'/'+name+'_'+categ)
        if plot:
            plt.show()
        else:
            plt.close()

def scatter_features(df,features, plot=True,save=False,categ=''):
    """Plot the histograms of the all features"""
    dim = len(features)
    if dim!=2 and dim!=3:
        print("Features must be either 2 or 3")
        return 0

    print("\nCalculating scatter combination: {}".format(', '.join(features)))
    basic_path = FIGURES_DIR+'/Scatter'

    if save:
        if not os.path.exists(basic_path+'/'+categ):
            os.makedirs(basic_path+'/'+categ)

    name = '_'.join(features)+'_{}dPlot'.format(dim)
    if dim == 3:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(df[features[0]],df[features[1]],df[features[2]], edgecolor='black')
        ax.set_zlabel(features[2])
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(df[features[0]],df[features[1]], edgecolor='black')
    ax.set_title('Combination of: {}'.format(','.join(features)))
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    if save:
        print("Saving scatter of combination...")
        if categ:
            plt.savefig(basic_path+'/'+categ+'/'+name+'_'+categ)
        else:
            plt.savefig(basic_path+'/'+categ+'/'+name)

    if plot:
        plt.show()
    else:
        plt.close()

if __name__ == '__main__':
    #Our numerical features
    numerical_features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']

    #Getting the histograms of our samples
    hist_features(df,numerical_features,plot=True,save=False,categ='Original')
    hist_features(df_norm,numerical_features,plot=True,save=False,categ='Normalized')
    hist_features(df_stand,numerical_features,plot=True,save=False,categ='Standarized')

    # #Some intresting combinations between our attributes
    scatter_combinations = [['longitude','latitude'],['housing_median_age','median_house_value'],['median_income','population'],
    ['households','median_income','median_house_value'],['longitude','latitude','median_income'],
    ['housing_median_age','total_rooms','total_bedrooms']]

    # #Getting the graph of the combinations
    for combination in scatter_combinations:
        scatter_features(df,combination,plot=False,save=True)

    #Getting the graph of the 3-dimensional dataset produced by PCA
    scatter_features(df_pca,df_pca.columns,plot=False,save=True)
