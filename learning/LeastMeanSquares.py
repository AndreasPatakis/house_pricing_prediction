import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd


DATA_PATH = '../preprocess/housing_normalized.csv'

def scatter(data,fold='',categ='',plot=True, save=False):
    path = './MSE_Graphs'
    if fold:
        path+='/Fold {}'.format(fold)

    if save:
            if not os.path.exists(path):
                os.makedirs(path)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    title = 'Gradient Descent Convergence'
    title += ' for Fold: {}'.format(fold) if fold else ''
    title += '\n\nMSE:{:.4f}'.format(min(data))
    ax.scatter(list(range(0,len(data))),data, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('Num of Iterations')
    ax.set_ylabel('Mean Squared Error')
    ax.set_yticks(np.arange(0,max(data),0.1))

    if save:
        graph_name = '/{}'.format(categ)
        print('Saving convergence graph..')
        plt.savefig(path+graph_name,format='jpg')
        print('Saved!')
    if plot:
        print("Plotting Gradient Descent's convergence graph..")
        plt.show()
        print("Done!")
    else:
        plt.close()

def k_fold_cross_validation(all_inputs,k_fold):
    data_num = len(all_inputs)
    each_fold = int(data_num/k_fold)
    training_set = []
    testing_set = []
    for fold in range(k_fold):
        start_test = fold*each_fold
        for m in range(0,data_num,each_fold):
            if(m == start_test):
                testing_set.append(all_inputs[m:m+each_fold])
            else:
                if(len(training_set) < fold+1):
                    training_set.append(all_inputs[m:m+each_fold])
                else:
                    training_set[fold] += all_inputs[m:m+each_fold]
    return np.array(training_set),np.array(testing_set)

def hypothesis(x,w):
    return np.dot(x,w)

def robbins_monro(x,y,w,min_learning):
    num_data = len(x)
    learning_rate = 0.1
    convergence = []
    iter = 0
    for set in range(1,num_data):
        if(learning_rate <= min_learning):
            return w
        else:
            if(iter == 100):
                convergence.append(MSE(x[1:set],y[1:set],w))
                iter = 0
            learning_rate = learning_rate/set #+0.001 makes algorithm converge a little bit better
            print(learning_rate)
            J = y-hypothesis(x,w)
            step = np.multiply(J,learning_rate)
            w = w + step[set]
            iter += 1
    return w, convergence

def MSE(x,y,w):
    return np.sum((y-hypothesis(x,w))**2)/len(x)

def train(x,y,min_learning=0.000001):
    #Initializing random weights
    w = np.zeros(len(x.T))
    w,convergence = robbins_monro(x,y,w,min_learning)
    return w, convergence

def predict(x,w):
    return np.dot(x,w)

def test(x,y,w,print=False):
    results = predict(x,w)
    if print:
        for i,result in enumerate(results):
            print('Observed value: {:.4f} | Predicted vale" {:.4f}'.format(y[i],result))
    return results, MSE(x,y,w)




if __name__ == '__main__':

    #Setting up data
    pd.set_option('display.max_columns', None)
    df_norm = pd.read_csv(DATA_PATH)
    #Extractiong our features
    x = df_norm.drop(columns=['median_house_value']).to_numpy()
    #Adding another 'feature' of ones to help us calculate the intercept(bias) later
    x = np.c_[x,np.ones(len(x))]
    #Extractiong our observed value
    y = df_norm['median_house_value'].to_numpy()


    #Learning

    #Producing k-folds
    k=10
    x_training_set,x_testing_set = k_fold_cross_validation(list(x),k)
    y_training_set,y_testing_set = k_fold_cross_validation(list(y),k)

    weights_by_fold = []
    fold_score = []

    for i in range(k):
        print('-'*60)
        print('\nTraining started for fold {}'.format(i+1))
        final_w, convergence = train(x_training_set[i],y_training_set[i],min_learning=0.000001)
        weights_by_fold.append(final_w)
        results, testing_set_mse = test(x_testing_set[i],y_testing_set[i],final_w)
        fold_score.append(np.amin(testing_set_mse))

        print('Training completed for fold {} with Test Set MSE: {:.5f}'.format(i+1,testing_set_mse))
        scatter(convergence,i+1,plot=True,save=False,categ='NORMALIZED')
        print('\n')

    winning_fold = fold_score.index(min(fold_score))
    print('-'*60)
    print('\nBest results were observed while using Fold {}, with Mean Squared Error being minimum at {:.4f}.\n'.format(winning_fold+1,fold_score[winning_fold]))
    print('-'*60)
