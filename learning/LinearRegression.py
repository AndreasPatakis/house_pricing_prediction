import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd

DATA_PATH = '../preprocess/housing_normalized.csv'

def scatter(data,batch_size,learning_rate,fold='',categ='',plot=True, save=False):
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
    title += '\n\nbatch_size:{}, learning_rate:{}, MSE:{:.4f}'.format(batch_size,learning_rate,min(data))
    ax.scatter(list(range(0,len(data))),data, edgecolor='black')
    ax.set_title(title)
    ax.set_xlabel('Num of Iterations')
    ax.set_ylabel('Mean Squared Error')
    ax.set_yticks(np.arange(0,max(data),0.1))

    if save:
        graph_name = '/{} L_R={} BATCH={}'.format(categ,batch_size,learning_rate)
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

def hypothesis_fun(x,w):
    return np.dot(x,w)

def MSE(x,y,w):
    return np.sum((y-hypothesis_fun(x,w))**2)/len(x)

def squared_error_derivative(x,y,w):
    hypothesis_sumed = hypothesis_fun(x,w)
    ch_rule1 = np.multiply(y-hypothesis_sumed,2)
    weights_derivative = np.dot(x.T,ch_rule1)
    return weights_derivative


def GradientDescent(x,y,w,learning_rate,batch_size,max_iter,max_conv):
    variable_converge = [False]*len(x)
    curr_iter = 0
    start = 0
    convergence = []
    while curr_iter < max_iter and not all(variable_converge):
        convergence.append(MSE(x,y,w))
        stop = start+batch_size if start+batch_size<len(x) else start+len(x)-start
        w_der = -2*squared_error_derivative(x[start:stop],y[start:stop],w)
        step_sizes = w_der * learning_rate
        w = w - step_sizes
        variable_converge = (w <= max_conv)
        curr_iter += 1
        start = 0 if stop==len(x) else stop
    return w, convergence

def train(x,y,learning_rate=0.001,batch_size=100,max_iter=1000,max_conv=0.001):
    #Initializing random weights
    w = np.random.rand(len(x.T))
    w,convergence = GradientDescent(x,y,w,learning_rate,batch_size,max_iter,max_conv)
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

    learning_rate = 0.0001
    batch_size = 150
    weights_by_fold = []
    fold_score = []

    for i in range(k):
        print('-'*60)
        print('\nTraining started for fold {}'.format(i+1))
        print('Batch size: {}, Learning rate: {}'.format(batch_size,learning_rate))
        final_w, convergence = train(x_training_set[i],y_training_set[i],learning_rate=learning_rate,batch_size=batch_size,max_iter=600)
        weights_by_fold.append(final_w)
        results, testing_set_mse = test(x_testing_set[i],y_testing_set[i],final_w)
        fold_score.append(np.amin(testing_set_mse))

        print('Training completed for fold {} with Test Set MSE: {:.5f}'.format(i+1,testing_set_mse))
        scatter(convergence,batch_size,learning_rate,i+1,plot=True,save=False,categ='NORMALIZED')
        print('\n')

    winning_fold = fold_score.index(min(fold_score))
    print('-'*60)
    print('\nBest results were observed while using Fold {}, with Mean Squared Error being minimum at {:.4f}.\n'.format(winning_fold+1,fold_score[winning_fold]))
    print('-'*60)



























































    #test(x_testing_set[0],y_testing_set[0],final_w)
