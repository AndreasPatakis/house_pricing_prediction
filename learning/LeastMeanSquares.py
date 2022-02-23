import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd

DATA = 'housing_'
SCALING = 'normalized'
DATASET = DATA+SCALING
DATA_PATH = '../preprocess/{}.csv'.format(DATASET)

def scatter(data,fold='',sampling=100,categ='',plot=True, save=False):
    path = './Convergence_Graphs_LMS'
    if fold:
        path+='/Fold {}'.format(fold)

    if type(sampling is dict):
        sampling['rest'] = sampling.pop(0)

    metrics = list(data.keys())
    for metric in metrics:
        if save:
            metric_path = path+'/'+metric
            if not os.path.exists(metric_path):
                os.makedirs(metric_path)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        title = 'Robbins Monro Convergence'
        title += ' for Fold: {}'.format(fold) if fold else ''
        title += '\n\nSampling rate: {},  {}:{:.4f}'.format(sampling,metric.upper(),min(data[metric]))
        ax.scatter(list(range(0,len(data[metric]))),data[metric], edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel('Num of Iterations')
        ax.set_ylabel(metric)
        ax.set_yticks(np.arange(0,max(data[metric]),0.1))

        if save:
            graph_name = '/{} {}={:.4f}'.format(categ,metric.upper(),min(data[metric]))
            print('Saving convergence graph..')
            plt.savefig(metric_path+graph_name,format='jpg')
            print('Saved!')
        if plot:
            print("Plotting Robbins Monro's convergence graph..")
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

def plot_graph(x,y,w,variables,dataset,method='Least Mean Squares'):
    xx, yy = np.meshgrid(range(2), range(2))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z = (w[0]*xx + w[1]*yy + w[2])
    ax.plot_surface(xx, yy, z, color = 'pink',alpha = 0.5)
    ax.scatter(x[0],x[1],y, color='blue')
    plt.title("Prediction plane for dataset: {}\n Using {}".format(dataset,method))
    ax.set_xlabel(variables['x'])
    ax.set_ylabel(variables['y'])
    ax.set_zlabel(variables['z'])
    plt.legend()
    plt.tight_layout()

    plt.show()

def hypothesis(x,w):
    return np.dot(x,w)

def robbins_monro(x,y,w,min_learning,sample_rate):
    num_data = len(x)
    learning_rate = 1
    iter = 0
    curr_key = 0
    sample_iter = list(sample_rate.keys())
    convergence = {'MSE':[],'MAE':[]}
    for k in range(num_data-1):
        sample_iter_count = sample_iter[curr_key]
        if sample_iter_count != 0:
            if k >= sample_iter_count:
                curr_key += 1
        if iter%sample_rate[sample_iter_count] == 0:
            convergence['MSE'].append(MSE(x,y,w))
            convergence['MAE'].append(MAE(x,y,w))
            iter=0
        if(learning_rate/(k+1) <= min_learning):
            return w
        else:
            J = y[k+1:k+2]-hypothesis(x[k+1:k+2],w)
            step = np.multiply(J,x[k+1])
            step = np.multiply(step,learning_rate/(k+1))
            w = np.add(w,step)
            iter += 1
    return w, convergence

def MSE(x,y,w):
    return np.sum((y-hypothesis(x,w))**2)/len(x)

def MAE(x,y,w):
    return np.sum(abs(y-hypothesis(x,w)))/len(x)

def train(x,y,min_learning=0.000001,sample_rate=100):
    #Initializing random weights
    w = np.zeros(len(x.T))
    w = robbins_monro(x,y,w,min_learning,sample_rate)
    return w

def predict(x,w):
    return np.dot(x,w)

def test(x,y,w,show=False):
    results = predict(x,w)
    if show:
        for i,result in enumerate(results):
            print('Observed value: {:.4f} | Predicted value: {:.4f}'.format(y[i],result))
    return results




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
    sampling_rate = {100:1,0:300}

    for i in range(k):
        print('-'*60)
        print('\nTraining started for fold {}'.format(i+1))
        final_w, convergence = train(x_training_set[i],y_training_set[i],min_learning=0.000001,sample_rate=sampling_rate)
        weights_by_fold.append(final_w)
        results = test(x_testing_set[i],y_testing_set[i],final_w)
        testing_set_mse = MSE(x_testing_set[i],y_testing_set[i],final_w)
        testing_set_mae = MAE(x_testing_set[i],y_testing_set[i],final_w)
        fold_score.append(np.amin(testing_set_mse))
        print('Training completed for fold {} with Test Set MSE: {:.4f} and MAE: {:.4f}'.format(i+1,testing_set_mse,testing_set_mae))
        scatter(convergence,i+1,sampling_rate.copy(),plot=False,save=True,categ=SCALING.upper())
        print('\n')

    winning_fold = fold_score.index(min(fold_score))
    print('-'*60)
    print('\nBest results were observed while using Fold {}, with Mean Squared Error being minimum at {:.4f}.\n'.format(winning_fold+1,fold_score[winning_fold]))
    print('-'*60)

    while(True):
        ans = input('\nDo you want to see the prediction for every observed value?[y/n]\n')
        if ans == 'y' or ans == 'yes':
            test(x_testing_set[winning_fold],y_testing_set[winning_fold],weights_by_fold[winning_fold],show=True)
            break
        elif ans == 'n' or ans == 'no':
            print('OK BYEE:)')
            break
        else:
            print("You have to answer yes or no. Let's go again..")

    if(weights_by_fold[winning_fold].shape == (3,)):
        print('\nPlotting the graph before i go..')
        variables = {'x':'PCA_1','y':'PCA_2','z':'median_house_value'}
        plot_graph(x_training_set[winning_fold].T,y_training_set[winning_fold],weights_by_fold[winning_fold],variables,DATASET)
