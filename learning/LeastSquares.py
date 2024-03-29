import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd

DATA = 'housing_'
SCALING = 'PCA'
DATASET = DATA+SCALING
DATA_PATH = '../preprocess/{}.csv'.format(DATASET)

def scatter(data,batch_size,learning_rate,fold='',categ='',plot=True, save=False):
    path = './Convergence_Graphs_LS'
    if fold:
        path+='/Fold {}'.format(fold)


    metrics = list(data.keys())
    for metric in metrics:
        if save:
            metric_path = path+'/'+metric
            if not os.path.exists(metric_path):
                os.makedirs(metric_path)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        title = 'Gradient Descent Convergence'
        title += ' for Fold: {}'.format(fold) if fold else ''
        title += '\n\nbatch_size:{}, learning_rate:{}, {}:{:.4f}'.format(batch_size,learning_rate,metric.upper(),min(data[metric]))
        ax.scatter(list(range(0,len(data[metric]))),data[metric], edgecolor='black')
        ax.set_title(title)
        ax.set_xlabel('Num of Iterations')
        ax.set_ylabel(metric)
        ax.set_yticks(np.arange(0,max(data[metric]),0.1))

        if save:
            graph_name = '/{} L_R={} BATCH={}'.format(categ,learning_rate,batch_size)
            print('Saving convergence graph..')
            plt.savefig(metric_path+graph_name,format='jpg')
            print('Saved!')
        if plot:
            print("Plotting Gradient Descent's convergence graph..")
            plt.show()
            print("Done!")
        else:
            plt.close()

def plot_graph(x,y,w,variables,dataset,method='Least Squares'):
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

def MAE(x,y,w):
    return np.sum(abs(y-hypothesis_fun(x,w)))/len(x)

def squared_error_derivative(x,y,w):
    hypothesis_sumed = hypothesis_fun(x,w)
    ch_rule1 = y-hypothesis_sumed
    weights_derivative = np.dot(x.T,ch_rule1)
    return weights_derivative

def GradientDescent(x,y,w,learning_rate,batch_size,max_iter,max_conv):
    variable_converge = [False]*len(x)
    curr_iter = 0
    start = 0
    convergence = {'MSE':[],'MAE':[]}
    while curr_iter < max_iter and not all(variable_converge):
        convergence['MSE'].append(MSE(x,y,w))
        convergence['MAE'].append(MAE(x,y,w))
        stop = start+batch_size if start+batch_size<len(x) else start+len(x)-start
        w_der = np.multiply(squared_error_derivative(x[start:stop],y[start:stop],w),-2/(stop-start))
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

    #We will apply least squares for each pair of the test values
    tests = {'l_r':[0.01,0.001,0.001,0.0001],'batch':[50,80,120,350],'max_iter':[700,700,700,1500]}
    num_of_tests = len(tests['l_r'])

    weights_by_fold = []
    fold_score = []
    for j in range(num_of_tests):
        learning_rate = tests['l_r'][j]
        batch_size = tests['batch'][j]
        max_iter = tests['max_iter'][j]
        print("\nTesting for values:\nLearning rate: {}, Batch size: {}, Max iteration: {}.\n".format(learning_rate,batch_size,max_iter))
        for i in range(k):
            print('-'*60)
            print('\nTraining started for fold {}'.format(i+1))
            print('Batch size: {}, Learning rate: {}'.format(batch_size,learning_rate))
            final_w, convergence = train(x_training_set[i],y_training_set[i],learning_rate=learning_rate,batch_size=batch_size,max_iter=max_iter)
            weights_by_fold.append(final_w)
            results = test(x_testing_set[i],y_testing_set[i],final_w)
            testing_set_mse = MSE(x_testing_set[i],y_testing_set[i],final_w)
            testing_set_mae = MAE(x_testing_set[i],y_testing_set[i],final_w)
            fold_score.append(np.amin(testing_set_mse))

            print('Training completed for fold {} with Test Set MSE: {:.4f} and MAE: {:.4f}'.format(i+1,testing_set_mse,testing_set_mae))
            scatter(convergence,batch_size,learning_rate,i+1,plot=False,save=True,categ=SCALING.upper())
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
























































    #test(x_testing_set[0],y_testing_set[0],final_w)
