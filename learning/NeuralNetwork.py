import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd
import tensorflow as tf


DATA_PATH = '../preprocess/housing_normalized.csv'


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

def show_predictions(observed,predicted):
    for set in range(len(predicted)):
        print('Observed value: {:.4f} | Predicted vale" {:.4f}'.format(observed[set],predicted[set]))

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

    best_score = None
    features_num = len(x.T)
    batch_size = 250


    for i in range(k):
        print('-'*60)
        print('\nTraining started for fold {}\n'.format(i+1))
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(int(features_num*1.5),kernel_initializer='normal',input_dim=features_num,activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(features_num//2,kernel_initializer='normal',activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(1,kernel_initializer='normal',activation=tf.nn.relu))
        model.compile(optimizer='adam',
                     loss='mean_squared_error',
                     metrics =['mean_squared_error'])
        history = model.fit(x_training_set[i],y_training_set[i],batch_size=batch_size, epochs=20)
        mse = history.history['mean_squared_error'][-1]

        if not best_score:
            best_score = mse
            winning_fold = i+1
            testing_predictions = model.predict(x_testing_set[i])
        elif mse < best_score:
            best_score = mse
            winning_fold = i+1
            testing_predictions = model.predict(x_testing_set[i])

        print('\nTraining completed for fold {} with MSE of: {:.4f}'.format(i+1,mse))
        print('\n')

    print('-'*60)
    print('\nBest results were observed while using Fold {}, with Mean Squared Error being minimum at {:.4f}.\n'.format(winning_fold,best_score))
    print('-'*60)

    while(True):
        ans = input('\nDo you want to see the prediction for every observed value?[y/n]\n')
        if ans == 'y' or ans == 'yes':
            show_predictions(y_testing_set[winning_fold],testing_predictions.flatten())
            break
        elif ans == 'n' or ans == 'no':
            print('OK BYEE:)')
            break
        else:
            print("You have to answer yes or no. Let's go again..")
