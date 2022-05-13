from cProfile import label
import pandas as pd
import numpy as np

def filter_label(df, values):
   
    df_mask = df.label.isin(values)

    filtered_df = df[df_mask]

    return filtered_df

def generate_x_y(df, principal_label, use_principal=True):
    def isPrincipal(label):
        if label == principal_label:
            return 1
        else:
            return -1
    y = []
    if use_principal:
        y = df.apply(lambda row: isPrincipal(int(row[0])), axis=1) 
    else:
        y = df['label']

    y = y.to_numpy() 

    x = df[['intensity', 'symmetry']].to_numpy()

    return x, y

def generate_datasets_one_for_all(test, train, size_test = None, size_train = None):

    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8 , 9]
    one_for_all = labels.copy()
    directory = '../data/filtered_digit_dataset/'

    for l in labels:
        if l == 9:
            break

        print('label = ', l)

        filtered_test = filter_label(test, one_for_all)
        filtered_train = filter_label(train, one_for_all)

        if size_test:
            filtered_test = filtered_test[:size_test]

        if size_train:
            filtered_train = filtered_train[:size_train]

        x_test, y_test = generate_x_y(filtered_test, l)
        x_train, y_train = generate_x_y(filtered_train, l)

        one_for_all.remove(l)

        filename_x_test = str(l) + '_x_test.txt'
        filename_y_test = str(l) + '_y_test.txt'

        filename_x_train = str(l) + '_x_train.txt'
        filename_y_train = str(l) + '_y_train.txt'

        np.savetxt(directory+filename_x_test, x_test)
        np.savetxt(directory+filename_y_test, y_test, fmt='%d')

        np.savetxt(directory+filename_x_train, x_train)
        np.savetxt(directory+filename_y_train, y_train, fmt='%d')

