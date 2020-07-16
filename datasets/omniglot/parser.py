import numpy as np
from PIL import Image
import os
from scipy.io import savemat

def create_arrays():
    for task_dir in os.listdir('images_background'):
        X = []
        y = []
        if not task_dir.startswith('.'):
            for label_dir in os.listdir('images_background/' + task_dir):
                if not label_dir.startswith('.'):
                    for file in os.listdir('images_background/' + task_dir + '/' + label_dir):
                        if not file.startswith('.'):
                            img = Image.open('images_background/' + task_dir + '/' + label_dir + '/' + file)
                            img_arr = np.array(img)
                            X.append(img_arr)
                            y.append(int(label_dir.lstrip('character')) - 1)
        
            X = np.array(X)
            y = np.array(y)
            idx_shuffle = np.random.permutation(len(y))
            num_train = int(len(y) * .8)
            num_val = int(len(y) * .1)
            X_train = X[idx_shuffle[:num_train]]
            y_train = y[idx_shuffle[:num_train]]
            X_val = X[idx_shuffle[num_train:num_train+num_val]]
            y_val = y[idx_shuffle[num_train:num_train+num_val]]
            X_test = X[idx_shuffle[num_train+num_val:]]
            y_test = y[idx_shuffle[num_train+num_val:]]
            os.makedirs('all_tasks/' + task_dir, exist_ok=True)
            savemat('all_tasks/' + task_dir + '/train.mat', {'X': X_train, 'y': y_train})
            savemat('all_tasks/' + task_dir + '/val.mat', {'X': X_val, 'y': y_val})
            savemat('all_tasks/' + task_dir + '/test.mat', {'X': X_test, 'y': y_test})

    for task_dir in os.listdir('images_evaluation'):
        X = []
        y = []
        if not task_dir.startswith('.'):
            for label_dir in os.listdir('images_evaluation/' + task_dir):
                if not label_dir.startswith('.'):
                    for file in os.listdir('images_evaluation/' + task_dir + '/' + label_dir):
                        if not file.startswith('.'):
                            img = Image.open('images_evaluation/' + task_dir + '/' + label_dir + '/' + file)
                            img_arr = np.array(img)
                            X.append(img_arr)
                            y.append(int(label_dir.lstrip('character')) - 1)
        
            X = np.array(X)
            y = np.array(y)
            idx_shuffle = np.random.permutation(len(y))
            num_train = int(len(y) * .8)
            num_val = int(len(y) * .1)
            X_train = X[idx_shuffle[:num_train]]
            y_train = y[idx_shuffle[:num_train]]
            X_val = X[idx_shuffle[num_train:num_train+num_val]]
            y_val = y[idx_shuffle[num_train:num_train+num_val]]
            X_test = X[idx_shuffle[num_train+num_val:]]
            y_test = y[idx_shuffle[num_train+num_val:]]
            os.makedirs('all_tasks/' + task_dir, exist_ok=True)
            savemat('all_tasks/' + task_dir + '/train.mat', {'X': X_train, 'y': y_train})
            savemat('all_tasks/' + task_dir + '/val.mat', {'X': X_val, 'y': y_val})
            savemat('all_tasks/' + task_dir + '/test.mat', {'X': X_test, 'y': y_test})

if __name__ == '__main__':
    create_arrays()

