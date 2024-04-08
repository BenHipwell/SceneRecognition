import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import xgboost as xgb
from keras.applications import ResNet152
from matplotlib import pyplot as plt
import pathlib

# # option to stop tensorflow from allocating all available GPU memory so that it does not run out for classification
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)

# obtain the file path of the scripts current location and append the training & testing folder location
rootdir = str(pathlib.Path().resolve()) + '\\training'
test_rootdir = str(pathlib.Path().resolve()) + '\\testing'

# list of the 15 different scene classes
data_classes = ["bedroom", "Coast", "Forest", "Highway", "industrial", "Insidecity", "kitchen", "livingroom", "Mountain", "Office", "OpenCountry", "store", "Street", "Suburb", "TallBuilding"]

# creates a dictionary with key being the data classes and the value being a number between 0 and 14
d = dict(zip(data_classes, range(0,15)))
# creates an inverse dictionary to the above to map the class number to the class name
inv_d = dict((v,k) for k,v in d.items())

images = []
y = []

# reads in the training images and their given label into the newly created arrays
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if (file.startswith('.')):
            print("DS store")
        else:
            images.append(cv2.imread(subdir +'/'+ file))
            y.append(d.get(subdir.split('\\')[-1]))

test_images = []
test_names = []

# reads in the testing images and their given label into the newly created arrays
for subdir, dirs, files in os.walk(test_rootdir):
    for file in files:
        if (file.startswith('.')):
            print("DS store")
        else:
            test_images.append(cv2.imread(subdir +'/'+ file))
            test_names.append(file)

# ensure label arrays are also of type numpy array
y = np.array(y)
test_names = np.array(test_names)

# to verify the images and labels have been loaded correctly
unique, counts = np.unique(y, return_counts=True)
print(f'Unique labels: {unique}')
print(f'Counts: {counts}')

# instantiate the ResNet 152 layer pre-trained architecture
res = ResNet152()

# resize the test and training images to the same size of images that ResNet has been trained to deal with
images = np.array([cv2.resize(img, (224, 224)) for img in images])
test_images = np.array([cv2.resize(img, (224, 224)) for img in test_images])

# using ResNet to process the testing and training images
train_res = res.predict(images)
test_res = res.predict(test_images)

# split the training data into training and validation sets (70/30 split) with the seed 1399
train_index, val_index = train_test_split(range(len(train_res)), test_size=0.3, random_state=1399)
X_train, X_val = train_res[train_index], train_res[val_index]
y_train, y_val = y[train_index], y[val_index]

# creates the XGBoost model, giving the parameters to fine tune it, along with specifying early stopping & evaluation metric and ensuring it uses the GPU instead of the CPU for performance
model = xgb.XGBClassifier(max_depth=2, n_estimators=3000, learning_rate=0.01, subsample=0.75, colsample_bytree=0.9, tree_method='gpu_hist', eval_metric='mlogloss', early_stopping_rounds=40)

# performs the training and saves the results into a variable
results = model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False).evals_result()

# calculate the accuracy of the trained model using the defined validation set
accuracy = model.score(X_val, y_val)

# using the trained model, classify the unseen testing set
test_pred = model.predict(test_res)
# combines the labels with the test predictions
test_results = zip(test_names, test_pred)

# obtain the evaluation metric from the saved results
train_metric = results['validation_0']['mlogloss']

# plot the evaluation metric over time during training
plt.plot(train_metric)
plt.ylabel('mlogloss')
plt.xlabel('Iteration')
plt.text(0.5, 0.5, f'Accuracy: {accuracy:.3f}', transform=plt.gca().transAxes)
plt.show()

# open and write to a (potentially new) text file to export the testing set classification predictions
with open('run3.txt', 'w') as f:
    for r in test_results:
        f.write(r[0] + " " + str(inv_d.get(r[1])).lower() + "\n")

