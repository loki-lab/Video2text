import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import pickle
import cv2


class Dataset:
    def __init__(self, folder_dataset):
        self.folder_dataset = folder_dataset
        self.list_file = os.listdir(folder_dataset)

    def __len__(self):
        return len(self.list_file)

    def __getitem__(self, index):
        path = os.path.join(self.folder_dataset, self.list_file[index])
        image = cv2.imread(path)
        if "down" in path:
            label = 0
        else:
            label = 1
        return image, label


my_dataset = Dataset("triangle_output")

encoder = OneHotEncoder()
list_img = []
list_label = []
for img, label in my_dataset:
    list_img.append(np.reshape(img, (100 * 100 * 3)))
    list_label.append(label)

data, label = shuffle(list_img, list_label, random_state=10)


data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.2)


model = KNeighborsClassifier(n_neighbors=2)
model.fit(data_train, label_train)

y_ = model.predict(data_test)

acc = accuracy_score(label_test, y_)
print(acc)

pickle.dump(model, open("knn_model.sav", "wb"))