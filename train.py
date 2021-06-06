import cv2
import pickle
import numpy as np
from imutils import paths
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix,accuracy_score

data = []
labels = []

for label in range(2):
    im_paths = np.array(list(paths.list_images(f"{label}/")))
    for i,path in enumerate(im_paths):
        image = cv2.imread(path,0)
        data.append(image)
        labels.append(label)
print(f"Total images found : {len(data)}")
data,labels = np.array(data), np.array(labels)
data = data.reshape(-1,64*64) / 255.0
x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.25)
svm = SVC(kernel='rbf',C=5,gamma=0.005,probability=True)
print("Traing has started...")
svm.fit(x_train, y_train)
print("Training has ended")
print("Evaluation has started")
preds = svm.predict(x_test)
cm = confusion_matrix(y_test, preds)
print(classification_report(y_test, preds))
plot_confusion_matrix(svm,x_test,y_test)
print("Accuracy:",accuracy_score(y_test,preds))
plt.show()
pickle.dump(svm, open("smile_svm.sav", 'wb'))
