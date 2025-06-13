# weld_classifier.py

# ---------------------------
# 📦 Import Libraries
# ---------------------------
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# ---------------------------
# 🧠 Classification Categories
# ---------------------------
Categories = ['A', 'B', 'C', 'D']  # Default categories

# Optionally let user input custom categories
print("Type 'y' to enter your own categories or 'n' to use default [A, B, C, D]:")
while True:
    check = input().strip().lower()
    if check in ['y', 'n']:
        break
    print("Please enter a valid input (y/n)")

if check == 'y':
    n = int(input("Enter how many types of images you want to classify: "))
    Categories = []
    print(f"Please enter {n} category names:")
    for i in range(n):
        name = input(f"Category {i+1}: ")
        Categories.append(name)

print(f"Using categories: {Categories}")

# ---------------------------
# 📂 Load Images and Labels
# ---------------------------
datadir = '/content/drive/MyDrive/temporary'  # Update if needed
flat_data_arr = []
target_arr = []

for category in Categories:
    print(f'Loading category: {category}')
    path = os.path.join(datadir, category)
    for img_name in os.listdir(path):
        img_array = imread(os.path.join(path, img_name))
        img_resized = resize(img_array, (150, 150, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(Categories.index(category))
    print(f'Loaded category {category} successfully.')

flat_data = np.array(flat_data_arr)
target = np.array(target_arr)
df = pd.DataFrame(flat_data)
df['Target'] = target
df

# ---------------------------
# 🔀 Split Data
# ---------------------------
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=77)
print("Dataset split successfully")

# ---------------------------
# 🌲 Train Random Forest
# ---------------------------
rfc = RandomForestClassifier(n_estimators=1800)
rfc.fit(x_train, y_train)
rfc_predict = rfc.predict(x_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rfc_predict))

# ---------------------------
# 🌳 Train Decision Tree
# ---------------------------
dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)
dtree_pred = dtree.predict(x_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, dtree_pred))

# ---------------------------
# 💾 Save Trained Model
# ---------------------------
pickle.dump(rfc.fit(x_train,y_train), open('model.p','wb'))
print("Pickle is dumped successfully")

# ---------------------------
# 📷 Predict on New Image
# ---------------------------
    model = pickle.load(open("model", 'rb'))
    param_grid = {
      'n_estimators':[200,700],
      'max_features':['auto','sqrt','log2']
    }
    url=input('enter url of image')
    img = imread(url)
    plt.imshow(img)
    plt.show()
    img_resized = resize(img, (150, 150, 3))
    img_flattened = img_resized.flatten().reshape(1, -1)
    prediction = model.predict(img_flattened)
    probability = model.predict_proba(img_flattened)

    print("Prediction Probabilities:")
    for ind, val in enumerate(Categories):
        print(f'{val}: {probability[0][ind]*100:.2f}%')

    print("The predicted image is : "+Categories[model.predict(l)[0]]) print(f'Is the image a {Categories[model.predict(l)[0]]} ?(y/n)') while(True): 
b=input() 
if(b=="y" or b=="n"): 
break 
print("please enter either y or n") 
 
if(b=='n'): 
print("What is the image?") 
for i in range(len(Categories)): 
print(f"Enter {i} for {Categories[i]}") 
k=int(input()) 
while(k<0 or k>=len(Categories)): 
print(f"Please enter a valid number between 0-{len(Categories)-1}") 
k=int(input()) 
print("Please wait for a while for the model to learn from this image :)") flat_arr=flat_data_arr.copy() 
tar_arr=target_arr.copy() 
tar_arr.append(k) 
flat_arr.extend(l) 
tar_arr=np.array(tar_arr) 
flat_df=np.array(flat_arr) 
df1=pd.DataFrame(flat_df) 
df1['Target']=tar_arr 
model1=GridSearchCV(ada,param_grid) x1=df1.iloc[:,:-1] 
y1=df1.iloc[:,-1] 
x_train1,x_test1,y_train1,y_test1=train_test_split(x1,y1,test_size=0.20,random_state=77,strat ify=y1) 
d={} 
for i in model.best_params_: 
d[i]=[model.best_params_[i]] 
model1=GridSearchCV(ada_model,d) 
model1.fit(x_train1,y_train1) 
y_pred1=model.predict(x_test1) 
print(f"The model is now {accuracy_score(y_pred1,y_test1)*100}% accurate") pickle.dump(model2,open(model,'wb')) 
print("Thank you for your feedback") 

# predict_image('path_to_your_test_image.jpg')
