#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
print(os.listdir("Desktop/Projet_Harispe/ChestXRay2017/chest_xray/"))
# On voudrait s'assurer que toutes les images manipulées soient de même dimension si jamais ce n'est pas initialement le cas.
taille_img = 128


# In[41]:


# On va stocker les images et les labels récupérés dans des arrays
# Déclaration de listes vides
train_images = []
train_labels = [] 
# On déclare le lien vers les images d'entrainement
for directory_path in glob.glob("Desktop/Projet_Harispe/ChestXRay2017/chest_xray/train/*"):
    # On récupère les noms des labels
    label = directory_path.split("\\")[-1]
    # Et on les visualise pour s'assurer que tout marche bien pour l'instant.
    print(label)
    
    for img_path in glob.glob(os.path.join(directory_path, "*.jpeg")):
            print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) #Reading color images
            img = cv2.resize(img, (taille_img, taille_img)) #Resize images
            
            train_images.append(img)
            train_labels.append(label)
        
train_images = np.array(train_images)
train_labels = np.array(train_labels)
print(train_images)
print(train_labels) 


# In[42]:


# On va stocker les images et les labels récupérés dans des arrays (on fait comme pour les images d'entraînement)
# Déclaration de listes vides
test_images = []
test_labels = [] 
# On déclare le lien vers les images de test
for directory_path in glob.glob("Desktop/Projet_Harispe/ChestXRay2017/chest_xray/test/*"):
    # On récupère les noms des labels
    label = directory_path.split("\\")[-1]
    # Et on les visualise pour s'assurer que tout marche bien pour l'instant.
    print(label)

    for img_path in glob.glob(os.path.join(directory_path, "*.jpeg")):
            print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR) #Reading color images
            img = cv2.resize(img, (taille_img, taille_img)) #Resize images
    
            test_images.append(img)
            test_labels.append(label)
# On stocke les liens des images et les labels dans des array
test_images = np.array(test_images)
test_labels = np.array(test_labels)
# On print pour vérifier que tout est ok
print(test_images)
print(test_labels) 

# On encode les labels sous forme d'entier (facilite l'interprétation par la machine)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

# On sépare les données et labels en données d'entrainement et de test
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

# On normalise les valeurs de pixels entre 0 et 1
x_train, x_test = x_train / 255.0, x_test / 255.0
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)


# In[50]:


#
def feature_extractor(dataset):
    num_images = dataset.shape[0]
    image_size = dataset.shape[1] * dataset.shape[2] * dataset.shape[3]
    image_dataset = np.zeros((num_images, image_size))

    for image in range(num_images):
        input_img = dataset[image, :, :, :]
        pixel_values = input_img.reshape(-1)
        image_dataset[image, :] = pixel_values

    return image_dataset


# In[49]:


#Extrait les valeurs de caractéristiques des images d'entraînement
image_features = feature_extractor(x_train)

# Met les caractéristiques sous forme de vecteur pour les modèles RandomForest et arbre de décision
n_features = image_features.shape[1]
image_features = np.expand_dims(image_features, axis=0)
X_for_RF = np.reshape(image_features, (x_train.shape[0], -1))

# On importe les classificateurs RandomForest et DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators=100, random_state=42)
estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)

# On fit chaque modèle sur les données d'entraînement
RF_model.fit(X_for_RF, y_train.ravel()) 
estimator.fit(X_for_RF, y_train.ravel())

# Prédictions sur les données test
# On extrait les features des données test, comme pour les données d'entraînement, et on les reshape en vecteur
test_features = feature_extractor(x_test)
test_features = np.expand_dims(test_features, axis=0)
test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))

# Prédictions sur les données tests avec les deux modèles
test_prediction = RF_model.predict(test_for_RF)
test_prediction1 = estimator.predict(test_for_RF)
# On décode les valeurs numériques de label en valeur catégorielle 
test_prediction = le.inverse_transform(test_prediction)
test_prediction1 = le.inverse_transform(test_prediction1)

# Impression de la métrique d'exactitude "accuracy" pour chaque modèle
from sklearn import metrics
print("Random Forest Accuracy = ", metrics.accuracy_score(test_labels.ravel(), test_prediction))
print("Decision Tree Accuracy = ", metrics.accuracy_score(test_labels.ravel(), test_prediction1))

# On imprime deux matrices de confusion (une par modèle) 
#afin d'évaluer la précision et le rappel du modèle (faux positifs et faux négatifs)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels.ravel(), test_prediction)
fig, ax = plt.subplots(figsize=(6,6))         # Sample figsize in inches
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, ax=ax)

cm1 = confusion_matrix(test_labels.ravel(), test_prediction1)
fig, ax = plt.subplots(figsize=(6,6))         # Sample figsize in inches
sns.set(font_scale=1.6)
sns.heatmap(cm1, annot=True, ax=ax)

# On effectue les prédictions sur chaque image de l'ensemble test
for i in range(x_test.shape[0]):
    img = x_test[i]
    # Extraction des features et reshaping dans les bonnes dimensions
    input_img = np.expand_dims(img, axis=0) # On rajoute une dimension pour que l'input soit de la forme(num images, x, y, c)
    input_img_features=feature_extractor(input_img)
    input_img_features = np.expand_dims(input_img_features, axis=0)
    input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))

    # Predictions pour chaque modèle
    img_prediction = RF_model.predict(input_img_for_RF)
    img_prediction = le.inverse_transform([img_prediction])  #Reverse the label encoder to original name
    print("The prediction for image ", i, " is: ", img_prediction)
    print("The actual label for image ", i, " is: ", test_labels[i])
    plt.imshow(img)
    plt.show()
    
    img_prediction1 = estimator.predict(input_img_for_RF)
    img_prediction1 = le.inverse_transform([img_prediction1])  #Reverse the label encoder to original name
    print("The prediction for image ", i, " is: ", img_prediction1)
    print("The actual label for image ", i, " is: ", test_labels[i])
    #plt.imshow(img) pour voir l'image associée à la prédiction
    #plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




