
# coding: utf-8

# In[8]:


import keras
import cv2
from keras.models import load_model, Model
import numpy as np


# In[9]:


TRAINED_MODEL_PATH ="Model-path"
model = load_model(TRAINED_MODEL_PATH)


# In[10]:


file_path = "file path"


# In[11]:


img = cv2.imread(file_path )


# In[12]:


imgs = cv2.resize(img, (300,300))


# In[13]:


img1 = imgs.reshape(1,300,300,3)


# In[15]:


predicted_classes = []
predicted_probs = []
classes = ['symptoms','nosymptoms'] 
num_classes = len(classes)
print (classes)
probabilities=model.predict(img1)
sorted_prob_idxs = (-probabilities).argsort()[0]
predicted_prob = np.amax(probabilities)
predicted_probs.append(predicted_prob)
predicted_class = classes[sorted_prob_idxs[0]]
predicted_classes.append(predicted_class)
print (probabilities)
print predicted_class

