#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install idx2numpy


# In[ ]:





# In[ ]:


import gzip
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from sklearn.preprocessing import normalize


# In[ ]:





# In[21]:


import idx2numpy

imagefile = 't10k-images-idx3-ubyte'
train_x = idx2numpy.convert_from_file(imagefile)

imagefile = 't10k-labels-idx1-ubyte'
train_y = idx2numpy.convert_from_file(imagefile)

imagefile = 'train-images-idx3-ubyte'
test_x = idx2numpy.convert_from_file(imagefile)

imagefile = 'train-labels-idx1-ubyte'
test_y = idx2numpy.convert_from_file(imagefile)


# In[41]:


from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 7))
rows = 3
columns = 6

j = 1 
for i in range(0,30):
    fig.add_subplot(rows, columns, j)
    plt.imshow(train_x[i])
    plt.axis('off')
    plt.title(str(train_y[i]))
    j += 1


# In[ ]:


selected_images_indices = []
for i in range(0, 10):
    label_df = df[df['label'] == i]
    selected_ids = list(np.random.choice(label_df.index, size=10))
    selected_images_indices.extend(selected_ids)
df_with_100_images = df[df.index.isin(selected_images_indices)].reset_index(0, drop=True)


# In[ ]:





# In[ ]:


def training_images(sample):  # for extracting images
    with gzip.open(sample, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)            .reshape((image_count, row_count, column_count))
        return images
    
def training_labels(): # for extracting labels
    with gzip.open(sample, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels


# In[ ]:


train_x = get_images('train-images-idx3-ubyte.gz')
train_y = get_labels('train-labels-idx1-ubyte.gz')
test_y = get_images('t10k-images-idx3-ubyte.gz')
test_y = get_labels('t10k-labels-idx1-ubyte.gz')


# In[ ]:





# In[ ]:


train_x = np.concatenate((train_x[train_y==5], train_x[ train_y==6], train_x[train_y==8]))
test_x = np.concatenate((test_x[test_y==5], test_x[test_y==6], test_x[test_y==8]))
train_y = np.concatenate((train_y[train_y==5], train_y[ train_y==6], train_y[train_y==8]))
test_y = np.concatenate((test_y[test_y==5], test_y[test_y==6], test_y[test_y==8]))


# In[ ]:





# In[ ]:


train_x = train_x[np.random.random_integers(0,len(train_x)-1, 15000)]
train_y = train_y[np.random.random_integers(0,len(train_y)-1, 15000)]
test_y = test_x[np.random.random_integers(0,len(test_x)-1, 2500)]
test_y = test_y[np.random.random_integers(0,len(test_y)-1, 2500)]


# In[ ]:





# In[ ]:


def dist(Xtraining, Xtesting):
    distance =[[None for _ in range(len(Xtesting))]for _ in range(len(Xtraining))]
    for idxi, imgi in enumerate(Xtraining):
        for idxj, imgj in enumerate(Xtesting):
            distance[idxi][idxj]=np.linalg.norm(imgi-imgj) 
    return distance


# In[ ]:





# In[ ]:


def retrieve_k(i, M, k):
    s=[]
    for row, val in enumerate(M):
        s.append(val[i])
    ls = []
    s1=sorted(s)
    while len(ls)<k:
        if [ls.index(s1[0]), s1[0]] not in final_list:
            ls.append([ls.index(s1[0]), s1[0]])
        s1.pop(0)
    return ls


# In[ ]:





# In[ ]:


def precision_K(y, ytrain, I):
    vals=[ytrain[i] for i in I]
    prec=0
    for i in vals:
        if i==y:
            prec+=1
    prec/=len(I)
    return prec

def avg_precision_k(Xtraining, ytrain, Xtesting, ytest, k):
    M = dist(Xtraining, Xtesting)
    precs=[]
    for test_idx, test_val in enumerate(Xtesting):
        s_time=time.time()
        indices=[i[0] for i in retrieve_k(test_idx, M, k)]
        precision=precision_K(ytest[test_idx], ytrain, indices)
        precs.append(precision)
    return sum(precs)/len(Xtesting)


# In[ ]:





# In[ ]:


avg=[]
b_time=time.time()
for k in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
    iteration_time=time.time()
    avg.append(avg_precision_k(norm_X, trainY, norm_X, testY, k))
    print(f"K = {k} finished in {time.time()-itr_time}")
print(f"Run time: {time.time()-b_time}")


# In[ ]:





# In[ ]:


plt.title("Average Precision vs K ")
plt.xlabel("K")
plt.ylabel("Average Precision")
plt.plot([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], avg)
plt.savefig("avg_prec.jpg")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




