from sklearn.cluster import KMeans
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


ds = pd.read_csv('./kMeans_dataset.csv')
# ds


# In[3]:


X = pd.DataFrame({'X': ds['A'],
                  'Y': ds['B'],
                  })
np.random.seed(200)
k = 3

centroids = {
    i+1: [np.random.randint(0, 10), np.random.randint(0, 10)] for i in range(k)
}

fig = plt.figure(figsize=(5, 5))
plt.scatter(X['X'], X['Y'], color='k')
colmap = {1: 'r', 2: 'g', 3: 'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.show()


# In[4]:


def assignment(ds, centroids):
    for i in centroids.keys():

        ds['distance_from_{}'.format(i)] = (np.sqrt(
            (ds['X'] - centroids[i][0])**2 + (ds['Y'] - centroids[i][1])**2
        ))

    centroid_distance_cols = [
        'distance_from_{}'.format(i) for i in centroids.keys()]
    ds['closest'] = ds.loc[:, centroid_distance_cols].idxmin(axis=1)
    ds['closest'] = ds['closest'].map(
        lambda x: int(x.lstrip('distance_from_')))
    ds['color'] = ds['closest'].map(lambda x: colmap[x])
    return ds


X = assignment(X, centroids)
print(X.head())

fig = plt.figure(figsize=(5, 5))
plt.scatter(X['X'], X['Y'], color=X['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.show()


# In[5]:

old_centroids = copy.deepcopy(centroids)


def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(X[X['closest'] == i]['X'])
        centroids[i][1] = np.mean(X[X['closest'] == i]['Y'])
    return k


centroids = update(centroids)

fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(X['X'], X['Y'], color=X['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids, color=colmap[i])
plt.xlim(0, 20)
plt.ylim(0, 20)

for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]

    dx = (centroids[i][0] - old_centroids[i][0])*0.75
    dy = (centroids[i][1] - old_centroids[i][1])*0.75
#	ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc=colmap[i], ec=colmap[i])
plt.show()

X = assignment(X, centroids)

fig = plt.figure(figsize=(5, 5))
plt.scatter(X['X'], X['Y'], color=X['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.show()


# In[7]:


while True:
    closest_centroids = X['closest'].copy(deep=True)
    centroids = update(centroids)
    X = assignment(X, centroids)
    if closest_centroids.equals(X['closest']):
        break

fig = plt.figure(figsize=(5, 5))
plt.scatter(X['X'], X['Y'], color=X['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 15)
plt.ylim(0, 15)
plt.show()


# In[8]:


closest_centroids


# In[9]:


Y = pd.DataFrame({'X': ds['A'],
                  'Y': ds['B'],
                  })

kmeans = KMeans(n_clusters=2)
kmeans.fit(Y)


# In[10]:


labels = kmeans.predict(Y)
centroids = kmeans.cluster_centers_


# In[11]:


fig = plt.figure(figsize=(5, 5))

colors = map(lambda x: colmap[x+1], labels)

plt.scatter(Y['X'], Y['Y'], color=list(colors), alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(0, 15)
plt.ylim(0, 15)
plt.show()


# In[12]:


centroids
