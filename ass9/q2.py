from numpy import linalg as la
import numpy as np
import pandas as pd


# In[2]:


ds = pd.read_csv('./pca_dataset.csv')
# ds


# In[16]:


dt = pd.DataFrame({"1": ds['English'],
                   "2": ds['Hindi'],
                   "3": ds['Maths'],
                   "4": ds['Science']
                   })

mean_1 = dt["1"].mean()
mean_2 = dt["2"].mean()
mean_3 = dt["3"].mean()
mean_4 = dt["4"].mean()

dr = pd.DataFrame({
    "1": dt["1"] - mean_1,
    "2": dt["2"] - mean_2,
    "3": dt["3"] - mean_3,
    "4": dt["4"] - mean_4,
})

# mean_1, mean_2, mean_3, mean_4, dr


# In[17]:


# matrix, np.transpose(matrix) np.array(dr).shape
# matrix = []
# matrix.append(np.array([dr]))
# matrix


# In[18]:
matrix = dr.to_numpy()
# print(matrix)

dr_transpose = np.transpose(dr)
# print(dr_transpose)
# matrix_traspose = []
matrix_traspose = dr_transpose.to_numpy()
# print(matrix_traspose)
# cor_matrix = np.matmul(dr_transpose, dr)
cor_matrix = np.matmul(matrix, matrix_traspose)
cor_matrix /= 4
print(cor_matrix)

# cor_matrix = np.matmul(np.transpose(dr), dr)
# cor_matrix = np.matmul(np.transpose(dr), dr)
# cor_matrix /= 4
# cor_matrix


# In[19]:


# In[20]:


eig_val, eig_vect = la.eig(cor_matrix)
# eig_val, eig_vect


# In[21]:


max_component = max(eig_val)
# max_component


# In[22]:


eig_vect[0]


# In[26]:


final_dataset = [np.matmul(matrix, eig_vect[i]) for i in range(4)]
final_dataset
