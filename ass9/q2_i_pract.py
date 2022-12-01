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
print(dr)

# matrix = []
# matrix.append(np.array([dr]))
matrix = dr.to_numpy()
print(matrix)

dr_transpose = np.transpose(dr)
# print(dr_transpose)
# matrix_traspose = []
matrix_traspose = dr_transpose.to_numpy()
print(matrix_traspose)
# cor_matrix = np.matmul(dr_transpose, dr)
cor_matrix = np.matmul(matrix, matrix_traspose)
cor_matrix /= 4
print(cor_matrix)
