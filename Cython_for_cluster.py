#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


NUM_NEIGHB = 27

def get_site(coord, L):
    """Get the site index from the 3-vector of coordinates."""
    # XXX: 3D hardcoded, can do N-D
    return coord[0] * L[1] * L[2] + coord[1] * L[2] + coord[2]


def get_coord(site, L):
    """Get the 3-vector of coordinates from the site index."""
    # XXX: 3D hardcoded, can do N-D
    x = site // (L[1]*L[2])
    yz = site % (L[1]*L[2])
    y = yz // L[2]
    z = yz % L[2]
    return [x, y, z]


def get_neighbors(site, L):
    neighb = set()
    x, y, z = get_coord(site, L)
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                x1 = (x + i) % L[0]
                y1 = (y + j) % L[1]
                z1 = (z + k) % L[2]
                neighb.add(get_site([x1, y1, z1], L))
    
    return list(neighb)
    
def tabulate_neighbors(L):
    """Tabulate the root-2 neighbors on the 3D cubic lattice with PBC."""
    Nsite = L[0]*L[1]*L[2]
    neighb = np.empty((Nsite, NUM_NEIGHB), dtype=int)
    for site in range(Nsite):
        neighb[site, :] = get_neighbors(site, L)
    return neighb


# In[4]:


import evolve3D_2_C


# In[5]:


B = set()
for i in range(28):
    for j in range(1, 28):
        if i/j > 1:
            B.add(i/j + 0.0000000001)
B = sorted(list(B))
len(B)


# In[6]:


SIZE = 60
L = (SIZE, SIZE, SIZE)
neighbors = tabulate_neighbors(L)


# In[9]:


rndm = np.random.RandomState(17)
C_PROB = 0.9 # test 0.1, 0.5, 0.9
N_STEPS = 100 
N_MEASUR = 100 
N_FIELDS = 1 
BURN_IN_STEPS = 2000
DIR_NAME = "RUN_10_all_B"

for i in range(len(B)):
    b = B[i]
    for k in range(N_FIELDS):
        results = np.zeros((N_MEASUR), dtype=float)
        field = (rndm.uniform(size=L) > C_PROB).astype('int16')
        field = evolve3D_2_C.evolve3D_2_C(field, neighbors, b, num_steps=BURN_IN_STEPS)
        for measure in range(N_MEASUR):
            field = evolve3D_2_C.evolve3D_2_C(field, neighbors, b, num_steps=N_STEPS)
            results[measure] =  1 - (field.sum() / (L[0] * L[1] * L[2]))
        fname = DIR_NAME + "/" + str(i) + "_" + str(k)
        np.save(fname, results) 





