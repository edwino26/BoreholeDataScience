import glob
import lasio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#Clustering packages
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.model_selection import train_test_split


files = glob.glob('./*.las')
las = lasio.read(files[0])

well = las.well
headers = las.curves
params = las.params
logs = las.data
No_logs = len(headers)
dims = las.data.shape

print(dims)
i=0
for curve in las.curves:
    print(curve.mnemonic + ": " + str(curve.data) + " " + str(i))
    i += 1

data = pd.DataFrame(las.data)

DEPTH = las.index
GR = data[53] #las["GR_EDTC"]
RESD = data[225] #las["AT90"]
RHOB = data[108] #las["RHOZ"]
NPHI= data[96] #las["NPHI"]


BD = 3000
TD = 3880


# ======== Manual Neural Net ============




plt.figure()
plt.subplot(141)
plt.plot(GR, DEPTH, 'green'); plt.axis([0, 120, BD, TD]); plt.gca().invert_yaxis()
plt.subplot(142)
plt.plot(RESD, DEPTH); plt.axis([0.1, 100, BD, TD]); plt.gca().invert_yaxis();plt.gca().yaxis.set_visible(False); plt.grid(True,which='minor',axis='x'); plt.xscale('log')
plt.subplot(143)
plt.plot(RHOB, DEPTH, 'red'); plt.axis([1.65, 2.65, BD, TD]); plt.gca().invert_yaxis(); plt.gca().yaxis.set_visible(False)
plt.subplot(144)
plt.plot(NPHI, DEPTH, 'blue')
plt.gca().invert_yaxis(); plt.axis([0.6, 0, BD, TD]); plt.gca().invert_yaxis();plt.gca().yaxis.set_visible(False)

plt.suptitle('Well logs for ' + las.well['WELL']['value'])

# Plot Input Logs
#plt.show()

#Explore
#data_o = data.dropna()
print(data.head())
data_o = data.iloc[:, [53, 225, 108, 96]]  #Removed Depth 0
data_o = data_o.dropna()
data_o = data_o.rename(columns={53: "GR", 225: 'RESD', 108: "RHOB", 96: "NPHI"})
print("Size is:")
print(data_o.size)
print("Header")
print(data_o.head())

print(data_o.shape)

#CLustering
#https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
#https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c


train, test = train_test_split(data_o, test_size=0.2)

inputs = train
print(inputs)
weights = np.random.rand(train.size)

bias = 3

ii = [1, 2, 3, 2.5]
ww1 = [0.2, 0.8, -0.5, 1.0]
ww2 = [0.5, -0.91, -0.26, -0.5]
ww3 = [-0.26, -0.27, 0.17, 0.87]
b1 = 2
b2 = 2
b3 = 2


output = [ii[0]*ww1[0] + ii[1]*ww1[1] + ii[2]*ww1[2] + ii[3]*ww1[3] + b1,
          ii[0]*ww2[0] + ii[1]*ww2[1] + ii[2]*ww2[2] + ii[3]*ww2[3] + b2,
          ii[0]*ww3[0] + ii[1]*ww3[1] + ii[2]*ww3[2] + ii[3]*ww3[3] + b3]

print(output)




