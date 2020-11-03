import glob
import lasio
import matplotlib.pyplot as plt
import pandas as pd


#Clustering packages
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d


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
data_o = data.iloc[:, [0, 53, 225, 108, 96]]
data_o = data_o.dropna()
data_o = data_o.rename(columns={0: 'DEPTH', 53: "GR", 225: 'RESD', 108: "RHOB", 96: "NPHI"})
print("Size is:")
print(data_o.size)
print("Header")
print(data_o.head())


#CLustering
#https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html