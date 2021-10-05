import pandas as pd
import xarray as xr
import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

source_path = Path('P:/11207539-001-undeepwaves/')


#Initialize dictionaries and store paths
s = 0
paths = {}
errors = []
res = {}
names = {}
param = {}
for a in source_path.glob("*"):
    if a.is_dir() and str(a.parts[-1][0:5]) == 'bathy':
        paths[str(a.parts[-1])] = a
        res[str(a.parts[-1])] = {}
        names[str(a.parts[-1])] = {}
        s += 1
      
        
#Add first 200 datasets and filenames of each directory to dictionary  
for p in paths:
    s = 0
    for a in paths[p].glob("*/*"):
        if a.is_dir():
            string = a.parts[-2]
            if s >= 0 and s <= 200:
                try:
                    res[p][s] = xr.open_dataset(a.joinpath(string+'.nc'),engine='scipy')
                    names[p][s] = string
                    s += 1
                except:
                    errors.append(str(a))
                    s += 1
            #elif s<201:
                #s += 1
            else:
                break
            
#Store parameters dataframes
params = {}
for k in res:
    params[k] = pd.read_hdf(source_path.joinpath(k+'.h5'))

#P:\11207539-001-undeepwaves\bathy-gebco-b-runs\7278e0a2-d885-4726-a432-12cb491dd30e\results error

#%% saving

#import numpy as np
#np.save('C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Results/Results1.npy',res)
#np.save('C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Results/Names1.npy',names)
#np.save('C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Results/Parameters.npy',params)


#%% Loading
import numpy as np
import pandas as pd
import pathlib
from pathlib import Path
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
source_path = Path('C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Results')
source_path2 = Path('C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Analysis Results')

#source_path2 = Path('P:/11207539-001-undeepwaves/')
results1 = np.load(source_path.joinpath('Results1.npy'),allow_pickle=True).item()
results2 = np.load(source_path.joinpath('Results2.npy'),allow_pickle=True).item()
results3 = np.load(source_path.joinpath('Results3.npy'),allow_pickle=True).item()
results4 = np.load(source_path.joinpath('Results4.npy'),allow_pickle=True).item()
results5 = np.load(source_path.joinpath('Results5.npy'),allow_pickle=True).item()

names1 = np.load(source_path.joinpath('Names1.npy'),allow_pickle=True).item()
names2 = np.load(source_path.joinpath('Names2.npy'),allow_pickle=True).item()
names3 = np.load(source_path.joinpath('Names3.npy'),allow_pickle=True).item()
names4 = np.load(source_path.joinpath('Names4.npy'),allow_pickle=True).item()
names5 = np.load(source_path.joinpath('Names5.npy'),allow_pickle=True).item()

params = np.load(source_path.joinpath('Parameters.npy'),allow_pickle=True).item()

dfNoNaN = pd.read_csv(source_path2.joinpath('NoNaN.txt'),sep=" ")
dfFull = pd.read_csv(source_path2.joinpath('Full.txt'),sep=" ")
NaNIndex = np.load(source_path2.joinpath('NaNIndex.npy'), allow_pickle=True).item()

DFEmoda = pd.read_pickle(source_path2.joinpath('DataEmoda.npy'))

Tussendoortje = pd.read_pickle("C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Analysis Results/DataNoNaN2.npy")


#%% Reordering dataframes

names = [names1,names2,names3,names4,names5]
names_full = {}
for key in names1:
    temp = []    
    for i in names:
        for j in i[key]:
            temp.append(i[key][j])
    names_full[key] = temp
    
# Reorder parameter dataframe to allign with results dataset
for key in names_full:
    dftemp = pd.DataFrame(names_full[key])
    dftemp = dftemp.sort_values(0)
    DfOrd = dftemp.index.values
    params[key] = params[key].sort_values('uuid')
    params[key] = params[key].set_index(DfOrd,drop=False)
    


#%% Combining Data
import xarray as xr

# Combine the dictionaries containing the results
merge = {}
for key in results1:
    merge[key] = {**results1[key], **results2[key], **results3[key], **results4[key], **results5[key]}

#y = {}
#for key in ['bathy-emodnet-c-runs','bathy-emodnet-d-runs']:
    #y[key] = xr.concat([merge[key][i] for i in merge[key]],dim='run')
x = xr.concat([merge['bathy-emodnet-c-runs'][i] for i in merge['bathy-emodnet-c-runs']],dim='run')

#y['bathy-emodnet-c-runs'].to_netcdf("C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Results/XRDictEmodc.nc")
#np.save("C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Results/XRDict1.npy",y)

Tussendoortje = params['bathy-emodnet-a-runs']
CheeseBreak = merge['bathy-emodnet-a-runs']

Tussendoortje = Tussendoortje.drop(NaNIndex['bathy-emodnet-a-runs'])

a = np.arange(0,1016,1)
a = np.delete(a,NaNIndex['bathy-emodnet-a-runs'])

b = {}
for i in Tussendoortje.index:
    b[i] = CheeseBreak[i].hs.data[0]
    
Tussendoortje = Tussendoortje.assign(hs=b.values())

for i in Tussendoortje.index:
    Tussendoortje['bathy'][i] = Tussendoortje['bathy'][i].T
    

#Tussendoortje.to_pickle(r'C:\Users\hemert\OneDrive - Stichting Deltares\Programmas\Data\Analysis Results\DataNoNaN2.npy')
#Tussendoortje.to_csv(r"C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Analysis Results/DataNoNaN1.txt")
Tussendoortje = Tussendoortje.drop(['bathy_file','run_id','uuid','bathy_source'],axis=1)
Tussendoortje.to_hdf(r"C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Analysis Results/DataNoNaN1.h5",key='df',index = False)
Hey = pd.read_hdf(r"C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Analysis Results/DataNoNaN1.h5")

#%%
from pathlib import Path
import xarray as xr

source_path = Path("C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Results/")
path="C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Analysis Results/DataEmoda.npy"
df = pd.read_pickle(path)

hs = {}
tm01 = {}
for i in merge['bathy-schematic-a-runs']:
    hs[i] = merge['bathy-schematic-a-runs'][i].hs.data[0]
    tm01[i] = merge['bathy-schematic-a-runs'][i].tm01.data[0]
    if i%100 == 0:
        print(i)
        
'''
for i in range(800,1016):
    b[i] = merge['bathy-emodnet-a-runs'][i].hs.data[0]
    if i%100 == 0:
        print(i)
'''
        
DFemodb = params['bathy-schematic-a-runs']

c = {}
d = {}
for i in DFemodb.index:
    c[i] = hs[i]
    d[i] = tm01[i]

DFemodb = DFemodb.assign(hs = c.values(), tm01 = d.values())

for i in DFemodb.index:
    DFemodb['bathy'][i] = -DFemodb['bathy'][i].T                #Nog negatief maken voor a,b en c emod

DFemodb = DFemodb.drop(['bathy_file','run_id','uuid','bathy_source'],axis=1)
DFemodb.to_pickle(r'C:\Users\hemert\OneDrive - Stichting Deltares\Programmas\Data\Analysis Results\DataSchema.npy')

#%% Extracting features from data

#Checking for NaN in hs
NaNIndex = {}
for key in merge:
    NaNIndex[key] = []    
    for i in merge[key]:
        if np.isnan(merge[key][i].hs.data[0]).any():
            NaNIndex[key].append(i)

np.save('C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Analysis Results/NaNIndex.npy',NaNIndex)

# Extracting features from data without NaN values
minhsNoNaN = {}
minbotlNoNaN = {}
zetaNoNaN = {}
thetaNoNaN = {}
etaNoNaN = {}
maxhsNoNaN = {}
maxbotlNoNaN = {}
meanhsNoNaN = {}
meanbotlNoNaN = {}
for key in merge:
    minhsNoNaN[key], minbotlNoNaN[key], meanhsNoNaN[key], meanbotlNoNaN[key] = [], [], [], []
    zetaNoNaN[key], thetaNoNaN[key], etaNoNaN[key] = [], [], [], []
    for i in merge[key]:
        if i not in NaNIndex[key]:
            minhsNoNaN[key].append(np.min(merge[key][i].hs.data[0]))
            minbotlNoNaN[key].append(np.min(merge[key][i].botl.data[0]))
            maxhsNoNaN[key].append(np.max(merge[key][i].hs.data[0]))
            maxbotlNoNaN[key].append(np.max(merge[key][i].botl.data[0]))
            meanhsNoNaN[key].append(np.mean(merge[key][i].hs.data[0]))
            meanbotlNoNaN[key].append(np.mean(merge[key][i].botl.data[0]))
            zetaNoNaN[key].append(params[key]['$\zeta$'][i])
            thetaNoNaN[key].append(params[key]['$\theta_{wave}$'][i])
            etaNoNaN[key].append(params[key]['$\eta$'][i])
            
# Creating dataframe containing features from datasets without NaN
minhsNoNaNFull, minbotlNoNaNFull, zetaNoNaNFull, thetaNoNaNFull, etaNoNaNFull = [], [], [], [], []
maxhsNoNaNFull, maxbotlNoNaNFull, meanhsNoNaNFull, meanbotlNoNaNFull = [], [], [], []
for key in merge:
    minhsNoNaNFull += minhsNoNaN[key]
    minbotlNoNaNFull += minbotlNoNaN[key]
    zetaNoNaNFull += zetaNoNaN[key]
    thetaNoNaNFull += thetaNoNaN[key]
    etaNoNaNFull += etaNoNaN[key]
    maxhsNoNaNFull += maxhsNoNaN[key]
    maxbotlNoNaNFull +=  maxbotlNoNaN[key]
    meanhsNoNaNFull += meanhsNoNaN[key]
    meanbotlNoNaNFull += meanbotlNoNaN[key]

dfNoNaN = {'minhs':minhsNoNaNFull, 'maxhs':maxhsNoNaNFull, 'minbotl':minbotlNoNaNFull,
           'maxbotl':maxbotlNoNaNFull, 'meanhs':meanhsNoNaNFull, 'meanbotl':meanbotlNoNaNFull,
           'zeta':zetaNoNaNFull, 'theta':thetaNoNaNFull, 'eta':etaNoNaNFull}
dfNoNaN = pd.DataFrame(dfNoNaN)

# Extracting features from full datasets
minhs, minbotl, maxhs, maxbotl, meanhs, meanbotl, zeta, theta, eta = {}, {}, {}, {}, {}, {}, {}, {}, {}
for key in merge:
    minhs[key], minbotl[key], maxhs[key], maxbotl[key] = [], [], [], []
    meanhs[key], meanbotl[key], zeta[key], theta[key], eta[key] = [], [], [], [], []
    for i in merge[key]:
        minhs[key].append(np.nanmin(merge[key][i].hs.data[0]))
        minbotl[key].append(np.nanmin(merge[key][i].botl.data[0]))
        maxhs[key].append(np.nanmax(merge[key][i].hs.data[0]))
        maxbotl[key].append(np.nanmin(merge[key][i].botl.data[0]))
        meanhs[key].append(np.nanmean(merge[key][i].hs.data[0]))
        meanbotl[key].append(np.nanmean(merge[key][i].botl.data[0]))
        
        zeta[key].append(params[key]["$\zeta$"][i])
        theta[key].append(params[key]['$\theta_{wave}$'][i])
        eta[key].append(params[key]['$\eta$'][i])

# Creating dataframe containing data from full datasets
minhsFull, minbotlFull, maxhsFull, maxbotlFull, meanhsFull, meanbotlFull, zetaFull, thetaFull, etaFull = [], [], [], [], [], [], [], [], []
for key in merge:
    minhsFull += minhs[key]
    minbotlFull += minbotl[key]
    maxhsFull += maxhs[key]
    maxbotlFull += maxbotl[key]
    meanhsFull += meanhs[key]
    meanbotlFull += meanbotl[key]
    zetaFull += zeta[key]
    thetaFull += theta[key]
    etaFull += eta[key]

dfFull = {'minhs':minhsFull, 'maxhs':maxhsFull, 'minbotl':minbotlFull,
           'maxbotl':maxbotlFull, 'meanhs':meanhsFull, 'meanbotl': meanbotlFull, 'zeta':zetaFull, 'theta':thetaFull, 'eta':etaFull}
dfFull = pd.DataFrame(dfFull)
dfFull.to_csv(r'C:\Users\hemert\OneDrive - Stichting Deltares\Programmas\Data\Analysis Results\Full.txt',sep=" ")

#%% Data Exploration
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Plotting hs and botl
fig = plt.figure(figsize=(10,7))
columns = 5
rows = 2
for i in range(1,6):
    fig.add_subplot(rows,columns,i)
    merge['bathy-emodnet-a-runs'][i-1].botl[0].plot()
    fig.add_subplot(rows,columns,i+5)
    merge['bathy-emodnet-a-runs'][i-1].hs[0].plot()

#Scatter plot min values no NaN
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(dfNoNaN['minhs'],dfNoNaN['zeta'])
ax.set_xlabel('Min hs')
ax.set_ylabel('Min botl')


fig = plt.figure(figsize=(8,3))

ax = fig.add_subplot(1,2,1)
ax.set_title('colorMap')
plt.imshow(DFEmoda['hs'][0])

qx = fig.add_subplot(1,2,2)
plt.imshow(DFEmoda['bathy'][0])

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()







