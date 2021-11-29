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
DFEmodb = pd.read_pickle(source_path2.joinpath('DataEmodb.npy'))
DFEmodc = pd.read_pickle(source_path2.joinpath('DataEmodc.npy'))
DFEmodd = pd.read_pickle(source_path2.joinpath('DataEmodd.npy'))
DFEmode = pd.read_pickle(source_path2.joinpath('DataEmode.npy'))

DFGeba = pd.read_pickle(source_path2.joinpath('DataGeba.npy'))
DFGebb = pd.read_pickle(source_path2.joinpath('DataGebb.npy'))
DFGebc = pd.read_pickle(source_path2.joinpath('DataGebc.npy'))
DFGebd = pd.read_pickle(source_path2.joinpath('DataGebd.npy'))
DFGebe = pd.read_pickle(source_path2.joinpath('DataGebe.npy'))
DFGebf = pd.read_pickle(source_path2.joinpath('DataGebf.npy'))

DFSchema = pd.read_pickle(source_path2.joinpath('DataSchema.npy')) #Augment schematic bathymetry


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
#x = xr.concat([merge['bathy-emodnet-c-runs'][i] for i in merge['bathy-emodnet-c-runs']],dim='run')

#y['bathy-emodnet-c-runs'].to_netcdf("C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Results/XRDictEmodc.nc")
#np.save("C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Resu"lts/XRDict1.npy",y)

'''
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
'''   

#Tussendoortje.to_pickle(r'C:\Users\hemert\OneDrive - Stichting Deltares\Programmas\Data\Analysis Results\DataNoNaN2.npy')
#Tussendoortje.to_csv(r"C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Analysis Results/DataNoNaN1.txt")
#Tussendoortje = Tussendoortje.drop(['bathy_file','run_id','uuid','bathy_source'],axis=1)
#Tussendoortje.to_hdf(r"C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Analysis Results/DataNoNaN1.h5",key='df',index = False)
#Hey = pd.read_hdf(r"C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Analysis Results/DataNoNaN1.h5")

#%%
from pathlib import Path
import xarray as xr

source_path = Path("C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Results/")
path="C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Analysis Results/DataEmoda.npy"
df = pd.read_pickle(path)
folder = 'bathy-schematic-a-runs'


hs = {}
tm01 = {}
theta0 = {}
for i in merge[folder]:
    hs[i] = merge[folder][i].hs.data[0]
    tm01[i] = merge[folder][i].tm01.data[0]
    theta0[i] = merge[folder][i].theta0.data[0]
    if i%100 == 0:
        print(i)
        
'''
for i in range(800,1016):
    b[i] = merge['bathy-emodnet-a-runs'][i].hs.data[0]
    if i%100 == 0:
        print(i)
'''
        
DFSchema = params[folder]

c = {}
d = {}
e = {}
for i in DFSchema.index:
    c[i] = hs[i]
    d[i] = tm01[i]
    e[i] = theta0[i]

DFSchema = DFSchema.assign(hs = c.values(), tm01 = d.values(), theta0 = e.values())

for i in DFSchema.index:
    DFSchema['bathy'][i] = DFSchema['bathy'][i].T                

DFSchema = DFSchema.drop(['bathy_file','run_id','uuid','bathy_source'],axis=1)
#DFSchema.to_pickle(r'C:\Users\hemert\OneDrive - Stichting Deltares\Programmas\Data\Analysis Results\DataSchema.npy')

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

#%%
Dummy = DFSchema
theta_wavex = np.cos(Dummy['$\theta_{wave}$'])
theta_wavey = np.sin(Dummy['$\theta_{wave}$'])

theta0_rad = {}
theta0x = {}
theta0y = {}
for i in Dummy.index:
    theta0_rad[i] = np.deg2rad(Dummy['theta0'][i])
    theta0x[i] = np.cos(theta0_rad[i])
    theta0y[i] = np.sin(theta0_rad[i])

Dummy = Dummy.assign(theta_wavex = theta_wavex.values, theta_wavey = theta_wavey.values,
                     theta0x = theta0x.values(), theta0y = theta0y.values())

#Dummy.to_pickle(r'C:\Users\hemert\OneDrive - Stichting Deltares\Programmas\Data\Analysis Results\DataEmoda.npy')

#%% Data Exploration
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(42)

ds = DFSchema
sample = np.random.randint(0,ds['hs'].size, size = 10)
'''
Max = 999
Index = []
for i in range(ds['hs'].size):
    if np.max(ds['bathy'][i][:,0]) == 999:
        Max = np.max(ds['bathy'][i])
        Index.append(i)        

start = 20
for i in range(start,min(start+10,len(Index))):
    fig, axes = plt.subplots(ncols=3, figsize=(16, 4))
    #ds['bathy'][i][ds['bathy'][i]<5] = np.nan    
    im = axes[0].imshow(ds['bathy'][Index[i]])
    axes[0].set_title('bathy')
    plt.colorbar(im, ax=axes[0])
    im = axes[1].imshow(ds['hs'][Index[i]])
    axes[1].set_title('hs')
    plt.colorbar(im, ax=axes[1])
    im = axes[2].imshow(ds['tm01'][Index[i]])
    axes[2].set_title('tm01')
    plt.colorbar(im, ax=axes[2])
    fig.suptitle('eta = {}, zeta = {}'.format(ds['$\eta$'][i], ds['$\zeta$'][i]), fontsize=16)
'''      
for i in sample:
    fig, axes = plt.subplots(ncols=4, figsize=(16, 4))
    plt.subplots_adjust(wspace=0.5)
    #ds['bathy'][i][ds['bathy'][i]<5] = np.nan    
    im = axes[0].imshow(ds['bathy'][i])
    axes[0].set_title('bathy', fontsize=16)
    axes[0].arrow(128,128, 
                  70*np.cos(ds['$\theta_{wave}$'][i]*np.pi/180. - np.pi/2), 
                  70*np.sin(ds['$\theta_{wave}$'][i]*np.pi/180. - np.pi/2),
                  color='red', lw=3.7, overhang=0.80, head_width=6.8)
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.15)
    plt.colorbar(im, cax=cax)
    im = axes[1].imshow(ds['hs'][i])
    axes[1].set_title('hs', fontsize=16)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.15)
    plt.colorbar(im, cax=cax)
    #plt.colorbar(im, ax=axes[1])
    mask = np.zeros(ds['hs'][i].shape)
    mask = np.logical_or(mask, np.isnan(ds['hs'][i]))
    #im = axes[2].imshow(mask)
    #axes[2].set_title('mask')
    #plt.colorbar(im, ax=axes[2])
    im = axes[2].imshow(ds['tm01'][i])
    axes[2].set_title('tm01', fontsize=16)
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.15)
    plt.colorbar(im, ax=axes[2], cax=cax)
    #plt.colorbar(im, ax=axes[2])
    im = axes[3].imshow(np.deg2rad(ds['theta0'][i]))
    axes[3].set_title(r'$\theta_0$', fontsize=16)
    divider = make_axes_locatable(axes[3])
    cax = divider.append_axes("right", size="5%", pad=0.15)
    plt.colorbar(im, ax=axes[3], cax=cax)
    #plt.colorbar(im, ax=axes[3])
    #fig.suptitle(f'eta = {}, zeta = {}, theta_wave = {}')
    fig.suptitle('$\eta$ = {:.2f}m, $\zeta$ = {:.2f}m, $\\theta_{{wave}}$ = {:.2f}$\degree$, $\\theta_{{bathy, z}}$ = {:.2f}$\degree$'.format(ds['$\eta$'][i], ds['$\zeta$'][i], ds['$\theta_{wave}$'][i], ds['$\theta_{bathy, z}$'][i]), fontsize=16)

#%%
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







