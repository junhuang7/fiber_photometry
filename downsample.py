# developed by Jun Huang on
# updated by Jun Huang on 20.05.24
from time import perf_counter
tstart = perf_counter()
import numpy as np
#import cupy as cp
import pandas as pd
import gc
import os
from pathlib import Path
data_folder = Path("E:/data/")
file_to_open = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
file_to_save = [f for f in os.listdir(data_folder) if f.endswith('.csv')]

for i in range(len(file_to_open)):
    file_to_open[i]= os.path.join(data_folder, file_to_open[i])
    file_to_save[i] = "DS_" + file_to_save[i]
    file_to_save[i] =os.path.join(data_folder, file_to_save[i])

print(file_to_save)
gc.collect()

def downsample_file(file_to_open, file_to_save):
        
        data = pd.read_csv(file_to_open, error_bad_lines=False) 
        df =pd.DataFrame(data)
        del data
        gc.collect()
        len_df=len(df.columns)
        # Delete columns at index 1 & 2
        df = df.drop(df.columns[len_df-1] ,  axis='columns')
        
        if 'AOut-1' in df.columns:
            df = df.drop(['AOut-1'], axis=1)

        if 'AOut-2' in df.columns:
            df = df.drop(['AOut-2'], axis=1)

        if 'AOut-3' in df.columns:
            df = df.drop(['AOut-3'], axis=1)

        if 'AOut-4' in df.columns:
            df = df.drop(['AOut-4'], axis=1)

        time_df = np.array(df['Time(s)'])
        
        rg=range(int(df['Time(s)'].tail(1)) * 1000)
        
        # Python3 implementation to find  
        # the index of first '1' in a  
        # sorted array of 0's and 1's 
          
        # function to find the index of first '1' 
        def indexOfFirstOne(indice, n): 
          
            # traverse the array from left to right 
            for i in range(0, n): 
                  
                # if true, then return i 
                if (indice[i] == 1): 
                    return i 
          
            # 1's are not present in the array 
            return -1
        
        
        index_ds = np.zeros(len(rg)) # Pre-allocate matrix
        index_ds = index_ds.astype(int)
        
        for i in rg:
             indice=(time_df[0:16]>0.001*i).astype(np.int)
             index_ds1 = indexOfFirstOne(indice, len(indice))
             index_ds[i] = index_ds1
             time_df = time_df[index_ds[i]:]
        
        def zero_runs(a):
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
            iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
            absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
            ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
            return ranges
        
        gap=zero_runs(index_ds)
        gap=gap-1
        gap_len=gap[:,1] - gap[:,0] + 1
        
        for i in range(len(index_ds)-1):
            index_ds[i+1]=index_ds[i+1] + index_ds[i]
        
        df_np=df.to_numpy()
        df_ds = pd.DataFrame(df.loc[index_ds,:])*0
        df_ds_np=df_ds.to_numpy()

        for i in range(len(index_ds)-1):
            if index_ds[i] == index_ds[i+1]:
               df_ds_np[i,:]= df_np[index_ds[i],:]
            else:
               df_ds_np[i,:]=np.mean(df_np[index_ds[i]:index_ds[i+1],:], axis=0)
        
        if gap.size != 0:
            for i in range(gap.shape[0]-1):
                df_ds_np[gap[i,0]:gap[i,1]+1,:]=np.linspace(df_ds_np[gap[i,0]],df_ds_np[gap[i,1]],gap_len[i])
        
        timevec_ds=np.arange(0.001,float(int(df['Time(s)'].tail(1)))+0.001,0.001)
        
        del df_np
        gc.collect()
        df_ds = pd.DataFrame(df_ds_np)
        df_ds = pd.DataFrame(data=df_ds.values, columns=df.columns)
        df_ds.iloc[-1,:]=pd.DataFrame.mean(df.loc[index_ds[-1]:index_ds[-1]+12,:])
        df_ds['Time(s)']=timevec_ds
        del df, len_df, rg, time_df, index_ds, df_ds_np, timevec_ds
        df_ds["DI/O-1"]=(df_ds["DI/O-1"]>0.5)*1
        df_ds["DI/O-2"]=(df_ds["DI/O-2"]>0.5)*1
        
	    # saving the dataframe
        df_ds.to_csv(file_to_save, index=False)
        del df_ds
        gc.collect()

for i in range(len(file_to_open)):
    #Start the stopwatch / counter 
    tic = perf_counter()
    downsample_file(file_to_open[i],file_to_save[i])
    #Stop the stopwatch / counter 
    toc = perf_counter()
    print (file_to_save[i], "done, Time:", toc-tic, "seconds, great job `#^_^#′!")

#print("Elapsed time during the whole program in seconds:", toc-tic)
tend = perf_counter()
print("All done, Time:", tend-tstart, "seconds, now your computer needs a rest `#^_^#′!")