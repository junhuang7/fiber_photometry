from time import perf_counter 
#Start the stopwatch / counter 
tic = perf_counter()

def downsample_file(file_to_open, file_to_save):
    
        # Load the Pandas libraries with alias 'pd' 
        import numpy as np
        import pandas as pd
        
    
        data = pd.read_csv(file_to_open) 
        df =pd.DataFrame(data)
        
        len_df=len(df.columns)
        # Delete columns at index 1 & 2
        df = df.drop(df.columns[len_df-1] ,  axis='columns')
        
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
        
        
        time_df = np.array(df['Time(s)'])
        index_ds = np.zeros(len(rg)) # Pre-allocate matrix
        index_ds = index_ds.astype(int)
        
        for i in rg:
             indice=(time_df[0:16]>0.001*i).astype(np.int)
             if i == 0:
              index_ds[i] = indexOfFirstOne(indice, len(indice))
             else:
              index_ds1 = indexOfFirstOne(indice, len(indice))
              index_ds[i] = index_ds1
              time_df = time_df[index_ds[i]:]
        
        index_ds[0]=0
        for i in range(len(index_ds)-1):
            index_ds[i+1]=index_ds[i+1] + index_ds[i]
        
        df_ds = pd.DataFrame(df.loc[index_ds,:]) 
        # saving the dataframe 
        df_ds.to_csv(file_to_save, index=False) 

import os
from pathlib import Path
data_folder = Path("D:/")
file_to_open = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
file_to_save = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
  
for i in range(len(file_to_open)):
    file_to_open[i]= os.path.join(data_folder, file_to_open[i])
    file_to_save[i] = "DS_" + file_to_save[i]
    file_to_save[i] =os.path.join(data_folder, file_to_save[i])
print(file_to_save)


for i in range(len(file_to_open)):
    downsample_file(file_to_open[i],file_to_save[i])
    print (file_to_save[i], "done, Jun did a great job `#^_^#′ ")

#Stop the stopwatch / counter 
toc = perf_counter()
print("Elapsed time during the whole program in seconds:", toc-tic) 