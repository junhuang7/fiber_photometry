#created by Jun Huang

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# led_onset_frame=np.array([42, 96,   66,   104,   199,  112,   58,  183,  2520,   11,     73,    66,    106,   40,    40])
# led_onset=led_onset_frame/25
# snif_trim=np.array([89.28, 90.92, 92.52, 83, 107.24, 100.6, 85.64, 97.84, 397.28, 92.36, 110.53, 83.04, 87.88, 81.48,   83.68])
# mice_id= np.array([33024, 33025, 33026, 33027, 33028, 33032, 33033, 33034, 33037, 33052, 33053, 33055, 33056, 33057,  33062])
#special case mice_ID 33054, no observer data is shown, so the relative value is taken away for the moment (led_onset_frame 61, led_onset xxx)

# led_onset_frame=np.array([42, 96,   66,   104,   199,  112,   58,  183,    11,     73,    66,    106,   40,    40])
# led_onset=led_onset_frame/25
# snif_trim=np.array([89.28, 90.92, 92.52, 83, 107.24, 100.6, 85.64, 97.84,  92.36, 110.53, 83.04, 87.88, 81.48,   83.68])
# mice_id= np.array([33024, 33025, 33026, 33027, 33028, 33032, 33033, 33034,  33052, 33053, 33055, 33056, 33057,  33062])
# #special case mice_ID 33037, no observer data is shown, so the relative value is taken away for the moment (led_onset_frame 2520, snif_trim 397.28)


led_onset_frame=np.array([42, 96,   66,   104,   199,  112,   58,  183,    11,     73,    66,    40,    40])
led_onset=led_onset_frame/25
snif_trim=np.array([89.28, 90.92, 92.52, 83, 107.24, 100.6, 85.64, 97.84,  92.36, 110.53, 83.04, 81.48,   83.68])
mice_id= np.array([33024, 33025, 33026, 33027, 33028, 33032, 33033, 33034,  33052, 33053, 33055, 33057,  33062])
#special case mice_ID 33056, no urine sniffing, so the relative value is taken away for the moment (led_onset_frame 106, snif_trim 87.88)

# led_onset_frame=np.array([42, 96,   66,   104,   199,  112,   58,  183, 2542,    11,     73,    66,    40,    40])
# led_onset=led_onset_frame/25
# snif_trim=np.array([89.28, 90.92, 92.52, 83, 107.24, 100.6, 85.64, 97.84, 397.28, 92.36, 110.53, 83.04, 81.48,   83.68])
# mice_id= np.array([33024, 33025, 33026, 33027, 33028, 33032, 33033, 33034, 33037, 33052, 33053, 33055, 33057,  33062])
# #special case mice_ID 33056, no urine sniffing, so the relative value is taken away for the moment (led_onset_frame 106, snif_trim 87.88)


events_folder=r'C:\data\fust\snif_events'
events_files = glob.glob(events_folder + '/*.xlsx', recursive=True)
recordings_folder=r'C:\data\fust\recordings'
recordings = glob.glob(recordings_folder + '/*.csv', recursive=True)

dFF0_water_1=[]
dFF0_urine_1=[]
dFF0_water_2=[]
dFF0_urine_2=[]
dFF0_water_avg=[]
dFF0_urine_avg=[]

for j in np.arange(len(recordings)):
#for j in [12]:
	df_snif = pd.read_excel(events_files[j])
	    # Read data from file 'filename.csv' 
    # Control delimiters, rows, column names with read_csv (see later) 
	df = pd.read_csv(recordings[j])

	# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.contains.html
	snif_water_onset=df_snif['Behavior'].str.contains('sniffing water', regex=False)&df_snif['Event_Type'].str.contains('start', regex=False)
	snif_water_offset=df_snif['Behavior'].str.contains('sniffing water', regex=False)&df_snif['Event_Type'].str.contains('stop', regex=False)
	snif_urine_onset=df_snif['Behavior'].str.contains('sniffing urine', regex=False)&df_snif['Event_Type'].str.contains('start', regex=False)
	snif_urine_offset=df_snif['Behavior'].str.contains('sniffing urine', regex=False)&df_snif['Event_Type'].str.contains('stop', regex=False)
	# https://stackoverflow.com/questions/21448225/getting-indices-of-true-values-in-a-boolean-list
	water_on_indice=[i for i, x in enumerate(snif_water_onset) if x]
	water_off_indice=[i for i, x in enumerate(snif_water_offset) if x]
	urine_on_indice=[i for i, x in enumerate(snif_urine_onset) if x]
	urine_off_indice=[i for i, x in enumerate(snif_urine_offset) if x]
	#
	water_on=np.array(df_snif['Time_Relative_sf'][water_on_indice]+snif_trim[j]-led_onset[j])
	water_off=np.array(df_snif['Time_Relative_sf'][water_off_indice]+snif_trim[j]-led_onset[j])
	water_dur=np.array(df_snif['Duration_sf'][water_on_indice])
	urine_on=np.array(df_snif['Time_Relative_sf'][urine_on_indice]+snif_trim[j]-led_onset[j])
	urine_off=np.array(df_snif['Time_Relative_sf'][urine_off_indice]+snif_trim[j]-led_onset[j])
	urine_dur=np.array(df_snif['Duration_sf'][urine_on_indice])
	water_on=water_on[0:2]
	urine_on=urine_on[0:2]
    
	df.columns = ['times', 'ch1', 'chDIO1', 'ch2']
	miceNO=str(recordings[j][-16:-11])
	expdate=str(recordings[j][-10:-4])
	filename=str(recordings[j][-16:-4])
	print(filename)


	## trial basis definition

	SR=1000 #sampling rate in Hz
	prestart=3 # sec, duration before trial start relative to trigger
	poststart=5 # sec, duriation after trial start relative to trigger
	basestart=0 # sec, baseline start for calculating dFF0 relative to trial start trigger
	baseend=1 # sec, baseline end for calculating dFF0 relative to trial start trigger
	timevec=np.arange(-prestart, poststart, 1/SR)
	timevec1 = np.arange(0,(len(df['ch1'])/SR), 1/SR)
	t=np.arange(len(df.index))

	# http://dx.doi.org/10.1016/j.cell.2015.07.014
	# dFF using 405 fit as baseline
	reg= np.polyfit(df['ch2'], df['ch1'], 1)
	fit_405=reg[0]*df['ch2']+reg[1]
	dFF=(df['ch1']-fit_405)/fit_405 #this gives deltaF/F
	df['fit_405']=fit_405
	df['dFF']=dFF

	Trace_water=np.zeros((water_on.size, (poststart-(-prestart))*SR));
	Trace_urine=np.zeros((urine_on.size, (poststart-(-prestart))*SR));
	dFF0_water=np.zeros((water_on.size, (poststart-(-prestart))*SR));
	dFF0_urine=np.zeros((urine_on.size, (poststart-(-prestart))*SR));

	for i in range(len(water_on)):
	    Trace_water[i:]=dFF[int(water_on[i]*SR)-SR*prestart:int(water_on[i]*SR)+SR*poststart]
	for i in range(len(urine_on)):
	    Trace_urine[i:]=dFF[int(urine_on[i]*SR)-SR*prestart:int(urine_on[i]*SR)+SR*poststart]

	    ## calculate water sniffing traces
	MeanTrace_water=np.mean(Trace_water,axis=0)

	for i in range(water_on.size):
	    dFF0_water[i,:]=Trace_water[i,:]-(np.mean(Trace_water[i][int(basestart*SR):int(baseend*SR)], axis=0))

	meandFF0_water=np.mean(dFF0_water,axis=0);

	## calculate urine traces
	MeanTrace_urine=np.mean(Trace_urine,axis=0)

	for i in range(urine_on.size):
	    dFF0_urine[i,:]=Trace_urine[i,:]-(np.mean(Trace_urine[i][int(basestart*SR):int(baseend*SR)], axis=0))

	meandFF0_urine=np.mean(dFF0_urine,axis=0);

	plt.plot(timevec,meandFF0_water,'b',timevec,meandFF0_urine,'r',[0, 0],[-0.02, 0.02])
	plt.xlabel('seconds to sniffing onset')
	plt.ylabel('dFF0')
	plt.title(recordings[j][-16:-4] + 'grand average')
	plt.legend('WU0')
	plt.savefig(recordings[j][-16:-4] +'_avg.jpg', dpi=300)
	plt.close()

	plt.plot(timevec,dFF0_water[0],'b',timevec,dFF0_urine[0],'r',[0, 0],[-0.02, 0.02])
	plt.xlabel('seconds to sniffing onset')
	plt.ylabel('dFF0')
	plt.title(recordings[j][-16:-4] + '_1st sniffing')
	plt.legend('WU0')
	plt.savefig(recordings[j][-16:-4] +'_1st.jpg', dpi=300)
	plt.close()

	plt.plot(timevec,dFF0_water[1],'b',timevec,dFF0_urine[1],'r',[0, 0],[-0.02, 0.02])
	plt.xlabel('seconds to sniffing onset')
	plt.ylabel('dFF0')
	plt.title(recordings[j][-16:-4] + '_2nd sniffing')
	plt.legend('WU0')
	plt.savefig(recordings[j][-16:-4] +'_2nd.jpg', dpi=300)
	plt.close()

	# plt.plot(timevec,dFF0_water[2],'b',timevec,dFF0_urine[2],'r',[0, 0],[-0.02, 0.02])
	# plt.xlabel('seconds to sniffing onset')
	# plt.ylabel('dFF0')
	# plt.title(recordings[j][-16:-4] + '3rd sniffing')
	# plt.legend('WU0')
	# plt.savefig(recordings[j][-16:-4] +'_3rd.jpg', dpi=300)
	# plt.close()
    
	dFF0_water_1.append(dFF0_water[0])
	dFF0_urine_1.append(dFF0_urine[0])
	dFF0_water_2.append(dFF0_water[1])
	dFF0_urine_2.append(dFF0_urine[1])
	dFF0_water_avg.append(meandFF0_water)
	dFF0_urine_avg.append(meandFF0_urine)

	del df 
	del Trace_water
	del Trace_urine

# for i in np.arange(len(dFF0_urine_1)):
#     plt.plot(timevec,dFF0_urine_1[i])

# plt.plot(np.mean(dFF0_urine_1,0))

# from scipy import stats
# x=stats.zscore(dFF0_urine_1, axis=1, ddof=1)


# for i in np.arange(len(dFF0_urine_1)):
#     plt.plot(timevec,x[i])
    
plt.plot(timevec, np.mean(dFF0_water_1,0),'b', timevec, np.mean(dFF0_urine_1,0),'r')
plt.xlabel('seconds to sniffing onset')
plt.ylabel('dFF0')
plt.title('1st sniffing average')
plt.legend('WU0')
plt.savefig('1st_avg.jpg', dpi=300)
plt.close()



plt.plot(timevec, np.mean(dFF0_water_2,0),'b', timevec, np.mean(dFF0_urine_2,0),'r')
plt.xlabel('seconds to sniffing onset')
plt.ylabel('dFF0')
plt.title('2nd sniffing average')
plt.legend('WU0')
plt.savefig('2nd_avg.jpg', dpi=300)
plt.close()
    





