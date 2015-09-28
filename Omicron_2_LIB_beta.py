#!/usr/bin/python

import numpy as np
import os
import LLRT_object_beta
import time

#############################################
#Define Functions

###
def collect_trigs(rawdir, ifo, channel_name, ppdir):
	"""
	Collect omicron trigs from rawdir and compile them into a single, time-sorted list
	"""
	#Find all omicron raw output files
	os.chdir("%s/%s/%s:%s/"%(rawdir,ifo,ifo,channel_name))
	os.system("rm tmp.txt")
	files_all = os.listdir("%s/%s/%s:%s"%(rawdir,ifo,ifo,channel_name))
	
	#Collect trigger info from all raw output files
	os.system("> tmp.txt")
	for i, f in enumerate(files_all):
		if f.split('.')[1] == 'txt':
			os.system("sed '/#/d' %s >> tmp.txt"%f)
			print "Collected file %s of %s for %s"%(i+1, len(files_all), ifo)
	 
	#Sort the compiled file
	os.system("sort -n tmp.txt > triggers_unclustered_%s.txt"%ifo)
	
	#Move the file to the unclustered post-processing folder
	if not os.path.exists("%s/unclustered/"%ppdir):
		os.makedirs("%s/unclustered/"%ppdir)
	os.system("mv triggers_unclustered_%s.txt %s/unclustered/"%(ifo, ppdir))
	os.system("rm tmp.txt")
	
	return "%s/unclustered/triggers_unclustered_%s.txt"%(ppdir, ifo)

###
def cluster_trigs(ts_file, t_clust, ifo, ppdir):
	"""
	Cluster set of neighboring identical-template triggers (same f_0 and Q) to the trigger with the highest SNR within a specified time window
	"""

	#open output file to write to
	if not os.path.exists("%s/clustered/"%ppdir):
		os.makedirs("%s/clustered/"%ppdir)
	clust_file_nm = "%s/clustered/triggers_clustered_%s_tc%s.txt"%(ppdir, ifo, t_clust)
	clust_file = open(clust_file_nm,'wt')
	
	#load in lines from time-sorted file
	with open(ts_file,'rt') as read_ts_file:
		lines = list(read_ts_file)
	i_list = range(len(lines))
	
	#iterate over triggers
	for i in i_list:
		if i == 'SKIP':
			continue
		
		#initialize tmp_dic to store relevant triggers
		tmp_dic = {}
		
		#get current trigger info
		current_elements = lines[i].split()
		t_current = float(current_elements[0])
		f_current = float(current_elements[1])
		snr_current = float(current_elements[2])
		Q_current = float(current_elements[3])
	
		#add current trig to tmp_dic, labeled by snr so that finding loudest event is trivial
		tmp_dic[snr_current] = current_elements
	
		#check to see if on last trigger
		if (i + 1) >= len(i_list):
			#if last trigger, write it and end loop
			final_elements = tmp_dic[max(tmp_dic)]
			t_final = float(final_elements[0])
			f_final = float(final_elements[1])
			snr_final = float(final_elements[2])
			Q_final = float(final_elements[3])
			clust_file.write('%10.10f %10.10f %10.10f %10.10f\n'%(t_final, f_final, snr_final, Q_final))
			break
		
		#compare to subsequent triggers within specified time window
		i_comp = i+1
		while i_list[i_comp] == 'SKIP':  #make sure the comparison line isn't meant to be skipped
			if (i_comp + 1) >= len(i_list):
				break
			i_comp += 1
		if 	i_list[i_comp] == 'SKIP':  #will trigger if the last line is labeled to be skipped
			break
		compare_elements = lines[i_comp].split()
		t_compare = float(compare_elements[0])
		f_compare = float(compare_elements[1])
		snr_compare = float(compare_elements[2])
		Q_compare = float(compare_elements[3])
		
		while abs(t_compare - t_current) <= t_clust:
			#check if comparison trigger is the same template
			if (f_current == f_compare) and (Q_current == Q_compare):
				#if same template, add to tmp_dic...
				tmp_dic[snr_compare] = compare_elements
				
				#...update current trigger...
				t_current = t_compare
				f_current = f_compare
				snr_current = snr_compare
				Q_current = Q_compare
				
				#...and remove comparison index from i_list
				i_list[i_comp] = 'SKIP'
				
			#move comparison trigger to next trigger
			if (i_comp + 1) >= len(i_list):  #break if on last trigger
				break
			i_comp += 1
			while i_list[i_comp] == 'SKIP':  #make sure the comparison line isn't meant to be skipped
				if (i_comp + 1) >= len(i_list):
					break
				i_comp += 1
			if 	i_list[i_comp] == 'SKIP':  #will trigger if the last line is labeled to be skipped
				break
			
			compare_elements = lines[i_comp].split()
			t_compare = float(compare_elements[0])
			f_compare = float(compare_elements[1])
			snr_compare = float(compare_elements[2])
			Q_compare = float(compare_elements[3])
		
		#write maximum snr trigger to file
		final_elements = tmp_dic[max(tmp_dic)]  #choose trigger with loudest SNR
		t_final = float(final_elements[0])
		f_final = float(final_elements[1])
		snr_final = float(final_elements[2])
		Q_final = float(final_elements[3])
		clust_file.write('%10.10f %10.10f %10.10f %10.10f\n'%(t_final, f_final, snr_final, Q_final))
	
	clust_file.close()
	
	os.system("sort %s -o %s"%(clust_file_nm, clust_file_nm))
	
	return clust_file_nm
		
###
def constrain_2_eff_segs(trig_file, seg_file, t_clust, ifo, ppdir):
	"""
	Constrain list of time-sorted triggers to lie within the passed time-sorted effective segments (i.e., apply vetoes)
	"""
	#import clustered trigs and vetoes
	trig_array = np.genfromtxt(trig_file).reshape((-1,4))
	trig_start_stop = np.zeros((len(trig_array), 2))
	delta_t = trig_array[:,3] * 1.253059 / (np.sqrt(2.)*np.pi*trig_array[:,1])  #This appears to be the conversion between time duration and Q
	trig_start_stop[:,0] = trig_array[:,0] - delta_t/2.
	trig_start_stop[:,1] = trig_array[:,0] + delta_t/2.

	seg_start_stop = np.genfromtxt(seg_file).reshape((-1,2))

	#create arrays of start and end times of trigs (keeping all trigger info) and sort by tigger start time
	tot_list = zip(trig_start_stop, trig_array)
	tot_list = sorted(tot_list, key = lambda x:x[0][0])

	#open output file
	survive_file_nm = "%s/clustered/triggers_clustered_%s_tc%s_postveto.txt"%(ppdir, ifo, t_clust)
	survive_file = open(survive_file_nm,'wt')

	i_seg = 0
	end_flag = 0
	seg_start = seg_start_stop[i_seg,0]
	seg_stop = seg_start_stop[i_seg,1]

	#starting at beginning, find next trig
	for i_trig in xrange(len(tot_list)):
			
		#Get start and stop times from trig:
		trig_start = tot_list[i_trig][0][0]
		trig_stop = tot_list[i_trig][0][1]
		
		#Find first segment that has an end after the trigger start time (only iterate once all trig start times have passed segment end)
		while seg_stop < trig_start:
			#Check to see if there is another seg
			if i_seg >= len(seg_start_stop)-1:
				#Not another seg so end the loop over trigs
				end_flag = 1
				break
			else:
				#Move to next seg
				i_seg += 1
				seg_start = seg_start_stop[i_seg,0]
				seg_stop = seg_start_stop[i_seg,1]
		if end_flag:
			break
			
		#Check to see if trig lies completely within segment
		if (trig_start >= seg_start) and (trig_stop <= seg_stop):
			#Trig lies completely within segment, so write to file
			survive_file.write("%10.10f %10.10f %10.10f %10.10f\n"%(tot_list[i_trig][1][0],tot_list[i_trig][1][1],tot_list[i_trig][1][2],tot_list[i_trig][1][3]))
		else:
			#Trig does not completely lie within segment, so disregard it
			continue
			
	survive_file.close()
	
	#Sorting by start time undoes sorting by central time, so sort by central time
	os.system("sort %s -o %s"%(survive_file_nm, survive_file_nm))
	
	return survive_file_nm


###	
def coincidence(trig_file_1, trig_file_2, t_coin, snr_coin, ifos, t_shift, ppdir):
	"""
	Coincide two sets of triggers, using timing and snr parameters and constraining concident triggers to have identical f_0 and Q
	"""
	#Open file to write coincident triggers to
	coin_file_nm = "%s/coincident/triggers_coincident_%s%s_tc%s_snr%s_ts%s_.txt"%(ppdir, ifos[0], ifos[1], t_coin, snr_coin, t_shift)
	if not os.path.exists("%s/coincident/"%ppdir):
		os.makedirs("%s/coincident/"%ppdir)
	coin_file = open(coin_file_nm,'wt')
	
	#Open each file and read in lines
	read_trig_file_1 = open(trig_file_1,'rt')
	with open(trig_file_2,'rt') as read_trig_file_2:
		lines_2 = list(read_trig_file_2)
	
	#Iterate through list 1, comparing to list 2
	i2_min = 0
	
	for lin_1 in read_trig_file_1:
		current_line = lin_1.split()
		t_current = float(current_line[0])
		f_current = float(current_line[1])
		snr_current = float(current_line[2])
		Q_current = float(current_line[3])
		
		#Update reference index for list 2
		t2_min = float(lines_2[i2_min].split()[0])
		while abs(t_current - t2_min) > t_coin:
			if (t2_min - t_current) >  t_coin:
				break
			elif (i2_min + 1) >= len(lines_2):
				break
			else:
				i2_min += 1
				t2_min = float(lines_2[i2_min].split()[0])
		
		#Search for coincident templates within time window
		i2_compare = i2_min
		compare_line = lines_2[i2_compare].split()
		t_compare = float(compare_line[0])
		f_compare = float(compare_line[1])
		snr_compare = float(compare_line[2])
		Q_compare = float(compare_line[3])
		
		while abs(t_current - t_compare) <= t_coin:
			if (f_current == f_compare) and (Q_current == Q_compare) and (np.sqrt(snr_current**2. + snr_compare**2.) >= snr_coin):
				coin_file.write( "%10.10f %10.10f %10.10f %10.10f %10.10f %10.10f %10.10f %10.10f %10.10f %10.10f %10.10f %10.10f\n"%((t_current+t_compare)/2., (f_current+f_compare)/2., np.sqrt( snr_current**2. + snr_compare**2. ), (Q_current+Q_compare)/2., t_current, f_current, snr_current, Q_current, t_compare, f_compare, snr_compare, Q_compare) )

			if (i2_compare + 1) >= len(lines_2):
				break

			i2_compare += 1
			compare_line = lines_2[i2_compare].split()
			t_compare = float(compare_line[0])
			f_compare = float(compare_line[1])
			snr_compare = float(compare_line[2])
			Q_compare = float(compare_line[3])
	
	coin_file.close()
	read_trig_file_1.close()
	
	os.system("sort %s -o %s"%(coin_file_nm, coin_file_nm))
				
	return coin_file_nm
	
###
def time_slide(trig_file_1, trig_file_2, t_coin, snr_coin, ifos, t_shift_start, t_shift_stop, t_shift_num, ppdir):
	"""
	Create a set of coincided timeslides
	"""
		
	#create list of time slides
	t_shift_array = np.linspace(start=t_shift_start, stop=t_shift_stop, num=t_shift_num)
		
	for t_shift in t_shift_array:
		#add time shifts in a temporary file
		trig_file_2_tmp = trig_file_2 + '_tmp'
		write_trig_file_2_tmp = open(trig_file_2_tmp,'wt')
		
		#time shift is applied to trig_file_2
		read_trig_file_2 = open(trig_file_2,'rt')
		for line in read_trig_file_2:
			elements = line.split()
			write_trig_file_2_tmp.write("%10.10f %10.10f %10.10f %10.10f\n"%(float(elements[0])+t_shift, float(elements[1]), float(elements[2]), float(elements[3])))
		 
		write_trig_file_2_tmp.close()
		read_trig_file_2.close()
		
		#do coincidence with time-shifted files
		coin_file_tmp = coincidence(trig_file_1=trig_file_1, trig_file_2=trig_file_2_tmp, t_coin=t_coin, snr_coin=snr_coin, ifos=ifos, t_shift=t_shift, ppdir=ppdir)

		#remove the temporary time-shifted cluster file
		os.system('rm %s'%trig_file_2_tmp)
		
		print "Coincided trigs for %s and %s for time slide of %s"%(ifos[0], ifos[1], t_shift)

###
def log_likelihood_ratio_test(signal_kde_coords, signal_kde_values, noise_kde_coords, noise_kde_values, log_like_thresh, LIB_window, ifos, ppdir):
	"""
	Downselect triggers by performing log likelihood ratio thresholding test
	"""
		
	if not os.path.exists("%s/LIB_trigs/"%ppdir):
		os.makedirs("%s/LIB_trigs/"%ppdir)
	
	#Save the log likelihood used for this LLRT test
	thresh_log_like_ratio = log_like_thresh
	np.savetxt("%s/LIB_trigs/threshold_log_likelihood_ratio.txt"%ppdir, np.array([thresh_log_like_ratio]))
	
	#Open write files for LIB trigs and their corresponding timeslides for both 0-lags and timeslide triggers
	lib_0lag_times = open('%s/LIB_trigs/LIB_0lag_times_%s%s.txt'%(ppdir, ifos[0], ifos[1]), 'wt')
	lib_0lag_timeslides = open('%s/LIB_trigs/LIB_0lag_timeslides_%s%s.txt'%(ppdir, ifos[0], ifos[1]), 'wt')
	lib_ts_times = open('%s/LIB_trigs/LIB_ts_times_%s%s.txt'%(ppdir, ifos[0], ifos[1]), 'wt')
	lib_ts_timeslides = open('%s/LIB_trigs/LIB_ts_timeslides_%s%s.txt'%(ppdir, ifos[0], ifos[1]), 'wt')
	
	#Build LLRT object for 0-lag and for each timeslide
	files_all = sorted(os.listdir("%s/coincident/"%ppdir))
	for f in files_all:
		#Load in coincident omicron data for each timeslide
		terms = f.split("_")
		tshift = float(terms[5].split("ts")[1])
		try:
			data_array = np.genfromtxt("%s/coincident/%s"%(ppdir,f)).reshape((-1,12))
		except IOError:
			data_array = np.array([])

		#Build calc_info dictionary
		calc_info = {}
		calc_info['interp method'] = 'Grid Linear'
		calc_info['extrap method'] = 'Grid Nearest'

		#Build param_info dictionary
		param_info = {}
		param_info['delta_t'] = {}
		param_info['delta_t']['dimension'] = 1
		param_info['delta_t']['param names'] = ['delta_t']
		param_info['delta_t']['interp range'] = np.array([[-0.05, 0.05]])
		
		#Load likelihood estimate for signal
		train_signal_data = {}
		train_signal_data['delta_t'] = {}
		train_signal_data['delta_t']['KDE'] = ([np.load(signal_kde_coords),np.load(signal_kde_values)])
		
		#Load likelihood estimate for noise
		train_noise_data = {}
		train_noise_data['delta_t'] = {}
		train_noise_data['delta_t']['KDE'] = ([np.load(noise_kde_coords),np.load(noise_kde_values)])

		#Build foreground_data dictionary
		try:
			dt_tmp = data_array[:,4] - data_array[:,8]
		except IndexError:
			dt_tmp = np.array([])

		foreground_data = {}
		foreground_data['npoints'] = len(dt_tmp)		
		foreground_data['delta_t'] = {}
		foreground_data['delta_t']['data'] = np.transpose(np.array([dt_tmp]))
				
		if foreground_data['delta_t']['data'].any():
			#Initialize the LLRT object
			LLRT = LLRT_object_beta.LLRT(calc_info=calc_info, param_info=param_info, train_signal_data=train_signal_data, train_noise_data=train_noise_data, foreground_data=foreground_data)
		
			#Find foreground trigs that are above the passed log likelihood threshold and save them to file
			trigs_above_thresh = data_array[LLRT.LLR_above_thresh(threshold=thresh_log_like_ratio, groundtype='Foreground')]
			trigs_above_thresh = cluster_LIB_trigs(LIB_trig_array=trigs_above_thresh, LIB_window=LIB_window)
			
			#Save LIB triggers
			np.savetxt('%s/LIB_trigs/LIB_trigs_%s%s_ts%s.txt'%(ppdir, ifos[0], ifos[1], tshift), trigs_above_thresh)
			if tshift == 0.:
				for i in xrange(len(trigs_above_thresh)):
					lib_0lag_times.write('%10.10f\n'%trigs_above_thresh[i,0])
					lib_0lag_timeslides.write('0. %s\n'%(tshift))
			else:
				for i in xrange(len(trigs_above_thresh)):
					lib_ts_times.write('%10.10f\n'%trigs_above_thresh[i,0])
					lib_ts_timeslides.write('0. %s\n'%(tshift))
		else:
			os.system('touch %s/LIB_trigs/LIB_trigs_%s%s_ts%s.txt'%(ppdir, ifos[0], ifos[1], tshift))
	
	lib_0lag_times.close()
	lib_0lag_timeslides.close()
	lib_ts_times.close()
	lib_ts_timeslides.close()

###
def cluster_LIB_trigs(LIB_trig_array, LIB_window):
	"""
	Cluster LIB trigs so that the LIB trig times are those of the loudest trig within a given LIB window length
	"""
	#Initialize for first loop over trigs
	iterations = 0
	clust_flag = 1  #Do this to start loop below
	in_array = LIB_trig_array

	#Loop over algorithm until no clustering is done
	while clust_flag:
		
		#Count number of iterations through algorithm
		iterations += 1
		
		#Initialize out_array
		out_array = np.array([])
		
		#Initialize clust_flag as 0, marking no clustering yet done
		clust_flag = 0

		#Iterate over lines, clustering them into windows of specified length, centered on highest SNR trigger within window
		t_start = float('inf')
		window_dic = {}
		found = 0

		for line in in_array:
			#Read in necessary data parameters
			t_current = line[0]
			snr_current = line[2]
			
			#Compare current time to window start time to see if current trig is in window
			if abs(t_current - t_start) <= 0.5*LIB_window:
				#If in current window, save trigger in dic with snr as key
				window_dic[snr_current] = line
				#Note that clustering was done during this iteration
				clust_flag = 1
			else:
				#Trigger outside of window, so write old window and start new window
				if window_dic:
					#If there are triggers in the old window, write to file
					found += 1
					max_line = window_dic[max(window_dic)]
					out_array = np.append(out_array, max_line)
				
				#Set start time of new window
				t_start = t_current
				
				#Initiate new window
				window_dic = {}
				window_dic[snr_current] = line
				
		#Check and write last window
		if window_dic:
			#If there are triggers in the old window, write to file
			found += 1
			max_line = window_dic[max(window_dic)]
			out_array = np.append(out_array, max_line)	

		#Resize out_array
		try:
			out_array = out_array.reshape((found,-1))
		except ValueError:
			if len(out_array) == 0.:
				#No trigs, can break clustering loop
				break

		#Replace initial trig array with down-selected trig array
		in_array = out_array
			
	print "Finished down selection of LIB trigs after %s iterations"%iterations
	return out_array

###
def crop_segs(seg_file, overlap, ifo, ppdir):
	"""
	Crop out the Omicron overlap from merged segments to get segments in which triggers can actually occur (note that these segments should be merged first!)
	"""
	#import segments
	seg_start_stop = np.genfromtxt(seg_file).reshape((-1,2))
	
	#open output file
	cropped_segs_file_nm = "%s/live_segs/segments_%s_cropped.seg"%(ppdir, ifo)
	if not os.path.exists("%s/live_segs/"%ppdir):
		os.makedirs("%s/live_segs/"%ppdir)
	cropped_segs_file = open(cropped_segs_file_nm,'wt')
	
	#loop through segments, cropping off the Omicron overlap from the edges
	for seg in seg_start_stop:
		tmp_start = seg[0] + int(overlap/2.)
		tmp_stop = seg[1] - int(overlap/2.)
		#write cropped segment start and stop times if still valid after cropping
		if tmp_start < tmp_stop:
			cropped_segs_file.write("%10.10f %10.10f\n"%(tmp_start,tmp_stop))
			
	cropped_segs_file.close()
	return cropped_segs_file_nm
		
###
def effective_segs(seg_file, veto_file, ifo, ppdir):
	"""
	Remove vetoes from segment list for a given ifo, thus creating effective segments
	"""
	#import segments and vetoes
	seg_start_stop = np.genfromtxt(seg_file).reshape((-1,2))

	try:
		veto_start_stop = np.genfromtxt(veto_file).reshape((-1,2))
		if not veto_start_stop.any():
			veto_start_stop = np.array([[float('inf'), float('inf')]])
	except IOError:
		veto_start_stop = np.array([[float('inf'), float('inf')]])

	#sort segments and vetoes, this sorting will be conserved
	seg_start_stop = np.array(sorted(seg_start_stop, key = lambda x:x[0]))
	veto_start_stop = np.array(sorted(veto_start_stop, key = lambda x:x[0]))

	#open output file
	eff_segs_file_nm = "%s/live_segs/segments_%s_postveto.seg"%(ppdir, ifo)
	if not os.path.exists("%s/live_segs/"%ppdir):
		os.makedirs("%s/live_segs/"%ppdir)
	eff_segs_file = open(eff_segs_file_nm,'wt')

	#Start at first veto
	i_veto = 0
	veto_start = veto_start_stop[i_veto,0]
	veto_stop = veto_start_stop[i_veto,1]

	#Loop through segments
	for i_seg in xrange(len(seg_start_stop)):
		seg_start = seg_start_stop[i_seg,0]
		seg_stop = seg_start_stop[i_seg,1]
		
		#Interate through vetoes until we find one that either intersects with our segment or occurs after
		while (veto_start < seg_start) and (veto_stop < seg_start):
			if i_veto >= len(veto_start_stop)-1:
				#if at end of vetoes, set veto times to infinity
				veto_start = float('inf')
				veto_stop = float('inf')
			else:
				#increment veto
				i_veto += 1
				veto_start = veto_start_stop[i_veto,0]
				veto_stop = veto_start_stop[i_veto,1]
		
		###At this point, at least veto_stop occurs after seg_start###
		
		#Choose initial segment start based on location of veto
		if (veto_start <= seg_start) and (veto_stop >= seg_start):
			start = veto_stop
		elif (veto_start > seg_start) and (veto_stop > seg_start):	
			start = seg_start
		else:
			raise ValueError, "Encountered situation that is not considered, exiting"
		
		#Loop through vetoes occuring within segment
		while (veto_start < seg_stop) and (veto_stop < seg_stop):
			if (veto_start <= seg_start) and (veto_stop < seg_stop):
				#Increment veto so that veto occurs completely after segment start
				if i_veto >= len(veto_start_stop)-1:
					#if at end of vetoes, set to infinity
					veto_start = float('inf')
					veto_stop = float('inf')
				else:
					#increment veto
					i_veto += 1
					veto_start = veto_start_stop[i_veto,0]
					veto_stop = veto_start_stop[i_veto,1]
			elif (veto_start > seg_start) and (veto_stop < seg_stop):
			#Record the effective segments
				stop = veto_start
				if start < stop:
					eff_segs_file.write("%10.10f %10.10f\n"%(start,stop))
				#Change to new start and iterate veto
				start = veto_stop
				if i_veto >= len(veto_start_stop)-1:
					#if at end of vetoes, set to infinity
					veto_start = float('inf')
					veto_stop = float('inf')
				else:
					#increment veto
					i_veto += 1
					veto_start = veto_start_stop[i_veto,0]
					veto_stop = veto_start_stop[i_veto,1]
			else:
				raise ValueError, "Encountered situation that is not considered, exiting"
			
		###At this point, at least veto_stop occurs after seg_stop###
			
		#We should now be at last relevant veto for this segment
		if (veto_start <= seg_start) and (veto_stop >= seg_stop):
			pass	
		elif (veto_start <= seg_stop) and (veto_stop >= seg_stop):
		#Record the effective segments
			stop = veto_start
			if start < stop:
				eff_segs_file.write("%10.10f %10.10f\n"%(start,stop))
		elif (veto_start >= seg_stop) and (veto_stop >= seg_stop):
			stop = seg_stop
			if start < stop:
				eff_segs_file.write("%10.10f %10.10f\n"%(start,stop))
					
	eff_segs_file.close()
	return eff_segs_file_nm

###
def merge_segs(seg_file):
	"""
	For a time-sorted segment list, combine segments that are divided at a common start/stop time
	"""
	#Load segments into an array
	seg_array = np.genfromtxt(seg_file).reshape((-1,2))
	
	#Open seg_file for overwriting
	write_seg_file = open(seg_file,'wt')
	
	#If only 1 segment, nothing needs to be merged
	if np.shape(seg_array) == (1,2):
		write_seg_file.write("%10.10f %10.10f\n"%(seg_array[0,0],seg_array[0,1]))	
	
	#If more than 1 segment, loop over segments
	else:
		#Initialize first pair of neighboring segs
		i_seg = 0
		start_current = seg_array[i_seg,0]
		stop_current = seg_array[i_seg,1]
		
		i_seg += 1
		start_next = seg_array[i_seg,0]
		stop_next = seg_array[i_seg,1]
		
		#Loop over all pairs of neighboring segs
		while i_seg < len(seg_array):
			#Check if segments need to be merged
			if stop_current == start_next:
				#merge current and next segments together
				stop_current = stop_next
			else:
				#write current segment start and stop times
				if start_current < stop_current:
					write_seg_file.write("%10.10f %10.10f\n"%(start_current,stop_current))
				#make next segment the current segment
				start_current = start_next
				stop_current = stop_next
			
			#iterate to next segment
			if i_seg >= len(seg_array) - 1:
				#if we've reached last seg, then break and end loop
				break
			else:
				#else iterate to next segment
				i_seg += 1
				start_next = seg_array[i_seg,0]
				stop_next = seg_array[i_seg,1]
				
		#Write final segment
		if start_current < stop_current:
			write_seg_file.write("%10.10f %10.10f\n"%(start_current,stop_current))
				
	#Close seg_file
	write_seg_file.close()
	
	
###
def intersect_segments(seg_file1, seg_file2, ifos, t_shift, ppdir):
	"""
	For time-sorted segments, find intersection of livetime between two detectors
	"""
	seg_array1 = np.genfromtxt(seg_file1).reshape((-1,2))
	seg_array2 = np.genfromtxt(seg_file2).reshape((-1,2))
	
	seg_intersect_nm = "%s/live_segs/intersect_%s%s_ts%s.seg"%(ppdir, ifos[0], ifos[1], t_shift)
	if not os.path.exists("%s/live_segs/"%ppdir):
		os.makedirs("%s/live_segs/"%ppdir)
	seg_intersect = open(seg_intersect_nm,'wt')
	
	i1 = 0
	i2 = 0
	
	while (i1 <= len(seg_array1)-1) and (i2 <= len(seg_array2)-1):
		#Take highest start time and lowest end time
		t_low_tmp = max(seg_array1[i1,0], seg_array2[i2,0])
		t_high_tmp = min(seg_array1[i1,1], seg_array2[i2,1])
		
		#Print intersecting segment to file
		if t_low_tmp < t_high_tmp:
			seg_intersect.write("%10.10f %10.10f\n"%(t_low_tmp, t_high_tmp))
		
		#Advance whichever current segment ends first
		if (seg_array1[i1,1] <= seg_array2[i2,1]):
			i1 += 1
		elif (seg_array2[i2,1] <= seg_array1[i1,1]):
			i2 += 1
	
	seg_intersect.close()
	return seg_intersect_nm

###
def calculate_livetimes(seg_file1, seg_file2, t_shift_start, t_shift_stop, t_shift_num, ifos, ppdir, infodir, train_runmode):
	"""
	Add up effective segments for 0-lag and for each timeslide to calculate effective livetimes
	"""	
	#Create list of time slides
	t_shift_array = np.linspace(start=t_shift_start, stop=t_shift_stop, num=t_shift_num)
	
	#Initiate running sums for livetime calculations
	zero_lag_lt = 0.
	timeslide_lt = 0.
	
	#Load 2nd set segments (which need to be timeslided) into an array
	seg_array2 = np.genfromtxt(seg_file2).reshape((-1,2))
	tmp_seg_file2_nm = "%s/live_segs/tmp_seg_file.seg"%ppdir
	if not os.path.exists("%s/live_segs/"%ppdir):
		os.makedirs("%s/live_segs/"%ppdir)
	
	#Loop over timeslides, intersecting segs and calculating overlapping livetime for each
	for ts in t_shift_array:
		#Initiate running sum for this particular timeslide
		ts_lt = 0.
		
		#Apply time shifts to 2nd set of segments
		tmp_seg_file2 = open(tmp_seg_file2_nm,'wt')
		for line in seg_array2:
			tmp_seg_file2.write("%10.10f %10.10f\n"%(line[0]+ts, line[1]+ts))
		tmp_seg_file2.close()
		
		#Intersect the timeslided segments
		tmp_intersected_segs = intersect_segments(seg_file1=seg_file1, seg_file2=tmp_seg_file2_nm, ifos=ifos, t_shift=ts, ppdir=ppdir)
		
		#Add up the livetime from the intersected segments
		read_tmp_intersected_segs = open(tmp_intersected_segs,'rt')
		for line in read_tmp_intersected_segs:
			elements = line.split()
			ts_lt += float(elements[1]) - float(elements[0])
		read_tmp_intersected_segs.close()
		
		#Add livetime to appropriate sum
		if ts == 0.:
			zero_lag_lt += ts_lt
		else:
			timeslide_lt += ts_lt
	
	#Remove the temporary seg file
	os.system('rm %s'%tmp_seg_file2_nm)
	
	#Write summed livetimes to file
	np.savetxt("%s/live_segs/livetime_0lag_%s%s.txt"%(ppdir,ifos[0],ifos[1]), np.array([zero_lag_lt]))
	np.savetxt("%s/live_segs/livetime_timeslides_%s%s.txt"%(ppdir,ifos[0],ifos[1]), np.array([timeslide_lt]))
	
	#Add summed livetimes to running collection if not in a training mode
	if (train_runmode == 'None'):
		#Check to make sure another function isn't currently updating the livetimes
		while os.path.isfile('%s/result_dics/livetimes.txt_lock'%infodir):
			time.sleep(5)
			
		#Lock livetime files while updating them
		if not os.path.exists('%s/result_dics/'%infodir):
			os.makedirs('%s/result_dics/'%infodir)
		os.system('> %s/result_dics/livetimes.txt_lock'%infodir)
		
		#Load foreground and background livetimes
		if os.path.isfile('%s/result_dics/foreground_livetime.txt'%infodir):
			foreground_livetime = np.genfromtxt('%s/result_dics/foreground_livetime.txt'%infodir)
		else:
			foreground_livetime = 0.	
		if os.path.isfile('%s/result_dics/background_livetime.txt'%infodir):
			background_livetime = np.genfromtxt('%s/result_dics/background_livetime.txt'%infodir)
		else:
			background_livetime = 0.
		
		#Add summed livetimes to running foreground and background totals
		foreground_livetime += zero_lag_lt
		background_livetime += timeslide_lt
		
		#Save new foreground and background livetimes
		np.savetxt('%s/result_dics/foreground_livetime.txt'%infodir, np.array([foreground_livetime]))
		np.savetxt('%s/result_dics/background_livetime.txt'%infodir, np.array([background_livetime]))
		
		#Remove livetime lock file now that the update is complete
		os.system('rm %s/result_dics/livetimes.txt_lock'%infodir)

##############################################
if __name__=='__main__':

	from optparse import OptionParser

	usage = None
	parser = OptionParser(usage=usage)

	#general options
	parser.add_option("-p", "--ppdir", default=None, type="string", help="Path to post-processing directory, where results will be written")
	parser.add_option("-i","--infodir", default=None, type="string", help="Path to info directory (where sub files, etc. are stored)")
	parser.add_option("-I","--ifolist", default=None, type="string", help="Comma separated list of ifos. E.g., H1,L1")
	parser.add_option("-r", "--rawdir", default=None, type="string", help="Path to raw directory")
	parser.add_option("-c", "--channel-names", default=None, type="string", help="Comma separated names of channel to analyze for each ifo")
	parser.add_option("", "--cluster-t", default=None, type="float", help="Time window for clustering (in sec)")
	parser.add_option("","--coin-t", default=None, type="float", help="Time window for coincidence (in sec)")
	parser.add_option("","--coin-snr", default=None, type="float", help="SNR threshold for coincidence")
	parser.add_option("","--t-shift-start", default=None, type="float", help="Starting time shift to apply to IFO 2")
	parser.add_option("","--t-shift-stop", default=None, type="float", help="Ending time shift to apply to IFO 2")
	parser.add_option("","--t-shift-num", default=None, type="float", help="Number of time shifts to apply to IFO 2")
	parser.add_option("","--segs", default=None, type="string", help="Comma separated list of paths to files containing segment start/stop times for each ifo")
	parser.add_option("","--veto-files", default=None, type="string", help="Comma separated list of paths to veto data file for each ifo")
	parser.add_option("","--overlap", default=None, type='int', help="Overlap of segments used for Omicron")
	parser.add_option("","--log-like-thresh", default=None, type="float", help="Threshold log likelihood ratio value")
	parser.add_option("","--LIB-window", default=None, type="float", help="Length of window (in s) for LIB runs, cluster trigs to 1 trig per LIB window")
	parser.add_option("","--signal-kde-coords", default=None, type='string', help='Path to file containing coodinates of the KDE likelihood estimate for signals')
	parser.add_option("","--signal-kde-values", default=None, type='string', help='Path to file containing values of the KDE likelihood estimate for signals')
	parser.add_option("","--noise-kde-coords", default=None, type='string', help='Path to file containing coodinates of the KDE likelihood estimate for noise')
	parser.add_option("","--noise-kde-values", default=None, type='string', help='Path to file containing values of the KDE likelihood estimate for noise')
	parser.add_option("","--train-runmode", default=None, type='string', help='Either "Signal", "Noise", or "None" depending on if user wants to run in training mode or not')

	#----------------------------------------------

	opts, args = parser.parse_args()

	ppdir = opts.ppdir
	infodir=opts.infodir
	ifos = opts.ifolist.split(',')
	rawdir = opts.rawdir
	channel_names = opts.channel_names.split(',')
	t_clust = opts.cluster_t
	t_coin = opts.coin_t
	snr_coin = opts.coin_snr
	t_shift_start = opts.t_shift_start
	t_shift_stop = opts.t_shift_stop
	t_shift_num = opts.t_shift_num
	segs = opts.segs.split(",")
	veto_files = opts.veto_files.split(',')
	overlap = opts.overlap
	log_like_thresh = opts.log_like_thresh
	LIB_window = opts.LIB_window
	signal_kde_coords = opts.signal_kde_coords
	signal_kde_values = opts.signal_kde_values
	noise_kde_coords = opts.noise_kde_coords
	noise_kde_values = opts.noise_kde_values
	train_runmode = opts.train_runmode
	
	#--------------------------------------------------------------
	
	#MAIN

	#collect
	if not os.path.exists("%s/unclustered/"%ppdir):
		os.makedirs("%s/unclustered/"%ppdir)
	os.system('rm %s/unclustered/*'%ppdir)
	ts_files = {}
	for i,ifo in enumerate(ifos):
		ts_files[ifo] = collect_trigs(rawdir=rawdir, ifo=ifo, channel_name=channel_names[i], ppdir=ppdir)
		print "Collected and sorted trigs for %s"%ifo

	#cluster
	if not os.path.exists("%s/clustered/"%ppdir):
		os.makedirs("%s/clustered/"%ppdir)
	os.system('rm %s/clustered/*'%ppdir)
	clust_files = {}
	for ifo in ifos:		
		clust_files[ifo] = cluster_trigs(ts_file=ts_files[ifo], t_clust=t_clust, ifo=ifo, ppdir=ppdir)
		print "Clustered trigs for %s"%ifo

	#cropped and effective segments
	if not os.path.exists("%s/live_segs/"%ppdir):
		os.makedirs("%s/live_segs/"%ppdir)
	os.system('rm %s/live_segs/*'%ppdir)
	seg_files = {}
	for i,ifo in enumerate(ifos):
		merge_segs(seg_file=segs[i])
		seg_files[ifo] = crop_segs(seg_file=segs[i], overlap=overlap, ifo=ifo, ppdir=ppdir)
		seg_files[ifo] = effective_segs(seg_file=seg_files[ifo], veto_file=veto_files[i], ifo=ifo, ppdir=ppdir)
		print "Cropped Omicron overlaps and removed vetoes to create effective segments for %s"%ifo
	
	#apply vetoes by constraining triggers to effective segments
	for ifo in ifos:	
		clust_files[ifo] = constrain_2_eff_segs(trig_file=clust_files[ifo], seg_file=seg_files[ifo], t_clust=t_clust, ifo=ifo, ppdir=ppdir)
		print "Applied vetoes by constraining triggers to effective segments for %s"%ifo

	#Live time intersection
	calculate_livetimes(seg_file1=seg_files[ifos[0]], seg_file2=seg_files[ifos[1]], t_shift_start=t_shift_start, t_shift_stop=t_shift_stop, t_shift_num=t_shift_num, ifos=ifos, ppdir=ppdir, infodir=infodir, train_runmode=train_runmode)
	print "Calculated total coincident live time for 0-lag and timeslides"

	#coincidence
	if not os.path.exists("%s/coincident/"%ppdir):
		os.makedirs("%s/coincident/"%ppdir)
	os.system('rm %s/coincident/*'%ppdir)
	time_slide(trig_file_1=clust_files[ifos[0]], trig_file_2=clust_files[ifos[1]], t_coin=t_coin, snr_coin=snr_coin, ifos=ifos, t_shift_start=t_shift_start, t_shift_stop=t_shift_stop, t_shift_num=t_shift_num, ppdir=ppdir)

	#Likelihood ratio thresholding
	if not os.path.exists("%s/LIB_trigs/"%ppdir):
		os.makedirs("%s/LIB_trigs/"%ppdir)
	os.system('rm %s/LIB_trigs/*'%ppdir)
	
	log_likelihood_ratio_test(signal_kde_coords=signal_kde_coords, signal_kde_values=signal_kde_values, noise_kde_coords=noise_kde_coords, noise_kde_values=noise_kde_values, log_like_thresh=log_like_thresh, LIB_window=LIB_window, ifos=ifos, ppdir=ppdir)		
	print "Down-selected trigs using log likelihood ratio test"
	print "Complete"
