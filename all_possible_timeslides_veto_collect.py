import numpy as np
import os
import pickle

#=======================================================================
### DEFINE FUNCTIONS ###

###
def gather_posterior_stats(tmp_path,tmp_dic,ikey,lag):
	#First gather all waveform parameter samples
	f = tmp_path + "/LIB_%s/posterior_samples/posterior_H1L1_%s-%s.dat"%(lag,tmp_dic[ikey]['gpstime'],ikey)
	with open(f) as pos_samp_file:
		pos_samps = list(pos_samp_file)
	
	freq_ind = np.nan
	qual_ind = np.nan
	hrss_ind = np.nan
	
	freq_samps = np.ones(len(pos_samps)-1)*np.nan
	qual_samps = np.ones(len(pos_samps)-1)*np.nan
	hrss_samps = np.ones(len(pos_samps)-1)*np.nan
	
	for iline,line in enumerate(pos_samps):
		#loop over all samples
		params = line.split()
		if iline == 0:
			#on header line, look for necessary indices
			for ipar, par in enumerate(params):
				if par == 'frequency':
					freq_ind = ipar
				elif par == 'quality':
					qual_ind = ipar
				elif par == 'loghrss':
					hrss_ind = ipar
		else:
			#now on sample lines
			freq_samps[iline-1] = float(params[freq_ind])
			qual_samps[iline-1] = float(params[qual_ind])
			hrss_samps[iline-1] = np.exp(float(params[hrss_ind]))
		
	#With samples, add necessary estimators to dictionaries
	tmp_dic[ikey]['frequency'] = {}
	tmp_dic[ikey]['frequency']['posterior mean'] = np.mean(freq_samps)
	tmp_dic[ikey]['frequency']['posterior median'] = np.median(freq_samps)
	
	tmp_dic[ikey]['quality'] = {}
	tmp_dic[ikey]['quality']['posterior mean'] = np.mean(qual_samps)
	tmp_dic[ikey]['quality']['posterior median'] = np.median(qual_samps)
	
	tmp_dic[ikey]['hrss'] = {}
	tmp_dic[ikey]['hrss']['posterior mean'] = np.mean(hrss_samps)
	tmp_dic[ikey]['hrss']['posterior median'] = np.median(hrss_samps)
			
	return tmp_dic

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
def effective_segs(seg_file, veto_file, outfile):
	"""
	Remove vetoes from segment list, thus creating effective segments
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
	eff_segs_file_nm = outfile
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
def constrain_2_eff_segs(trig_dic, seg_array):
	"""
	Constrain list of time-sorted triggers to lie within the passed time-sorted effective segments (i.e., apply vetoes)
	"""
	#import clustered trigs and vetoes
	tau = float(trig_dic['quality']['posterior mean']) / (np.sqrt(2.)*np.pi*float(trig_dic['frequency']['posterior mean']))  #conversion between time duration and Q
	trig_start = float(trig_dic['gpstime']) #- tau/np.sqrt(2.)
	trig_stop = float(trig_dic['gpstime']) #+ tau/np.sqrt(2.)
		
	#Check to see if trig lies completely within any segment
	if ((trig_start >= seg_array[:,0]) * (trig_stop <= seg_array[:,1])).any():
		#Trig lies completely within segment, so it survives
		survive_flag = True
	else:
		#Trig does not completely lie within segment, so disregard it
		survive_flag = False
			
	return survive_flag

#=======================================================================
### RUN COLLECTION ###
if __name__=='__main__':	
	from optparse import OptionParser

	usage = None
	parser = OptionParser(usage=usage)

	#general options
	parser.add_option("","--datadir", default=None, type='string', help="Path to folder where all PostProc directories live")
	parser.add_option("","--tshift-start", default=None, type="float", help="Starting timslide")
	parser.add_option("","--tshift-stop", default=None, type="float", help="Ending timslide")
	parser.add_option("","--tshift-num", default=None, type="float", help="Number of 1s timslides in each PostProc job")
	parser.add_option("","--outdir", default=None, type='string', help="Path to directory in which to write results")
	parser.add_option("","--veto-files", default=None, type='string', help="Comma-separated path to veto files for H1 and L1 respectively")
	parser.add_option("","--prev-dic", default=None, type='string', help="Path to dictionary with previously collected LIB results")
	parser.add_option("","--label", default=None, type='string', help="Label to use when saving results")

	#-----------------------------------------------------------------------

	opts, args = parser.parse_args()

	datadir = opts.datadir
	tshift_start = opts.tshift_start
	tshift_stop = opts.tshift_stop
	tshift_num = opts.tshift_num
	outdir = opts.outdir
	veto_files = opts.veto_files.split(',')
	prev_dic = pickle.load(open(opts.prev_dic))
	label = opts.label

	#=======================================================================

	### INITIALIZE ###
	#Initialize dictionary
	tshifts = np.linspace(tshift_start,tshift_stop,int(tshift_stop - tshift_start + 1.))
	num_jobs = int(float(tshift_stop - tshift_start + 1.)/float(tshift_num))
	dic = {}
	for ts in tshifts:
		dic[int(ts)] = {}
		dic[int(ts)]['trigs'] = {}
		dic[int(ts)]['segs'] = np.array([])
		dic[int(ts)]['length'] = 0
		
	#Initialize sanity-check variables
	fore_ntrigs = 0.
	back_ntrigs = 0.
	fore_nvetoes = 0.
	back_nvetoes = 0.

	### COLLECT RESULTS ###
	#Loop over PostProc jobs, collecting all segments for each 1s timelag
	for job in xrange(num_jobs):
		for subjob in xrange(tshift_num):
			#Collect segments
			ts = float(tshift_start+job*tshift_num + subjob)
			try:
				dic[int(ts)]['segs'] = np.append(dic[int(ts)]['segs'], np.genfromtxt('%s/PostProc_%s_%s/live_segs/intersect_H1L1_ts%s.seg'%(datadir,tshift_start+job*tshift_num,tshift_start+(job+1.)*tshift_num-1.,ts)).reshape(-1))
			except IOError:
				pass
			#Collect number of trigs for foreground and background
			try:
				fore_ntrigs += len(np.genfromtxt('%s/PostProc_%s_%s/LIB_trigs/LIB_0lag_times_H1L1.txt'%(datadir,tshift_start+job*tshift_num,tshift_start+(job+1.)*tshift_num-1.)))
			except (IOError,TypeError):
				pass
			try:
				back_ntrigs += len(np.genfromtxt('%s/PostProc_%s_%s/LIB_trigs/LIB_ts_times_H1L1.txt'%(datadir,tshift_start+job*tshift_num,tshift_start+(job+1.)*tshift_num-1.)))
			except (IOError,TypeError):
				pass
	
	#Loop over trigs in dictionary, sorting by timelag
	for ikey in prev_dic:
		#Collect dictionary	
		try:
			ts = float(prev_dic[ikey]['timeslides']['L1'])
			dic[int(ts)]['trigs'][ dic[int(ts)]['length'] ] = prev_dic[ikey]
			dic[int(ts)]['length'] += 1
		except KeyError:
			pass
	
	### COLLECT VETOES ###
	#For now just import a veto file and save it to outdir
	os.system('cp %s %s/vetos_H1.seg'%(veto_files[0],outdir))
	os.system('cp %s %s/vetos_L1.seg'%(veto_files[1],outdir))

	###  APPLY VETOES ###
	fore_dic = {}
	back_dic = {}
	fore_key = 0
	back_key = 0
	fore_livetime = 0.
	back_livetime = 0.

	for ts in tshifts:
		#Reconstruct segments into a two-columned format and save
		dic[int(ts)]['segs'] = dic[int(ts)]['segs'].reshape(-1,2)
		np.savetxt('%s/intersect_H1L1_ts%s_pre_veto.seg'%(outdir,ts),dic[int(ts)]['segs'])

		#Merge segs
		try:
			merge_segs(seg_file='%s/intersect_H1L1_ts%s_pre_veto.seg'%(outdir,ts))
		except IOError:
			continue

		#Extract vetoes from segments for each ifo and calculate livetime
		effective_segs(seg_file='%s/intersect_H1L1_ts%s_pre_veto.seg'%(outdir,ts), veto_file='%s/vetos_H1.seg'%outdir, outfile='%s/intersect_H1L1_ts%s_post_Hveto.seg'%(outdir,ts))
		try:
			tmp_ts_vetoes = np.genfromtxt('%s/vetos_L1.seg'%outdir).reshape((-1,2))
			if not tmp_ts_vetoes.any():
				tmp_ts_vetoes = np.array([[float('inf'), float('inf')]])
		except IOError:
			tmp_ts_vetoes = np.array([[float('inf'), float('inf')]])
		tmp_ts_vetoes += ts
		np.savetxt('%s/vetos_L1.seg_ts_tmp'%outdir, tmp_ts_vetoes)
		effective_segs(seg_file='%s/intersect_H1L1_ts%s_post_Hveto.seg'%(outdir,ts), veto_file='%s/vetos_L1.seg_ts_tmp'%outdir, outfile='%s/intersect_H1L1_ts%s_post_HLveto.seg'%(outdir,ts))
		os.system('rm %s/vetos_L1.seg_ts_tmp'%outdir)

		#Calculate livetime
		seg_array = np.genfromtxt('%s/intersect_H1L1_ts%s_post_HLveto.seg'%(outdir,ts)).reshape(-1,2)
		if int(ts) == 0:
			fore_livetime += np.sum(seg_array[:,1]-seg_array[:,0])
		else:
			back_livetime += np.sum(seg_array[:,1]-seg_array[:,0])
		
		#Remove vetoed triggers
		for key in dic[int(ts)]['trigs']:
			if constrain_2_eff_segs(trig_dic=dic[int(ts)]['trigs'][key], seg_array=seg_array) == True:
				if int(ts) == 0:
					fore_dic[fore_key] = dic[int(ts)]['trigs'][key]
					fore_key += 1
				else:
					back_dic[back_key] = dic[int(ts)]['trigs'][key]
					back_key += 1
			else:
				if int(ts) == 0:
					fore_nvetoes += 1
				else:
					back_nvetoes += 1
			
	### SAVE ###

	#Print collection results
	print "Foreground livetime = ", fore_livetime
	print "Background livetime = ", back_livetime
	print "Foreground LIB trigs = ", fore_ntrigs
	print "Background LIB trigs = ", back_ntrigs
	print "Foreground dic entries = ", len(fore_dic)
	print "Background dic entires = ", len(back_dic)
	print "Foreground vetoed trigs = ", fore_nvetoes
	print "Background vetoed trigs = ", back_nvetoes

	#Save collection results
	np.savetxt('%s/foreground_livetime_%s.txt'%(outdir,label), np.array([fore_livetime]))
	np.savetxt('%s/background_livetime_%s.txt'%(outdir,label), np.array([back_livetime]))
	pickle.dump(fore_dic, open('%s/foreground_events_%s.pkl'%(outdir,label),'wt'))
	pickle.dump(back_dic, open('%s/background_events_%s.pkl'%(outdir,label),'wt'))
