#!/usr/bin/python

import numpy as np
import os
import pickle
from ligo.gracedb.rest import GraceDb
import json
import LLRT_object_beta
import commands
import time

#==============================================================
#Parse user options
from optparse import OptionParser

usage = None
parser = OptionParser(usage=usage)

parser.add_option("-I","--IFOs", default=None, type="string", help="Comma separated list of ifos. E.g., H1,L1")
parser.add_option("-r", "--rundir", default=None, type="string", help="Path to run directory containing LIB and LIB_rr folders")
parser.add_option("-i","--infodir", default=None, type="string", help="Path to info directory (where sub files, etc. are stored)")
parser.add_option("-b","--bindir", default=None, type="string", help="Path to bin directory for LIB executables")
parser.add_option("","--gdb", default=False, action="store_true", help="Write above-threshold events to GraceDb")
parser.add_option("","--LIB-followup", default=False, action="store_true", help="Run in-depth LIB follow-up on preliminary LIB triggers exceeding FAR threshold")
parser.add_option("-l","--lib-label", default=None, type="string", help="Title for labeling LIB runs")
parser.add_option("","--start", default=None, type='int', help="Start time of analysis")
parser.add_option("","--stride", default=None, type='int', help="Stride length of each segment")
parser.add_option("","--overlap", default=None, type='int', help="Overlap of segments")
parser.add_option("","--lag", default=None, type="string", help="Lag type (either 0lag or ts)")
parser.add_option("","--FAR-thresh", default=None, type='float', help="FAR treshold, below which events will be followed up with LIB")
parser.add_option("","--background-dic", default=None, type='string', help='Path to dictionary containing search statistics of background events')
parser.add_option("","--background-livetime", default=None, type='string', help='Path to file containing the livetime (in s) of background events')
parser.add_option("","--signal-kde-coords", default=None, type='string', help='Path to file containing coodinates of the KDE likelihood estimate for signals')
parser.add_option("","--signal-kde-values", default=None, type='string', help='Path to file containing values of the KDE likelihood estimate for signals')
parser.add_option("","--noise-kde-coords", default=None, type='string', help='Path to file containing coodinates of the KDE likelihood estimate for noise')
parser.add_option("","--noise-kde-values", default=None, type='string', help='Path to file containing values of the KDE likelihood estimate for noise')
parser.add_option("","--train-runmode", default=None, type='string', help='Either "Signal", "Noise", or "None" depending on if user wants to run in training mode or not')
parser.add_option("","--LIB-window", default=None, type="float", help="Length of window (in s) for LIB runs")

#--------------------------------------------------------------

opts, args = parser.parse_args()

ifos = opts.IFOs.split(',')
rundir = opts.rundir
infodir = opts.infodir
bindir=opts.bindir
gdb_flag = opts.gdb
LIB_followup_flag = opts.LIB_followup
lib_label=opts.lib_label
actual_start = opts.start
stride = opts.stride
overlap = opts.overlap
lag = opts.lag
FAR_thresh = opts.FAR_thresh
back_dic_path = opts.background_dic
back_livetime = opts.background_livetime
signal_kde_coords = opts.signal_kde_coords
signal_kde_values = opts.signal_kde_values
noise_kde_coords = opts.noise_kde_coords
noise_kde_values = opts.noise_kde_values
train_runmode = opts.train_runmode
LIB_window = opts.LIB_window

#===============================================================

#Initialize GraceDb
if gdb_flag:
	gdb = GraceDb()

#Initialize dictionary
dictionary = {}

#Initialize peparser
#peparser=bppu.PEOutputParser('common')

#Find trigtimes and timeslides and add to dictionary		
timeslide_array = np.genfromtxt('%s/PostProc/LIB_trigs/LIB_%s_timeslides_%s.txt'%(rundir,lag,"".join(ifos))).reshape(-1,len(ifos))
trigtime_array = np.genfromtxt('%s/PostProc/LIB_trigs/LIB_%s_times_%s.txt'%(rundir,lag,"".join(ifos))).reshape(-1,1)
for event in xrange(len(trigtime_array)):
	dictionary[event] = {}
	dictionary[event]['gpstime'] = str(trigtime_array[event,0])
	dictionary[event]['timeslides'] = {}
	for i, ifo in enumerate(ifos):
		dictionary[event]['timeslides'][ifo] = str(timeslide_array[event,i])

#Find BSNs and waveform params
posterior_files = os.listdir("%s/LIB_%s/posterior_samples/"%(rundir,lag))
for f in posterior_files:
	if (f.split('_')[1] == "".join(ifos)) and (f.split('.')[2] == 'dat_B'):
		#Initialize dictionary for event
		event = int(f.split('-')[1].split(".")[0])
		
		#Add basic info to dictionary
		dictionary[event]['instruments'] = opts.IFOs
		dictionary[event]['nevents'] = 1
		dictionary[event]['likelihood']= None
		
		#Add BSN to dictionary
		post_file = open("%s/LIB_%s/posterior_samples/%s"%(rundir,lag,f), 'rt')
		for line in post_file:
			bsn = float(line.split()[0])
		dictionary[event]['BSN'] = bsn
		post_file.close()
		
		#First gather all waveform parameter samples
		with open("%s/LIB_%s/posterior_samples/%s"%(rundir,lag,f.split('_B.txt')[0]),'rt') as pos_samp_file:
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
		dictionary[event]['frequency'] = np.mean(freq_samps)
		dictionary[event]['quality'] = np.mean(qual_samps)
		dictionary[event]['hrss'] = np.mean(hrss_samps)

#Find BCIs		
coherence_files = os.listdir("%s/LIB_%s/coherence_test/"%(rundir,lag))
for f in coherence_files:
		#Get event
		event = int(f.split('-')[1].split(".")[0])
		
		#Add BCI to dictionary
		coh_file = open("%s/LIB_%s/coherence_test/%s"%(rundir,lag,f), 'rt')
		for line in coh_file:
			bci = float(line.split()[0])
		dictionary[event]['BCI'] = bci
		coh_file.close()

#Find Omicron SNR
event = 0
if lag == '0lag':
	#Only consider 0lag events if lag is 0lag
	try:
		trig_info_array = np.genfromtxt("%s/PostProc/LIB_trigs/LIB_trigs_%s_ts0.0.txt"%(rundir,"".join(ifos))).reshape((-1,12))
		for line in trig_info_array:
			if np.absolute(float(dictionary[event]['gpstime']) - line[0]) <= 0.01:
				dictionary[event]['Omicron SNR'] = line[2]
				event += 1
			else:
				raise ValueError("The event and trig time do not match up when finding Omicron SNR")
	except IOError:
		pass
		
elif lag == 'ts':
	#Only consider timeslide events if lag is ts
	trig_files = sorted(os.listdir("%s/PostProc/LIB_trigs/"%rundir))
	for f in trig_files:
		f_split = f.split('ts')
		if (f_split[0] == 'LIB_trigs_%s_'%("".join(ifos))) and (float(f_split[1].split('.txt')[0]) != 0.0):
			try:
				trig_info_array = np.genfromtxt("%s/PostProc/LIB_trigs/%s"%(rundir,f)).reshape((-1,12))
				for line in trig_info_array:
					if np.absolute(float(dictionary[event]['gpstime']) - line[0]) <= 0.01:
						dictionary[event]['Omicron SNR'] = line[2]
						event += 1
					else:
						raise ValueError("The event and trig time do not match up when finding Omicron SNR")
			except IOError:
				continue

#Construct LLRT object for the set of events, first gathering the foreground coordinates
#Build calc_info dictionary
calc_info = {}
calc_info['interp method'] = 'Grid Linear'
calc_info['extrap method'] = 'Grid Nearest'

#Build param_info dictionary
param_info = {}
param_info['BSN_and_BCI'] = {}
param_info['BSN_and_BCI']['dimension'] = 2
param_info['BSN_and_BCI']['param names'] = ['BSN','BCI']
param_info['BSN_and_BCI']['interp range'] = np.array([[-30., 300.],[-35., 35.]])

#Load likelihood estimate for signal
train_signal_data = {}
train_signal_data['BSN_and_BCI'] = {}
train_signal_data['BSN_and_BCI']['KDE'] = ([np.load(signal_kde_coords),np.load(signal_kde_values)])

#Load likelihood estimate for noise
train_noise_data = {}
train_noise_data['BSN_and_BCI'] = {}
train_noise_data['BSN_and_BCI']['KDE'] = ([np.load(noise_kde_coords),np.load(noise_kde_values)])

#Build foreground_data dictionary
BSNs = np.zeros(len(dictionary))
BCIs = np.zeros(len(dictionary))
oSNRs = np.zeros(len(dictionary))
for event in dictionary:
	BSNs[event] = dictionary[event]['BSN']
	BCIs[event] = dictionary[event]['BCI']
	oSNRs[event] = dictionary[event]['Omicron SNR']

foreground_data = {}
foreground_data['npoints'] = len(dictionary)
foreground_data['BSN'] = {}
foreground_data['BSN']['data'] = np.transpose(np.array([BSNs]))
foreground_data['BCI'] = {}
foreground_data['BCI']['data'] = np.transpose(np.array([BCIs]))
foreground_data['oSNR'] = {}
foreground_data['oSNR']['data'] = np.transpose(np.array([oSNRs]))

#Build background_data dictionary
try:
	float_back_livetime = float(np.genfromtxt(back_livetime))
except IOError:
	float_back_livetime = np.nan
back_dic = pickle.load(open(back_dic_path))
back_coords = np.ones((len(back_dic),3))*np.nan

for i, key in enumerate(back_dic):
	try:
		if back_dic[key]['BCI'] <= 20.:
			back_coords[i,0] = back_dic[key]['BSN']
			back_coords[i,1] = back_dic[key]['BCI']
			back_coords[i,2] = back_dic[key]['Omicron SNR']
		else:
			print 'Outrageous noise BCI for event: ', key, back_dic[key]['BCI']
	except KeyError:
		print 'Failure for noise background event: ',i

back_coords = back_coords[ back_coords >= -np.inf ].reshape(-1,3)

background_data = {}
background_data['npoints'] = len(back_coords)
background_data['BSN'] = {}
background_data['BSN']['data'] = back_coords[:,0]
background_data['BCI'] = {}
background_data['BCI']['data'] = back_coords[:,1]
background_data['oSNR'] = {}
background_data['oSNR']['data'] = back_coords[:,2]

#Initialize the LLRT object
LLRT = LLRT_object_beta.LLRT(calc_info=calc_info, param_info=param_info, train_signal_data=train_signal_data, train_noise_data=train_noise_data, foreground_data=foreground_data, background_data=background_data)

#Calculate FAR for each foreground event wrt background events
event_LLRs = LLRT.log_likelihood_ratios(groundtype='Foreground')
for event, event_LLR in enumerate(event_LLRs):
	dictionary[event]['FAR'] = LLRT.calculate_FAR_of_thresh(threshold=event_LLR, livetime=float_back_livetime, groundtype='Background')

#Save dictionary
pickle.dump(dictionary, open('%s/PostProc/LIB_trigs/LIB_%s_dic_%s.pkl'%(rundir, lag, "".join(ifos)),'wt'))

#---------------------------------------------------------------------------------------

#Prepare for handling triggers exceeding FAR threshold
rr_trigtimes = open(rundir+'/PostProc/LIB_trigs/LIB_%s_times_rr_%s.txt'%(lag,"".join(ifos)),'wt')
rr_timeslides = open(rundir+'/PostProc/LIB_trigs/LIB_%s_timeslides_rr_%s.txt'%(lag,"".join(ifos)),'wt')
e_new = -1  #e_new will mark the event number of the follow-up run

#Check to see if triggers should be collected for foreground collection
if (train_runmode == 'None') and (lag == '0lag'):
	#Check to make sure another function isn't currently updating the foreground dictionary
	while os.path.isfile('%s/result_dics/foreground_events.pkl_lock'%infodir):
		time.sleep(5)

	#Lock foreground dictionary while updating it
	if not os.path.exists('%s/result_dics/'%infodir):
		os.makedirs('%s/result_dics/'%infodir)
	os.system('> %s/result_dics/foreground_events.pkl_lock'%infodir)

	#load foreground training dictionary
	if os.path.isfile('%s/result_dics/foreground_events.pkl'%infodir):
		foreground_dic = pickle.load(open('%s/result_dics/foreground_events.pkl'%infodir))
	else:
		foreground_dic = {}
		
	#get next event key
	if len(foreground_dic.keys()) == 0:
		foreground_key = 0
	else:
		foreground_key = np.max(foreground_dic.keys()) + 1

#Check to see if triggers should be collected for background collection
elif (train_runmode == 'None') and (lag == 'ts'):
	#Check to make sure another function isn't currently updating the background dictionary
	while os.path.isfile('%s/result_dics/background_events.pkl_lock'%infodir):
		time.sleep(5)
		
	#Lock background dictionary while updating it
	if not os.path.exists('%s/result_dics/'%infodir):
		os.makedirs('%s/result_dics/'%infodir)
	os.system('> %s/result_dics/background_events.pkl_lock'%infodir)
	
	#load background training dictionary
	if os.path.isfile('%s/result_dics/background_events.pkl'%infodir):
		background_dic = pickle.load(open('%s/result_dics/background_events.pkl'%infodir))
	else:
		background_dic = {}
		
	#get next event key
	if len(background_dic.keys()) == 0:
		background_key = 0
	else:
		background_key = np.max(background_dic.keys()) + 1

#Check to see if triggers should be collected for noise training
elif (train_runmode == 'Noise') and (lag == 'ts'):
	#Check to make sure another function isn't currently updating the noise training dictionary
	while os.path.isfile('%s/training_dics/new_noise_training_points.pkl_lock'%infodir):
		time.sleep(5)
	
	#Lock noise training dictionary while updating it
	if not os.path.exists('%s/training_dics/'%infodir):
		os.makedirs('%s/training_dics/'%infodir)
	os.system('> %s/training_dics/new_noise_training_points.pkl_lock'%infodir)
	
	#load noise training dictionary
	if os.path.isfile('%s/training_dics/new_noise_training_points.pkl'%infodir):
		noise_train_dic = pickle.load(open('%s/training_dics/new_noise_training_points.pkl'%infodir))
	else:
		noise_train_dic = {}
	
	#get next event key
	if len(noise_train_dic.keys()) == 0:
		noise_train_key = 0
	else:
		noise_train_key = np.max(noise_train_dic.keys()) + 1

#Check to see if triggers should be collected for signal training	
elif (train_runmode == 'Signal') and (lag == '0lag'):
	#Check to make sure another function isn't currently updating the signal training dictionary
	while os.path.isfile('%s/training_dics/new_signal_training_points.pkl_lock'%infodir):
		time.sleep(5)
	
	#Lock signal training dictionary while updating it
	if not os.path.exists('%s/training_dics/'%infodir):
		os.makedirs('%s/training_dics/'%infodir)
	os.system('> %s/training_dics/new_signal_training_points.pkl_lock'%infodir)
	
	#load signal training dictionary
	if os.path.isfile('%s/training_dics/new_signal_training_points.pkl'%infodir):
		sig_train_dic = pickle.load(open('%s/training_dics/new_signal_training_points.pkl'%infodir))
	else:
		sig_train_dic = {}
	
	#get next event key
	if len(sig_train_dic.keys()) == 0:
		sig_train_key = 0
	else:
		sig_train_key = np.max(sig_train_dic.keys()) + 1
	
	#find the time of the training injection
	inj_time = float(commands.getstatusoutput('%s/ligolw_print %s/training_injections/raw/*.xml -c "time_geocent_gps" -r 0:1'%(bindir,rundir))[1])

#loop over all events
for event in dictionary:
	#if pipeline is not in training mode, check to see if event is in 0-lag, if so add to foreground dictionary
	if (train_runmode == 'None') and (lag == '0lag'):
		#only add 0-lag events to foreground
		foreground_dic[foreground_key] = dictionary[event]
		foreground_key += 1
	
	#if pipeline is not in training mode, check to see if event is in ts, if so add to background dictionary
	elif (train_runmode == 'None') and (lag == 'ts'):
		#only add ts events to background
		background_dic[background_key] = dictionary[event]
		background_key += 1
	
	#if pipeline is in noise training mode, check to see if event is in ts, if so add to noise training dictionary
	elif (train_runmode == 'Noise') and (lag == 'ts'):
		#only want to train noise on timeslided events
		noise_train_dic[noise_train_key] = dictionary[event]
		noise_train_key += 1
	
	#if pipeline is in signal training mode, check to see if signal is found in 0-lag, if so add to signal training dictionary if most significant event
	elif (train_runmode == 'Signal') and (lag == '0lag'):
		#Only want to train signal on 0-lag events
		if np.absolute(inj_time - float(dictionary[event]['gpstime'])) <= 0.5*LIB_window:
			#LIB needs to have run over at least part of the injection for it to be considered found
			if sig_train_dic.has_key(sig_train_key):
				#Found event has already been entered into the training dictionary
				if sig_train_dic[sig_train_key]['Omicron SNR'] < dictionary[event]['Omicron SNR']:
					#Update the training dictionary with the found event with the loudest oSNR
					sig_train_dic[sig_train_key] = dictionary[event]
			else:
				#No other events have entered the training dictionary, so enter this one
				sig_train_dic[sig_train_key] = dictionary[event]
	
	#check Bayes factors against specified thresholds
	if dictionary[event]['FAR'] <= FAR_thresh:			
		#FAR threshold is exceeded so write trigtimes and timeslides to files
		e_new += 1
		rr_trigtimes.write('%s\n'%dictionary[event]['gpstime'])
		rr_timeslides.write('%s\n'%" ".join([dictionary[event]['timeslides'][ifo] for ifo in ifos]))
		
		#If event is 0-lag, write to GraceDb if enabled
		if lag == '0lag' and gdb_flag:
			#Save dictionary as json file
			dic_path = rundir+'/GDB/%s.json'%('%s-%s'%(dictionary[event]['gpstime'],event))
			with open(dic_path, 'wt') as fp:
				json.dump(dictionary[event], fp)
			
			#Upload dictionary to GraceDb
			response = gdb.createEvent('Burst','LIB',dic_path, search='AllSky', filecontents=None)
			
			#Parse GraceDb ID
			response = json.loads(response.read())
			gid = str(response["graceid"])
			
			#Update GraceDb log with post-proc pages
			gdb.writeLog(gid, message="oLIB preliminary estimates: frequency = %s, quality = %s, hrss = %s, logBCI = %s, logBSN= %s, oSNR = %s"%(dictionary[event]['frequency'],dictionary[event]['quality'],dictionary[event]['hrss'],dictionary[event]['BCI'],dictionary[event]['BSN'],dictionary[event]['Omicron SNR']), tagname='pe')
			if LIB_followup_flag:
				gdb.writeLog(gid, message="Follow-up results will be written: https://ldas-jobs.ligo.caltech.edu/~ryan.lynch/%s/%s/followup/%s/%s/%s/posplots.html"%(lib_label,lag,'%s_%s_%s'%("".join(ifos),actual_start,stride-overlap),'%s-%s'%(dictionary[event]['gpstime'],e_new),"".join(ifos)), tagname='pe')

#store results of loop over events
rr_trigtimes.close()
rr_timeslides.close()

if (train_runmode == 'None') and (lag == '0lag'):
	#Save foreground dictionary and test that it is not corrupt
	pickle.dump(foreground_dic, open('%s/result_dics/foreground_events.pkl_tmp'%infodir,'wt'))
	if pickle.load(open('%s/result_dics/foreground_events.pkl_tmp'%infodir)) == foreground_dic:
		os.system('mv %s/result_dics/foreground_events.pkl_tmp %s/result_dics/foreground_events.pkl'%(infodir,infodir))
	else:
		os.system('rm %s/result_dics/foreground_events.pkl_tmp'%infodir)
		os.system('rm %s/result_dics/foreground_events.pkl_lock'%infodir)
		raise ValueError, "Something corrupt in foreground dictionary"
	#Unlock foreground dictionary now that it's updated
	os.system('rm %s/result_dics/foreground_events.pkl_lock'%infodir)
	
elif (train_runmode == 'None') and (lag == 'ts'):
	#Save background dictionary and test that it is not corrupt
	pickle.dump(background_dic, open('%s/result_dics/background_events.pkl_tmp'%infodir,'wt'))
	if pickle.load(open('%s/result_dics/background_events.pkl_tmp'%infodir)) == background_dic:
		os.system('mv %s/result_dics/background_events.pkl_tmp %s/result_dics/background_events.pkl'%(infodir,infodir))
	else:
		os.system('rm %s/result_dics/background_events.pkl_tmp'%infodir)
		os.system('rm %s/result_dics/background_events.pkl_lock'%infodir)
		raise ValueError, "Something corrupt in background dictionary"
	#Unlock background dictionary now that it's updated
	os.system('rm %s/result_dics/background_events.pkl_lock'%infodir)

elif (train_runmode == 'Noise') and (lag == 'ts'):
	#Save noise training dictionary and test that it is not corrupt
	pickle.dump(noise_train_dic, open('%s/training_dics/new_noise_training_points.pkl_tmp'%infodir,'wt'))
	if pickle.load(open('%s/training_dics/new_noise_training_points.pkl_tmp'%infodir)) == noise_train_dic:
		os.system('mv %s/training_dics/new_noise_training_points.pkl_tmp %s/training_dics/new_noise_training_points.pkl'%(infodir,infodir))
	else:
		os.system('rm %s/training_dics/new_noise_training_points.pkl_tmp'%infodir)
		os.system('rm %s/training_dics/new_noise_training_points.pkl_lock'%infodir)
		raise ValueError, "Something corrupt in noise training dictionary"
	#Unlock noise training dictionary now that it's updated
	os.system('rm %s/training_dics/new_noise_training_points.pkl_lock'%infodir)
	
elif (train_runmode == 'Signal') and (lag == '0lag'):
	#Save signal training dictionary and test that it is not corrupt
	pickle.dump(sig_train_dic, open('%s/training_dics/new_signal_training_points.pkl_tmp'%infodir,'wt'))
	if pickle.load(open('%s/training_dics/new_signal_training_points.pkl_tmp'%infodir)) == sig_train_dic:
		os.system('mv %s/training_dics/new_signal_training_points.pkl_tmp %s/training_dics/new_signal_training_points.pkl'%(infodir,infodir))
	else:
		os.system('rm %s/training_dics/new_signal_training_points.pkl_tmp'%infodir)
		os.system('rm %s/training_dics/new_signal_training_points.pkl_lock'%infodir)
		raise ValueError, "Something corrupt in signal training dictionary"
	#Unlock signal training dictionary now that it's updated
	os.system('rm %s/training_dics/new_signal_training_points.pkl_lock'%infodir)

#run pipeline to make dag for reruns
if LIB_followup_flag:
	os.system('%s/lalinference_2ndpipe_beta.py %s/runfiles/LIB_%s_reruns_beta.ini -r %s/LIB_%s_rr/ -p /usr1/ryan.lynch/logs/ -g %s/PostProc/LIB_trigs/LIB_%s_times_rr_%s.txt'%(infodir, rundir, lag, rundir, lag, rundir, lag, "".join(ifos)))
