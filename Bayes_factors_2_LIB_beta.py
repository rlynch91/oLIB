#!/usr/bin/python

import numpy as np
import os
import pickle
from ligo.gracedb.rest import GraceDb
import json
#from pylal import bayespputils as bppu
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
parser.add_option("","--gdb", default=False, action="store_true", help="Write above-threshold events to GraceDb")
parser.add_option("","--LIB-followup", default=False, action="store_true", help="Run in-depth LIB follow-up on preliminary LIB triggers exceeding FAR threshold")
parser.add_option("-l","--lib-label", default=None, type="string", help="Title for labeling LIB runs")
parser.add_option("","--start", default=None, type='int', help="Start time of analysis")
parser.add_option("","--stride", default=None, type='int', help="Stride length of each segment")
parser.add_option("","--overlap", default=None, type='int', help="Overlap of segments")
parser.add_option("","--lag", default=None, type="string", help="Lag type (either 0lag or ts)")
parser.add_option("","--FAR-thresh", default=None, type='float', help="FAR treshold, below which events will be followed up with LIB")
parser.add_option("","--background-dic", default=None, type='string', help='Path to dictionary containing search statistics of background events')
parser.add_option("","--background-livetime", default=None, type='float', help='Livetime (in s) of background events')
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
		
		#Find waveform params
#		commonResultsObj=peparser.parse(open("%s/LIB_%s/posterior_samples/%s"%(rundir,lag,f.split('_B.txt')[0]),'rt'),info=[None,None])
#		pos = bppu.BurstPosterior(commonResultsObj)
#		statmax_pos,max_j=pos._posMap()
#		for param in ['frequency','quality']:
#			dictionary[event][param]=pos[param].samples[max_j][0]
			

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
		trig_info_array = np.genfromtxt("%s/PostProc/LIB_trigs/LIB_trigs_H1L1_ts0.0.txt"%rundir).reshape((-1,12))
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
		if (f_split[0] == 'LIB_trigs_H1L1_') and (float(f_split[1].split('.txt')[0]) != 0.0):
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
param_info['BSN_and_BCI_and_oSNR'] = {}
param_info['BSN_and_BCI_and_oSNR']['dimension'] = 3
param_info['BSN_and_BCI_and_oSNR']['param names'] = ['BSN','BCI','oSNR']
param_info['BSN_and_BCI_and_oSNR']['interp range'] = np.array([[0.,40.],[-10., 10.],[7.7, 12.]])

#Load likelihood estimate for signal
train_signal_data = {}
train_signal_data['BSN_and_BCI_and_oSNR'] = {}
train_signal_data['BSN_and_BCI_and_oSNR']['KDE'] = ([np.load(signal_kde_coords),np.load(signal_kde_values)])

#Load likelihood estimate for noise
train_noise_data = {}
train_noise_data['BSN_and_BCI_and_oSNR'] = {}
train_noise_data['BSN_and_BCI_and_oSNR']['KDE'] = ([np.load(noise_kde_coords),np.load(noise_kde_values)])

#Build foreground_data dictionary
BSNs = np.zeros(len(dictionary))
BCIs = np.zeros(len(dictionary))
oSNRs = np.zeros(len(dictionary))
for event in dictionary:
	BSNs[event] = dictionary[event]['BSN']
	BCIs[event] = dictionary[event]['BCI']
	oSNRs[event] = dictionary[event]['Omicron SNR']

foreground_data={}
foreground_data = {}
foreground_data['npoints'] = len(dictionary)
foreground_data['BSN'] = {}
foreground_data['BSN']['data'] = np.transpose(np.array([BSNs]))
foreground_data['BCI'] = {}
foreground_data['BCI']['data'] = np.transpose(np.array([BCIs]))
foreground_data['oSNR'] = {}
foreground_data['oSNR']['data'] = np.transpose(np.array([oSNRs]))

#Build background_data dictionary
back_dic = pickle.load(open(back_dic_path))
back_coords = np.zeros((len(back_dic),3))

for i, key in enumerate(back_dic):
	try:
		if back_dic[key]['BCI'] <= 100:
			back_coords[i,0] = back_dic[key]['BSN']
			back_coords[i,1] = back_dic[key]['BCI']
			back_coords[i,2] = back_dic[key]['Omicron SNR']
		else:
			print 'Outrageous noise BCI for event: ', key, back_dic[key]['BCI']
	except KeyError:
		print 'Failure for noise background event: ',i

back_coords = back_coords[(back_coords != np.array([0,0,0])).any(axis=1)]

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
	dictionary[event]['FAR'] = LLRT.calculate_FAR_of_thresh(threshold=event_LLR, livetime=back_livetime, groundtype='Background')

#Save dictionary
pickle.dump(dictionary, open('%s/PostProc/LIB_trigs/LIB_%s_dic_%s.pkl'%(rundir, lag, "".join(ifos)),'wt'))

#---------------------------------------------------------------------------------------

#Prepare for handling triggers exceeding FAR threshold
rr_trigtimes = open(rundir+'/PostProc/LIB_trigs/LIB_%s_times_rr_%s.txt'%(lag,"".join(ifos)),'wt')
rr_timeslides = open(rundir+'/PostProc/LIB_trigs/LIB_%s_timeslides_rr_%s.txt'%(lag,"".join(ifos)),'wt')
e_new = -1  #e_new will mark the event number of the follow-up run

#Check to see if triggers should be collected for noise training
if (train_runmode == 'Noise') and (lag == 'ts'):
	#Check to make sure another function isn't currently updating the noise training dictionaries
	while os.path.isfile('%s/training_dics/new_noise_training_points.pkl_lock'%infodir):
		time.sleep(5)
	
	#Lock noise training dictionaries while updating them
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
if (train_runmode == 'Signal') and (lag == '0lag'):
	#Check to make sure another function isn't currently updating the signal training dictionaries
	while os.path.isfile('%s/training_dics/new_signal_training_points.pkl_lock'%infodir):
		time.sleep(5)
	
	#Lock signal training dictionaries while updating them
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
	inj_time = float(commands.getstatusoutput('ligolw_print %s/training_injections/raw/*.xml -c "time_geocent_gps" -r 0:1'%rundir)[1])

#loop over all events
for event in dictionary:
	#if pipeline is in noise training mode, check to see if event is in ts, if so add to training dictionary
	if (train_runmode == 'Noise') and (lag == 'ts'):
		#only want to train noise on timeslided events
		noise_train_dic[noise_train_key] = dictionary[event]
		noise_train_key += 1
	
	#if pipeline is in signal training mode, check to see if signal is found in 0-lag, if so add to training dictionary if most significant event
	if (train_runmode == 'Signal') and (lag == '0lag'):
		#Only want to train signal on 0-lag events
		if np.absolute(inj_time - dictionary[event]['gpstime']) <= 0.5*LIB_window:
			#LIB needs to have run over at least part of the injection for it to be considered found
			if signal_train_dic.has_key(signal_train_key):
				#Found event has already been entered into the training dictionary
				if signal_train_dic[signal_train_key]['Omicron SNR'] < dictionary[event]['Omicron SNR']:
					#Update the training dictionary with the found event with the loudest oSNR
					signal_train_dic[signal_train_key] = dictionary[event]
			else:
				#No other events have entered the training dictionary, so enter this one
				signal_train_dic[signal_train_key] = dictionary[event]
	
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
#			gdb.writeLog(gid, message="Preliminary results: f_0 = %s, Q = %s"%(dictionary[event]['frequency'],dictionary[event]['quality']))
			gdb.writeLog(gid, message="Preliminary results: BSN = %s, BCI = %s, oSNR = %s"%(dictionary[event]['BSN'],dictionary[event]['BCI'],dictionary[event]['Omicron SNR']))
			if LIB_followup_flag:
				gdb.writeLog(gid, message="Follow-up results will be written: https://ldas-jobs.ligo.caltech.edu/~ryan.lynch/%s/%s/followup/%s/%s/%s/posplots.html"%(lib_label,lag,'%s_%s_%s'%("".join(ifos),actual_start,stride-overlap),'%s-%s'%(dictionary[event]['gpstime'],e_new),"".join(ifos)))

#store results of loop over events
rr_trigtimes.close()
rr_timeslides.close()

if (train_runmode == 'Noise') and (lag == 'ts'):
	#Save noise training dicitonary
	pickle.dump(noise_train_dic, open('%s/training_dics/new_noise_training_points.pkl'%infodir,'wt'))
	#Unlock noise training dictionaries now that they're updated
	os.system('rm %s/training_dics/new_noise_training_points.pkl_lock'%infodir)
	
if (train_runmode == 'Signal') and (lag == '0lag'):
	#Save signal training dictionary
	pickle.dump(signal_train_dic, open('%s/training_dics/new_signal_training_points.pkl'%infodir,'wt'))
	#Unlock signal training dictionaries now that they're updated
	os.system('rm %s/training_dics/new_signal_training_points.pkl_lock'%infodir)

#run pipeline to make dag for reruns
if LIB_followup_flag:
	os.system('%s/lalinference_2ndpipe_beta.py %s/runfiles/LIB_%s_reruns_beta.ini -r %s/LIB_%s_rr/ -p /usr1/ryan.lynch/logs/ -g %s/PostProc/LIB_trigs/LIB_%s_times_rr_%s.txt'%(infodir, rundir, lag, rundir, lag, rundir, lag, "".join(ifos)))
