#!/usr/bin/python

import numpy as np
import os
import pickle
from ligo.gracedb.rest import GraceDb
import json
#from pylal import bayespputils as bppu
import LLRT_object_beta

#==============================================================
#Parse user options
from optparse import OptionParser

usage = None
parser = OptionParser(usage=usage)

parser.add_option("-I","--IFOs", default=None, type="string", help="Comma separated list of ifos. E.g., H1,L1")
parser.add_option("-r", "--rundir", default=None, type="string", help="Path to run directory containing LIB and LIB_rr folders")
parser.add_option("-i","--infodir", default=None, type="string", help="Path to info directory (where sub files, etc. are stored)")
parser.add_option("","--gdb", default=False, action="store_true", help="Write above-threshold events to GraceDb")
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

#--------------------------------------------------------------

opts, args = parser.parse_args()

ifos = opts.IFOs.split(',')
rundir = opts.rundir
infodir = opts.infodir
gdb_flag = opts.gdb
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
print dictionary

#---------------------------------------------------------------------------------------

#loop over all events, checking Bayes factors against specified thresholds
rr_trigtimes = open(rundir+'/PostProc/LIB_trigs/LIB_%s_times_rr_%s.txt'%(lag,"".join(ifos)),'wt')
rr_timeslides = open(rundir+'/PostProc/LIB_trigs/LIB_%s_timeslides_rr_%s.txt'%(lag,"".join(ifos)),'wt')
e_new = -1  #e_new will mark the event number of the follow-up run
for event in dictionary:
	if dictionary[event]['FAR'] <= FAR_thresh:			
		#if FAR threshold is exceeded, write trigtimes and timeslides to files
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
			gdb.writeLog(gid, message="Follow-up results will be written: https://ldas-jobs.ligo.caltech.edu/~ryan.lynch/%s/%s/followup/%s/%s/%s/posplots.html"%(lib_label,lag,'%s_%s_%s'%("".join(ifos),actual_start,stride-overlap),'%s-%s'%(dictionary[event]['gpstime'],e_new),"".join(ifos)))

rr_trigtimes.close()
rr_timeslides.close()

#run pipeline to make dag for reruns
os.system('%s/lalinference_2ndpipe_beta.py %s/runfiles/LIB_%s_reruns_beta.ini -r %s/LIB_%s_rr/ -p /usr1/ryan.lynch/logs/ -g %s/PostProc/LIB_trigs/LIB_%s_times_rr_%s.txt'%(infodir, rundir, lag, rundir, lag, rundir, lag, "".join(ifos)))
