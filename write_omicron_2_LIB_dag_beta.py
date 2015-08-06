#!/usr/bin/python

import numpy as np
import os

##############################################
from optparse import OptionParser

usage = None
parser = OptionParser(usage=usage)

parser.add_option("-I","--IFOs", default=None, type="string", help="Comma separated list of ifos. E.g., H1,L1")
parser.add_option("-r","--rundir", default=None, type="string", help="Path to run directory")
parser.add_option("-i","--infodir", default=None, type="string", help="Path to info directory (where sub files, etc. are stored)")
parser.add_option("-b","--bindir", default=None, type="string", help="Path to bin directory for LIB executables")
parser.add_option("-l","--lib-label", default=None, type="string", help="Title for labeling LIB runs")
parser.add_option("-s","--seg-files", default=None, type="string", help="Comma separated list of paths to files containing segment start/stop times for each ifo")
parser.add_option("-c","--cache-files", default=None, type="string", help="Comma separated list of paths to frame cache files for each ifo")
parser.add_option("","--start", default=None, type='int', help="Start time of analysis")
parser.add_option("","--stride", default=None, type='int', help="Stride length of each segment")
parser.add_option("","--overlap", default=None, type='int', help="Overlap of segments")
parser.add_option("","--channel-names", default=None, type='string', help="Comma separated name of channels to run on (frames) for each ifo")
parser.add_option("","--channel-types", default=None, type='string', help="Comma separated types of channel to run on (for purposes of ligo-data-find) for each ifo")
parser.add_option("","--gdb", default=False, action="store_true", help="Write above-threshold events to GraceDb")
parser.add_option("","--LIB", default=False, action="store_true", help="Run LIB to follow up Omicron triggers")
parser.add_option("","--LIB-followup", default=False, action="store_true", help="Run in-depth LIB follow-up on preliminary LIB triggers exceeding FAR threshold")
parser.add_option("","--t-shift-start", default=None, type="float", help="Starting time shift to apply to IFO 2")
parser.add_option("","--t-shift-stop", default=None, type="float", help="Ending time shift to apply to IFO 2")
parser.add_option("","--t-shift-num", default=None, type="float", help="Number of time shifts to apply to IFO 2")
parser.add_option("","--dt-signal-kde-coords", default=None, type='string', help='Path to file containing coodinates of the KDE likelihood estimate of dt for signals')
parser.add_option("","--dt-signal-kde-values", default=None, type='string', help='Path to file containing values of the KDE likelihood estimate of dt for signals')
parser.add_option("","--dt-noise-kde-coords", default=None, type='string', help='Path to file containing coodinates of the KDE likelihood estimate of dt for noise')
parser.add_option("","--dt-noise-kde-values", default=None, type='string', help='Path to file containing values of the KDE likelihood estimate of dt for noise')
parser.add_option("","--FAR-thresh", default=None, type='float', help="FAR treshold, below which events will be followed up with LIB")
parser.add_option("","--background-dic", default=None, type='string', help='Path to dictionary containing search statistics of background events')
parser.add_option("","--background-livetime", default=None, type='float', help='Livetime (in s) of background events')
parser.add_option("","--oLIB-signal-kde-coords", default=None, type='string', help='Path to file containing coodinates of the KDE likelihood estimate of oLIB for signals')
parser.add_option("","--oLIB-signal-kde-values", default=None, type='string', help='Path to file containing values of the KDE likelihood estimate of oLIB for signals')
parser.add_option("","--oLIB-noise-kde-coords", default=None, type='string', help='Path to file containing coodinates of the KDE likelihood estimate of oLIB for noise')
parser.add_option("","--oLIB-noise-kde-values", default=None, type='string', help='Path to file containing values of the KDE likelihood estimate of oLIB for noise')
parser.add_option("","--train-runmode", default=None, type='string', help='Either "Signal", "Noise", or "None" depending on if user wants to run in training mode or not')

#---------------------------------------------

opts, args = parser.parse_args()

ifos=opts.IFOs.split(",")
rundir=opts.rundir
infodir=opts.infodir
bindir=opts.bindir
lib_label=opts.lib_label
seg_files=opts.seg_files.split(",")
cache_files=opts.cache_files.split(",")
actual_start = opts.start
stride = opts.stride
overlap = opts.overlap
channel_names = opts.channel_names.split(',')
channel_types = opts.channel_types.split(',')
gdb_flag = opts.gdb
LIB_flag = opts.LIB
LIB_followup_flag = opts.LIB_followup
t_shift_start = opts.t_shift_start
t_shift_stop = opts.t_shift_stop
t_shift_num = opts.t_shift_num
dt_signal_kde_coords = opts.dt_signal_kde_coords
dt_signal_kde_values = opts.dt_signal_kde_values
dt_noise_kde_coords = opts.dt_noise_kde_coords
dt_noise_kde_values = opts.dt_noise_kde_values
FAR_thresh = opts.FAR_thresh
back_dic_path = opts.background_dic
back_livetime = opts.background_livetime
oLIB_signal_kde_coords = opts.oLIB_signal_kde_coords
oLIB_signal_kde_values = opts.oLIB_signal_kde_values
oLIB_noise_kde_coords = opts.oLIB_noise_kde_coords
oLIB_noise_kde_values = opts.oLIB_noise_kde_values
train_runmode = opts.train_runmode

#############################################

##################
### Initialize ###
##################

#make log directory
if not os.path.exists("%s/log/"%rundir):
	os.makedirs("%s/log/"%rundir)
	
#make dag directory
if not os.path.exists("%s/dag/"%rundir):
	os.makedirs("%s/dag/"%rundir)
	
#make runfiles directory
if not os.path.exists("%s/runfiles/"%rundir):
	os.makedirs("%s/runfiles/"%rundir)

#open dag file
dagfile = open("%s/dag/2ndPipeDag_%s_%s_%s.dag"%(rundir,"".join(ifos),actual_start,stride-overlap),'wt')

#initialize job number
job = 0

#gather channel types and channel names into dictionaries
chtypes_dic = {}
chnames_dic = {}
for i,ifo in enumerate(ifos):
	chtypes_dic[ifo] = "%s"%channel_types[i]
	chnames_dic[ifo] = "%s:%s"%(ifo,channel_names[i])

#############################################
### Write DAG to run omicron for each ifo ###
#############################################

omicron_jobs=[]

#Loop over all ifos
for i,ifo in enumerate(ifos):
	
	#make raw directory
	if not os.path.exists("%s/raw/%s"%(rundir,ifo)):
		os.makedirs("%s/raw/%s"%(rundir,ifo))

	#replace all IFO in omicron sub file with ifo
	os.system('sed -e "s|IFO|%s|g" -e "s|RUNDIR|%s|g" %s/omicron_beta.sub > %s/runfiles/omicron_%s_beta.sub'%(ifo,rundir,infodir,rundir,ifo))
	#replace all necessary variables in params file
	os.system('sed -e "s|IFO|%s|g" -e "s|FRAMECACHE|%s|g" -e "s|CHNAME|%s|g" -e "s|OLAP|%s|g" -e "s|STRIDE|%s|g" -e "s|RAWDIR|%s|g" %s/omicron_params_beta.txt > %s/runfiles/omicron_params_%s_beta.txt'%(ifo, cache_files[i], channel_names[i], overlap, stride, rundir+'/raw/'+ifo, infodir, rundir, ifo))
	if train_runmode == 'Signal':
		os.system('sed -e "s|//**INJECTION|INJECTION|g" %s/runfiles/omicron_params_%s_beta.txt > %s/tmp.txt; mv %s/tmp.txt %s/runfiles/omicron_params_%s_beta.txt'%(rundir,ifo,rundir,rundir,rundir,ifo))

	#write JOB
	dagfile.write('JOB %s %s/runfiles/omicron_%s_beta.sub\n'%(job,rundir,ifo))
	#write VARS
	dagfile.write('VARS %s macroid="omicron-%s-%s" macroarguments="%s %s"\n'%(job, ifo, job, seg_files[i], rundir+'/runfiles/omicron_params_%s_beta.txt'%ifo))
	#write RETRY
	dagfile.write('RETRY %s 0\n\n'%job)
	
	#Record omicron job numbers
	omicron_jobs += [job]
	
	#Done with job
	job += 1

####################################
### Write DAG to run omicron2LIB ###
####################################

omicron2LIB_jobs = []

#Copy omicron2LIB sub file to rundir
os.system('sed "s|RUNDIR|%s|g" %s/omicron2LIB_beta.sub > %s/runfiles/omicron2LIB_beta.sub'%(rundir,infodir,rundir))

#Create PostProc folder
if not os.path.exists("%s/PostProc/"%rundir):
	os.makedirs("%s/PostProc/"%rundir)

#Create vetoes folder with empty veto file
if not os.path.exists("%s/vetoes/"%rundir):
	os.makedirs("%s/vetoes/"%rundir)
os.system('touch %s/vetoes/null_vetoes.txt'%rundir)
	
#Write JOB
dagfile.write('JOB %s %s/runfiles/omicron2LIB_beta.sub\n'%(job,rundir))
#Write VARS
dagfile.write('VARS %s macroid="omicron2LIB-%s" macroarguments="-p %s/PostProc -I %s -r %s/raw -c %s --cluster-t=0.1 --coin-t=0.05 --coin-snr=0. --t-shift-start=%s --t-shift-stop=%s --t-shift-num=%s --segs=%s --veto-files=%s/vetoes/null_vetoes.txt,%s/vetoes/null_vetoes.txt --log-like-thresh=0. --LIB-window=0.1 --signal-kde-coords=%s --signal-kde-values=%s --noise-kde-coords=%s --noise-kde-values=%s"\n'%(job,job,rundir,opts.IFOs,rundir,",".join(channel_names),t_shift_start,t_shift_stop,t_shift_num,opts.seg_files,rundir,rundir,dt_signal_kde_coords,dt_signal_kde_values,dt_noise_kde_coords,dt_noise_kde_values))
#Write RETRY
dagfile.write('RETRY %s 0\n\n'%job)

#Record omicron2LIB job number
omicron2LIB_jobs += [job]

#Done with job
job += 1

####################################
### Check if supposed to run LIB ###
####################################

if LIB_flag:

	####################################################
	### Write DAG to run lalinference_pipe on 0-lags ###
	####################################################

	lalinference_pipe_0lag_jobs = []

	#Create LIB_0lag folder
	if not os.path.exists("%s/LIB_0lag/"%rundir):
		os.makedirs("%s/LIB_0lag/"%rundir)

	#replace all necessary fields in LIB_runs_beta.ini file
	os.system('sed -e "s|IFOSCOMMA|%s|g" -e "s|IFOSTOGETHER|%s|g" -e "s|LIBLABEL|%s|g" -e "s|SEGNAME|%s|g" -e "s|RUNDIR|%s|g" -e "s|BINDIR|%s|g" -e "s|CHANNELTYPES|%s|g" -e "s|CHANNELNAMES|%s|g" -e "s|LAG|0lag|g" %s/LIB_runs_beta.ini > %s/runfiles/LIB_0lag_runs_beta.ini'%(ifos,"".join(ifos),lib_label,"%s_%s_%s"%("".join(ifos),actual_start,stride-overlap),rundir,bindir,chtypes_dic,chnames_dic,infodir,rundir))
	if train_runmode == 'Signal':
		os.system('sed -e "s|START|%s|g" -e "s|STOP|%s|g" -e "s|#mdc|mdc|g" -e "s|#MDC|MDC|g" %s/runfiles/LIB_0lag_runs_beta.ini > %s/tmp.txt; mv %s/tmp.txt %s/runfiles/LIB_0lag_runs_beta.ini'%(actual_start-int(0.5*overlap),actual_start-int(0.5*overlap)+stride,rundir,rundir,rundir,rundir))
		
	#Copy lalinference_pipe sub file to rundir
	os.system('sed "s|RUNDIR|%s|g" %s/lalinference_pipe_beta.sub > %s/runfiles/lalinference_pipe_beta.sub'%(rundir,infodir,rundir))

	#Write JOB
	dagfile.write('JOB %s %s/runfiles/lalinference_pipe_beta.sub\n'%(job,rundir))
	#Write VARS
	dagfile.write('VARS %s macroid="lalinference_pipe_0lag-%s" macroarguments="%s -r %s -p /usr1/ryan.lynch/logs/ -g %s/PostProc/LIB_trigs/LIB_0lag_times_%s.txt"\n'%(job,job,rundir+'/runfiles/LIB_0lag_runs_beta.ini',rundir+'/LIB_0lag/',rundir,"".join(ifos)))
	#Write RETRY
	dagfile.write('RETRY %s 0\n\n'%job)

	#Record lalinference_pipe job number
	lalinference_pipe_0lag_jobs += [job]

	#Done with job
	job += 1
	
	########################################################
	### Write DAG to run lalinference_pipe on timeslides ###
	########################################################

	lalinference_pipe_ts_jobs = []

	#Create LIB_ts folder
	if not os.path.exists("%s/LIB_ts/"%rundir):
		os.makedirs("%s/LIB_ts/"%rundir)

	#replace all necessary fields in LIB_runs_beta.ini file
	os.system('sed -e "s|IFOSCOMMA|%s|g" -e "s|IFOSTOGETHER|%s|g" -e "s|LIBLABEL|%s|g" -e "s|SEGNAME|%s|g" -e "s|RUNDIR|%s|g" -e "s|BINDIR|%s|g" -e "s|CHANNELTYPES|%s|g" -e "s|CHANNELNAMES|%s|g" -e "s|LAG|ts|g" %s/LIB_runs_beta.ini > %s/runfiles/LIB_ts_runs_beta.ini'%(ifos,"".join(ifos),lib_label,"%s_%s_%s"%("".join(ifos),actual_start,stride-overlap),rundir,bindir,chtypes_dic,chnames_dic,infodir,rundir))
	if train_runmode == 'Signal':
		os.system('sed -e "s|START|%s|g" -e "s|STOP|%s|g" -e "s|#mdc|mdc|g" -e "s|#MDC|MDC|g" %s/runfiles/LIB_ts_runs_beta.ini > %s/tmp.txt; mv %s/tmp.txt %s/runfiles/LIB_ts_runs_beta.ini'%(actual_start-int(0.5*overlap),actual_start-int(0.5*overlap)+stride,rundir,rundir,rundir,rundir))
	
	#Write JOB
	dagfile.write('JOB %s %s/runfiles/lalinference_pipe_beta.sub\n'%(job,rundir))
	#Write VARS
	dagfile.write('VARS %s macroid="lalinference_pipe_ts-%s" macroarguments="%s -r %s -p /usr1/ryan.lynch/logs/ -g %s/PostProc/LIB_trigs/LIB_ts_times_%s.txt"\n'%(job,job,rundir+'/runfiles/LIB_ts_runs_beta.ini',rundir+'/LIB_ts/',rundir,"".join(ifos)))
	#Write RETRY
	dagfile.write('RETRY %s 0\n\n'%job)

	#Record lalinference_pipe job number
	lalinference_pipe_ts_jobs += [job]

	#Done with job
	job += 1

	###############################################
	### Write DAG to point to LIB_0lag_runs dag ###
	###############################################

	LIB_0lag_runs_jobs = []

	#Write SUBDAG EXTERNAL
	dagfile.write('SUBDAG EXTERNAL %s %s/LIB_0lag/LIB_runs.dag\n'%(job,rundir))
	#Write RETRY
	dagfile.write('RETRY %s 0\n\n'%job)

	#Record LIB_runs job number
	LIB_0lag_runs_jobs += [job]

	#Done with job
	job += 1
	
	#############################################
	### Write DAG to point to LIB_ts_runs dag ###
	#############################################

	LIB_ts_runs_jobs = []

	#Write SUBDAG EXTERNAL
	dagfile.write('SUBDAG EXTERNAL %s %s/LIB_ts/LIB_runs.dag\n'%(job,rundir))
	#Write RETRY
	dagfile.write('RETRY %s 0\n\n'%job)

	#Record LIB_runs job number
	LIB_ts_runs_jobs += [job]

	#Done with job
	job += 1

	######################################################
	### Write DAG to run Bayes_factors_2_LIB for 0-lag ###
	######################################################

	Bayes2LIB_0lag_jobs = []

	#Create LIB_0lag_rr folder
	if not os.path.exists("%s/LIB_0lag_rr/"%rundir):
		os.makedirs("%s/LIB_0lag_rr/"%rundir)
	
	#Create GraceDb folder
	if not os.path.exists("%s/GDB/"%rundir):
		os.makedirs("%s/GDB/"%rundir)

	#replace all necessary fields in LIB_reruns_beta.ini file if running follow-up
	if LIB_followup_flag:
		os.system('sed -e "s|IFOSCOMMA|%s|g" -e "s|IFOSTOGETHER|%s|g" -e "s|LIBLABEL|%s|g" -e "s|SEGNAME|%s|g" -e "s|RUNDIR|%s|g" -e "s|BINDIR|%s|g" -e "s|CHANNELTYPES|%s|g" -e "s|CHANNELNAMES|%s|g" -e "s|LAG|0lag|g" %s/LIB_reruns_beta.ini > %s/runfiles/LIB_0lag_reruns_beta.ini'%(ifos,"".join(ifos),lib_label,"%s_%s_%s"%("".join(ifos),actual_start,stride-overlap),rundir,bindir,chtypes_dic,chnames_dic,infodir,rundir))
		if train_runmode == 'Signal':
			os.system('sed -e "s|START|%s|g" -e "s|STOP|%s|g" -e "s|#mdc|mdc|g" -e "s|#MDC|MDC|g" %s/runfiles/LIB_0lag_reruns_beta.ini > %s/tmp.txt; mv %s/tmp.txt %s/runfiles/LIB_0lag_reruns_beta.ini'%(actual_start-int(0.5*overlap),actual_start-int(0.5*overlap)+stride,rundir,rundir,rundir,rundir))

	#Copy Bayes2LIB sub file to rundir
	os.system('sed "s|RUNDIR|%s|g" %s/Bayes2LIB_beta.sub > %s/runfiles/Bayes2LIB_beta.sub'%(rundir,infodir,rundir))

	#Write JOB
	dagfile.write('JOB %s %s/runfiles/Bayes2LIB_beta.sub\n'%(job,rundir))
	#Write VARS
	B2L_args = "-I %s -r %s -i %s --lib-label=%s --start=%s --stride=%s --overlap=%s --lag=0lag --FAR-thresh=%s --background-dic=%s --background-livetime=%s --signal-kde-coords=%s --signal-kde-values=%s --noise-kde-coords=%s --noise-kde-values=%s --train-runmode=%s --LIB-window=0.1"%(",".join(ifos),rundir,infodir,lib_label,actual_start,stride,overlap,FAR_thresh,back_dic_path,back_livetime,oLIB_signal_kde_coords,oLIB_signal_kde_values,oLIB_noise_kde_coords,oLIB_noise_kde_values,train_runmode)
	if gdb_flag:
		B2L_args += " --gdb"
	if LIB_followup_flag:
		B2L_args += " --LIB-followup"
	dagfile.write('VARS %s macroid="Bayes2LIB_0lag-%s" macroarguments="%s"\n'%(job,job,B2L_args))
	#Write RETRY
	dagfile.write('RETRY %s 0\n\n'%job)

	#Record lalinference_pipe job number
	Bayes2LIB_0lag_jobs += [job]

	#Done with job
	job += 1
	
	###########################################################
	### Write DAG to run Bayes_factors_2_LIB for timeslides ###
	###########################################################

	Bayes2LIB_ts_jobs = []
		
	#Create LIB_ts_rr folder
	if not os.path.exists("%s/LIB_ts_rr/"%rundir):
		os.makedirs("%s/LIB_ts_rr/"%rundir)
		
	#Create GraceDb folder
	if not os.path.exists("%s/GDB/"%rundir):
		os.makedirs("%s/GDB/"%rundir)

	#replace all necessary fields in LIB_reruns_beta.ini file if running follow-up
	if LIB_followup_flag:
		os.system('sed -e "s|IFOSCOMMA|%s|g" -e "s|IFOSTOGETHER|%s|g" -e "s|LIBLABEL|%s|g" -e "s|SEGNAME|%s|g" -e "s|RUNDIR|%s|g" -e "s|BINDIR|%s|g" -e "s|CHANNELTYPES|%s|g" -e "s|CHANNELNAMES|%s|g" -e "s|LAG|ts|g" %s/LIB_reruns_beta.ini > %s/runfiles/LIB_ts_reruns_beta.ini'%(ifos,"".join(ifos),lib_label,"%s_%s_%s"%("".join(ifos),actual_start,stride-overlap),rundir,bindir,chtypes_dic,chnames_dic,infodir,rundir))
		if train_runmode == 'Signal':
			os.system('sed -e "s|START|%s|g" -e "s|STOP|%s|g" -e "s|#mdc|mdc|g" -e "s|#MDC|MDC|g" %s/runfiles/LIB_ts_reruns_beta.ini > %s/tmp.txt; mv %s/tmp.txt %s/runfiles/LIB_ts_reruns_beta.ini'%(actual_start-int(0.5*overlap),actual_start-int(0.5*overlap)+stride,rundir,rundir,rundir,rundir))

	#Write JOB
	dagfile.write('JOB %s %s/runfiles/Bayes2LIB_beta.sub\n'%(job,rundir))
	#Write VARS
	B2L_args = "-I %s -r %s -i %s --lib-label=%s --start=%s --stride=%s --overlap=%s --lag=ts --FAR-thresh=%s --background-dic=%s --background-livetime=%s --signal-kde-coords=%s --signal-kde-values=%s --noise-kde-coords=%s --noise-kde-values=%s --train-runmode=%s --LIB-window=0.1"%(",".join(ifos),rundir,infodir,lib_label,actual_start,stride,overlap,FAR_thresh,back_dic_path,back_livetime,oLIB_signal_kde_coords,oLIB_signal_kde_values,oLIB_noise_kde_coords,oLIB_noise_kde_values,train_runmode)
	if LIB_followup_flag:
		B2L_args += " --LIB-followup"
	dagfile.write('VARS %s macroid="Bayes2LIB_ts-%s" macroarguments="%s"\n'%(job,job,B2L_args))
	#Write RETRY
	dagfile.write('RETRY %s 0\n\n'%job)

	#Record lalinference_pipe job number
	Bayes2LIB_ts_jobs += [job]

	#Done with job
	job += 1	

	##############################################
	### Check if supposed to run LIB follow-up ###
	##############################################

	if LIB_followup_flag:
		
		#################################################
		### Write DAG to point to LIB_0lag_reruns dag ###
		#################################################

		LIB_0lag_reruns_jobs = []

		#Write SUBDAG EXTERNAL
		dagfile.write('SUBDAG EXTERNAL %s %s/LIB_0lag_rr/LIB_runs.dag\n'%(job,rundir))
		#Write RETRY
		dagfile.write('RETRY %s 0\n\n'%job)

		#Record LIB_runs job number
		LIB_0lag_reruns_jobs += [job]

		#Done with job
		job += 1
		
		###############################################
		### Write DAG to point to LIB_ts_reruns dag ###
		###############################################

		LIB_ts_reruns_jobs = []

		#Write SUBDAG EXTERNAL
		dagfile.write('SUBDAG EXTERNAL %s %s/LIB_ts_rr/LIB_runs.dag\n'%(job,rundir))
		#Write RETRY
		dagfile.write('RETRY %s 0\n\n'%job)

		#Record LIB_runs job number
		LIB_ts_reruns_jobs += [job]

		#Done with job
		job += 1


####################################
### Write parent-child relations ###
####################################

#make each omicron job a parent to each omicron2LIB job
for parent in omicron_jobs:
	for child in omicron2LIB_jobs:
		dagfile.write('PARENT %s CHILD %s\n'%(parent,child))

if LIB_flag:

	#make each omicron2LIB job a parent to each lalinference_pipe_0lag job
	for parent in omicron2LIB_jobs:
		for child in lalinference_pipe_0lag_jobs:
			dagfile.write('PARENT %s CHILD %s\n'%(parent,child))
			
	#make each omicron2LIB job a parent to each lalinference_pipe_ts job
	for parent in omicron2LIB_jobs:
		for child in lalinference_pipe_ts_jobs:
			dagfile.write('PARENT %s CHILD %s\n'%(parent,child))
			
	#make each lalinference_pipe_0lag job a parent to each LIB_0lag_runs job
	for parent in lalinference_pipe_0lag_jobs:
		for child in LIB_0lag_runs_jobs:
			dagfile.write('PARENT %s CHILD %s\n'%(parent,child))
	
	#make each lalinference_pipe_ts job a parent to each LIB_ts_runs job
	for parent in lalinference_pipe_ts_jobs:
		for child in LIB_ts_runs_jobs:
			dagfile.write('PARENT %s CHILD %s\n'%(parent,child))
			
	#make each LIB_0lag_runs job a parent to each Bayes2LIB_0lag job
	for parent in LIB_0lag_runs_jobs:
		for child in Bayes2LIB_0lag_jobs:
			dagfile.write('PARENT %s CHILD %s\n'%(parent,child))
	
	#make each LIB_ts_runs job a parent to each Bayes2LIB_ts job
	for parent in LIB_ts_runs_jobs:
		for child in Bayes2LIB_ts_jobs:
			dagfile.write('PARENT %s CHILD %s\n'%(parent,child))
			
	if LIB_followup_flag:
	
		#make each Bayes2LIB_0lag job a parent to each LIB_0lag_reruns job
		for parent in Bayes2LIB_0lag_jobs:
			for child in LIB_0lag_reruns_jobs:
				dagfile.write('PARENT %s CHILD %s\n'%(parent,child))
				
		#make each Bayes2LIB_ts job a parent to each LIB_ts_reruns job
		for parent in Bayes2LIB_ts_jobs:
			for child in LIB_ts_reruns_jobs:
				dagfile.write('PARENT %s CHILD %s\n'%(parent,child))

#################
### Close DAG ###
#################
dagfile.close()


