#!/usr/bin/python

import numpy as np
import time
import os
import commands

##############################################
from optparse import OptionParser

usage = None
parser = OptionParser(usage=usage)

parser.add_option("-I","--IFOs", default=None, type="string", help="Comma separated list of ifos. E.g., H1,L1")
parser.add_option("-r","--rundir", default=None, type="string", help="Path to run directory")
parser.add_option("-i","--infodir", default=None, type="string", help="Path to info directory (where sub files, etc. are stored)")
parser.add_option("-b","--bindir", default=None, type="string", help="Path to bin directory for LIB executables")
parser.add_option("-l","--lib-label", default=None, type="string", help="Title for labeling LIB runs")
parser.add_option("","--channel-types", default=None, type='string', help="Comma separated types of channel to run on (for purposes of ligo-data-find) for each ifo")
parser.add_option("","--channel-names", default=None, type='string', help="Comma separated names of channel type to run on (frames) for each ifo")
parser.add_option("","--state-channels", default=None, type='string', help="Comma separated names of channels containing the state vector for each ifo")
parser.add_option("","--start", default=None, type='int', help="Start time of analysis (if not given will use current gps time)")
parser.add_option("","--stride", default=32, type='int', help="Stride length of each segment (default is 32s)")
parser.add_option("","--overlap", default=2, type='int', help="Overlap of segments (default is 2s)")
parser.add_option("","--wait", default=5, type='int', help="Wait time for run script for fetching data (default is 2s)")
parser.add_option("","--max-wait", default=900, type='int', help="Maximum amount of time to wait for data before moving forward")
parser.add_option("","--gdb", default=False, action="store_true", help="Write above-threshold events to GraceDb")
parser.add_option("","--LIB", default=False, action="store_true", help="Run LIB on the Omicron triggers")
parser.add_option("","--LIB-followup", default=False, action="store_true", help="Run in-depth LIB follow-up on preliminary LIB triggers exceeding FAR threshold")
parser.add_option("","--t-shift-start", default=0, type="float", help="Starting time shift to apply to IFO 2")
parser.add_option("","--t-shift-stop", default=0, type="float", help="Ending time shift to apply to IFO 2")
parser.add_option("","--t-shift-num", default=1, type="float", help="Number of time shifts to apply to IFO 2")
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
parser.add_option("","--bitmask", default=None, type='int', help='Number corresponding to the bitmask to use for data quality when building segments')
parser.add_option("","--inj-runmode", default=None, type='string', help='Either "Inj" or "NonInj" depending on if user wants to run on injections or not')
parser.add_option("","--train-runmode", default=None, type='string', help='Either "Signal", "Noise", or "None" depending on if user wants to run in training mode or not')
parser.add_option("","--min-hrss", default=None, type='float', help="Minimum hrss for injections when in signal training mode")
parser.add_option("","--max-hrss", default=None, type='float', help="Maximum hrss for injections when in signal training mode")

#---------------------------------------------

opts, args = parser.parse_args()

ifos = opts.IFOs.split(",")
rundir = opts.rundir
infodir = opts.infodir
bindir = opts.bindir
lib_label = opts.lib_label
channel_types = opts.channel_types.split(',')
channel_names = opts.channel_names.split(',')
state_channels = opts.state_channels.split(',')
actual_start = opts.start
stride = opts.stride
overlap = opts.overlap
wait = opts.wait
max_wait = opts.max_wait
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
bitmask = opts.bitmask
inj_runmode = opts.inj_runmode
train_runmode = opts.train_runmode
min_hrss = opts.min_hrss
max_hrss = opts.max_hrss

#############################################

#initialize start time (if no start time is given, default is current gps time) and stop times for first segment
if not actual_start:
	actual_start = int(commands.getstatusoutput('lalapps_tconvert now')[1])

np.savetxt(rundir+'/current_start.txt', np.array([actual_start]))

start = actual_start - int(0.5*overlap)
stop = start + stride
running_wait = 0

#make rundir for first segment
segdir = "%s/%s/%s_%s_%s"%(rundir,int(actual_start/100000.),"".join(ifos), actual_start, stride-overlap)
if not os.path.exists(segdir):
	os.makedirs(segdir)

#initialize dics holding frame files for each ifo
frame_files = {}
frame_times = {}
frame_lengths = {}
seg_files = {}
cache_files = {}
for ifo in ifos:
	frame_files[ifo] = [np.nan]
	frame_times[ifo] = [np.nan]
	frame_lengths[ifo] = [np.nan]
	seg_files[ifo] = None
	cache_files[ifo] = None

#initialize success flags for each ifo and inj_go_ahead
success_flags = {}
inj_go_ahead = {}
for ifo in ifos:
	success_flags[ifo] = False
	inj_go_ahead[ifo] = False

#start a while loop
while True:
	#wait for designated amount of time
	time.sleep(wait)
	running_wait += wait
	
	#run loop for each ifo
	for i_ifo, ifo in enumerate(ifos):
		#do some checks to see if we filled time (with frames) between start and stop for each ifo
		check_before_start = False
		check_after_stop = False
			
		if frame_times[ifo][0] <= start:
			check_before_start = True
		if (frame_times[ifo][-1] + frame_lengths[ifo][-1]) >= stop:
			check_after_stop = True
				
		#if we haven't reached stop time yet, fetch frames again
		if not check_after_stop:
			print "haven't reached stop time, fetching frames for", ifo, start, stop
			frame_files[ifo] = commands.getstatusoutput("gw_data_find --server=datafind.ldas.cit:80 --observatory=%s --url-type=file --type=%s --gps-start-time=%s --gps-end-time=%s"%(ifo.strip("1"), channel_types[i_ifo], start, stop))[1].split("\n")
			if frame_files[ifo] == [""]:
				print "no frames found for ifo", ifo, start, stop
				frame_times[ifo] = [np.nan]
				frame_lengths[ifo] = [np.nan]
			else:
				print "frames found for ifo", ifo, start, stop
				frame_times[ifo] = [int(f.split(channel_types[i_ifo]+'-')[-1].split('-')[0]) for f in frame_files[ifo]]
				frame_lengths[ifo] = [int(f.split(channel_types[i_ifo]+'-')[-1].split('-')[-1].split('.')[0]) for f in frame_files[ifo]]
				print frame_times[ifo]
				
		#here we have passed stop time, if we still have the start time frame then we can prepare to launch the condor jobs for that ifo
		elif check_before_start:
			print "filled time for ifo", ifo, start, stop
			#check if ifo has already been flagged as ready for condor submission
			if not success_flags[ifo]:
				#create necessary framecache for each ifo
				if not os.path.exists("%s/framecache"%segdir):
					os.makedirs("%s/framecache"%segdir)
				cache_files[ifo] = "%s/framecache/%s_%s_%s.cache"%(segdir,ifo,start,stop)
				cache_file = open(cache_files[ifo],'wt')
				for i in xrange(len(frame_files[ifo])):
					cache_file.write("%s %s %s %s %s\n"%(ifo.strip("1"), channel_types[i_ifo], frame_times[ifo][i], frame_lengths[ifo][i], frame_files[ifo][i]))
				cache_file.close()

				#write segment file for ifo
				if not os.path.exists("%s/segments"%segdir):
					os.makedirs("%s/segments"%segdir)
				inj_flag = commands.getstatusoutput('%s/framecache2segs_beta.py --cache-file=%s --state-channel=%s --start=%s --stop=%s -o %s -I %s -b %s'%(infodir, cache_files[ifo], state_channels[i_ifo], start, stop, "%s/segments/"%segdir, ifo, bitmask))[1].split('\n')[-1]
				seg_files[ifo] = "%s/segments/%s_%s_%s.seg"%(segdir,ifo,start,stop)
								
				#check inj_flag corresponds to inj_runmode
				if (inj_flag == 'True' and inj_runmode == 'Inj') or (inj_flag == 'False' and inj_runmode == 'NonInj'):
					inj_go_ahead[ifo] = True
				
				#flag that ifo is ready for condor submission
				success_flags[ifo] = True

			#check to see if all ifos have been flagged as ready for condor submission
			if np.prod([success_flags[ifo_test] for ifo_test in ifos]):
				#if in signal training mode, inject signals and point to new cache files
				if not os.path.exists("%s/training_injections"%segdir):
					os.makedirs("%s/training_injections"%segdir)
				os.system("%s/inject_signal_training_beta.py"%(infodir))
				for ifo in ifos:
					cache_files[ifo] = "%s/framecache/MDC_DatInjMerge_%s_%s_%s.lcf"%(segdir,ifo,start,stop)
				print "Injected event for signal training"
					
				#write pipeline dag and runfolders
				write_args = "-I %s -r %s -i %s -b %s -l %s -s %s -c %s --start=%s --stride=%s --overlap=%s --channel-names=%s --channel-types=%s --t-shift-start=%s --t-shift-stop=%s --t-shift-num=%s --dt-signal-kde-coords=%s --dt-signal-kde-values=%s --dt-noise-kde-coords=%s --dt-noise-kde-values=%s --FAR-thresh=%s --background-dic=%s --background-livetime=%s --oLIB-signal-kde-coords=%s --oLIB-signal-kde-values=%s --oLIB-noise-kde-coords=%s --oLIB-noise-kde-values=%s --train-runmode=%s"%(",".join(ifos), segdir, infodir, bindir, lib_label, ",".join([seg_files[ifo_tmp] for ifo_tmp in ifos]), ",".join([cache_files[ifo_tmp] for ifo_tmp in ifos]), actual_start, stride, overlap, ",".join(channel_names), ",".join(channel_types), t_shift_start, t_shift_stop, t_shift_num, dt_signal_kde_coords, dt_signal_kde_values, dt_noise_kde_coords, dt_noise_kde_values, FAR_thresh, back_dic_path, back_livetime, oLIB_signal_kde_coords, oLIB_signal_kde_values, oLIB_noise_kde_coords, oLIB_noise_kde_values, train_runmode)
				if gdb_flag:
					write_args += " --gdb"
				if LIB_flag:
					write_args += " --LIB"
				if LIB_followup_flag:
					write_args += " --LIB-followup"
				os.system("%s/write_omicron_2_LIB_dag_beta.py %s"%(infodir, write_args))

				#launch pipeline dag if data state vector matched with injection run mode
				if np.any([inj_go_ahead[ifo_test] for ifo_test in ifos]):
					os.system("condor_submit_dag %s/dag/2ndPipeDag_%s_%s_%s.dag"%(segdir,"".join(ifos),actual_start,stride-overlap))
					print "Submitted dag"
				else:
					print "Data does not fit with injection runmode, not submitting to condor"
												
				#reset success flags and inj_go_ahead
				for ifo_tmp in ifos:
					success_flags[ifo_tmp] = False
					inj_go_ahead[ifo_tmp] = False
				
				#move on to next time segment
				start += (stride - overlap)
				stop = start + stride
				actual_start = start + int(0.5*overlap)
				np.savetxt(rundir+'/current_start.txt', np.array([actual_start]))
				running_wait = 0
				
				#make rundir for next segment
				segdir = "%s/%s/%s_%s_%s"%(rundir,int(actual_start/100000.),"".join(ifos), actual_start, stride-overlap)
				if not os.path.exists(segdir):
					os.makedirs(segdir)
			
		#here we have passed stop time, if we don't still have the start time frame then we will skip this segment
		else:
			print "lost start time for ifo", ifo, start, stop

			#reset success flags
			for ifo_tmp in ifos:
				success_flags[ifo_tmp] = False
				inj_go_ahead[ifo_tmp] = False
			
			#move on to next time interval
			start += (stride - overlap)
			stop = start + stride
			actual_start = start + int(0.5*overlap)
			np.savetxt(rundir+'/current_start.txt', np.array([actual_start]))
			running_wait = 0
			
			#make rundir for next segment
			segdir = "%s/%s/%s_%s_%s"%(rundir,int(actual_start/100000.),"".join(ifos), actual_start, stride-overlap)
			if not os.path.exists(segdir):
				os.makedirs(segdir)
		
		#Check to see if we have exceeded the maximum wait
		if running_wait > max_wait:
			#catch up to real time
			actual_start = int(commands.getstatusoutput('lalapps_tconvert now')[1])
			np.savetxt(rundir+'/current_start.txt', np.array([actual_start]))
			start = actual_start - int(0.5*overlap)
			stop = start + stride
			running_wait = 0
			
			#make rundir for next segment
			segdir = "%s/%s/%s_%s_%s"%(rundir,int(actual_start/100000.),"".join(ifos), actual_start, stride-overlap)
			if not os.path.exists(segdir):
				os.makedirs(segdir)

