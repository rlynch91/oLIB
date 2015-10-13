#!/usr/bin/python

import numpy as np
import os
import commands

########################################################################

#Config file for launching the oLIB pipeline

ifos = 'H1,L1'  #Comma-separated list of IFOs to run over
rundir = '/home/ryan.lynch/2nd_pipeline/pipeline_beta_dev/test'  #Directory in which to write oLIB results
infodir = '/home/ryan.lynch/2nd_pipeline/pipeline_beta_dev/'  #Directory containing config/template files needed for running oLIB
bindir = '/home/ryan.lynch/lalsuites/LIB/opt/bin/'  #Directory containing LIB executables
lib_label = 'TEST_beta'  #Label to use for LIB summary pages
channel_types = 'H1_llhoft,L1_llhoft'  #Comma-separated list of channel types to search for for each ifo when running ligo_data_find
channel_names = 'GDS-CALIB_STRAIN,GDS-CALIB_STRAIN'  #Comma-separated list of channel names to run Omicron on for each IFO
state_channels = 'GDS-CALIB_STATE_VECTOR,GDS-CALIB_STATE_VECTOR'  #Comma-separated list of channel names for checking data quality for each ifo
stride = 32  #Stride length of Omicron segments (in s)
overlap = 2  #Overlap for Omicron segments (in s)
wait = 5  #Wait stride between serches for new data (in s)
max_wait = 1200  #Maximum wait interval before skipping data and moving on
gdb_flag = False  #Upload to GraceDB if True
LIB_flag = True  #Run LIB portion of oLIB if true
LIB_followup_flag = False
t_shift_start = -20  #Maximum "leftward" timeslide shift (in s)
t_shift_stop = 20  #Maximum "rightward" timeslide shift (in s)
t_shift_num = 41  #Number of timeslides in total
dt_signal_kde_coords = '/home/ryan.lynch/2nd_pipeline/pipeline_beta_dev/delta_t_HL_Signal_KDE_coords.npy'
dt_signal_kde_values = '/home/ryan.lynch/2nd_pipeline/pipeline_beta_dev/delta_t_HL_Signal_KDE_values.npy'
dt_noise_kde_coords = '/home/ryan.lynch/2nd_pipeline/pipeline_beta_dev/delta_t_HL_Noise_KDE_coords.npy'
dt_noise_kde_values = '/home/ryan.lynch/2nd_pipeline/pipeline_beta_dev/delta_t_HL_Noise_KDE_values.npy'
FAR_thresh = 1.e-5
back_dic_path = '/home/ryan.lynch/2nd_pipeline/pipeline_beta_dev/noise_snrcut_6d5sqrt2_ts.pkl'
back_livetime = '/home/ryan.lynch/2nd_pipeline/pipeline_beta_dev/result_dics/background_livetime.txt'
oLIB_signal_kde_coords = '/home/ryan.lynch/2nd_pipeline/pipeline_beta_dev/BSN_and_BCI_Signal_KDE_coords.npy'
oLIB_signal_kde_values = '/home/ryan.lynch/2nd_pipeline/pipeline_beta_dev/BSN_and_BCI_Signal_KDE_values.npy'
oLIB_noise_kde_coords = '/home/ryan.lynch/2nd_pipeline/pipeline_beta_dev/BSN_and_BCI_Noise_KDE_coords.npy'
oLIB_noise_kde_values = '/home/ryan.lynch/2nd_pipeline/pipeline_beta_dev/BSN_and_BCI_Noise_KDE_values.npy'
bitmask = 8
inj_runmode = "NonInj"
train_runmode = "None" #either Signal, Noise, None
min_hrss = 1e-22
max_hrss = 20e-22
asd_file = '/home/ryan.lynch/2nd_pipeline/pipeline_beta_dev/test_asd'

#Remove lock files on training dictionaries in case they remain after a failed run
if train_runmode == "None":
	if not os.path.exists('%s/result_dics/'%infodir):
		os.makedirs('%s/result_dics/'%infodir)
	os.system('rm %s/result_dics/foreground_events.pkl_lock'%infodir)
	os.system('rm %s/result_dics/background_events.pkl_lock'%infodir)
	os.system('rm %s/result_dics/livetimes.txt_lock'%infodir)
elif train_runmode == "Signal":
	if not os.path.exists('%s/training_dics/'%infodir):
		os.makedirs('%s/training_dics/'%infodir)
	os.system('rm %s/training_dics/new_signal_training_points.pkl_lock'%infodir)
elif train_runmode == "Noise":
	if not os.path.exists('%s/training_dics/'%infodir):
		os.makedirs('%s/training_dics/'%infodir)
	os.system('rm %s/training_dics/new_noise_training_points.pkl_lock'%infodir)

#Decide what time to start running on
if os.path.exists(rundir+'/current_start.txt'):
	#Continue past run based on saved timestamp
	actual_start = int(np.genfromtxt(rundir+'/current_start.txt'))
else:
	#Start running on current timestamp
	actual_start = int(commands.getstatusoutput('%s/lalapps_tconvert now'%bindir)[1]) - 500

#Launch oLIB
run_args = '-I %s -r %s -i %s -b %s -l %s --channel-types=%s --channel-names=%s --state-channels=%s --start=%s --stride=%s --overlap=%s --wait=%s --max-wait=%s --t-shift-start=%s --t-shift-stop=%s --t-shift-num=%s --dt-signal-kde-coords=%s --dt-signal-kde-values=%s --dt-noise-kde-coords=%s --dt-noise-kde-values=%s --FAR-thresh=%s --background-dic=%s --background-livetime=%s --oLIB-signal-kde-coords=%s --oLIB-signal-kde-values=%s --oLIB-noise-kde-coords=%s --oLIB-noise-kde-values=%s --bitmask=%s --inj-runmode=%s --train-runmode=%s --min-hrss=%s --max-hrss=%s --asd-file=%s'%(ifos, rundir, infodir, bindir, lib_label, channel_types, channel_names, state_channels, actual_start, stride, overlap, wait, max_wait, t_shift_start, t_shift_stop, t_shift_num, dt_signal_kde_coords, dt_signal_kde_values, dt_noise_kde_coords, dt_noise_kde_values, FAR_thresh, back_dic_path, back_livetime, oLIB_signal_kde_coords, oLIB_signal_kde_values, oLIB_noise_kde_coords, oLIB_noise_kde_values, bitmask, inj_runmode, train_runmode, min_hrss, max_hrss, asd_file)
if gdb_flag:
	run_args += " --gdb"
if LIB_flag:
	run_args += " --LIB"
if LIB_followup_flag:
	run_args += " --LIB-followup"
os.system('%s/run_2ndpipe_beta.py %s'%(infodir, run_args))
