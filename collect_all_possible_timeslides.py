import numpy as np
import os

#=======================================================================

from optparse import OptionParser

usage = None
parser = OptionParser(usage=usage)

#general options
parser.add_option("", "--out-name", default=None, type="string", help="Path and name of base file name where collected times, timeslides, and livetime will be written")
parser.add_option("", "--home-dir", default=None, type="string", help="Path to folder where all post-processing directories live")
parser.add_option("", "--ts-num", default=None, type="int", help="Number of 1s timeslides each job ran over")
parser.add_option("", "--ts-start", default=None, type="float", help="Absolute starting timeslide to collect")
parser.add_option("", "--ts-stop", default=None, type="float", help="Absolute stopping timeslide to collect")
parser.add_option("", "--snr-cut", default=None, type='float', help="Network SNR threshold required of coincidences")

#-----------------------------------------------------------------------

opts, args = parser.parse_args()

out_name = opts.out_name
home_dir = opts.home_dir
ts_num = opts.ts_num
ts_start = opts.ts_start
ts_stop = opts.ts_stop
snr_cut = opts.snr_cut

#=======================================================================

#initialize files to write trigger times and timeslides to
if snr_cut:
	trig_file = open('%s_trigtimes_%s_%s_SNR%s.txt'%(out_name,ts_start,ts_stop,snr_cut),'wt')
	ts_file = open('%s_timeslides_%s_%s_SNR%s.txt'%(out_name,ts_start,ts_stop,snr_cut),'wt')
else:
	os.system('> %s_trigtimes_%s_%s.txt'%(out_name,ts_start,ts_stop))
	os.system('> %s_timeslides_%s_%s.txt'%(out_name,ts_start,ts_stop))

#initialize variable for total livetime
total_lt = 0.

#find number of post_proc folders we have to loop over
num_jobs = int(float(ts_stop - ts_start + 1.)/float(ts_num))

#loop over jobs, collecting all necessary data
for job in xrange(num_jobs):
	if snr_cut:
		#SNR thresh specified, meaning we have to check every trigger
		for j in xrange(ts_num):
			ts = ts_start+job*ts_num+j
			print ts
			tmp_file = open('%s/PostProc_%s_%s/LIB_trigs/LIB_trigs_H1L1_ts%s.txt'%(home_dir,ts_start+job*ts_num,ts_start+(job+1.)*ts_num-1.,ts))
			for line in tmp_file:
				array = line.split()
				if (float(array[2])) >= snr_cut:
					#only write if network snr exceeds the given threshold
					trig_file.write('%10.10f\n'%float(array[0]))
					ts_file.write('0. %s\n'%ts)
			tmp_file.close()		
	else:
		#No SNR thresh specified, can just use already collected coincidences
		os.system('cat %s/PostProc_%s_%s/LIB_trigs/LIB_ts_times_H1L1.txt >> %s_trigtimes_%s_%s.txt'%(home_dir,ts_start+job*ts_num,ts_start+(job+1.)*ts_num-1.,out_name,ts_start,ts_stop))
		os.system('cat %s/PostProc_%s_%s/LIB_trigs/LIB_ts_timeslides_H1L1.txt >> %s_timeslides_%s_%s.txt'%(home_dir,ts_start+job*ts_num,ts_start+(job+1.)*ts_num-1.,out_name,ts_start,ts_stop))
	total_lt += np.genfromtxt('%s/PostProc_%s_%s/live_segs/livetime_timeslides_H1L1.txt'%(home_dir,ts_start+job*ts_num,ts_start+(job+1.)*ts_num-1.))
	
#save total collected livetime
np.savetxt('%s_total_lt_%s_%s.txt'%(out_name,ts_start,ts_stop),np.array([total_lt]))

#close files if necessary
if snr_cut:
	trig_file.close()
	ts_file.close()
