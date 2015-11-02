import numpy as np

#=======================================================================

from optparse import OptionParser

usage = None
parser = OptionParser(usage=usage)

#general options
parser.add_option("", "--out-name", default=None, type="string", help="Path and name of of dag file to write")
parser.add_option("", "--sub-name", default=None, type="string", help="Path and name of .sub file for all_possible_timeslides.py")
parser.add_option("", "--home-dir", default=None, type="string", help="Path to folder where all post-processing directories will be written")
parser.add_option("", "--coin-snr", default=None, type="float", help="Lower-network-SNR threshold for coincident events")
parser.add_option("", "--ts-num", default=None, type="int", help="Number of timeslides to run with each job")
parser.add_option("", "--total-lt", default=None, type="float", help="Total livetime spanned by analysis window (i.e., length of time that data could be present in either IFO)")
parser.add_option("", "--clustered-files", default=None, type="string", help="Comma-separated paths to files containing the clustered triggers for each ifo")

#-----------------------------------------------------------------------

opts, args = parser.parse_args()

out_name = opts.out_name
sub_name = opts.sub_name
home_dir = opts.home_dir
coin_snr = opts.coin_snr
ts_num = opts.ts_num
total_lt = opts.total_lt
clustered_files = opts.clustered_files

#=======================================================================

#open file to write to
dag_file = open(out_name,'wt')

#do 0-lag
job=0
dag_file.write('JOB %s %s\n'%(job,sub_name))
dag_file.write('VARS %s macroid="ts-0.-0._%s" macroarguments="-p %s/PostProc_0._0./ -i /home/ryan.lynch/2nd_pipeline/pipeline_ER8/ -I H1,L1 -r /shouldnt/matter/ -c GDS-CALIB_STRAIN,GDS-CALIB_STRAIN --cluster-t=0.1 --coin-t=0.05 --coin-snr=%s --t-shift-start=0. --t-shift-stop=0. --t-shift-num=1. --segs=/home/ryan.lynch/public_html/ER8_significant_event/RUNS/segments/H1.seg,/home/ryan.lynch/public_html/ER8_significant_event/RUNS/segments/L1.seg --veto-files=/home/ryan.lynch/public_html/ER8_significant_event/vetoes/null_vetoes.txt,/home/ryan.lynch/public_html/ER8_significant_event/vetoes/null_vetoes.txt --overlap=2 --log-like-thresh=0. --LIB-window=0.1 --signal-kde-coords=/home/ryan.lynch/2nd_pipeline/pipeline_ER8/delta_t_HL_Signal_KDE_coords.npy --signal-kde-values=/home/ryan.lynch/2nd_pipeline/pipeline_ER8/delta_t_HL_Signal_KDE_values.npy --noise-kde-coords=/home/ryan.lynch/2nd_pipeline/pipeline_ER8/delta_t_HL_Noise_KDE_coords.npy --noise-kde-values=/home/ryan.lynch/2nd_pipeline/pipeline_ER8/delta_t_HL_Noise_KDE_values.npy --train-runmode=None --clustered=%s"\n'%(job,coin_snr,home_dir,coin_snr,clustered_files))
dag_file.write('RETRY %s 0\n\n'%job)

#find number of negative/positive jobs to write
negpos_jobs = int(float(total_lt)/ts_num)

#do negative timeslides
for job in xrange(negpos_jobs):
	#figure out which timeslide we're on
	tmp_ts_start = -total_lt + ts_num*job
	tmp_ts_stop = tmp_ts_start + ts_num -1.
	
	#write dag
	dag_file.write('JOB %s %s\n'%(job+1,sub_name))
	dag_file.write('VARS %s macroid="ts-%s-%s_%s" macroarguments="-p %s/PostProc_%s_%s/ -i /home/ryan.lynch/2nd_pipeline/pipeline_ER8/ -I H1,L1 -r /shouldnt/matter/ -c GDS-CALIB_STRAIN,GDS-CALIB_STRAIN --cluster-t=0.1 --coin-t=0.05 --coin-snr=%s --t-shift-start=%s --t-shift-stop=%s --t-shift-num=%s --segs=/home/ryan.lynch/public_html/ER8_significant_event/RUNS/segments/H1.seg,/home/ryan.lynch/public_html/ER8_significant_event/RUNS/segments/L1.seg --veto-files=/home/ryan.lynch/public_html/ER8_significant_event/vetoes/null_vetoes.txt,/home/ryan.lynch/public_html/ER8_significant_event/vetoes/null_vetoes.txt --overlap=2 --log-like-thresh=0. --LIB-window=0.1 --signal-kde-coords=/home/ryan.lynch/2nd_pipeline/pipeline_ER8/delta_t_HL_Signal_KDE_coords.npy --signal-kde-values=/home/ryan.lynch/2nd_pipeline/pipeline_ER8/delta_t_HL_Signal_KDE_values.npy --noise-kde-coords=/home/ryan.lynch/2nd_pipeline/pipeline_ER8/delta_t_HL_Noise_KDE_coords.npy --noise-kde-values=/home/ryan.lynch/2nd_pipeline/pipeline_ER8/delta_t_HL_Noise_KDE_values.npy --train-runmode=None --clustered=%s"\n'%(job+1,tmp_ts_start,tmp_ts_stop,coin_snr,home_dir,tmp_ts_start,tmp_ts_stop,coin_snr,tmp_ts_start,tmp_ts_stop,ts_num,clustered_files))
	dag_file.write('RETRY %s 0\n\n'%(job+1))

#do positive timeslides
for job in xrange(negpos_jobs):
	#figure out which timeslide we're on
	tmp_ts_start = ts_num*job + 1.
	tmp_ts_stop = tmp_ts_start + ts_num -1.
	
	#write dag
	dag_file.write('JOB %s %s\n'%(job+negpos_jobs+1,sub_name))
	dag_file.write('VARS %s macroid="ts-%s-%s_%s" macroarguments="-p %s/PostProc_%s_%s/ -i /home/ryan.lynch/2nd_pipeline/pipeline_ER8/ -I H1,L1 -r /shouldnt/matter/ -c GDS-CALIB_STRAIN,GDS-CALIB_STRAIN --cluster-t=0.1 --coin-t=0.05 --coin-snr=%s --t-shift-start=%s --t-shift-stop=%s --t-shift-num=%s --segs=/home/ryan.lynch/public_html/ER8_significant_event/RUNS/segments/H1.seg,/home/ryan.lynch/public_html/ER8_significant_event/RUNS/segments/L1.seg --veto-files=/home/ryan.lynch/public_html/ER8_significant_event/vetoes/null_vetoes.txt,/home/ryan.lynch/public_html/ER8_significant_event/vetoes/null_vetoes.txt --overlap=2 --log-like-thresh=0. --LIB-window=0.1 --signal-kde-coords=/home/ryan.lynch/2nd_pipeline/pipeline_ER8/delta_t_HL_Signal_KDE_coords.npy --signal-kde-values=/home/ryan.lynch/2nd_pipeline/pipeline_ER8/delta_t_HL_Signal_KDE_values.npy --noise-kde-coords=/home/ryan.lynch/2nd_pipeline/pipeline_ER8/delta_t_HL_Noise_KDE_coords.npy --noise-kde-values=/home/ryan.lynch/2nd_pipeline/pipeline_ER8/delta_t_HL_Noise_KDE_values.npy --train-runmode=None --clustered=%s"\n'%(job+negpos_jobs+1,tmp_ts_start,tmp_ts_stop,coin_snr,home_dir,tmp_ts_start,tmp_ts_stop,coin_snr,tmp_ts_start,tmp_ts_stop,ts_num,clustered_files))
	dag_file.write('RETRY %s 0\n\n'%(job+negpos_jobs+1))

dag_file.close()
