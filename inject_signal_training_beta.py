#!/usr/bin/python

import numpy as np
import os
import commands

#=======================================================================

parser.add_option("-I","--IFOs", default=None, type="string", help="Comma separated list of ifos. E.g., H1,L1")
parser.add_option("","--start", default=None, type='int', help="Start time of frame")
parser.add_option("","--stop", default=None, type='int', help="Stop time of frame")
parser.add_option("","--overlap", default=None, type='int', help="Overlap of segments")
parser.add_option("","--segdir", default=None, type="string", help="Path to directory at which the pipeline is analyzing a stride")
parser.add_option("","--min-hrss", default=None, type='float', help="Minimum hrss for injections")
parser.add_option("","--max-hrss", default=None, type='float', help="Maximum hrss for injections")
parser.add_option("","--cache-files", default=None, type="string", help="Comma separated list of paths to data cache files corresponding to the ifos")

#---------------------------------------------

opts, args = parser.parse_args()

ifos = opts.IFOs
start = opts.start
stop = opts.stop
overlap = opts.overlap
segdir = opts.segdir
min_hrss = opts.min_hrss
max_hrss = opts.max_hrss
cache_files_list = opts.cache_files
cache_files = {}
for i,ifo in enumerate(ifos):
	cache_files[ifo] = cache_files_list[i]

#=======================================================================

#Initialize mdc parameters
ifos = repr(",".join(ifos))  #"H1,L1"
num_mdc = 1  #number of mdc injection frames generated
mdc_start_time = start  #start time of mdc injection frame
mdc_end_time = stop  #end time of mdc injection frame
padding = int(overlap/2.)  # buffer between edges of frame and injections, value is padding on each end

mdc_duration = mdc_end_time - mdc_start_time
trig_end_time = mdc_end_time - padding
trig_start_time = mdc_start_time + padding + np.random.randint(low=0, high=(mdc_duration-2*padding+1))
seed = trig_start_time

mdc_par={
"ifos":"["+ifos+"]",
"duration":mdc_duration,
"pad":padding
"gps-start":mdc_start_time
}

#Initialize injection parameters
#Randomly select SG vs WNB
types = ["SG","WNB"]
inj_type = types[np.random.randint(low=0, high=len(types))]

if inj_type == "SG":
	par={
	"population":"all_sky_sinegaussian", # time domain SG
	"q-distr":"uniform",
	"min-q":3,
	"max-q":30,
	"f-distr":"uniform",
	"min-f":40,
	"max-f":1500,
	"hrss-distr":"volume",
	'min-hrss':min_hrss, # approximate lower limit of detectability
	'max-hrss':max_hrss,
	"polar-angle-distr":"uniform",
	"min-polar-angle":0.0,
	"max-polar-angle":2.0*np.pi,
	"polar-eccentricity-distr":"uniform",
	"min-polar-eccentricity":0.0,
	"max-polar-eccentricity":1.0,
	"seed":seed,
	"gps-start-time":trig_start_time,
	"gps-end-time":trig_end_time,
	"time-step":10000.,
	"output": "%s/training_injections/raw/SG_seed_%s_hrss_%s_%s_time_%s_%s.xml"%(segdir,seed,min_hrss,max_hrss,mdc_start_time,mdc_end_time)
	}
elif inj_type == "WNB"
	par={
	"population":"all_sky_btlwnb", # time domain white noise burst
	'min-duration':0.005,
	'max-duration':0.1,
	'hrss-distr':'volume',
	'min-hrss':min_hrss, #approximate lower limit of detectability
	'max-hrss':max_hrss,
	"f-distr":"uniform",
	"min-f":40,
	"max-f":1500,
	"min-e-over-r2":1e-10, #won't be used anyway. Hrss will decide the amplitude
	"max-e-over-r2":1e-10, #won't be used anyway. Hrss will decide the amplitude
	"min-bandwidth":10,
	"max-bandwidth":500,
	"seed":seed,
	"gps-start-time":trig_start_time,
	"gps-end-time":trig_end_time,
	"time-step":10000.,
	"output": "%s/training_injections/raw/WNB_seed_%s_hrss_%s_%s_time_%s_%s.xml"%(segdir,seed,min_hrss,max_hrss,mdc_start_time,mdc_end_time)
	}
else:
	raise ValueError, "No injection type selected"

#Create timeslide file
os.chdir('%s/training_injections/raw/'%segdir)
os.system("ligolw_tisi --instrument H1=0:0:0 --instrument L1=0:0:0 --instrument H2=0:0:0 --instrument V1=0:0:0 %s/training_injections/raw/time_slides.xml.gz"%segdir) # if this file doesn't exist, the main function will complain

#Run lalapps_binj to create injection xml file
binj_string="lalapps_binj --time-slide-file %s/training_injections/raw/time_slides.xml.gz"%segdir

for key in par:
	if type(par[key])==str:
		binj_string += " --" + key + " " + par[key]
	else:
		binj_string += " --" + key + " " + repr(par[key])

os.chdir('%s/training_injections/raw/'%segdir)
os.system(binj_string)

#Run lalapps_simburst_to_frame to create injection frames
frame_string="lalapps_simburst_to_frame --simburst-file %s --channels [Science,Science]" % par["output"]

for key in mdc_par:
  if type(mdc_par[key])==str:
	frame_string += " --" + key + " " + mdc_par[key]
  else:
	frame_string += " --" + key + " " + repr(mdc_par[key])

os.chdir('%s/training_injections/raw/'%segdir)
os.system(frame_string)

#Make cache file for the injection frames
os.system('ls %s/training_injections/raw/*.gwf | lalapps_path2cache >> %s/framecache/MDC_Injections_%s_%s.lcf'%(segdir,segdir,mdc_start_time,mdc_stop_time))

#Combine data frames and injection frame, putting it in a cache file
for IFO in ifos.split(','):
	cache = open('%s/framecache/MDC_DatInjMerge_%s_%s_%s.lcf'%(segdir,IFO,mdc_start_time,mdc_stop_time),'wt')

	#Get paths of data frames
	dat_files = []
	dat_cache = open(cache_files[IFO],'rt')
	for line in dat_cache:
		#Get frame_file from data cache
		words = line.split()
		frame_file = words[4].split('file://localhost')[1]
		dat_files.append(frame_file)
	dat_files = " ".join(dat_files).strip()

	#Get paths of injection frames
	inj_files = commands.getstatusoutput("ls %s/training_injections/raw/*%s*.gwf"%(segdir,IFO))[1]
	inj_files = re.split('\n', inj_files)
	inj_files = " ".join(inj_files).strip()

	out_file = "%s/training_injections/merged/%s-DatInjMerge-%u-%u.gwf"%(segdir, IFO, mdc_start_time, mdc_duration)
	os.system("FrCopy -i %s %s -o %s"%(inj_files, dat_files, out_file))
	out_file_actual = commands.getstatusoutput("readlink -f %s"%out_file)[1]
	cache.write("%s DatInjMerge %s %s %s\n"%(IFO, mdc_start_time, mdc_duration, "file://localhost"+out_file_actual))
		
	cache.close()
