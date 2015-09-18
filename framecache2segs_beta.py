#!/usr/bin/python

import numpy as np
import os
import commands

###
def framecache2segs(framecache_file, chname, abs_start, abs_stop, outdir, ifo, bitmask):
	"""
	Takes a file containing the data quality state vetor and converts it to segments of a desired bitmask, returning injection status
	"""
	#Initialize injection status as True (will change if 'No injection' bits are on)
	inj_flag = False
	
	#Open framecache file and segment file to write to
	cache = open(framecache_file, 'rt')
	segfile = open(outdir+'/%s_%s_%s.seg'%(ifo,abs_start,abs_stop),'wt')

	#Define start and stop of current segment
	current_start = abs_start
	current_stop = None

	#Loop over frames in the cache
	for line in cache:
		#Get frame_start, frame_stride, and frame_file
		words = line.split()
		frame_start = int(words[2])
		frame_stride = int(words[3])
		frame_file = words[4].split('file://localhost')[1]
			
		#Output state vector contents into outdir/tmp.txt
		os.system('FrameDataDump -I%s -C%s:%s -d -a > %s/tmp.txt'%(frame_file,ifo,chname,outdir))
		
		#Open state vector as array
		state_array = np.genfromtxt('%s/tmp.txt'%outdir).reshape(-1)
		
		#Calculate sample rate
		samp_rate = len(state_array)/float(frame_stride)
		
		#Loop over state vetor
		for i, value in enumerate(state_array):
			#Check to make sure we've passed the absolute start
			if (frame_start + i/float(samp_rate)) < abs_start:
				continue
				
			#Check to make sure we haven't passed the absolute stop
			elif (frame_start + i/float(samp_rate)) > abs_stop:
				break
				
			#Check if state vector corresponds to desired bitmask
			elif (int(value) & bitmask) == bitmask:  #(e.g., 0b00011 = 3 and we want bits 0 and 1 to be on, so we do & with 3)
				#Data is good, start new seg if needed
				if not current_start:
					current_start = int(np.ceil(frame_start + i/float(samp_rate) ))  #data good starting at ith sample, use ceiling so don't underestimate start
			else:
				#Data not good, end current seg if needed
				if current_start:
					current_stop = int(np.floor(frame_start + i/float(samp_rate) ))  #data goes bad at ith sample but good until then, use floor so don't overestimate stop
					if current_start < current_stop:
						segfile.write('%s %s\n'%(current_start, current_stop))
					#Wait to start next segment until find good data
					current_start = None

			#Check if state vector denotes that an injection is present
			if ((int(value) & 448) != 448):  #(e.g., we want bits 6, 7, or 8 to be on)
				inj_flag = True

	#Write final segment if needed
	if current_start:
		if current_start < abs_stop:
			segfile.write('%s %s\n'%(current_start, abs_stop))

	cache.close()
	os.system('rm %s/tmp.txt'%outdir)
	
	#Return injection flag
	return inj_flag

##############################################
if __name__=='__main__':
	from optparse import OptionParser

	usage = None
	parser = OptionParser(usage=usage)

	parser.add_option("","--cache-file", default=None, type='string', help="Path to framecache file")
	parser.add_option("","--state-channel", default=None, type='string', help="Name of channel containing ifo state vector")
	parser.add_option("","--start", default=None, type='int', help="Absolute start time of segments")
	parser.add_option("","--stop", default=None, type='int', help="Absolute stop time of segments")
	parser.add_option("-o","--outdir", default=None, type="string", help="Path to directory in which to output segments")
	parser.add_option("-I","--IFO", default=None, type="string", help="Name of ifo, e.g., H1")
	parser.add_option("-b","--bitmask", default=None, type="int", help="Number corresponding to bitmask to search for")

	#---------------------------------------------

	opts, args = parser.parse_args()

	framecache_file = opts.cache_file
	chname = opts.state_channel
	abs_start = opts.start
	abs_stop = opts.stop
	outdir = opts.outdir
	ifo = opts.IFO
	bitmask = opts.bitmask
	
	#---------------------------------------------
	
	inj_status = framecache2segs(framecache_file=framecache_file, chname=chname, abs_start=abs_start, abs_stop=abs_stop, outdir=outdir, ifo=ifo, bitmask=bitmask)
	print inj_status

