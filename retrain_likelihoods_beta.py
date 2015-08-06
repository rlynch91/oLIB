#!/usr/bin/python

import numpy as np
import pickle
import LLRT_object_beta
import os
import time

#=======================================================================
#Parse user options
from optparse import OptionParser

usage = None
parser = OptionParser(usage=usage)

parser.add_option("", "--dic-dir", default=None, type="string", help="Path to directory containing training dictionaries")
parser.add_option("", "--like-dir", default=None, type="string", help="Path to directory containing likelihood estimates")
parser.add_option("", "--max-signal-size", default=None, type="int", help="Maximum number of points to train signal likelihoods on")
parser.add_option("", "--max-noise-size", default=None, type="int", help="Maximum number of points to train noise likelihoods on")
parser.add_option("", "--LRT-param-group", default=None, type="string", help="Group of parameters to train likelihoods on ('BSN_and_BCI_and_oSNR' or 'BSN_and_BCI')")

#-----------------------------------------------------------------------

opts, args = parser.parse_args()

dic_dir = opts.dic_dir
like_dir = opts.like_dir
max_signal_size = opts.max_signal_size
max_noise_size = opts.max_noise_size
LRT_param_group = opts.LRT_param_group

#=======================================================================

################################
# Update training dictionaries #
################################

#Check to make sure another function isn't currently updating the training dictionaries
while os.path.isfile('%s/new_signal_training_points.pkl_lock'%dic_dir) or os.path.isfile('%s/new_noise_training_points.pkl_lock'%dic_dir):
	time.sleep(5)
	
#Lock training dictionaries while updating them
os.system('> %s/new_signal_training_points.pkl_lock'%dic_dir)
os.system('> %s/new_noise_training_points.pkl_lock'%dic_dir)

#Load in current training dictionaries
current_signal_dic = pickle.load(open('%s/current_signal_training_points.pkl'%dic_dir))
current_noise_dic = pickle.load(open('%s/current_noise_training_points.pkl'%dic_dir))

#Load in new training dictionaries
new_signal_dic = pickle.load(open('%s/new_signal_training_points.pkl'%dic_dir))
new_noise_dic = pickle.load(open('%s/new_noise_training_points.pkl'%dic_dir))

#Create updated dictionary
key_shift_signal = len(new_signal_dic)
key_shift_noise = len(new_noise_dic)
updated_signal_dic = {}
updated_noise_dic = {}

###SIGNAL###
#If more than max new training points, need to downselect to the proper amount
if len(new_signal_dic) >= max_signal_size:
	downselect_keys = numpy.random.choice(new_signal_dic.keys(), size=max_signal_size, replace=False)
	for i,key in enumerate(downselect_keys):
		updated_signal_dic[i] = new_signal_dic[key]

#If less than max new training points, add new training points, replacing oldest training points with new ones if need be
else:
	for key in current_signal_dic:
		if (key+key_shift_signal) < max_signal_size:
			updated_signal_dic[key+key_shift_signal] = current_signal_dic[key]
		else:
			continue
	for key in new_signal_dic:
		updated_signal_dic[key] = new_signal_dic[key]

###NOISE###
#If more than max new training points, need to downselect to the proper amount
if len(new_noise_dic) >= max_noise_size:
	downselect_keys = numpy.random.choice(new_noise_dic.keys(), size=max_noise_size, replace=False)
	for i,key in enumerate(downselect_keys):
		updated_noise_dic[i] = new_noise_dic[key] 

#If less than max new training points, add new training points, replacing oldest training points with new ones if need be
else:
	for key in current_noise_dic:
		if (key+key_shift_noise) < max_noise_size:
			updated_noise_dic[key+key_shift_noise] = current_noise_dic[key]
		else:
			continue
	for key in new_noise_dic:
		updated_noise_dic[key] = new_noise_dic[key]

#Save both updated dictionary and an empty dictionary for new training points
pickle.dump(updated_signal_dic,open('%s/current_signal_training_points.pkl'%dic_dir,'wt'))
pickle.dump(updated_noise_dic,open('%s/current_noise_training_points.pkl'%dic_dir,'wt'))
pickle.dump({},open('%s/new_signal_training_points.pkl'%dic_dir,'wt'))
pickle.dump({},open('%s/new_noise_training_points.pkl'%dic_dir,'wt'))

#Unlock training dictionaries now that they're updated
os.system('rm %s/new_signal_training_points.pkl_lock'%dic_dir)
os.system('rm %s/new_noise_training_points.pkl_lock'%dic_dir)

#-----------------------------------------------------------------------

#######################
# Retrain likelihoods #
#######################

#Build calc_info dictionary
calc_info = {}
calc_info['interp method'] = 'Grid Linear'
calc_info['extrap method'] = 'Grid Nearest'

#Build param_info dictionary

if LRT_param_group == 'BSN_and_BCI_and_oSNR':
	param_info = {}
	param_info['BSN_and_BCI_and_oSNR'] = {}
	param_info['BSN_and_BCI_and_oSNR']['dimension'] = 3
	param_info['BSN_and_BCI_and_oSNR']['param names'] = ['BSN','BCI','oSNR']
	param_info['BSN_and_BCI_and_oSNR']['interp range'] = np.array([[-20.,150.],[-25., 25.],[6., 20.]])
	
	optimize_signal_training = {}
	optimize_signal_training['BSN_and_BCI_and_oSNR']['optimization grid dimensions'] = np.array([10.,10.,10.])
	optimize_signal_training['BSN_and_BCI_and_oSNR']['optimization grid ranges'] = np.array([[1.0,100.],[0.1,10.],[0.01,1.0]])
	
	optimize_noise_training = {}
	optimize_noise_training['BSN_and_BCI_and_oSNR']['optimization grid dimensions'] = np.array([10.,10.,10.])
	optimize_noise_training['BSN_and_BCI_and_oSNR']['optimization grid ranges'] = np.array([[0.1,10.],[0.1,10.],[0.01,1.0]])

elif LRT_param_group == 'BSN_and_BCI':
	param_info = {}
	param_info['BSN_and_BCI'] = {}
	param_info['BSN_and_BCI']['dimension'] = 2
	param_info['BSN_and_BCI']['param names'] = ['BSN','BCI']
	param_info['BSN_and_BCI']['interp range'] = np.array([[-20., 150.],[-25., 25.]])
	
	optimize_signal_training = {}
	optimize_signal_training['BSN_and_BCI']['optimization grid dimensions'] = np.array([50.,50.])
	optimize_signal_training['BSN_and_BCI']['optimization grid ranges'] = np.array([[1.0,100.],[0.1,10.]])
	
	optimize_noise_training = {}
	optimize_noise_training['BSN_and_BCI']['optimization grid dimensions'] = np.array([50.,50.])
	optimize_noise_training['BSN_and_BCI']['optimization grid ranges'] = np.array([[0.1,10.],[0.1,10.]])

#Collect all signal training coordinates from dictionaries and put in arrays
sig_BSN = np.ones(len(updated_signal_dic))*np.nan
sig_BCI = np.ones(len(updated_signal_dic))*np.nan
sig_oSNR = np.ones(len(updated_signal_dic))*np.nan
for i, key in enumerate(updated_signal_dic):
	try:
		 sig_BSN[i] = updated_signal_dic[key]['BSN']
		 sig_BCI[i] = updated_signal_dic[key]['BCI']
		 sig_oSNR[i] = updated_signal_dic[key]['Omicron SNR']
	except KeyError:
		print 'Failure for signal train event: ',i
sig_BSN = sig_BSN[ sig_BSN >= -np.inf ]
sig_BCI = sig_BCI[ sig_BCI >= -np.inf ]
sig_oSNR = sig_oSNR[ sig_oSNR >= -np.inf ]

train_signal_data = {}

train_signal_data['BSN'] = {}
train_signal_data['BSN']['data'] = np.transpose(np.array([sig_BSN]))
train_signal_data['BSN']['KDE ranges'] = np.array([[-20., 150.]])
train_signal_data['BSN']['KDE bandwidths'] = np.array([np.nan])
train_signal_data['BSN']['KDE points'] = np.array([75])

train_signal_data['BCI'] = {}
train_signal_data['BCI']['data'] = np.transpose(np.array([sig_BCI]))
train_signal_data['BCI']['KDE ranges'] = np.array([[-25., 25.]])
train_signal_data['BCI']['KDE bandwidths'] = np.array([np.nan])
train_signal_data['BCI']['KDE points'] = np.array([75])

train_signal_data['oSNR'] = {}
train_signal_data['oSNR']['data'] = np.transpose(np.array([sig_oSNR]))
train_signal_data['oSNR']['KDE ranges'] = np.array([[6., 20.]])
train_signal_data['oSNR']['KDE bandwidths'] = np.array([np.nan])
train_signal_data['oSNR']['KDE points'] = np.array([75])

#Collect all noise training coordinates from dictionaries and put in arrays
noise_BSN = np.ones(len(updated_noise_dic))*np.nan
noise_BCI = np.ones(len(updated_noise_dic))*np.nan
noise_oSNR = np.ones(len(updated_noise_dic))*np.nan
for i, key in enumerate(updated_noise_dic):
	try:
		 noise_BSN[i] = updated_noise_dic[key]['BSN']
		 noise_BCI[i] = updated_noise_dic[key]['BCI']
		 noise_oSNR[i] = updated_noise_dic[key]['Omicron SNR']
	except KeyError:
		print 'Failure for noise train event: ',i
noise_BSN = noise_BSN[ noise_BSN >= -np.inf ]
noise_BCI = noise_BCI[ noise_BCI >= -np.inf ]
noise_oSNR = noise_oSNR[ noise_oSNR >= -np.inf ]

train_noise_data = {}

train_noise_data['BSN'] = {}
train_noise_data['BSN']['data'] = np.transpose(np.array([noise_BSN]))
train_noise_data['BSN']['KDE ranges'] = np.array([[-20., 150.]])
train_noise_data['BSN']['KDE bandwidths'] = np.array([np.nan])
train_noise_data['BSN']['KDE points'] = np.array([75])

train_noise_data['BCI'] = {}
train_noise_data['BCI']['data'] = np.transpose(np.array([noise_BCI]))
train_noise_data['BCI']['KDE ranges'] = np.array([[-25., 25.]])
train_noise_data['BCI']['KDE bandwidths'] = np.array([np.nan])
train_noise_data['BCI']['KDE points'] = np.array([75])

train_noise_data['oSNR'] = {}
train_noise_data['oSNR']['data'] = np.transpose(np.array([noise_oSNR]))
train_noise_data['oSNR']['KDE ranges'] = np.array([[6., 20.]])
train_noise_data['oSNR']['KDE bandwidths'] = np.array([np.nan])
train_noise_data['oSNR']['KDE points'] = np.array([75])

#Initialize LLRT object (which launches likelihood training), and then save the trained likelihoods to temporary files
LLRT = LLRT_object_beta.LLRT(calc_info=calc_info, param_info=param_info, train_signal_data=train_signal_data, train_noise_data=train_noise_data, foreground_data=None, background_data=None, optimize_signal_training=optimize_signal_training, optimize_noise_training=optimize_noise_training)
LLRT.save_all_KDE(like_dir)
LLRT.save_all_bandwidths(like_dir)
