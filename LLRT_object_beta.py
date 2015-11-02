import sys
sys.path.insert(1,'/home/ryan.lynch/numpy/numpy-1.8.2-INSTALL/lib64/python2.6/site-packages')
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import re
from sys import exit
import pickle
import matplotlib.lines as mlines
from matplotlib import cm
import scipy.ndimage.interpolation as ndimage_interp

#=================================================
#
#        Log Likelihood Ratio Test class
#
#=================================================

class LLRT(object):
	"""
	An object that handles log likelihood ratio testing for a given set of parameters 
	"""
	
	###
	def __init__(self, calc_info, param_info, train_signal_data, train_noise_data, foreground_data=None, background_data=None, optimize_signal_training=None, optimize_noise_training=None):
		"""
		*The calc_info option should be a dictionary with the following structure:
			-['interp method'] --> Method of interpolation to use (i.e, 'Linear', ...)
			-['extrap method'] --> Method of extrapolation to use (i.e, 'Nearest', ...)
		
		*The param_info option should be a dictionary with the following structure:
			-['group name']
				*['dimension'] --> Number of parameters in group
				*['param names'] --> Names of each parameter in group
				*['interp range'] --> interpolation ranges for each parameter in group
			
		*The *_data options should be dictionaries with the following structure:
			-['npoints'] -->  number of data points (must be passed with data for foreground and background)
			-['param name/group name']
				*['data'] -->  raw data
				*['KDE ranges'] --> range over which to do KDE smoothing (must be passed with data for training)
				*['KDE bandwidths'] --> bandwidth to use for KDE smoothing (must be passed with data for training)
				*['KDE points'] --> number of points to use for KDE smoothing (must be passed with data for training)
				*['KDE'] --> Set of KDE-smoothed points to be interpolated among (for training only)
				
		*The optimize_*_training options should be dictionaries with the following structure:
			-['group name']
				*['optimization grid dimensions'] -->  The number of points to grid over for each parameter in the group
				*['optimization grid ranges'] -->  The range of values to logarithmically grid over for each parameter in the group
		"""
		#Should do necessary checks on passed stuff here
		
		#Locally define some useful variables
		self.group_names = param_info.keys()
		self.interp_method = calc_info['interp method']
		self.extrap_method = calc_info['extrap method']
		
		###Signal###
		
		#Take the passed signal data and group properly
		self.signal = {}
		for key in self.group_names:
			self.signal[key] = {}
			self.signal[key]['dimension'] = int(param_info[key]['dimension'])
			self.signal[key]['param names'] = param_info[key]['param names']	
			self.signal[key]['interp range'] = param_info[key]['interp range']
			self.signal[key]['KDE'] = None
						
			#See if data has been passed for group
			if train_signal_data.has_key(key):
				#First check to see if we have KDE data
				if train_signal_data[key].has_key('KDE'):
					self.signal[key]['KDE'] = train_signal_data[key]['KDE']
				
				#If not, check to see if we have group data	
				elif train_signal_data[key].has_key('data'):
					self.signal[key]['data'] = np.reshape( train_signal_data[key]['data'], (-1,self.signal[key]['dimension']) )
					self.signal[key]['KDE ranges'] = np.reshape( train_signal_data[key]['KDE ranges'], (self.signal[key]['dimension'],2) )
					self.signal[key]['KDE bandwidths'] = np.reshape( train_signal_data[key]['KDE bandwidths'], (self.signal[key]['dimension']) ) 
					self.signal[key]['KDE points'] = np.reshape( train_signal_data[key]['KDE points'], (self.signal[key]['dimension']) )
			
			#If not passed for group, see if data has been passed for params
			else:
				self.signal[key]['data'] = [None]*self.signal[key]['dimension']
				self.signal[key]['KDE ranges'] = [None]*self.signal[key]['dimension']
				self.signal[key]['KDE bandwidths'] = [None]*self.signal[key]['dimension']
				self.signal[key]['KDE points'] = [None]*self.signal[key]['dimension']
				for i,param in enumerate(self.signal[key]['param names']):
						self.signal[key]['data'][i] = train_signal_data[param]['data']
						self.signal[key]['KDE ranges'][i] = train_signal_data[param]['KDE ranges']
						self.signal[key]['KDE bandwidths'][i] = train_signal_data[param]['KDE bandwidths']
						self.signal[key]['KDE points'][i] = train_signal_data[param]['KDE points']
				self.signal[key]['data'] = np.array(self.signal[key]['data'])
				self.signal[key]['data'] = np.reshape( np.transpose(self.signal[key]['data']), (-1,self.signal[key]['dimension']) )
				self.signal[key]['KDE ranges'] = np.reshape( np.array(self.signal[key]['KDE ranges']), (self.signal[key]['dimension'],2) )
				self.signal[key]['KDE bandwidths'] = np.reshape( np.array(self.signal[key]['KDE bandwidths']), (self.signal[key]['dimension']) )
				self.signal[key]['KDE points'] = np.reshape( np.array(self.signal[key]['KDE points']), (self.signal[key]['dimension']) )
			
			#If flagged, find optimal bandwidths to train with
			if optimize_signal_training:
				self.signal[key]['KDE bandwidths'] = self.find_opt_gaussian_band_grid(groupname=key, model="Signal", grid_points=np.reshape( optimize_signal_training[key]['optimization grid dimensions'], (self.signal[key]['dimension']) ), grid_ranges=np.reshape( optimize_signal_training[key]['optimization grid ranges'], (self.signal[key]['dimension'],2) ))
			
			#If no KDE data passed, do KDE smoothing
			if self.signal[key]['KDE'] == None:
				self.signal[key]['KDE'] = self.log_KDE_smoothing(model='Signal', groupname=key)
			
		###NOISE###
				
		#Take the passed noise data and group properly
		self.noise = {}
		for key in self.group_names:
			self.noise[key] = {}
			self.noise[key]['dimension'] = int(param_info[key]['dimension'])
			self.noise[key]['param names'] = param_info[key]['param names']	
			self.noise[key]['interp range'] = param_info[key]['interp range']
			self.noise[key]['KDE'] = None
						
			#See if data has been passed for group
			if train_noise_data.has_key(key):
				#First check to see if we have KDE data
				if train_noise_data[key].has_key('KDE'):
					self.noise[key]['KDE'] = train_noise_data[key]['KDE']
				
				#If not, check to see if we have group data	
				elif train_noise_data[key].has_key('data'):
					self.noise[key]['data'] = np.reshape( train_noise_data[key]['data'], (-1,self.noise[key]['dimension']) )
					self.noise[key]['KDE ranges'] = np.reshape( train_noise_data[key]['KDE ranges'], (self.noise[key]['dimension'],2) )
					self.noise[key]['KDE bandwidths'] = np.reshape( train_noise_data[key]['KDE bandwidths'], (self.noise[key]['dimension']) )
					self.noise[key]['KDE points'] = np.reshape( train_noise_data[key]['KDE points'], (self.noise[key]['dimension']) )
			
			#If not passed for group, see if data has been passed for params
			else:
				self.noise[key]['data'] = [None]*self.noise[key]['dimension']
				self.noise[key]['KDE ranges'] = [None]*self.noise[key]['dimension']
				self.noise[key]['KDE bandwidths'] = [None]*self.noise[key]['dimension']
				self.noise[key]['KDE points'] = [None]*self.noise[key]['dimension']
				for i,param in enumerate(self.noise[key]['param names']):
						self.noise[key]['data'][i] = train_noise_data[param]['data']
						self.noise[key]['KDE ranges'][i] = train_noise_data[param]['KDE ranges']
						self.noise[key]['KDE bandwidths'][i] = train_noise_data[param]['KDE bandwidths']
						self.noise[key]['KDE points'][i] = train_noise_data[param]['KDE points']
				self.noise[key]['data'] = np.array(self.noise[key]['data'])
				self.noise[key]['data'] = np.reshape( np.transpose(self.noise[key]['data']), (-1,self.noise[key]['dimension']) )
				self.noise[key]['KDE ranges'] = np.reshape( np.array(self.noise[key]['KDE ranges']), (self.noise[key]['dimension'],2) )
				self.noise[key]['KDE bandwidths'] = np.reshape( np.array(self.noise[key]['KDE bandwidths']), (self.noise[key]['dimension']) )
				self.noise[key]['KDE points'] = np.reshape( np.array(self.noise[key]['KDE points']), (self.noise[key]['dimension']) )
		
			#If flagged, find optimal bandwidths to train with
			if optimize_noise_training:
				self.noise[key]['KDE bandwidths'] = self.find_opt_gaussian_band_grid(groupname=key, model="Noise", grid_points=np.reshape( optimize_noise_training[key]['optimization grid dimensions'], (self.noise[key]['dimension']) ), grid_ranges=np.reshape( optimize_noise_training[key]['optimization grid ranges'], (self.noise[key]['dimension'],2) ))
				
			#If no KDE data passed, do KDE smoothing
			if self.noise[key]['KDE'] == None:
				self.noise[key]['KDE'] = self.log_KDE_smoothing(model='Noise', groupname=key)
		
		###FOREGROUND###
				
		#If passed, take the foreground data and group properly
		if foreground_data:
			self.foreground = {}
			self.foreground['npoints'] = int(foreground_data['npoints'])
			for key in self.group_names:
				self.foreground[key] = {}
				self.foreground[key]['dimension'] = int(param_info[key]['dimension'])
				self.foreground[key]['param names'] = param_info[key]['param names']
				self.foreground[key]['data'] = None
							
				#See if data has been passed for group
				if foreground_data.has_key(key):
					self.foreground[key]['data'] = np.reshape( foreground_data[key]['data'], (-1,self.foreground[key]['dimension']) )
				
				#If not passed for group, see if data has been passed for params
				else:
					self.foreground[key]['data'] = [None]*self.foreground[key]['dimension']
					for i,param in enumerate(self.foreground[key]['param names']):
						self.foreground[key]['data'][i] = foreground_data[param]['data']
					self.foreground[key]['data'] = np.array(self.foreground[key]['data'])
					self.foreground[key]['data'] = np.reshape( np.transpose(self.foreground[key]['data']), (-1,self.foreground[key]['dimension']) )
					
			#Calculate value of total log likelihood ratio for all foreground data points
			self.foreground['LLR'] = self.calculate_totalLLR(groundtype='Foreground')
			
		###BACKGROUND###
		
		#If passed, take the background data and group properly
		if background_data:
			self.background = {}
			self.background['npoints'] = int(background_data['npoints'])
			for key in self.group_names:
				self.background[key] = {}
				self.background[key]['dimension'] = int(param_info[key]['dimension'])
				self.background[key]['param names'] = param_info[key]['param names']
				self.background[key]['data'] = None
							
				#See if data has been passed for group
				if background_data.has_key(key):
					self.background[key]['data'] = np.reshape( background_data[key]['data'], (-1,self.background[key]['dimension']) )
				
				#If not passed for group, see if data has been passed for params
				else:
					self.background[key]['data'] = [None]*self.background[key]['dimension']
					for i,param in enumerate(self.background[key]['param names']):
						self.background[key]['data'][i] = background_data[param]['data']
					self.background[key]['data'] = np.array(self.background[key]['data'])
					self.background[key]['data'] = np.reshape( np.transpose(self.background[key]['data']), (-1,self.background[key]['dimension']) )
					
			#Calculate value of total log likelihood ratio for all background data points
			self.background['LLR'] = self.calculate_totalLLR(groundtype='Background')
			
	###
	def KDE_smoothing(self, model, groupname):
		"""
		Perform KDE smoothing in specificied number of dimensions (ndim = n_params)
		"""
		#Load in necessary info for either data or noise depending on the passed model
		if model == 'Signal':
			dic = self.signal
		elif model == 'Noise':
			dic = self.noise
		else:
			#Raise error here
			pass
		
		n_params = dic[groupname]['dimension']
		KDE_ranges = dic[groupname]['KDE ranges']
		bandwidths = dic[groupname]['KDE bandwidths']
		data = dic[groupname]['data']
		grid_points = dic[groupname]['KDE points']
		
		#Initialize grid over which to do KDE binning
		grid_values = [None]*n_params
		for i in xrange(n_params):
			grid_values[i] = np.linspace(start=KDE_ranges[i,0], stop=KDE_ranges[i,1], num=grid_points[i])
		
		if n_params == 1:
			location_kde = np.transpose(np.array(grid_values))
		else:
			location_kde = np.array(np.meshgrid(*grid_values,indexing='ij'))
			location_kde = np.rollaxis(location_kde, 0, (n_params + 1))
		
		grid_shape = ()
		for i in xrange(n_params):
			grid_shape += (grid_points[i],)
		height_kde = np.zeros(grid_shape)
		
		#Calculate height contribution of each data point over the grid
		for i in xrange(len(data)):
			height_kde +=  np.exp( -0.5 * np.sum( ((data[i,:] - location_kde)/bandwidths)**2., axis=-1) ) / (np.product(bandwidths) * (2.*np.pi)**(n_params/2.))	

		height_kde /= float(len(data))
		log_height_kde = np.log10(height_kde)
		
		return location_kde, log_height_kde
	
	###
	def log_KDE_smoothing(self, model, groupname):
		"""
		Perform log10 KDE smoothing in specificied number of dimensions (ndim = n_params)
		"""
		#Load in necessary info for either data or noise depending on the passed model
		if model == 'Signal':
			dic = self.signal
		elif model == 'Noise':
			dic = self.noise
		else:
			#Raise error here
			pass
		
		n_params = dic[groupname]['dimension']
		KDE_ranges = dic[groupname]['KDE ranges']
		bandwidths = dic[groupname]['KDE bandwidths']
		data = dic[groupname]['data']
		grid_points = dic[groupname]['KDE points']
		
		#Initialize grid over which to do KDE binning
		grid_values = [None]*n_params
		for i in xrange(n_params):
			grid_values[i] = np.linspace(start=KDE_ranges[i,0], stop=KDE_ranges[i,1], num=grid_points[i])
		
		if n_params == 1:
			location_kde = np.transpose(np.array(grid_values))
		else:
			location_kde = np.array(np.meshgrid(*grid_values,indexing='ij'))
			location_kde = np.rollaxis(location_kde, 0, (n_params + 1))
		
		grid_shape = ()
		for i in xrange(n_params):
			grid_shape += (grid_points[i],)
		log_height_kde = np.zeros(grid_shape)
		
		#Calculate height contribution of each data point over the grid
		log_height_kde += -np.log10(np.product(bandwidths) * (2.*np.pi)**(n_params/2.) * float(len(data)))
		for i in xrange(len(data)):
			if i == 0:
				A =  -0.5 * np.log10(np.e) * np.sum( ((data[i,:] - location_kde)/bandwidths)**2., axis=-1)
				log_height_kde += A
			else:
				B = -A - 0.5 * np.log10(np.e) * np.sum( ((data[i,:] - location_kde)/bandwidths)**2., axis=-1)
				C = B
				C[B <= 300.] = np.log10( 1. + 10.**(B[B <= 300.]) )
				A += C
				log_height_kde += C
		
		return location_kde, log_height_kde
	
	###
	def interpolate(self, known_coords, known_values, interp_coords, groupname):
		"""
		Delegates to interpolation method based on user's choice
		"""
		#Find which interpolation method is desired
		if self.interp_method == "Grid Linear":
			interp_values = self.grid_interpolation_linear(known_coords=known_coords, known_values=known_values, interp_coords=interp_coords, groupname=groupname)
		elif self.interp_method == "Linear":
			from scipy.interpolate import griddata
			interp_values = self.interpolate_linear(known_coords=known_coords, known_values=known_values, interp_coords=interp_coords, groupname=groupname)
		else:
			#Should raise error
			pass
		
		return interp_values
	
	###
	def grid_interpolation_linear(self, known_coords, known_values, interp_coords, groupname):
		"""
		Interpolate between known values on a regular grid of coordinates
		*known_coords is array of shape \product(ith grid length) x (# of dimensions)
		*known_values is array of shape \product(ith grid length)
		*interp_coords is array of shape (# of events) x (# of dimensions)
		"""
		#known_coords is array of shape \product(ith grid length) x (# of dimensions), use to find mapping between coords and grid indices
		ndim = np.shape(known_coords)[-1]
		grid_lens = np.array(np.shape(known_coords)[:-1])
		
		grid_starts = np.zeros(ndim)
		grid_factors = np.zeros(ndim)
		for dim in xrange(ndim):
			#Find min and max of grid coordinates for each dimension and use these to construct mapping to grid indices
			ind_min = np.zeros(ndim+1,dtype='int')
			ind_min[-1] = dim
			
			ind_max = np.zeros(ndim+1,dtype='int')
			ind_max[dim] = -1
			ind_max[-1] = dim
			
			grid_starts[dim] = known_coords[tuple(ind_min)]
			grid_factors[dim] = (known_coords[tuple(ind_max)] - known_coords[tuple(ind_min)]) / float(grid_lens[dim] - 1.)
		
		#known_values is array of shape \product(ith grid length), values map fine to grid indices as is
		grid_known_values = known_values
		
		#interp_coords is array of shape (# of events) x (# of dimensions), need to map into grid indices and then take transpose
		grid_interp_coords = np.transpose( (interp_coords - grid_starts) / grid_factors )
		
		#With everything mapped to grid indices, we can now interpolate between "pixels"
		return ndimage_interp.map_coordinates(input=grid_known_values, coordinates=grid_interp_coords, output=float, order=1, mode='nearest')	
	
	###
	def interpolate_linear(self, known_coords, known_values, interp_coords, groupname):
		"""
		Does linear interpolation among known data points
		"""
		#First need to reshape known_coords and known_values
		n_params = self.signal[groupname]['dimension']
		known_coords = np.reshape( known_coords, (-1,n_params) )
		known_values = np.reshape( known_values, (-1) )		
		return griddata(known_coords, known_values, interp_coords, method='linear')
	
	###
	def extrapolate(self, known_coords, known_values, extrap_coords, groupname):
		"""
		Delegates to extrapolation method based on user's choice
		"""
		#Find which extrapolation method is desired
		if self.extrap_method == "Grid Nearest":
			extrap_values = self.grid_interpolation_linear(known_coords=known_coords, known_values=known_values, interp_coords=extrap_coords, groupname=groupname)
		elif self.extrap_method == "Nearest":
			extrap_values = self.extrapolate_nearest(known_coords=known_coords, known_values=known_values, extrap_coords=extrap_coords, groupname=groupname)
		else:
			#Should raise error
			pass
		
		return extrap_values
		
	###
	def extrapolate_nearest(self, known_coords, known_values, extrap_coords, groupname):
		"""
		Does extrapolation to the nearest known boundary point
		"""
		#First need to reshape known_coords and known_values
		n_params = self.signal[groupname]['dimension']
		known_coords = np.reshape( known_coords, (-1,n_params) )
		known_values = np.reshape( known_values, (-1) )
				
		#Different methods if dimension is (!/=)= 1
		if self.signal[groupname]['dimension'] == 1:
			#Reshape extrap_coords
			extrap_coords = np.reshape(extrap_coords, (-1))
			
			#Initialize array for extrapolation values
			extrap_values = np.ones(len(extrap_coords))*np.nan
			
			#Find minimum and maximum coordinates and the respective values at those coordinates
			min_coord_index = np.argmin(known_coords)
			min_coord = known_coords[min_coord_index]
			min_coord_value = known_values[min_coord_index]
			
			max_coord_index = np.argmax(known_coords)
			max_coord = known_coords[max_coord_index]
			max_coord_value = known_values[max_coord_index]
			
			#Do the nearest-bound extrapolation
			extrap_values[np.absolute(extrap_coords - max_coord) >= np.absolute(extrap_coords - min_coord)] = min_coord_value
			extrap_values[np.absolute(extrap_coords - max_coord) < np.absolute(extrap_coords - min_coord)] = max_coord_value
			
		elif self.signal[groupname]['dimension'] > 1:
			extrap_values = griddata(known_coords, known_values, extrap_coords, method='nearest')
		
		return extrap_values
		
	###
	def calculate_groupLLR(self, groundtype, groupname):
		"""
		Calculate the values of the log likelihood ratio for a parameter group for all data points, interpolating if data points are within a certain range, extrapolating elsewise
		"""
		#Load in data to compute group LLR for
		if groundtype == 'Foreground':
			data = self.foreground[groupname]['data']
		elif groundtype == 'Background':
			data = self.background[groupname]['data']
		
		#Find number of parameters within group
		n_params = self.signal[groupname]['dimension']
		
		#Divide data points into those which need interpolation and those that need extrapolation
		interp_array_sig = np.product((data >= self.signal[groupname]['interp range'][:,0]) * (data <= self.signal[groupname]['interp range'][:,1]), axis=-1, dtype=bool)
		extrap_array_sig = ~interp_array_sig
		
		interp_array_noise = np.product((data >= self.noise[groupname]['interp range'][:,0]) * (data <= self.noise[groupname]['interp range'][:,1]), axis=-1, dtype=bool)
		extrap_array_noise = ~interp_array_noise
		
		#Initialize arrays to store likelihood values
		log_likelihood_signal = np.zeros(len(data))
		log_likelihood_noise = np.zeros(len(data))
		
		#Get necessary coordinates and values
		coords_sig = self.signal[groupname]['KDE'][0]
		values_sig = self.signal[groupname]['KDE'][1]
		coords_noise = self.noise[groupname]['KDE'][0]
		values_noise = self.noise[groupname]['KDE'][1]		
				
		#Calculate interpolated likelihoods
		log_likelihood_signal[interp_array_sig] = self.interpolate(known_coords=coords_sig, known_values=values_sig, interp_coords=data[interp_array_sig], groupname=groupname)
		log_likelihood_noise[interp_array_noise] = self.interpolate(known_coords=coords_noise, known_values=values_noise, interp_coords=data[interp_array_noise], groupname=groupname)
		
		#Calculate extrapolated likelihoods
		log_likelihood_signal[extrap_array_sig] = self.extrapolate(known_coords=coords_sig, known_values=values_sig, extrap_coords=data[extrap_array_sig], groupname=groupname)
		log_likelihood_noise[extrap_array_noise] = self.extrapolate(known_coords=coords_noise, known_values=values_noise, extrap_coords=data[extrap_array_noise], groupname=groupname)
		
		return log_likelihood_signal - log_likelihood_noise
		
	###
	def calculate_totalLLR(self, groundtype):
		"""
		Calculate the values of the total log likelihood ratio for all data points
		"""
		#Initialize total log likelihood ratio
		if groundtype == 'Foreground':
			tot_LLR = np.zeros(self.foreground['npoints'])
		elif groundtype == 'Background':
			tot_LLR = np.zeros(self.background['npoints'])
			
		#Iterate over all parameter groups
		for group in self.group_names:
			tot_LLR += self.calculate_groupLLR(groundtype=groundtype, groupname=group)
		
		return tot_LLR
		
	###
	def log_likelihood_ratios(self, groundtype):
		"""
		Returns values of log likelihood ratio for all data points
		"""
		if groundtype == 'Foreground':
			LLR = self.foreground['LLR']
		elif groundtype == 'Background':
			LLR = self.background['LLR']
		
		return LLR
		
	###
	def LLR_above_thresh(self, threshold, groundtype):
		"""
		Return a boolean array of log likelihood ratios above/below a given threshold
		"""
		LLR = self.log_likelihood_ratios(groundtype=groundtype)
		return (LLR >= threshold)
	
	###
	def calculate_FAR_of_thresh(self, threshold, livetime, groundtype):
		"""
		Calculates the FAR of a given threshold given an amount of livetime wrt to given groundtype
		"""
		#Find FAR from number of events above a threshold
		FAR = float(np.sum(self.LLR_above_thresh(threshold=threshold, groundtype=groundtype)))/float(livetime)
		
		#If FAR is null, choose FAR to be that of the most extreme background event
		if FAR == 0.:
			FAR = 1./float(livetime)
		
		return FAR
	
	###
	def calculate_threshold_from_background(self, FAR, livetime):
		"""
		Calculate the log likelihood ratio threshold that will produce the given FAR over the extent of the background's livetime
		"""
		#Calculate the number of false alarms expected over a time interval at a given FAR
		num_back_above_thresh = int( FAR * livetime)
		
		#Sort the background LLRs from smallest to largest
		back_LLRs = np.sort(self.log_likelihood_ratios(groundtype='Background'))
		
		if back_LLRs.size == 0:
			raise ValueError, "Error: no background events so cannot determine threshold log likelihood ratio."
		elif num_back_above_thresh > len(back_LLRs):
			print "Warning: the FAR is too high for the given number of background events to correctly determine threshold log likelihood ratio, so using smallest found LLR."
			threshold = back_LLRs[0]
		elif num_back_above_thresh == 0.:
			print "Warning: the FAR is too low for the given number of background events to correctly determine threshold log likelihood ratio, so using the largest found LLR."
			threshold = back_LLRs[-1]
		else:
			threshold = back_LLRs[ len(back_LLRs) - num_back_above_thresh ]
		
		return threshold
	
	###		
	def largest_background_LLR(self):
		"""
		Return the largest log likelihood ratio found by a background event
		"""
		return np.sort(self.log_likelihood_ratios(groundtype='Background'))[-1]
	
	###	
	def save_group_KDE(self, model, groupname, outdir):
		"""
		Save KDE estimations for a given parameter group for a given model (signal or noise)
		"""
		#Load in necessary info for either data or noise depending on the passed model
		if model == 'Signal':
			dic = self.signal
		elif model == 'Noise':
			dic = self.noise
		else:
			#Raise error here
			pass
		
		#Save binary files cotaining both KDE coordinates and KDE values
		np.save('%s/%s_%s_log_KDE_coords.npy'%(outdir,groupname,model), dic[groupname]['KDE'][0])
		np.save('%s/%s_%s_log_KDE_values.npy'%(outdir,groupname,model), dic[groupname]['KDE'][1])
		
	###	
	def save_all_KDE(self, outdir):
		"""
		Save all KDE estimations, for all parameter groups for both signal and noise
		"""
		#Iterate over all parameter groups
		for group in self.group_names:
			self.save_group_KDE(model='Signal', groupname=group, outdir=outdir)
			self.save_group_KDE(model='Noise', groupname=group, outdir=outdir)
	
	###	
	def save_group_bandwidths(self, model, groupname, outdir):
		"""
		Save KDE bandwidths for a given parameter group for a given model (signal or noise)
		"""
		#Load in necessary info for either data or noise depending on the passed model
		if model == 'Signal':
			dic = self.signal
		elif model == 'Noise':
			dic = self.noise
		else:
			#Raise error here
			pass
		
		#Save binary files cotaining both KDE coordinates and KDE values
		np.save('%s/%s_%s_KDE_bandwidths.npy'%(outdir,groupname,model), dic[groupname]['KDE bandwidths'])
		
	###	
	def save_all_bandwidths(self, outdir):
		"""
		Save all KDE bandwidths, for all parameter groups for both signal and noise
		"""
		#Iterate over all parameter groups
		for group in self.group_names:
			self.save_group_bandwidths(model='Signal', groupname=group, outdir=outdir)
			self.save_group_bandwidths(model='Noise', groupname=group, outdir=outdir)
	
	###		
	def plot_group_log_likelihoods(self, groupname, outdir):
		"""
		Make a plot of the likelihoods for signal and noise for a given parameter group over the KDE interpolation range
		"""
		#Get dimensions of group
		ndim = self.signal[groupname]['dimension']
		
		#Get info for signal
		signal_params = self.signal[groupname]['param names']
		signal_coords = self.signal[groupname]['KDE'][0]
		signal_values = self.signal[groupname]['KDE'][1]
		
		#Get info for noise
		noise_params = self.noise[groupname]['param names']
		noise_coords = self.noise[groupname]['KDE'][0]
		noise_values = self.noise[groupname]['KDE'][1]
		
		#If the parameter group is 1-D, make plot
		if ndim == 1:
			myfig=plt.figure()
			plt.plot(signal_coords, signal_values, 'b-', label='Signal')
			plt.plot(noise_coords, noise_values, 'r-', label='Noise')
			plt.xlabel(signal_params[0])
			plt.ylabel('Log10 Probability Density')
			plt.grid(True,which="both")
			plt.legend(loc='best')
			myfig.savefig('%s/%s_log_likelihoods'%(outdir,groupname), bbox_inches='tight')
		
		#If the parameter group is 2-D, make plot
		elif ndim == 2:
			#Specify coordinates for both signal and noise
			x_sig = signal_coords[:,:,0]
			y_sig = signal_coords[:,:,1]
			z_sig = signal_values
			
			x_noise = noise_coords[:,:,0]
			y_noise = noise_coords[:,:,1]
			z_noise = noise_values
			
			#Make 3-D plot
			myfig=plt.figure()
			ax = plt.axes(projection='3d')
			ax.plot_wireframe(x_sig,y_sig,z_sig,color='b',label='Signal', rstride=1, cstride=1, alpha=0.5)
			ax.plot_wireframe(x_noise,y_noise,z_noise,color='r',label='Noise', rstride=1, cstride=1, alpha=0.5)
			ax.set_xlabel(signal_params[0])
			ax.set_ylabel(signal_params[1])
			ax.set_zlabel('Log 10 Probability Density')
			myfig.savefig('%s/%s_log_likelihoods_3D'%(outdir,groupname), bbox_inches='tight')
			
			#Make contour plot
			myfig=plt.figure()
			plt.contour(x_sig, y_sig, z_sig, linestyles='solid')
			plt.contour(x_noise, y_noise, z_noise, linestyles='dashed')
			plt.xlabel(signal_params[0])
			plt.ylabel(signal_params[1])
			sig_line = mlines.Line2D([],[],linestyle='solid',color='k')
			noise_line = mlines.Line2D([],[],linestyle='dashed',color='k')
			plt.legend([sig_line,noise_line],['Signal','Noise'])
			myfig.savefig('%s/%s_log_likelihoods_contour'%(outdir,groupname), bbox_inches='tight')
			
		#If the parameter group is >= 3-D, cannot make plot for now
		else:
			#Should add warning of some sort here
			pass
	
	###	
	def plot_all_log_likelihoods(self, outdir):
		"""
		Make a plot of the likelihoods for signal and noise over the KDE interpolation range for all parameter groups
		"""
		#Iterate over all parameter groups
		for group in self.group_names:
			self.plot_group_log_likelihoods(groupname=group, outdir=outdir)
	
	###		
	def plot_group_LLR(self, groupname, outdir):
		"""
		Make a plot of the log likelihood ratio for a given parameter group over the KDE interpolation range
		"""		
		#Get dimensions of group
		ndim = self.signal[groupname]['dimension']
		
		#Get info for signal
		signal_params = self.signal[groupname]['param names']
		signal_coords = self.signal[groupname]['KDE'][0]
		signal_values = self.signal[groupname]['KDE'][1]
		
		#Get info for noise
		noise_params = self.noise[groupname]['param names']
		noise_coords = self.noise[groupname]['KDE'][0]
		noise_values = self.noise[groupname]['KDE'][1]
		
		#If the KDE grid is the same for signal and noise, we can trivially calculate the LLR over a grid
		if (signal_coords - noise_coords).any():
			#Should raise some warning here
			pass
		else:
			LLR_params = signal_params
			LLR_coords = signal_coords
			LLR_values = signal_values - noise_values
		
			#If the parameter group is 1-D, make plot
			if ndim == 1:
				myfig=plt.figure()
				plt.plot(LLR_coords, LLR_values, 'b-')
				plt.xlabel(LLR_params[0])
				plt.ylabel('Log10 Likelihood Ratio')
				plt.grid(True,which="both")
				myfig.savefig('%s/%s_LLR'%(outdir,groupname), bbox_inches='tight')
			
			#If the parameter group is 2-D, make plot
			elif ndim == 2:
				#Specify coordinates for both signal and noise
				x_LLR = LLR_coords[:,:,0]
				y_LLR = LLR_coords[:,:,1]
				z_LLR = LLR_values
				
				#Make 3-D plot				
				myfig=plt.figure()
				ax = plt.axes(projection='3d')
				surf = ax.plot_surface(x_LLR,y_LLR,z_LLR,color='b', rstride=1, cstride=1, cmap=cm.jet, linewidth=0)
				ax.set_xlabel(LLR_params[0])
				ax.set_ylabel(LLR_params[1])
				ax.set_zlabel('Log10 Likelihood Ratio')
				myfig.colorbar(surf)
				myfig.savefig('%s/%s_LLR_3D'%(outdir,groupname), bbox_inches='tight')
				
				#Make contour plot
				myfig=plt.figure()
				plt.contour(x_LLR, y_LLR, z_LLR, linestyles='solid', levels=[-2., 0., 2.], colors=['b','k','r'])
				plt.xlabel(LLR_params[0])
				plt.ylabel(LLR_params[1])
				l_line = mlines.Line2D([],[],linestyle='solid',color='b')
				m_line = mlines.Line2D([],[],linestyle='solid',color='k')
				h_line = mlines.Line2D([],[],linestyle='solid',color='r')
				plt.legend([l_line,m_line,h_line],['-2','0','2'])
				myfig.savefig('%s/%s_LLR_contour'%(outdir,groupname), bbox_inches='tight')
				
			#If the parameter group is >= 3-D, cannot make plot for now
			else:
				#Should add warning of some sort here
				pass	

	###	
	def plot_all_LLRs(self, outdir):
		"""
		Make a plot of the log likelihoods ratio over the KDE interpolation range for all parameter groups
		"""
		#Iterate over all parameter groups
		for group in self.group_names:
			self.plot_group_LLR(groupname=group, outdir=outdir)
	
	###
	def KL_function_gaussian(self, H, data):
		"""
		Calculate the function that needs to be maximized to optimize the bandwidths according to the KL criteria
		"""
		#Initialize stuff
		n_data = float(len(data))
		n_dim = float(len(H))
		
		#Calculate the KL function by iterating
		KL_func = 0.
		for i in xrange(int(n_data)):
			KL_func += np.log10( -0.9999999999999999 + np.sum( np.exp(-0.5 * np.sum( ((data[i,:] - data[:,:])/H[:])**2., axis=-1)), axis=-1) )
		KL_func /= n_data
		KL_func -= np.log10( (n_data-1.) * np.sqrt( (2.*np.pi)**n_dim * np.prod(H[:])**2.) )

		return KL_func		

	###
	def oneD_rule_of_thumb_bandwidth_gaussian(self, data):
		"""
		Calculate the Silverman rule-of-thumb optimal bandwidth for a single parameter
		"""
		#Initialize stuff
		n_data = len(data)
		
		#Calculate sample variance
		samp_var = np.var(data,ddof=1)
		
		#Calculate 1-D r.o.t bandwidth
		rot_band = 1.06 * np.sqrt(samp_var) * n_data**(-1./5.)
		
		return rot_band
	
	###	
	def find_opt_gaussian_band_grid(self, groupname, model, grid_points, grid_ranges=None):
		"""
		Calculates optimal bandwidth for a parameter group using the KL criteria over a grid
		"""
		#Load in necessary info for either data or noise depending on the passed model
		if model == 'Signal':
			dic = self.signal
		elif model == 'Noise':
			dic = self.noise
		else:
			#Raise error here
			pass
		
		n_dim = dic[groupname]['dimension']
		data = dic[groupname]['data']
				
		#Initialize grid 
		if grid_ranges == None:
			#Find rule-of-thumb estimate of optimal bandwidth for each parameter in the group
			Ho = np.zeros(n_dim)
			for d in xrange(n_dim):
				Ho[d] = self.oneD_rule_of_thumb_bandwidth_gaussian(data[:,d])
			grid_ranges = np.zeros((n_dim,2))
			grid_ranges[:,0] = 0.1*Ho
			grid_ranges[:,1] = 10.*Ho
		
		grid_values = [None]*n_dim
		for d in xrange(n_dim):
			grid_values[d] = np.logspace(np.log10(grid_ranges[d,0]),np.log10(grid_ranges[d,1]),grid_points[d])
			
		if n_dim == 1:
			coords = np.transpose(np.array(grid_values))
		else:
			coords = np.array(np.meshgrid(*grid_values,indexing='ij'))
			coords = np.rollaxis(coords, 0, (n_dim + 1))
		
		coords = np.reshape(coords,(-1,n_dim))
		
		#Calculate the KL criteria function at each point on the grid, keeping track of the coordinates that yield the maximum value
		max_coords = None
		max_val = None
		for tmp_coords in coords:
			tmp_val = self.KL_function_gaussian(tmp_coords, data)
			if tmp_val > max_val:
				max_val = tmp_val
				max_coords = tmp_coords
		
		return max_coords
