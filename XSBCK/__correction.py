
## Copyright(c) 2022 / 2024 Yoann Robin
## 
## This file is part of XSBCK.
## 
## XSBCK is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## XSBCK is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with XSBCK.  If not, see <https://www.gnu.org/licenses/>.

#############
## Imports ##
#############

import sys
import os
import gc
import warnings
import logging
import datetime as dt

import numpy  as np
import xarray as xr

import SBCK as bc
import SBCK.ppp as bcp

import inspect

from .__XSBCKParams import xsbckParams
from .__logs import log_start_end


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


#############
## Classes ##
#############

###############
## Functions ##
###############


def yearly_window( tbeg_ , tend_ , wleft , wpred , wright , tleft_ , tright_ ):##{{{
	"""
	XSBCK.yearly_window
	===================
	Generator to iterate over the time axis between tbeg_ and tend_, with a
	fitting window of lenght wleft + wpred + wright, and a predict window of
	length wpred.
	
	Arguments
	---------
	tbeg_:
		Starting year
	tend_:
		Ending year
	wleft:
		Lenght of left window
	wpred:
		Lenght of middle / predict window
	wright:
		Lenght of right window
	tleft_:
		Left bound
	tright_:
		Right bound
	
	Returns
	-------
	The generator
	
	Examples
	--------
	>>> for tf0,tp0,tp1,tf1 in yearly_window( 2006 , 2100 , 5 , 50 , 5 , 1951 , 2100 ):
	>>> 	pass
	>>> ## Output:
	>>> ## Iterate over 5-50-5 window
	>>> ## * L-bound / Fit-left / Predict-left / Predict-right / Fit-right / R-Bound
	>>> ## *    1951 /     2001 /         2006 /          2055 /      2060 /    2100
	>>> ## *    1951 /     2041 /         2056 /          2100 /      2100 /    2100
	"""
	
	logger.info( f"Iterate over {wleft}-{wpred}-{wright} window" )
	logger.info( " * L-bound / Fit-left / Predict-left / Predict-right / Fit-right / R-Bound" )
	
	tbeg = int(tbeg_)
	tend = int(tend_)
	tleft  = int(tleft_)
	tright = int(tright_)
	
	tp0  = int(tbeg)
	tp1  = tp0 + wpred - 1
	tf0  = tp0 - wleft
	tf1  = tp1 + wright
	
	while not tp0 > tend:
		
		## Work on a copy, original used for iteration
		rtf0,rtp0,rtp1,rtf1 = tf0,tp0,tp1,tf1
		
		## Correction when the left window is lower than tleft
		if rtf0 < tleft:
			rtf1 = rtf1 + tleft - rtf0
			rtf0 = tleft
		
		## Correction when the right window is upper than tend
		if rtf1 > tright:
			rtf1 = tright
			rtf0 = rtf0 - (tf1 - tright)
		if rtp1 > tright:
			rtp1 = tright
		
		## The return
		logger.info( f" *    {tleft} /     {rtf0} /         {rtp0} /          {rtp1} /      {rtf1} /    {tright}" )
		yield [str(x) for x in [rtf0,rtp0,rtp1,rtf1]]
		
		## And iteration
		tp0 = tp1 + 1
		tp1 = tp0 + wpred - 1
		tf0 = tp0 - wleft
		tf1 = tp1 + wright
##}}}

def sbck_ns_ufunc( Y0 , X0 , X1f , X1p , cls , **kwargs ):##{{{
	"""
	XSBCK.sbck_ns_ufunc
	===================
	Non-stationary bias correction ufunc passed to xr.apply_ufunc.
	
	Arguments
	---------
	Y0:
		Ref in calibration period
	X0:
		Biased data in calibration period
	X1f:
		Biased data in projection period, for the fit
	X1p:
		Biased data in projection period, for the predict
	cls:
		class of bias correction method (class of SBCK or SBCK.ppp)
	kwargs:
		All keywords arguments are passed to cls
	
	Returns
	-------
	Z1p:
		Corrected data in projection period
	"""
	
	try:
		sbck_cls = cls(**kwargs)
		sbck_cls.fit( Y0 = Y0 , X0 = X0 , X1 = X1f )
		Z1p = sbck_cls.predict( X1 = X1p )
		if X1p.ndim == 1:
			Z1p = Z1p[:,0]
	except Exception as e:
		logger.error( f"Error in sbck_ns_ufunc: {e}" )
		Z1p = X1p.copy() + np.nan
	
	return Z1p
##}}}

def sbck_s_ufunc( Y0 , X0 , cls , **kwargs ):##{{{
	"""
	XSBCK.sbck_s_ufunc
	==================
	Stationary bias correction ufunc passed to xr.apply_ufunc.
	
	Arguments
	---------
	Y0:
		Ref in calibration period
	X0:
		Biased data in calibration period
	cls:
		class of bias correction method (class of SBCK or SBCK.ppp)
	kwargs:
		All keywords arguments are passed to cls
	
	Returns
	-------
	Z0:
		Corrected data in calibration period
	"""
	
	try:
		sbck_cls = cls(**kwargs)
		sbck_cls.fit( Y0 = Y0 , X0 = X0 )
		Z0 = sbck_cls.predict( X0 = X0 )
		
		if X0.ndim == 1:
			Z0 = Z0[:,0]
	except Exception as e:
		logger.error( f"Error in sbck_s_ufunc: {e}" )
		Z0 = X0.copy() + np.nan
	
	return Z0
##}}}


## spatial_chunked_correction ##{{{
@log_start_end(logger)
def spatial_chunked_correction( zX , zY , zZ , zc ):
	"""
	XSBCK.spatial_chunked_correction
	================================
	Main function for the correction of each spatial chunked
	
	Arguments
	---------
	zX:
		The XZarr of the model
	zY:
		The XZarr of the reference
	zZ:
		The XZarr of the output
	zc:
		The chunk identifier
	
	Returns
	-------
	None
	"""
	
	bc_n_kwargs = xsbckParams.bc_n_kwargs
	bc_s_kwargs = xsbckParams.bc_s_kwargs
	
	## Parameters
	months = [m+1 for m in range(12)]
	
	## Extract calibration period
	calib = xsbckParams.calibration
	Y0 = zY.sel_along_time( slice(*calib) , zc = zc ).rename( time = "timeY0" )
	X0 = zX.sel_along_time( slice(*calib) , zc = zc ).rename( time = "timeX0" )
	
	## Init time
	wleft,wpred,wright = xsbckParams.window
	tleft  = str(zX.time[0].values)[:4]
	tright = str(zX.time[-1].values)[:4]
	tbeg = xsbckParams.start_year
	tend = xsbckParams.end_year
	if tbeg is None:
		tbeg = tleft
		xsbckParams.start_year = tbeg
	if tend is None:
		tend = tright
		xsbckParams.end_year = tend
	
	
	## Loop over time for projection period
	for tf0,tp0,tp1,tf1 in yearly_window( tbeg , tend , wleft , wpred , wright , tleft , tright ):
		
		## Build data in projection period
		X1f = zX.sel_along_time( slice(tf0,tf1) , zc = zc ).rename( time = "timeX1f" )
		X1p = X1f.sel( timeX1f = slice(tp0,tp1) ).rename( timeX1f = "timeX1p" )
		
		## dim names
		input_core_dims  = [("timeY0" ,"cvar"),("timeX0","cvar"),("timeX1f","cvar"),("timeX1p","cvar")]
		output_core_dims = [("timeX1p","cvar")]
		bc_ufunc_kwargs  = { "cls" : bcp.PPPIgnoreWarnings , **bc_n_kwargs }
		
		## Correction
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			Z1 = xr.concat(
			        [ xr.apply_ufunc( sbck_ns_ufunc , Y0.groupby("timeY0.month")[m] ,
			                                          X0.groupby("timeX0.month")[m] ,
			                                          X1f.groupby("timeX1f.month")[m] ,
			                                          X1p.groupby("timeX1p.month")[m] ,
			                          input_core_dims  = input_core_dims ,
			                          kwargs           = bc_ufunc_kwargs ,
			                          output_core_dims = output_core_dims ,
			                          output_dtypes    = X1p.dtype ,
			                          vectorize        = True ,
			                          dask             = "parallelized" ,
			                          keep_attrs       = True ).rename( timeX1p = "time" ) for m in months] , dim = "time"
			        ).compute().sortby("time").transpose(*zZ.dims)
		
		zZ.set_along_time( Z1 , zc = zc )
		
		## Clean
		del X1f
		del X1p
		del Z1
		gc.collect()
	
	## And the calibration period
	logger.info( f"Correction in calibration period" )
	
	input_core_dims  = [("timeY0","cvar"),("timeX0","cvar")]
	output_core_dims = [("timeX0","cvar")]
	bc_ufunc_kwargs  = { "cls" : bcp.PPPIgnoreWarnings , **bc_s_kwargs }
	
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		Z0 = xr.concat(
		        [ xr.apply_ufunc( sbck_s_ufunc , Y0.groupby("timeY0.month")[m] ,
		                                         X0.groupby("timeX0.month")[m] ,
		                          input_core_dims  = input_core_dims ,
		                          kwargs           = bc_ufunc_kwargs ,
		                          output_core_dims = output_core_dims ,
		                          output_dtypes    = X0.dtype ,
		                          vectorize        = True ,
		                          dask             = "parallelized" ,
		                          keep_attrs       = True ).rename( timeX0 = "time" ) for m in months] , dim = "time"
		        ).compute().sortby("time").transpose(*zZ.dims)
	
	zZ.set_along_time( Z0 , zc = zc )
	del X0
	del Y0
	del Z0
	gc.collect()
##}}}

## global_correction ##{{{
@log_start_end(logger)
def global_correction( zX , zY , zZ ):
	"""
	XSBCK.global_correction
	================================
	Main function for the correction, just a loop on spatial chunk
	
	Arguments
	---------
	zX:
		The XZarr of the model
	zY:
		The XZarr of the reference
	zZ:
		The XZarr of the output
	
	"""
	
	for zc in zX.iter_zchunks():
		logger.info( f"zchunks ({zc[0]+1},{zc[1]+1}) / ({zX.data.cdata_shape[1]},{zX.data.cdata_shape[2]})" )
		spatial_chunked_correction( zX , zY , zZ , zc )
	
##}}}

