
## Copyright(c) 2022, 2023 Yoann Robin
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
import logging
import datetime as dt
import importlib

import numpy  as np
import xarray as xr

import SBCK as bc
import SBCK.ppp as bcp

import inspect

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

## build_pipe ##{{{
@log_start_end(logger)
def build_pipe( cvars , kwargs : dict ):
	"""
	XSBCK.build_pipe
	================
	Function used to build PrePostProcessing class from user input
	
	Arguments
	---------
	cvars:
		List of climate variables to be corrected
	kwargs:
		dict of all parameters of XSBCK
	
	Returns
	-------
	pipe:
		List of class based on SBCK.ppp.PrePostProcessing
	pipe_kwargs:
		List of kwargs passed to elements of pipe
	"""
	
	lppps = kwargs.get("ppp")
	
	if lppps is None:
		return [],[]
	
	## Init
	pipe        = []
	pipe_kwargs = []
	
	## Identify columns
	dcols = { cvar : [cvars.index(cvar)] for cvar in cvars }
	
	## Explore SBCK.ppp
	ppp_avail = [clsname for clsname in dir(bcp) if clsname.startswith("PPP") ]
	
	## Loop on ppp
	for i in range(len(lppps)):
		
		## Find cvar and list of ppp
		cvar,_,ppps_str = lppps[i].partition(",")
		
		## Split with ',', and remerge (e.g. 'A[B=2,C=3],K' => ['A[B=2,C=3]','K'] and not ['A[B=2','C=3]','K']
		split = ppps_str.split(",")
		lppp  = []
		while len(split) > 0:
			
			if "[" in split[0]:
				if "]" in split[0]:
					lppp.append(split[0])
					del split[0]
				else:
					for j in range(len(split)):
						if "]" in split[j]:
							break
					lppp.append( ",".join(split[:(j+1)]) )
					split = split[(j+1):]
			else:
				lppp.append(split[0])
				del split[0]
		
		## Loop on ppp
		for ppp in lppp:
			
			## Extract name / parameters
			if "[" in ppp:
				p_name,p_param = ppp.split("[")
				p_param = p_param.split("]")[0].split(",")
			else:
				p_name  = ppp
				p_param = []
			
			## Find the true name
			if p_name in ppp_avail:
				pass
			elif f"PPP{p_name}" in ppp_avail:
				p_name = f"PPP{p_name}"
			elif f"{p_name}Link" in ppp_avail:
				p_name = f"{p_name}Link"
			elif f"PPP{p_name}Link" in ppp_avail:
				p_name = f"PPP{p_name}Link"
			else:
				raise Exception(f"Unknow ppp {p_name}")
			
			## Define the class, and read the signature
			cls     = getattr(bcp,p_name)
			insp    = inspect.getfullargspec(cls)
			pkwargs = insp.kwonlydefaults
			
			## Special case, the cols parameter
			if "cols" in pkwargs and cvar in cvars:
				pkwargs["cols"] = dcols[cvar]
			
			## And others parameters
			for p in p_param:
				key,val = p.split("=")
				if key in insp.annotations:
					pkwargs[key] = insp.annotations[key](val)
				else:
					pkwargs[key] = val
					
					## Special case, val is a list (as sum) of cvar
					if len(set(val.split("+")) & set(cvars)) > 0:
						pkwargs[key] = []
						for v in val.split("+"):
							if not v in dcols:
								raise Exception(f"Unknow '{v}' as parameter for the ppp {p_name}")
							pkwargs[key] = pkwargs[key] + dcols[v]
			
			## Append
			pipe.append(cls)
			pipe_kwargs.append(pkwargs)
	
	logger.info( "PPP found:" )
	for p in pipe:
		logger.info( f" * {str(p)}" )
	
	return pipe,pipe_kwargs
##}}}

def checkf(X):##{{{
	return np.any(np.isfinite(X))
##}}}

## build_BC_method ##{{{
@log_start_end(logger)
def build_BC_method( cvars , kwargs : dict ):
	"""
	XSBCK.build_BC_method
	=====================
	Function used to build the bias correction class class from user input
	
	Arguments
	---------
	cvars:
		List of climate variables to be corrected
	kwargs:
		dict of all parameters of XSBCK
	
	Returns
	-------
	bc_n_kwargs:
		dict describing the non-stationary BC method used
	bc_s_kwargs:
		dict describing the stationary BC method used
	"""
	
	bc_method = bcp.PrePostProcessing
	
	## Find method kwargs
	dkwd = {}
	if kwargs.get("method_kwargs") is not None:
		dkwd = { k : v for (k,v) in [ kv.split("=") for kv in kwargs["method_kwargs"].split(",")] }
	
	## The method
	if "IdBC" in kwargs["method"]:
		bc_method_n_kwargs = { "bc_method" : bc.IdBC , "bc_method_kwargs" : {} }
		bc_method_s_kwargs = { "bc_method" : bc.IdBC , "bc_method_kwargs" : {} }
	if "CDFt" in kwargs["method"]:
		bc_method_n_kwargs = { "bc_method" : bc.CDFt , "bc_method_kwargs" : {} }
		bc_method_s_kwargs = { "bc_method" : bc.QM   , "bc_method_kwargs" : {} }
	if "dOTC" in kwargs["method"]:
		bc_method_n_kwargs = { "bc_method" : bc.dOTC , "bc_method_kwargs" : {} }
		bc_method_s_kwargs = { "bc_method" : bc.OTC  , "bc_method_kwargs" : {} }
	if "R2D2" in kwargs["method"]:
		col_cond   = [0]
		if "col_cond" in dkwd:
			col_cond = [cvars.index(cvar) for cvar in dkwd["col_cond"].split("+")]
		lag_keep   = int(kwargs["method"].split("-")[-1][:-1]) + 1
		lag_search = 2 * lag_keep
		bcmkwargs  = { "col_cond" : [0] , "lag_search" : lag_search , "lag_keep" : lag_keep , "reverse" : True }
		bc_method_n_kwargs = { "bc_method" : bc.AR2D2 , "bc_method_kwargs" : { **bcmkwargs , "bc_method" : bc.CDFt } }
		bc_method_s_kwargs = { "bc_method" : bc.AR2D2 , "bc_method_kwargs" : { **bcmkwargs , "bc_method" : bc.QM   } }
	
	## The pipe
	pipe,pipe_kwargs = build_pipe( cvars , kwargs )
	
	## Global arguments
	bc_n_kwargs = { "bc_method" : bc_method , "bc_method_kwargs" : bc_method_n_kwargs , "pipe" : pipe , "pipe_kwargs" : pipe_kwargs , "checkf" : checkf }
	bc_s_kwargs = { "bc_method" : bc_method , "bc_method_kwargs" : bc_method_s_kwargs , "pipe" : pipe , "pipe_kwargs" : pipe_kwargs , "checkf" : checkf }
	
	return bc_n_kwargs,bc_s_kwargs
##}}}


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
def spatial_chunked_correction( zX , zY , zZ , bc_n_kwargs , bc_s_kwargs , kwargs , zc ):
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
	bc_n_kwargs:
		dict describing the non-stationary BC method used
	bc_s_kwargs:
		dict describing the stationary BC method used
	kwargs:
		dict of all parameters of XSBCK
	zc:
		The chunk identifier
	
	Returns
	-------
	None
	"""
	
	## Parameters
	months = [m+1 for m in range(12)]
	
	## Extract calibration period
	calib = kwargs["calibration"]
	Y0 = zY.sel_along_time( slice(*calib) , zc = zc ).rename( time = "timeY0" )
	X0 = zX.sel_along_time( slice(*calib) , zc = zc ).rename( time = "timeX0" )
	
	## Init time
	wleft,wpred,wright = kwargs["window"]
	tleft  = str(zX.time[0].values)[:4]
	tright = str(zX.time[-1].values)[:4]
	tbeg = kwargs.get("start_year")
	tend = kwargs.get("end_year")
	if tbeg is None:
		tbeg = tleft
		kwargs["start_year"] = tbeg
	if tend is None:
		tend = tright
		kwargs["end_year"] = tend
	
	## Loop over time for projection period
	for tf0,tp0,tp1,tf1 in yearly_window( tbeg , tend , wleft , wpred , wright , tleft , tright ):
		
		## Build data in projection period
		X1f = zX.sel_along_time( slice(tf0,tf1) , zc = zc ).rename( time = "timeX1f" )
		X1p = X1f.sel( timeX1f = slice(tp0,tp1) ).rename( timeX1f = "timeX1p" )
		
		
		## dim names
		input_core_dims  = [("timeY0" ,"cvar"),("timeX0","cvar"),("timeX1f","cvar"),("timeX1p","cvar")]
		output_core_dims = [("timeX1p","cvar")]
		bc_ufunc_kwargs  = { "cls" : bcp.PrePostProcessing , **bc_n_kwargs }
		
		## Correction
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
	bc_ufunc_kwargs  = { "cls" : bcp.PrePostProcessing , **bc_s_kwargs }
	
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
def global_correction( zX , zY , zZ , bc_n_kwargs , bc_s_kwargs , kwargs ):
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
	bc_n_kwargs:
		dict describing the non-stationary BC method used
	bc_s_kwargs:
		dict describing the stationary BC method used
	kwargs:
		dict of all parameters of XSBCK
	
	"""
	
	for zc in zX.iter_zchunks():
		logger.info( f"zchunks ({zc[0]+1},{zc[1]+1}) / ({zX.data.cdata_shape[1]},{zX.data.cdata_shape[2]})" )
		spatial_chunked_correction( zX , zY , zZ , bc_n_kwargs , bc_s_kwargs , kwargs , zc )
	
##}}}

