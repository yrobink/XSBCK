
## Copyright(c) 2022 Yoann Robin
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

import numpy  as np
import xarray as xr

from xclim import sdba
import xclim.sdba.processing as sdbp

import SBCK as bc
import SBCK.ppp as bcp

import inspect

from .__io import Coordinates
from .__logs import log_start_end

##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


#############
## Classes ##
#############

class SAR2D2(bc.AR2D2):##{{{
	
	def __init__( self , col_cond = [0] , lag_search = 1 , lag_keep = 1 , bc_method = bc.QM , shuffle = "quantile" , reverse = False , **bckwargs ):##{{{
		bc.AR2D2.__init__( self , col_cond , lag_search , lag_keep , bc_method , shuffle , reverse , **bckwargs )
	##}}}
	
	def fit( self , Y0 , X0 ):##{{{
		bc.AR2D2.fit( self , Y0 , X0 )
	##}}}
	
	def predict( self , X0 ):##{{{
		return bc.AR2D2.predict( self , X0 = X0 )
	##}}}
	
##}}}


###############
## Functions ##
###############

## TODO : parameter of the BC method, i.e. col_cond for R2D2

## build_pipe ##{{{
@log_start_end(logger)
def build_pipe( coords : Coordinates , kwargs : dict ):
	
	lppps = kwargs.get("ppp")
	
	if lppps is None:
		return [],[]
	
	## Init
	pipe        = []
	pipe_kwargs = []
	
	## Identify columns
	dcols = { cvar : [coords.cvarsZ.index(cvar)] for cvar in coords.cvarsZ }
	
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
			if "cols" in pkwargs and cvar in coords.cvarsZ:
				pkwargs["cols"] = dcols[cvar]
			
			## And others parameters
			for p in p_param:
				key,val = p.split("=")
				if key in insp.annotations:
					pkwargs[key] = insp.annotations[key](val)
				else:
					pkwargs[key] = val
					
					## Special case, val is a list (as sum) of cvar
					if len(set(val.split("+")) & set(coords.cvarsZ)) > 0:
						pkwargs[key] = []
						for v in val.split("+"):
							if not v in dcols:
								raise Exception(f"Unknow '{v}' as parameter for the ppp {p_name}")
							pkwargs[key] = pkwargs[key] + dcols[v]
			
			## Append
			pipe.append(cls)
			pipe_kwargs.append(pkwargs)
	
	return pipe,pipe_kwargs
##}}}

## build_BC_method ##{{{
@log_start_end(logger)
def build_BC_method( coords : Coordinates , kwargs : dict ):
	
	bc_method        = bcp.PrePostProcessing
	
	## The method
	if "IdBC" in kwargs["method"]:
		bc_method_n_kwargs = { "bc_method" : bc.IdBC , "bc_method_kwargs" : {} }
		bc_method_s_kwargs = { "bc_method" : bc.IdBC , "bc_method_kwargs" : {} }
	if "CDFt" in kwargs["method"]:
		bc_method_n_kwargs = { "bc_method" : bc.CDFt , "bc_method_kwargs" : {} }
		bc_method_s_kwargs = { "bc_method" : bc.QM   , "bc_method_kwargs" : {} }
	if "R2D2" in kwargs["method"]:
		col_cond   = [0]
		lag_keep   = int(kwargs["method"].split("-")[-1][:-1]) + 1
		lag_search = 2 * lag_keep
		bc_method_n_kwargs = { "bc_method" : bc.AR2D2 , "bc_method_kwargs" : { "col_cond" : col_cond , "lag_search" : lag_search , "lag_keep" : lag_keep , "reverse" : True , "bc_method" : bc.CDFt } }
		bc_method_s_kwargs = { "bc_method" :   SAR2D2 , "bc_method_kwargs" : { "col_cond" : col_cond , "lag_search" : lag_search , "lag_keep" : lag_keep , "reverse" : True , "bc_method" : bc.QM   } }
	
	## The pipe
	pipe,pipe_kwargs = build_pipe( coords , kwargs )
	
	## Global arguments
	bc_n_kwargs = { "bc_method" : bc_method , "bc_method_kwargs" : bc_method_n_kwargs , "pipe" : pipe , "pipe_kwargs" : pipe_kwargs }
	bc_s_kwargs = { "bc_method" : bc_method , "bc_method_kwargs" : bc_method_s_kwargs , "pipe" : pipe , "pipe_kwargs" : pipe_kwargs }
	
	return bc_n_kwargs,bc_s_kwargs
##}}}


def yearly_window( tbeg_ , tend_ , wleft , wpred , wright ):##{{{
	
	logger.info( f"Iterate over {wleft}-{wpred}-{wright} window" )
	logger.info( " * Fit-left / Predict-left / Predict-right / Fit-right" )
	
	tbeg = int(tbeg_)
	tend = int(tend_)
	
	tp0  = int(tbeg)
	tp1  = tp0 + wpred - 1
	tf0  = tp0 - wleft
	tf1  = tp1 + wright
	
	while not tp0 > tend:
		
		## Work on a copy, original used for iteration
		rtf0,rtp0,rtp1,rtf1 = tf0,tp0,tp1,tf1
		
		## Correction when the left window is lower than tbeg
		if rtf0 < tbeg:
			rtf1 = rtf1 + tbeg - rtf0
			rtf0 = tbeg
		
		## Correction when the right window is upper than tend
		if rtf1 > tend:
			rtf1 = tend
			rtf0 = rtf0 - (tf1 - tend)
		if rtp1 > tend:
			rtp1 = tend
		
		## The return
		logger.info( f" *     {rtf0} /         {rtp0} /          {rtp1} /      {rtf1}" )
		yield [str(x) for x in [rtf0,rtp0,rtp1,rtf1]]
		
		## And iteration
		tp0 = tp1 + 1
		tp1 = tp0 + wpred - 1
		tf0 = tp0 - wleft
		tf1 = tp1 + wright
##}}}

def sbck_ns_ufunc( Y0 , X0 , X1f , X1p , cls , **kwargs ):##{{{
	
	sbck_cls = cls(**kwargs)
	sbck_cls.fit( Y0 = Y0 , X0 = X0 , X1 = X1f )
	Z1p = sbck_cls.predict( X1 = X1p )
	
	if X1p.ndim == 1:
		return Z1p[:,0]
	
	return Z1p
##}}}

def sbck_s_ufunc( Y0 , X0 , cls , **kwargs ):##{{{
	
	sbck_cls = cls(**kwargs)
	sbck_cls.fit( Y0 = Y0 , X0 = X0 )
	Z0 = sbck_cls.predict( X0 = X0 )
	
	if X0.ndim == 1:
		return Z0[:,0]
	
	return Z0
##}}}


## global_correction_zarr ##{{{
@log_start_end(logger)
def global_correction_zarr( dX , dY , coords , bc_n_kwargs , bc_s_kwargs , kwargs ):
	
	## Parameters
	months = [m+1 for m in range(12)]
	
	## Extract calibration period
	calib = kwargs["calibration"]
	Y0 = dY.sel_along_time(slice(*calib)).rename( time = "timeY0" )
	X0 = dX.sel_along_time(slice(*calib)).rename( time = "timeX0" )
	
	## And init output file
	dZ = dX.copy( os.path.join( kwargs["tmp"] , "Z.zarr" ) )
	
	## Init time
	wleft,wpred,wright = kwargs["window"]
	tbeg = str(coords.time[0].values)[:4]
	tend = str(coords.time[-1].values)[:4]
	
	## Loop over time for projection period
	for tf0,tp0,tp1,tf1 in yearly_window( tbeg , tend , wleft , wpred , wright ):
		
		## Build data in projection period
		X1f = dX.sel_along_time(slice(tf0,tf1)).rename( time = "timeX1f" )
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
		        ).compute().sortby("time").transpose(*dZ.dims)
		
		dZ.set_along_time(Z1)
		
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
	        ).compute().sortby("time").transpose(*dZ.dims)
	
	dZ.set_along_time(Z0)
	del X0
	del Y0
	del Z0
	gc.collect()
	
	return dZ
##}}}

## global_correction_nc ##{{{
@log_start_end(logger)
def global_correction_nc( dX , dY , coords , bc_n_kwargs , bc_s_kwargs , kwargs ):
	
	## Parameters
	months = [m+1 for m in range(12)]
	
	## Extract calibration period
	calib = kwargs["calibration"]
	dY0 = dY.sel( time = slice(*calib) )
	dX0 = dX.sel( time = slice(*calib) )
	
	## Prepare data in calibration period
	X0 = sdbp.stack_variables(dX0).rename( time = "timeX0" )
	Y0 = sdbp.stack_variables(dY0).rename( time = "timeY0" )
	if coords.ncvar > 1:
		X0 = X0.sel( multivar = coords.cvarsZ )
		Y0 = Y0.sel( multivar = coords.cvarsZ )
	
	## Init time
	wleft,wpred,wright = kwargs["window"]
	tbeg = str(coords.time[0].values)[:4]
	tend = str(coords.time[-1].values)[:4]
	
	## Loop over time
	for tf0,tp0,tp1,tf1 in yearly_window( tbeg , tend , wleft , wpred , wright ):
		
		## Build data in projection period
		dX1 = dX.sel( time = slice(tf0,tf1) )
		X1f  = sdbp.stack_variables(dX1).rename( time = "timeX1f" )
		if coords.ncvar > 1:
			X1f = X1f.sel( multivar = coords.cvarsZ )
		X1p = X1f.sel( timeX1f = slice(tp0,tp1) ).rename( timeX1f = "timeX1p" )
		
		## dim names
		input_core_dims  = [("timeY0","multivar"),("timeX0","multivar"),("timeX1f","multivar"),("timeX1p","multivar")]
		output_core_dims = [("timeX1p","multivar")]
		bc_ufunc_kwargs  = { "cls" : bcp.PrePostProcessing , **bc_n_kwargs }
		
		## Correction
		Z1 = xr.concat(
		        [ xr.apply_ufunc( sbck_ns_ufunc , Y0.groupby("timeY0.month")[m] ,
		                                          X0.groupby("timeX0.month")[m] ,
		                                          X1f.groupby("timeX1f.month")[m] ,
		                                          X1p.groupby("timeX1p.month")[m] ,
		                          input_core_dims = input_core_dims ,
		                          kwargs = bc_ufunc_kwargs ,
		                          output_core_dims = output_core_dims ,
		                          output_dtypes = X1p.dtype ,
		                          vectorize = True ,
		                          dask = "parallelized" ,
		                          keep_attrs = True ).rename( timeX1p = "time" ) for m in months] , dim = "time"
		        ).compute().sortby("time")
		
		## Split variables and save in a temporary folder
		dZ1 = sdbp.unstack_variables(Z1)
		for cvar in coords.cvarsZ:
			dZ1[[cvar]].to_netcdf( os.path.join( kwargs["tmp"] , f"{cvar}_Z1_{tp0}-{tp1}.nc" ) )
	
	## And the calibration period
	logger.info( f"Correction in calibration period" )
	
	input_core_dims  = [("timeY0","multivar"),("timeX0","multivar")]
	output_core_dims = [("timeX0","multivar")]
	bc_ufunc_kwargs  = { "cls" : bcp.PrePostProcessing , **bc_s_kwargs }
	
	Z0 = xr.concat(
	        [ xr.apply_ufunc( sbck_s_ufunc , Y0.groupby("timeY0.month")[m] ,
	                                         X0.groupby("timeX0.month")[m] ,
	                          input_core_dims = input_core_dims ,
	                          kwargs = bc_ufunc_kwargs ,
	                          output_core_dims = output_core_dims ,
	                          output_dtypes = X1p.dtype ,
	                          vectorize = True ,
	                          dask = "parallelized" ,
	                          keep_attrs = True ).rename( timeX0 = "time" ) for m in months] , dim = "time"
	        ).compute().sortby("time")
	
	## Split variables and save in a temporary folder
	dZ0 = sdbp.unstack_variables(Z0)
	for cvar in coords.cvarsZ:
		dZ0[[cvar]].to_netcdf( os.path.join( kwargs["tmp"] , f"{cvar}_Z0_{calib[0]}-{calib[1]}.nc" ) )
##}}}

## global_correction ##{{{
@log_start_end(logger)
def global_correction( dX , dY , coords , bc_n_kwargs , bc_s_kwargs , kwargs ):
	return global_correction_zarr( dX , dY , coords , bc_n_kwargs , bc_s_kwargs , kwargs )
##}}}


