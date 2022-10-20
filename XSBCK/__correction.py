
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
import logging

import numpy  as np
import xarray as xr

from xclim import sdba
import xclim.sdba.processing as sdbp

import SBCK as bc
import SBCK.ppp as bcp

import inspect


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

def build_pipe( coords , **kwargs ):##{{{
	
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

def build_BC_method( coords , **kwargs ):##{{{
	
	
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
	pipe,pipe_kwargs = build_pipe( coords , **kwargs )
	
	## Global arguments
	bc_n_kwargs = { "bc_method" : bc_method , "bc_method_kwargs" : bc_method_n_kwargs , "pipe" : pipe , "pipe_kwargs" : pipe_kwargs }
	bc_s_kwargs = { "bc_method" : bc_method , "bc_method_kwargs" : bc_method_s_kwargs , "pipe" : pipe , "pipe_kwargs" : pipe_kwargs }
	
	return bc_n_kwargs,bc_s_kwargs
##}}}

def yearly_window( tbeg_ , tend_ , wleft , wpred , wright ):##{{{
	
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
		yield [str(x) for x in [rtf0,rtp0,rtp1,rtf1]]
		
		## And iteration
		tp0 = tp1 + 1
		tp1 = tp0 + wpred - 1
		tf0 = tp0 - wleft
		tf1 = tp1 + wright
##}}}

def global_correction( dX , dY , coords , bc_n_kwargs , bc_s_kwargs , **kwargs ):##{{{
	
	logger.info( "global_correction:start" )
	
	## Parameters
	months = [m+1 for m in range(12)]
	
	## Extract calibration period
	calib = kwargs["calibration"]
	dY0 = dY.sel( time = slice(*calib) )
	dX0 = dX.sel( time = slice(*calib) )
	
	## Prepare data in calibration period
	X0 = sdbp.stack_variables(dX0)
	Y0 = sdbp.stack_variables(dY0)
	if coords.ncvar > 1:
		X0 = X0.sel( multivar = coords.cvarsZ )
		Y0 = Y0.sel( multivar = coords.cvarsZ )
	
	## Init time
	wleft,wpred,wright = kwargs["window"]
	tbeg = str(coords.time[0].values)[:4]
	tend = str(coords.time[-1].values)[:4]
	
	## Loop over time
	logger.info( f"Iterate over time {wleft}-{wpred}-{wright}" )
	logger.info( " * Fit-left / Predict-left / Predict-right / Fit-right" )
	for tf0,tp0,tp1,tf1 in yearly_window( tbeg , tend , wleft , wpred , wright ):
		
		logger.info( f" *     {tf0} /         {tp0} /          {tp1} /      {tf1}" )
		
		## Build data in projection period
		dX1 = dX.sel( time = slice(tf0,tf1) )
		X1  = sdbp.stack_variables(dX1)
		if coords.ncvar > 1:
			X1 = X1.sel( multivar = coords.cvarsZ )
		
		## Correction
		Z1  = xr.concat( [ sdba.adjustment.SBCK_XClimNPPP.adjust( Y0.groupby("time.month")[m] , X0.groupby("time.month")[m] , X1.groupby("time.month")[m] , multi_dim = "multivar" , **bc_n_kwargs ) for m in months ] , dim = "time" )
		Z1  = Z1.compute().sortby("time").sel( time = slice(tp0,tp1) )
		
		## Split variables and save in a temporary folder
		dZ1 = sdbp.unstack_variables(Z1)
		for cvar in coords.cvarsZ:
			dZ1[[cvar]].to_netcdf( os.path.join( kwargs["tmp"] , f"{cvar}_Z1_{tp0}-{tp1}.nc" ) )
	
	## And the calibration period
	logger.info( f"Correction in calibration period" )
	Z0  = xr.concat( [ sdba.adjustment.SBCK_XClimSPPP.adjust( Y0.groupby("time.month")[m] , X0.groupby("time.month")[m] , X0.groupby("time.month")[m] , multi_dim = "multivar" , **bc_s_kwargs ) for m in months ] , dim = "time" )
	Z0  = Z0.compute().sortby("time")
	
	## Split variables and save in a temporary folder
	dZ0 = sdbp.unstack_variables(Z0)
	for cvar in coords.cvarsZ:
		dZ0[[cvar]].to_netcdf( os.path.join( kwargs["tmp"] , f"{cvar}_Z0_{calib[0]}-{calib[1]}.nc" ) )
	
	logger.info( "global_correction:end" )
##}}}


