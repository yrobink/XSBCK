
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

##############
## Packages ##
##############

import sys
import os
import logging
import datetime as dt

import SBCK as bc
import xclim
import numpy as np
import xarray as xr
import dask


#############
## Imports ##
#############

from .__logs import LINE
from .__logs import init_logging

from .__release    import version
from .__curses_doc import print_doc

from .__input import read_inputs
from .__input import check_inputs

from .__exceptions import AbortException

from .__tmp import build_tmp_dir
from .__tmp import delete_tmp_dir

from .__io import load_data
from .__io import save_data
from .__correction import build_BC_method
from .__correction import global_correction


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


#########
## Dev ##
#########

###############
## Functions ##
###############

def run_xsbck( **kwargs ):##{{{
	"""
	XSBCK.run_xsbck
	===============
	
	Main execution, after the control of user input.
	
	"""
	#TODO add a parameter for year where correction start
	
	logger.info( "XSBCK::start" )
	
	## Init the distributed client
	wclient = dask.distributed.Client( n_workers = kwargs["n_workers"] , threads_per_worker = kwargs["threads_per_worker"] , memory_limit = kwargs["memory"] )
	
	## Load data
	dX,dY,coords = load_data(**kwargs)
	
	## Build the BC method (non-stationary and stationary)
	bc_n_kwargs,bc_s_kwargs = build_BC_method(**kwargs)
	
	## Correction
	global_correction( dX , dY , coords , bc_n_kwargs , bc_s_kwargs , **kwargs )
	
	## Save
	save_data( coords , **kwargs )
	
	logger.info( "XSBCK::end" )
	logger.info(LINE)
##}}}

def start_xsbck():##{{{
	"""
	XSBCK.start_xsbck
	=================
	
	Starting point of 'xsbck'.
	
	"""
	## Time counter
	walltime0 = dt.datetime.utcnow()
	
	## Read input
	kwargs = read_inputs()
	
	## Init logs
	init_logging(kwargs["log"])
	logger.info(LINE)
	logger.info( "Start: {}".format(str(walltime0)[:19] + " (UTC)") )
	logger.info(LINE)
	
	## Package version
	logger.info( "Packages version:" )
	logger.info( " * {:{fill}{align}{n}}".format( "XSBCK" , fill = " " , align = "<" , n = 8 ) + f"version {version}" )
	for name_pkg,pkg in zip(["SBCK","xclim","numpy","xarray","dask"],[bc,xclim,np,xr,dask]):
		logger.info( " * {:{fill}{align}{n}}".format( name_pkg , fill = " " , align = "<" , n = 8 ) +  f"version {pkg.__version__}" )
	logger.info(LINE)
	
	## Serious functions start here
	try:
		## Build temporary
		kwargs["tmp"] = build_tmp_dir(**kwargs)
		
		## List of all input
		logger.info("Input parameters:")
		keys = [key for key in kwargs]
		keys.sort()
		for key in keys:
			logger.info( " * {:{fill}{align}{n}}".format( key , fill = " ",align = "<" , n = 10 ) + ": {}".format(kwargs[key]) )
		logger.info(LINE)
		
		## Check inputs
		kwargs,abort = check_inputs(**kwargs)
		
		## User asks help
		if kwargs["help"]:
			print_doc()
			abort = True
		
		## In case of abort, raise Exception
		if abort:
			raise AbortException
		
		## Go!
		run_xsbck( **kwargs )
		
	except AbortException:
		pass
	except Exception as e:
		logger.error( f"Error: {e}" )
	finally:
		delete_tmp_dir(**kwargs)
	
	## End
	walltime1 = dt.datetime.utcnow()
	logger.info( "End: {}".format(str(walltime1)[:19] + " (UTC)") )
	logger.info( "Wall time: {}".format(walltime1 - walltime0) )
	logger.info(LINE)
##}}}

