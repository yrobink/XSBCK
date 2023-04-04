
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

##############
## Packages ##
##############

import sys
import os
import logging
import datetime as dt
import tempfile

import netCDF4
import SBCK as bc
import numpy as np
import xarray as xr
import dask
import distributed
import zarr

#############
## Imports ##
#############


from .__XSBCKParams import xsbckParams
from .__exceptions  import AbortForHelpException

from .__logs import LINE
from .__logs import log_start_end

from .__release    import version
from .__curses_doc import print_doc

from .__io import load_data
from .__io import save_data

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

## run_xsbck ##{{{

@log_start_end(logger)
def run_xsbck():
	"""
	XSBCK.run_xsbck
	===============
	
	Main execution, after the control of user input.
	
	"""
	
	xsbckParams.init_dask()
	try:
		## Load data
		zX,zY,zZ = load_data()
		
		## Build the BC strategy
		xsbckParams.init_BC_strategy()
		
		## Logs
		logger.info("PPP found:")
		for ppp in xsbckParams.pipe:
			logger.info( f" * {ppp}" )
		
		## Correction
		global_correction( zX , zY , zZ )
		
		## Save
		save_data( zX , zZ )
		
	except Exception as e:
		raise e
	finally:
		xsbckParams.stop_dask()
	
##}}}

def start_xsbck(*argv):##{{{
	"""
	XSBCK.start_xsbck
	=================
	
	Starting point of 'xsbck'.
	
	"""
	## Time counter
	walltime0 = dt.datetime.utcnow()
	
	## Read input
	xsbckParams.init_from_user_input(*argv)
	
	## Init logs
	xsbckParams.init_logging()
	
	## Logging
	logger.info(LINE)
	logger.info( "Start: {}".format( str(walltime0)[:19] + " (UTC)") )
	logger.info(LINE)
	
	## Package version
	pkgs = [("SBCK"       , bc ),
	        ("numpy"      , np ),
	        ("xarray"     , xr ),
	        ("dask"       , dask ),
	        ("distributed", distributed ),
	        ("zarr"       , zarr ),
	        ("netCDF4"    , netCDF4 )
	       ]
	
	logger.info( "Packages version:" )
	logger.info( " * {:{fill}{align}{n}}".format( "XSBCK" , fill = " " , align = "<" , n = 12 ) + f"version {version}" )
	for name_pkg,pkg in pkgs:
		logger.info( " * {:{fill}{align}{n}}".format( name_pkg , fill = " " , align = "<" , n = 12 ) +  f"version {pkg.__version__}" )
	logger.info(LINE)
	
	## Serious functions start here
	try:
		
		## Check inputs
		xsbckParams.check()
		
		## Init temporary
		xsbckParams.init_tmp()
		
		## List of all input
		logger.info("Input parameters:")
		for key in xsbckParams.keys():
			logger.info( " * {:{fill}{align}{n}}".format( key , fill = " ",align = "<" , n = 10 ) + ": {}".format(xsbckParams[key]) )
		logger.info(LINE)
		
		## User asks help
		if xsbckParams.help:
			print_doc()
		
		## In case of abort, raise Exception
		if xsbckParams.abort:
			raise xsbckParams.error
		
		## Go!
		run_xsbck()
		
	except AbortForHelpException:
		pass
	except Exception as e:
		logger.error( f"Error: {e}" )
	
	## End
	walltime1 = dt.datetime.utcnow()
	logger.info(LINE)
	logger.info( "End: {}".format(str(walltime1)[:19] + " (UTC)") )
	logger.info( "Wall time: {}".format(walltime1 - walltime0) )
	logger.info(LINE)
##}}}

