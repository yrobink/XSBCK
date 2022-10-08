
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

from .__logs import LINE
from .__logs import init_logging


#############
## Imports ##
#############

from .__release    import version
from .__curses_doc import print_doc

from .__input import read_input

## Init logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


###############
## Functions ##
###############

def run_xsbck( **kwargs ):##{{{
	"""
	XSBCK.run_xsbck
	===============
	
	Main execution, after the control of user input.
	
	"""
	logger.info( "Start the run of XSBCK" )
	
	logger.info( "End of the run of XSBCK" )
	logger.info(LINE)
##}}}

def start_xsbck( argv ):##{{{
	"""
	XSBCK.start_xsbck
	=================
	
	Starting point of 'xsbck'.
	
	"""
	## Time counter
	walltime0 = dt.datetime.utcnow()
	
	## Init logs
	init_logging(argv)
	logger.info(LINE)
	logger.info( "Start: {}".format(str(walltime0)[:19] + " (UTC)") )
	logger.info(LINE)
	
	## Package version
	logger.info( "Packages version:" )
	logger.info( " * {:{fill}{align}{n}}".format( "XSBCK" , fill = " " , align = "<" , n = 8 ) + f"version {version}" )
	for name_pkg,pkg in zip(["SBCK","xclim","numpy","xarray","dask"],[bc,xclim,np,xr,dask]):
		logger.info( " * {:{fill}{align}{n}}".format( name_pkg , fill = " " , align = "<" , n = 8 ) +  f"version {pkg.__version__}" )
	logger.info(LINE)
	
	## Read input
	kwargs,abort = read_input(argv)
	
	## List of all input
	logger.info("Input parameters:")
	keys = [key for key in kwargs]
	keys.sort()
	for key in keys:
		logger.info( "   * {:{fill}{align}{n}}".format( key , fill = " ",align = "<" , n = 10 ) + ": {}".format(kwargs[key]) )
	logger.info(LINE)
	
	## User asks help
	if kwargs["help"]:
		print_doc()
		abort = True
	
	## Go!
	if not abort:
		try:
			run_xsbck( **kwargs )
		finally:
			pass
	
	## End
	walltime1 = dt.datetime.utcnow()
	logger.info( "End: {}".format(str(walltime1)[:19] + " (UTC)") )
	logger.info( "Wall time: {}".format(walltime1 - walltime0) )
	logger.info(LINE)
##}}}

