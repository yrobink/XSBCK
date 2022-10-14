
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

import logging

from .__exceptions import UserDefinedLoggingLevelError

###############
## Functions ##
###############

LINE = "=" * 80

def init_logging(logs_):
	"""
	XSBCK.init_logging
	==================
	
	Function used to init the logging system
	
	Parameters
	----------
	logs_: A list, can contain nothing, or just the loglevel
	"""
	
	## If at least on args, this is the loglevel / filename (if file)
	logfile = None
	if len(logs_) > 0:
		loglevel = logs_[0]
		if len(logs_) > 1:
			logfile = logs_[1]
	else:
		loglevel = logging.DEBUG
	
	## loglevel can be an integet
	try:
		numlevel = int(loglevel)
	except:
		numlevel = getattr( logging , loglevel.upper() , None )
	
	## If it is not an integer, raise an error
	if not isinstance( numlevel , int ): 
		raise UserDefinedLoggingLevelError( f"Invalid log level: {loglevel}; nothing, an integer, 'debug', 'info', 'warning', 'error' or 'critical' expected" )
	
	##
	log_kwargs = { "format" : '%(message)s' , "level" : numlevel }
	if logfile is not None:
		log_kwargs["filename"] = logfile
#	logging.basicConfig( format = '%(levelname)s:%(name)s:%(funcName)s: %(message)s' , level = numlevel )
	logging.basicConfig(**log_kwargs)
	logging.captureWarnings(True)
