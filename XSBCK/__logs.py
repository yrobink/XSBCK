
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

def init_logging(argv):
	"""
	XSBCK.init_logging
	==================
	
	Function used to init the logging system
	
	Parameters
	----------
	argv: tuple of the user input, from sys.argv
	"""
	
	numlevel = logging.WARNING ## Default level: warning
	
	if "--log" in argv: ## User ask to see a specific level of log
		
		idx = argv.index("--log")+1
		if idx < len(argv) and not argv[idx][0] == "-": ## str after '--log' in argv exists, and is the parameter of '--log'
			loglevel = argv[idx]
			numlevel = getattr(logging, loglevel.upper(), None)
		else: ## Default value
			numlevel = 10
		if not isinstance(numlevel, int): ## If it is not an interger, raise an error
			raise UserDefinedLoggingLevelError( f"Invalid log level: {loglevel}; nothing, 'debug', 'info', 'warning', 'error' or 'critical' expected" )
	
#	logging.basicConfig( format = '%(levelname)s:%(name)s:%(funcName)s: %(message)s' , level = numlevel )
	logging.basicConfig( format = '%(message)s' , level = numlevel )

