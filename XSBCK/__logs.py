
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

import functools
import logging

import datetime as dt

###############
## Functions ##
###############

LINE = "=" * 80

def log_start_end(plog):##{{{
	"""
	XSBCK.log_start_end
	===================
	
	Decorator to add to the log the start / end of a function, and a walltime
	
	Parameters
	----------
	plog:
		A logger from logging
	
	"""
	def _decorator(f):
		
		@functools.wraps(f)
		def f_decor(*args,**kwargs):
			plog.info(f"XSBCK:{f.__name__}:start")
			time0 = dt.datetime.utcnow()
			out = f(*args,**kwargs)
			time1 = dt.datetime.utcnow()
			plog.info(f"XSBCK:{f.__name__}:walltime:{time1-time0}")
			plog.info(f"XSBCK:{f.__name__}:end")
			return out
		
		return f_decor
	
	return _decorator
##}}}


