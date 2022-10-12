
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


## Package
##########

import os
import logging
import argparse

from .__logs import LINE

## Init logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


## Functions
############

def read_inputs():##{{{
	"""
	XSBCK.read_inputs
	=================
	
	Function using argparse to read command line arguments. Return a dict
	of all parameters (see the documentation)
	"""
	
	parser = argparse.ArgumentParser( add_help = False )
	
	parser.add_argument( "-h" , "--help" , action = "store_const" , const = True , default = False )
	parser.add_argument( "--log" , nargs = '*' , default = ["WARNING"] )
	parser.add_argument( "--input-reference"  , "-iref"  , "-iY" , nargs = '+' )
	parser.add_argument( "--input-biased"     , "-ibias" , "-iX" , nargs = '+' )
	parser.add_argument( "--output-dir"       , "-odir"  , "-oZ" , nargs = '?' )
	parser.add_argument( "--method" , "-m" )
	parser.add_argument( "--n-workers"          , default = 1 , type = int )
	parser.add_argument( "--threads-per-worker" , default = 1 , type = int )
	parser.add_argument( "--memory" , default = "auto" )
	parser.add_argument( "--tmp-base" , nargs = 1 , default = "/tmp/" )
	parser.add_argument( "--tmp"      , nargs = 1 , default = None )
	
	kwargs = vars(parser.parse_args())
	
	##TODO
	kwargs["calibration"] = ("1976","2005")
	kwargs["window"]      = (5,10,5)
	#kwargs["chunk"]       = "?"
	
	##
	
	return kwargs
##}}}

def check_inputs(**kwargs):##{{{
	"""
	XSBCK.check_inputs
	==================
	
	Check the input read by read_inputs.
	
	"""
	
	logger.info("check_inputs:start")
	
	keys_input = ["input_biased","input_reference"]
	available_methods = ["CDFt","R2D2"]
	abort = False
	
	## Now the big try
	try:
		## Test if file list is not empty
		for key in keys_input:
			if kwargs[key] is None:
				raise Exception(f"File list '{key}' is empty, abort.")
		
		## Test if files really exist
		for key in keys_input:
			for f in kwargs[key]:
				if not os.path.isfile(f):
					raise Exception( f"File '{f}' from '{key}' doesn't exists, abort." )
		
		## Test of the output dir exist
		if kwargs["output_dir"] is None:
			raise Exception("Output directory must be given!")
		if not os.path.isdir(kwargs["output_dir"]):
			raise Exception( f"Output directory {kwargs['output_dir']} is not a path!" )
		
		## Test if the method is given
		m = kwargs["method"]
		if m is None:
			raise Exception( f"The method must be specified with the argument '--method'!" )
		
		## Test if the method is available
		if not any([ am in m for am in available_methods ]):
			raise Exception( f"The method {m} is not available, abort." )
		
		## Test if the tmp directory exists
		if kwargs["tmp"] is not None:
			if not os.path.isdir(kwargs["tmp"]):
				raise Exception( f"The temporary directory {kwargs['tmp']} is given, but doesn't exists!" )
			kwargs["tmp_base"] = False
		else:
			if not os.path.isdir(kwargs["tmp_base"]):
				raise Exception( f"The base temporary directory {kwargs['tmp_base']} doesn't exists!" )
	
	## All exceptions
	except Exception as e:
		logger.error( f"Error: {e}" )
		abort = True
	
	logger.info("check_inputs:end")
	logger.info(LINE)
	
	## And return
	return kwargs,abort
##}}}


