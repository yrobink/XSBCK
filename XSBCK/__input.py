
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
	parser.add_argument( "--output-corrected" , "-ocorr" , "-oZ" , nargs = '+' )
	parser.add_argument( "--method" , "-m" )
	parser.add_argument( "--n-workers"           , default = 1 , type = int )
	parser.add_argument( "--threads-per-workers" , default = 1 , type = int )
	parser.add_argument( "--memory" , default = "2gb" )
	
	kwargs = vars(parser.parse_args())
	
	return kwargs
##}}}

def check_inputs(**kwargs):##{{{
	"""
	XSBCK.check_inputs
	==================
	
	Check the input read by read_inputs.
	
	"""
	
	keys_files = ["input_biased","input_reference","output_corrected"]
	available_methods = ["CDFt","R2D2"]
	
	## Now the big try
	try:
		## Test if file list is not empty
		for key in keys_files:
			if kwargs[key] is None:
				raise Exception(f"File list '{key}' is empty, abort.")
		
		## Test if file number is the same for reference, biased and unbiased,
		n_file  = [len(kwargs[key]) for key in keys_files]
		if not max(n_file) == min(n_file): ## all values are equal
			raise Exception( f"File number not coincide:\n" + "\n".join( [f" * {key} : {n} file(s)" for key,n in zip(keys_files,n_file)] ) )
		
		## Now test if files really exist
		for key in keys_files:
			for f in kwargs[key]:
				if not os.path.isfile(f):
					raise Exception( f"File '{f}' from '{key}' doesn't exists, abort." )
		
		## Test if the method is given
		m = kwargs["method"]
		if m is None:
			raise Exception( f"The method must be specified with the argument '--method'!" )
		
		## Test if the method is available
		if not any([ am in m for am in available_methods ]):
			raise Exception( f"The method {m} is not available, abort." )
		
	
	## All exceptions
	except Exception as e:
		logger.error( f"Error: {e}" )
		return kwargs,True
	
	## And return
	return kwargs,False
##}}}


