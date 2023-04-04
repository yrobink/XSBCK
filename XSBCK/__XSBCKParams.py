
## Copyright(c) 2023 Yoann Robin
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

from __future__ import annotations

import sys
if sys.version_info >= (3, 11):
	from enum import StrEnum
else:
	from enum import Enum as StrEnum

import os
import argparse
import tempfile
import logging
import datetime as dt
import psutil

from dataclasses import dataclass, field
from pathlib import Path

import dask
import distributed
from .__utils import SizeOf
from .__exceptions  import AbortForHelpException


###############
## Variables ##
###############

class BCMethod(StrEnum):
	IdBC = "IdBC"
	CDFt = "CDFt"
	R2D2 = "R2D2"
	dOTC = "dOTC"

bcMethods = ["IdBC","CDFt","R2D2","dOTC"]



@dataclass
class XSBCKParams:
	
	## TODO:
	## input_reference      : list[Path] | None = None
	## input_biased         : list[Path] | None = None
	## output_dir           : Path | None       = None
	
	abort                : bool               = False
	error                : Exception | None   = None
	help                 : bool               = False
	log                  : tuple[str,str|None] = ("WARNING",None)
	input_reference      : list[str] | None   = None
	input_biased         : list[str] | None   = None
	output_dir           : str | None         = None
	method               : str | None         = None
	method_kwargs        : str | None         = None
	n_workers            : int                = 1
	threads_per_worker   : int                = 1
	memory_per_worker    : str                = "auto"
	frac_memory_per_array: float              = 0.2
	total_memory         : str                = "auto"
	tmp_base             : str | None         = None
	tmp_gen              : tempfile.TemporaryDirectory | None = None
	tmp                  : str | None         = None
	tmp_gen_dask         : tempfile.TemporaryDirectory | None = None
	tmp_dask             : str | None         = None
	window               : tuple[int,int,int] = (5,10,5)
	chunks               : int | None         = -1
	calibration          : tuple[str,str]     = ("1976","2005")
	disable_dask         : bool               = False
	cvarsX               : str | None         = None
	cvarsY               : str | None         = None
	cvarsZ               : str | None         = None
	start_year           : str | None         = None
	end_year             : str | None         = None
	ppp                  : list[str]          = None
	client               : distributed.client.Client | None = None
	
	def init_from_user_input( self , *argv ):##{{{
		
		## Parser for user input
		parser = argparse.ArgumentParser( add_help = False )
		
		parser.add_argument( "-h" , "--help" , action = "store_const" , const = True , default = False )
		parser.add_argument( "--log" , nargs = '*' , default = ("WARNING",None) )
		parser.add_argument( "--input-reference"  , "-iref"  , "-iY" , nargs = '+' )
		parser.add_argument( "--input-biased"     , "-ibias" , "-iX" , nargs = '+' )
		parser.add_argument( "--output-dir"       , "-odir"  , "-oZ" , nargs = '?' )
		parser.add_argument( "--method" , "-m" )
		parser.add_argument( "--n-workers"              , default = 1 , type = int )
		parser.add_argument( "--threads-per-worker"     , default = 1 , type = int )
		parser.add_argument( "--memory-per-worker"      , default = "auto" )
		parser.add_argument( "--frac-memory-per-array"  , default = 0.2 , type = float )
		parser.add_argument( "--total-memory"       , default = "auto" )
		parser.add_argument( "--tmp"         , default = None )
		parser.add_argument( "--window"      , default = (5,10,5) )
		parser.add_argument( "--chunks"      , default = None )
		parser.add_argument( "--calibration" , default = ("1976","2005") )
		parser.add_argument( "--disable-dask" , action = "store_const" , const = True , default = False )
		parser.add_argument( "--cvarsX" , default = None )
		parser.add_argument( "--cvarsY" , default = None )
		parser.add_argument( "--cvarsZ" , default = None )
		parser.add_argument( "--start-year" , default = None )
		parser.add_argument( "--end-year"   , default = None )
		parser.add_argument( "--ppp" , nargs = "+" , action = "extend" )
		
		## Transform in dict
		kwargs = vars(parser.parse_args(argv))
		
		## And store in the class
		for key in kwargs:
			if key not in self.__dict__:
				raise Exception("Parameter not present in the class")
			self.__dict__[key] = kwargs[key]
			
		
	##}}}
	
	def init_tmp(self):##{{{
		
		if self.tmp is None:
			self.tmp_base = tempfile.gettempdir()
		else:
			self.tmp_base     = self.tmp
		
		now               = str(dt.datetime.utcnow())[:19].replace("-","").replace(":","").replace(" ","-")
		prefix            = f"XSBCK_{now}_"
		self.tmp_gen      = tempfile.TemporaryDirectory( dir = self.tmp_base , prefix = prefix )
		self.tmp          = self.tmp_gen.name
		self.tmp_gen_dask = tempfile.TemporaryDirectory( dir = self.tmp_base , prefix = prefix + "DASK_" )
		self.tmp_dask     = self.tmp_gen_dask.name
	##}}}
	
	def init_logging(self):##{{{
		
		if len(self.log) == 0:
			self.log = ("INFO",None)
		elif len(self.log) == 1:
			
			try:
				level = int(self.log[0])
				lfile = None
			except:
				try:
					level = getattr( logging , self.log[0].upper() , None )
					lfile = None
				except:
					level = "INFO"
					lfile = self.log[0]
			self.log = (level,lfile)
		
		level,lfile = self.log
		
		## loglevel can be an integet
		try:
			level = int(level)
		except:
			level = getattr( logging , level.upper() , None )
		
		## If it is not an integer, raise an error
		if not isinstance( level , int ): 
			raise UserDefinedLoggingLevelError( f"Invalid log level: {level}; nothing, an integer, 'debug', 'info', 'warning', 'error' or 'critical' expected" )
		
		##
		log_kwargs = {
			"format" : '%(message)s',
#			"format" : '%(levelname)s:%(name)s:%(funcName)s: %(message)s',
			"level"  : level
			}
		
		if lfile is not None:
			log_kwargs["filename"] = lfile
		
		logging.basicConfig(**log_kwargs)
		logging.captureWarnings(True)
		
	##}}}
	
	def init_dask(self):##{{{
		
		if self.disable_dask:
			return
		
		dask_config  = { "temporary_directory" : self.tmp_dask } #, "array.slicing.split_large_chunks" : False }
		client_config = { "n_workers"          :self.n_workers ,
		                  "threads_per_worker" :self.threads_per_worker ,
		                  "memory_limit"       : f"{self.memory_per_worker.B}B" }
		
		dask.config.set(**dask_config)
		self.client = distributed.Client(**client_config)
	##}}}
	
	def stop_dask(self):##{{{
		if self.disable_dask:
			return
		
		self.client.close()
		del self.client
		self.client = None
	##}}}
	
	def check( self ): ##{{{
		
		try:
			if self.help:
				raise AbortForHelpException
			
			## Test if file list is not empty
			for key in ["input_biased","input_reference"]:
				if self.__dict__[key] is None:
					raise Exception(f"File list '{key}' is empty, abort.")
			
			## Test if files really exist
			for key in ["input_biased","input_reference"]:
				for f in self.__dict__[key]:
					if not os.path.isfile(f):
						raise Exception( f"File '{f}' from '{key}' doesn't exists, abort." )
			
			## Test of the output dir exist
			if self.output_dir is None:
				raise Exception("Output directory must be given!")
			if not os.path.isdir(self.output_dir):
				raise Exception( f"Output directory {self.output_dir} is not a path!" )
			
			## Check and set the memory
			if self.memory_per_worker == "auto":
				if self.total_memory == "auto":
					self.total_memory = SizeOf( n = int(0.8 * psutil.virtual_memory().total) , unit = "B" )
				else:
					self.total_memory = SizeOf(self.total_memory)
				self.memory_per_worker = self.total_memory // self.n_workers
			else:
				self.memory_per_worker = SizeOf(self.memory_per_worker)
				self.total_memory      = self.total_memory * self.n_workers
			
			## Test if the method is given
			if '[' in self.method:
				m = method.split("[")[0]
				self.method_kwargs = self.method.split("[")[1].split("]")[0]
				self.method = m
			m = self.method
			
			if m is None:
				raise Exception( f"The method must be specified with the argument '--method'!" )
			
			## Test if the method is available
			if not any([ am in m for am in bcMethods ]):
				raise Exception( f"The method {m} is not available, abort." )
			
			## Check the method configuration
			if m in bcMethods:
				if m == "R2D2":
					self.method = m + "-L-NV-2L"
				else:
					self.method = m + "-L-1V-0L"
			else:
				try:
					meth,cs,cv,cl = m.split("-")
					if not cs == "L":
						raise Exception
					if not cv in ["1V","NV"]:
						raise Exception
					if not cl[-1] == "L":
						raise Exception
					l = int(cl[:-1]) ## Test if an Error is raised
				except:
					raise Exception( f"Method configuration ({m}) not well formed" )
			
			## Test if the tmp directory exists
			if self.tmp is not None:
				if not os.path.isdir(self.tmp):
					raise Exception( f"The temporary directory {self.tmp} is given, but doesn't exists!" )
			
			## The window
			if type(self.window) is str:
				self.window = tuple([ int(s) for s in self.window.split(",") ])
				if not len(self.window) == 3:
					raise Exception( f"Bad arguments for the window ({self.window})" )
			
			## The calibration period
			if type(self.calibration) is str:
				self.calibration = tuple(self.calibration.split("/"))
			try:
				_ = [int(s) for s in self.calibration]
			except:
				raise Exception( f"Bad arguments for the calibration ({self.calibration})" )
			
			if not len(self.calibration) == 2:
				raise Exception( f"Bad arguments for the calibration ({self.calibration})" )
			
		except Exception as e:
			self.abort = True
			self.error = e
		
		
	##}}}
	
	def keys(self):##{{{
		keys = [key for key in self.__dict__]
		keys.sort()
		return keys
	##}}}
	
	def __getitem__( self , key ):##{{{
		return self.__dict__.get(key)
	##}}}
	
xsbckParams = XSBCKParams()

