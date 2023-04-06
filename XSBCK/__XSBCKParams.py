
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
import os
import inspect
import argparse
import tempfile
import logging
import datetime as dt
import psutil
import dataclasses

import dask
import distributed

import numpy as np
import SBCK as bc
import SBCK.ppp as bcp

from .__utils import SizeOf
from .__exceptions  import AbortForHelpException



###############
## Variables ##
###############

#if sys.version_info >= (3, 11):
#	from enum import StrEnum
#else:
#	from enum import Enum as StrEnum
#class BCMethod(StrEnum):
#	IdBC = "IdBC"
#	CDFt = "CDFt"
#	R2D2 = "R2D2"
#	dOTC = "dOTC"

bcMethods = ["IdBC","CDFt","R2D2","dOTC"]


@dataclasses.dataclass
class XSBCKParams:
	
	abort                : bool               = False
	error                : Exception | None   = None
	help                 : bool               = False
	log                  : tuple[str,str|None] = ("WARNING",None)
	
	input_reference      : list[str] | None   = None
	input_biased         : list[str] | None   = None
	output_dir           : str | None         = None
	time_axis            : str                = "time"
	
	n_workers            : int                = 1
	threads_per_worker   : int                = 1
	memory_per_worker    : str                = "auto"
	frac_memory_per_array: float              = 0.2
	total_memory         : str                = "auto"
	client               : distributed.client.Client | None = None
	disable_dask         : bool               = False
	chunks               : int | None         = -1
	
	tmp_base             : str | None         = None
	tmp_gen              : tempfile.TemporaryDirectory | None = None
	tmp                  : str | None         = None
	tmp_gen_dask         : tempfile.TemporaryDirectory | None = None
	tmp_dask             : str | None         = None
	
	window               : tuple[int,int,int] = (5,10,5)
	calibration          : tuple[str,str]     = ("1976","2005")
	cvarsX               : str | None         = None
	cvarsY               : str | None         = None
	cvarsZ               : str | None         = None
	start_year           : str | None         = None
	end_year             : str | None         = None
	
	method               : str | None         = None
	method_kwargs        : str | None         = None
	ppp                  : list[str]          = None
	bc_n_kwargs          : dict | None        = None
	bc_s_kwargs          : dict | None        = None
	pipe                 : list | None        = None
	pipe_kwargs          : list | None        = None
	
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
		parser.add_argument( "--chunks"      , default = -1 )
		parser.add_argument( "--calibration" , default = ("1976","2005") )
		parser.add_argument( "--disable-dask" , action = "store_const" , const = True , default = False )
		parser.add_argument( "--cvarsX" , default = None )
		parser.add_argument( "--cvarsY" , default = None )
		parser.add_argument( "--cvarsZ" , default = None )
		parser.add_argument( "--start-year" , default = None )
		parser.add_argument( "--end-year"   , default = None )
		parser.add_argument( "--time-axis"   , default = "time" )
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
	
	def _init_ppp(self):##{{{
		lppps = self.ppp
		
		## Init
		pipe        = []
		pipe_kwargs = []
		
		if lppps is None:
			return pipe,pipe_kwargs
		
		## Identify columns
		dcols = { cvar : [self.cvarsZ.index(cvar)] for cvar in self.cvarsZ }
		
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
				if "cols" in pkwargs and cvar in self.cvarsZ:
					pkwargs["cols"] = dcols[cvar]
				
				## And others parameters
				for p in p_param:
					key,val = p.split("=")
					if key in insp.annotations:
						pkwargs[key] = insp.annotations[key](val)
					else:
						pkwargs[key] = val
						
						## Special case, val is a list (as sum) of cvar
						if len(set(val.split("+")) & set(self.cvarsZ)) > 0:
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
	
	def init_BC_strategy(self):##{{{
		
		bc_method = bcp.PrePostProcessing
		
		## Find method kwargs
		dkwd = {}
		if self.method_kwargs is not None:
			dkwd = { k : v for (k,v) in [ kv.split("=") for kv in self.method_kwargs.split(",")] }
		
		## The method
		if "IdBC" in self.method:
			bc_method_n_kwargs = { "bc_method" : bc.IdBC , "bc_method_kwargs" : {} }
			bc_method_s_kwargs = { "bc_method" : bc.IdBC , "bc_method_kwargs" : {} }
		if "CDFt" in self.method:
			bc_method_n_kwargs = { "bc_method" : bc.CDFt , "bc_method_kwargs" : {} }
			bc_method_s_kwargs = { "bc_method" : bc.QM   , "bc_method_kwargs" : {} }
		if "dOTC" in self.method:
			bc_method_n_kwargs = { "bc_method" : bc.dOTC , "bc_method_kwargs" : {} }
			bc_method_s_kwargs = { "bc_method" : bc.OTC  , "bc_method_kwargs" : {} }
		if "R2D2" in self.method:
			col_cond   = [0]
			if "col_cond" in dkwd:
				col_cond = [self.cvarsZ.index(cvar) for cvar in dkwd["col_cond"].split("+")]
			lag_keep   = int(self.method.split("-")[-1][:-1]) + 1
			lag_search = 2 * lag_keep
			bcmkwargs  = { "col_cond" : [0] , "lag_search" : lag_search , "lag_keep" : lag_keep , "reverse" : True }
			bc_method_n_kwargs = { "bc_method" : bc.AR2D2 , "bc_method_kwargs" : { **bcmkwargs , "bc_method" : bc.CDFt } }
			bc_method_s_kwargs = { "bc_method" : bc.AR2D2 , "bc_method_kwargs" : { **bcmkwargs , "bc_method" : bc.QM   } }
		
		## The pipe
		pipe,pipe_kwargs = self._init_ppp()
		
		## Global arguments
		bc_n_kwargs = { "bc_method" : bc_method , "bc_method_kwargs" : bc_method_n_kwargs , "pipe" : pipe , "pipe_kwargs" : pipe_kwargs , "checkf" : lambda X: np.any(np.isfinite(X)) }
		bc_s_kwargs = { "bc_method" : bc_method , "bc_method_kwargs" : bc_method_s_kwargs , "pipe" : pipe , "pipe_kwargs" : pipe_kwargs , "checkf" : lambda X: np.any(np.isfinite(X)) }
		
		## Add to the class
		self.bc_n_kwargs = bc_n_kwargs
		self.bc_s_kwargs = bc_s_kwargs
		self.pipe        = pipe
		self.pipe_kwargs = pipe_kwargs
		
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
				self.total_memory      = self.memory_per_worker * self.n_workers
			
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

