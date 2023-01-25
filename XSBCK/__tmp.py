
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

import os
import random
import string
import tempfile
import datetime as dt

###############
## Functions ##
###############

def build_tmp_dir( kwargs : dict ):##{{{
	"""
	XSBCK.build_tmp_dir
	===================
	
	Build the temporary directories
	
	
	"""
	
	t = str(dt.datetime.utcnow())[:19].replace("-","").replace(":","").replace(" ","-")
	prefix = f"XSBCK_{t}_"
	kwargs["tmp_gen"]  = tempfile.TemporaryDirectory( dir = kwargs["tmp_base"] , prefix = prefix )
	kwargs["tmp"] = kwargs["tmp_gen"].name
	kwargs["tmp_gen_dask"] = tempfile.TemporaryDirectory( dir = kwargs["tmp_base"] , prefix = prefix + "dask_" )
	kwargs["tmp_dask"] = kwargs["tmp_gen_dask"].name
##}}}

