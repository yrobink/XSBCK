

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

import sys
import os
import logging

import numpy  as np
import xarray as xr


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


#########
## Dev ##
#########

class Coordinates:##{{{
	
	def __init__( self , dX , dY , lcvarsX = None , lcvarsY = None ):
		
		## Check the two dataset have the same coordinates
		coordsX = [key for key in dX.coords]
		coordsY = [key for key in dY.coords]
		coordsX.sort()
		coordsY.sort()
		
		if not all( [key in coordsX for key in coordsY] + [key in coordsY for key in coordsX] ):
			raise Exception( "Coordinates of input are differents: ref : {coordsY}, biased : {coordsX}" )
		
		self.coords = coordsX
		
		## Check that time is in coordinates
		if not "time" in self.coords:
			raise Exception( "No time axis detected" )
		self.time = dX.time
		
		## Check the spatial coordinates
		if not ( any([ s in self.coords for s in ["lat","latitude"]]) and any([ s in self.coords for s in ["lon","longitude"]]) ):
			raise Exception( "Latitude or longitude are not given" )
		
		## Now the variables
		if (lcvarsX is not None and lcvarsY is None) or (lcvarsY is not None and lcvarsX is None):
			raise Exception( "If cvars is given for the ref (or the biased), cvars must be given for the biased (or the ref)" )
		
		if lcvarsX is None:
			self.lcvarsX  = [key for key in dX.data_vars if len(dX[key].dims) == 3]
			self.lcvarsX.sort()
		else:
			self.lcvarsX = lcvarsX
		
		if lcvarsY is None:
			self.lcvarsY  = [key for key in dY.data_vars if len(dY[key].dims) == 3]
			self.lcvarsY.sort()
		else:
			self.lcvarsY = lcvarsY
		
		if not len(self.lcvarsY) == len(self.lcvarsX):
			raise Exception( "Different numbers of variables!" )
		
		self.ncvar = len(self.lcvarsX)
		
		## Now check if a grid mapping exists
		self.mapping = None
		if "grid_mapping" in dX[self.lcvarsX[0]].attrs:
			self.mapping = dX[self.lcvarsX[0]].attrs["grid_mapping"]
		
	
	def delete_mapping( self , *args ):
		if self.mapping is None:
			return args
		
		out = []
		for a in args:
			del a[self.mapping]
			out.append(a)
		
		return tuple(out)
	
	def summary(self):
		lstr = []
		lstr.append( "Coordinates:" )
		lstr = lstr + [ f" * {c}" for c in self.coords ]
		if self.mapping is not None:
			lstr.append( f"Mapping: {self.mapping}" )
		lstr.append( "Reference variables:" )
		lstr = lstr + [ f" * {cvarY}" for cvarY in self.lcvarsY ]
		lstr.append( "Biased variables:" )
		lstr = lstr + [ f" * {cvarX}" for cvarX in self.lcvarsX ]
		
		return "\n".join(lstr)
	
	@property
	def cvars(self):
		return [ (cvarX,cvarY) for cvarX,cvarY in zip(self.lcvarsX,self.lcvarsY) ]

##}}}

def load_data( **kwargs ):##{{{
	
	## Read the data
	dX = xr.open_mfdataset( kwargs["input_biased"]    , data_vars = "minimal" )
	dY = xr.open_mfdataset( kwargs["input_reference"] , data_vars = "minimal" )
	
	## Identify coordinates
	coords = Coordinates( dX , dY )
	dX,dY  = coords.delete_mapping(dX,dY)
	logger.info(coords.summary())
	
	## Now define chunk
	dX = dX.chunk( { "y" : 5 , "x" : 5 , "time" : -1 } )
	dY = dY.chunk( { "y" : 5 , "x" : 5 , "time" : -1 } )
	
	return dX,dY,coords
##}}}

def save_data( coords , **kwargs ):##{{{
	pass
##}}}

