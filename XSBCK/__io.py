

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
import datetime as dt

import numpy  as np
import xarray as xr
import cftime


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


#########
## Dev ##
#########

class Coordinates:##{{{
	
	def __init__( self , dX , dY , cvarsX = None , cvarsY = None ):
		
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
		if (cvarsX is not None and cvarsY is None) or (cvarsY is not None and cvarsX is None):
			raise Exception( "If cvars is given for the ref (or the biased), cvars must be given for the biased (or the ref)" )
		
		if cvarsX is None:
			self.cvarsX  = [key for key in dX.data_vars if len(dX[key].dims) == 3]
			self.cvarsX.sort()
		else:
			self.cvarsX = cvarsX
		self.dimsX = dX[self.cvarsX[0]].dims
		
		if cvarsY is None:
			self.cvarsY  = [key for key in dY.data_vars if len(dY[key].dims) == 3]
			self.cvarsY.sort()
		else:
			self.cvarsY = cvarsY
		
		if not len(self.cvarsY) == len(self.cvarsX):
			raise Exception( "Different numbers of variables!" )
		
		self.ncvar = len(self.cvarsX)
		
		## Now check if a grid mapping exists
		self.mapping = None
		if "grid_mapping" in dX[self.cvarsX[0]].attrs:
			self.mapping = dX[self.cvarsX[0]].attrs["grid_mapping"]
		
	
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
		lstr.append( f"Coordinates: {self.dimsX}" )
		lstr = lstr + [ f" * {c}" for c in self.coords ]
		if self.mapping is not None:
			lstr.append( f"Mapping: {self.mapping}" )
		lstr.append( "Reference variables:" )
		lstr = lstr + [ f" * {cvarY}" for cvarY in self.cvarsY ]
		lstr.append( "Biased variables:" )
		lstr = lstr + [ f" * {cvarX}" for cvarX in self.cvarsX ]
		
		return "\n".join(lstr)
	
	@property
	def cvars(self):
		return [ (cvarX,cvarY) for cvarX,cvarY in zip(self.cvarsX,self.cvarsY) ]

##}}}

def load_data( **kwargs ):##{{{
	
	## Read the data
	dX = xr.open_mfdataset( kwargs["input_biased"]    , data_vars = "minimal" )
	dY = xr.open_mfdataset( kwargs["input_reference"] , data_vars = "minimal" )
	
	## Identify coordinates
	coords = Coordinates( dX , dY , kwargs["cvarsX"] , kwargs["cvarsY"] )
	dX,dY  = coords.delete_mapping(dX,dY)
	logger.info(coords.summary())
	
	## Now define chunk
	dX = dX.chunk( { "y" : 5 , "x" : 5 , "time" : -1 } )
	dY = dY.chunk( { "y" : 5 , "x" : 5 , "time" : -1 } )
	
	return dX,dY,coords
##}}}

def build_reference( method ):##{{{
	
	ref = ""
	if "CDFt" in method:
		ref = "Michelangeli, P.-A., Vrac, M., and Loukos, H.: Probabilistic downscaling approaches: Application to wind cumulative distribution functions, Geophys. Res. Lett., 36, L11708, doi:10.1029/2009GL038401, 2009."
	
	if "R2D2" in method:
		ref = "Vrac, M. et S. Thao (2020). “R2 D2 v2.0 : accounting for temporal dependences in multivariate bias correction via analogue rank resampling”. In : Geosci. Model Dev. 13.11, p. 5367-5387. doi :10.5194/gmd-13-5367-2020."
	
	return ref
##}}}

def save_data( coords , **kwargs ):##{{{
	
	logger.info( "save_data:start" )
	
	## Read tmp files
	dZ = {}
	for cvar in coords.cvarsX:
		
		Z1 = xr.open_mfdataset( os.path.join( kwargs["tmp"] , f"{cvar}_Z1_*.nc" ) , data_vars = "minimal" )[cvar].transpose(*coords.dimsX)
		Z0 = xr.open_mfdataset( os.path.join( kwargs["tmp"] , f"{cvar}_Z0_*.nc" ) , data_vars = "minimal" )[cvar].transpose(*coords.dimsX)
		Z1.loc[Z0.time,:,:] = Z0
		dZ[cvar] = Z1
	
	dZ = xr.Dataset(dZ)
	
	## Now loop on input files to define output files
	for f in kwargs["input_biased"]:
		
		## Load data
		dX = xr.open_dataset(f)
		
		## Find calendar
		calendar = "gregorian"
		if isinstance(dX.time.values[0],cftime.DatetimeNoLeap):
			calendar = "365_day"
		if isinstance(dX.time.values[0],cftime.Datetime360Day):
			calendar = "360_day"
		
		## Find the variable
		for cvar in coords.cvarsX:
			if cvar in dX: break
		X = dX[cvar]
		
		## Build the output file
		avar = cvar + "Adjust"
		Z  = dZ[cvar].loc[X.time,:,:]
		oZ = { avar : Z }
		if coords.mapping is not None:
			oZ[coords.mapping] = 1
		odata = xr.Dataset(oZ)
		
		## Add global attributes
		odata.attrs = dX.attrs
		
		## Add variables attributes
		odata[avar].attrs = X.attrs
		odata[avar].attrs["long_name"] = "Bias Adjusted " + odata[avar].attrs["long_name"]
		
		## Add mapping? attributes
		if coords.mapping is not None:
			odata[coords.mapping].attrs = dX[coords.mapping].attrs
		
		## Add coords attributes
		for c in coords.coords:
			odata[c].attrs = dX[c].attrs
		
		## Add BC attributes
		odata.attrs["bc_creation_date"] = str(dt.datetime.utcnow())[:19] + " (UTC)"
		odata.attrs["bc_method"] = kwargs["method"]
		odata.attrs["bc_period_calibration"] = "/".join( [str(x) for x in kwargs["calibration"]] )
		odata.attrs["bc_window"] = ",".join( [str(x) for x in kwargs["window"]] )
		odata.attrs["bc_reference"] = build_reference(kwargs["method"])
		
		## The encoding
		encoding = {"time" : { "dtype" : "double" , "zlib" : True , "complevel" : 5 , "chunksizes" : (1,) , "calendar" : calendar , "units" : "days since " + str(dZ.time.values[0])[:10] } }
		for c in coords.coords:
			encoding[c] = { "dtype" : "double" , "zlib" : True , "complevel" : 5 , "chunksizes" : odata[c].shape }
		encoding[avar]  = { "dtype" : "float32" , "zlib" : True , "complevel" : 5 , "chunksizes" : (1,) + odata[avar].shape[1:] }
		if coords.mapping is not None:
			encoding[coords.mapping] = { "dtype" : "int32" }
		
		## ofile
		ifile  = os.path.basename(f)
		prefix = f"{avar}_{kwargs['method']}"
		if cvar in ifile:
			ofile = ifile.replace(cvar,prefix)
		else:
			ofile = f"{prefix}_{ifile}"
		
		## And save
		odata.to_netcdf( os.path.join( kwargs["output_dir"] , ofile ) , encoding = encoding )
	
	logger.info( "save_data:end" )
##}}}

