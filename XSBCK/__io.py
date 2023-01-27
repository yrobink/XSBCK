

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

#############
## Imports ##
#############

import sys
import itertools as itt
import os
import gc
import logging
import datetime as dt

import numpy  as np
import xarray as xr
import cftime
import dask
import distributed
import netCDF4

import zarr
import SBCK

from .__release import version
from .__logs    import log_start_end
from .__utils   import SizeOf


##################
## Init logging ##
##################

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


##============================================================================##
##                                                                            ##
##  Class to link zarr and xarray                                             ##
##                                                                            ##
##============================================================================##

## TODO time_axis
## TODO documentation
class XZarr:##{{{
	"""
	XSBCK.XZarr
	===========
	
	Class managing a zarr file, to access with the time axis
	
	"""
	
	def __init__( self , fzarr , persist = False ):##{{{
		"""
		XSBCK.XZarr.__init__
		====================
		
		Arguments
		---------
		fzarr:
			The file used as zarr file
		dX: [xarray.Dataset]
			Data to copy in the zarr file. If None, a zero file is init with
			the shape, dims and coords parameters
		shape:
			Only used if dX is None
		dims:
			Only used if dX is None
		coords:
			Only used if dX is None
		chunks:
			The chunk of the data
		persist:
			If False, the fzarr file is removed when the object is deleted.
		cvars:
			List of cvar
		dtype:
			The data type, default is float32
		avail_spatial_mem:
			Memory limit in octet of a single map (no time and variables). Used
			to spatially cut the dataset
		 
		"""
		
		self._fmatch = lambda a,b: [ b.index(x) if x in b else None for x in a ]
		
		self.fzarr   = fzarr
		self.persist = persist
		
		self.zarr_chunks = None
		self.dask_chunks = None
		
		self.shape  = None
		self.dims   = None
		self.coords = None
		
		self.dtype  = None
		self.data   = None
		self.ifiles = None
	##}}}
	
	## @staticmethod.from_dataset ##{{{
	
	@staticmethod
	def from_dataset( fzarr , xdata , ifiles , xcvars , zcvars , dask_chunks , zarr_chunks , time_axis = "time" , persist = False ):
		"""
		@staticmethod
		XSBCK.XZarr.from_dataset
		========================
		
		Arguments
		---------
		fzarr:
			The file used as zarr file
		xdata: [xarray.Dataset]
			Data to copy in the zarr file.
		ifiles:
			List of files to read the data. We have:
				xdata = xr.open_mfdataset(ifiles)
			But data are read directly from ifiles with the netCDF4 engine
		xcvars:
			List of cvar from xdata
		zcvars:
			List of cvar in the zarr file, can be used to rename xcvars
		dask_chunks:
			The chunk of the data
		zarr_chunks:
			The chunk of the data
		time_axis:
			Name of the time axis, not really used currently, use 'time'
		persist:
			If False, the fzarr file is removed when the object is deleted.
		 
		"""
		
		## The file
		xzarr = XZarr( fzarr = fzarr , persist = persist )
		
		## Init zarr chunks
		time_chunk = 4 * 365 + 1
		try:
			if isinstance(xdata[time_axis].values[0],cftime.DatetimeNoLeap):
				time_chunk = 365
			if isinstance(xdata[time_axis].values[0],cftime.Datetime360Day):
				time_chunk = 360
		except:
			pass
		zarr_chunks = list(zarr_chunks)
		zarr_chunks[0] = time_chunk
		xzarr.zarr_chunks = zarr_chunks
		
		## Init dask chunks
		xzarr.dask_chunks = dask_chunks
		
		## Init coordinates
		xzarr.shape  = xdata[xcvars[0]].shape  + (len(xcvars),)
		xzarr.dims   = xdata[xcvars[0]].dims   + ("cvar",)
		xzarr.coords = [xdata[c] for c in xzarr.dims[:-1]] + [zcvars,]
		xztime = xzarr.coords[0]
		if isinstance( xztime[0].values , np.datetime64 ):
			t0 = str(xztime[0].values)
			year,month,day = [int(s) for s in t0[:10].split("-")]
			h,m,sms        = t0[11:].split(":")
			h = int(h)
			m = int(m)
			s,ms = [int(k) for k in sms.split(".")]
			t0   = dt.datetime( year , month , day , h , m , s , ms )
			xztime = [ t0 + dt.timedelta(days = i) for i in range(xztime.size)]
		
		## And now build the zarr file
		xzarr.dtype = xdata[xcvars[0]].dtype
		xzarr.data  = zarr.open( xzarr.fzarr , mode = "w" , shape = xzarr.shape , chunks = xzarr.zarr_chunks , dtype = xzarr.dtype )
		
		## And copy netcdf file to zarr file
		xzarr.ifiles = ifiles
		for ifile in ifiles:
			with netCDF4.Dataset( ifile , "r" ) as ncfile:
				
				## Find the cvar
				for icvar,xcvar,zcvar in zip(range(len(zcvars)),xcvars,zcvars):
					if xcvar in ncfile.variables:
						break
				
				## Time idx
				itime     = cftime.num2date( ncfile.variables[time_axis] , ncfile.variables[time_axis].units , ncfile.variables[time_axis].calendar )
				num_itime = cftime.date2num( itime  , ncfile.variables[time_axis].units , ncfile.variables[time_axis].calendar )
				num_time  = cftime.date2num( xztime , ncfile.variables[time_axis].units , ncfile.variables[time_axis].calendar )
				t0,t1 = num_time[:2]
				idx   = np.array( np.ceil( (num_itime - t0) / (t1 - t0 ) ) , int ).tolist()
				
				## Now loop on zchunks
				for zc in xzarr.iter_zchunks():
					
					zc_y,zc_x = zc
					i0y =  zc_y    * xzarr.data.chunks[1]
					i1y = (zc_y+1) * xzarr.data.chunks[1]
					i0x =  zc_x    * xzarr.data.chunks[2]
					i1x = (zc_x+1) * xzarr.data.chunks[2]
					
					## And loop on time chunk to limit memory used
					for it in range(0,len(idx),time_chunk):
						sel = (idx[it:(it+time_chunk)],slice(i0y,i1y),slice(i0x,i1x),icvar)
						xzarr.data.set_orthogonal_selection( sel , ncfile.variables[xcvar][it:(it+time_chunk),i0y:i1y,i0x:i1x] )
		
		return xzarr
	##}}}
	
	def copy( self , fzarr , value = np.nan , persist = False ):##{{{
		
		## Init the copy
		copy = XZarr( fzarr = fzarr , persist = persist )
		
		## Copy the attributes
		copy.zarr_chunks = list(self.zarr_chunks)
		copy.dask_chunks = self.dask_chunks
		
		copy.shape  = list(self.shape)
		copy.dims   = list(self.dims)
		copy.coords = list(self.coords)
		
		copy.dtype  = self.dtype 
		copy.ifiles = list(self.ifiles)
		
		## Init the zarr file
		copy.data  = zarr.open( fzarr , mode = "w" , shape = copy.shape , chunks = copy.zarr_chunks , dtype = copy.dtype )
		
		## And fill it
		if value is None:
			copy.data[:] = self.data[:]
		else:
			copy.data[:] = value
		
		return copy
		
	##}}}
	
	def __del__( self ):##{{{
		if not self.persist:
			self.clean()
	##}}}
	
	def clean(self):##{{{
		"""
		XSBCK.XZarr.clean
		=================
		Remove the fzarr file
		 
		"""
		try:
			if os.path.isdir(self.fzarr):
				for f in os.listdir(self.fzarr):
					os.remove( os.path.join( self.fzarr , f ) )
				os.rmdir(self.fzarr)
		except:
			pass
	##}}}
	
	def __str__(self):##{{{
		
		out = []
		out.append( "<XSBCK.XZarr>" )
		out.append( "Dimensions: (" + ", ".join( [f"{d}: {s}" for d,s in zip(self.dims,self.shape)] ) + ")" )
		out.append( f"zarr_chunks  : {self.zarr_chunks}" )
		out.append( f"n_dask_chunks: {self.dask_chunks}" )
		out.append( "Coordinates:" )
		for d,coord in zip(self.dims,self.coords):
			line = str(coord).split("\n")[-1]
			if "*" in line:
				out.append(line)
			else:
				C = " ".join(coord)
				if len(C) > 24:
					C = C[:11] + " ... " + C[-11:]
				out.append( "  * {:{fill}{align}{n}}".format(d,fill=" ",align="<",n=9) + f"({d}) " + C )
		
		return "\n".join(out)
	##}}}
	
	def __repr__(self):##{{{
		return self.__str__()
	##}}}
	
	def iter_zchunks(self):##{{{
		"""
		Return a generator to iterate over spatial zarr chunks
		"""
		
		return itt.product(range(self.data.cdata_shape[1]),range(self.data.cdata_shape[2]))
	##}}}
	
	def sel_along_time( self , time , zc = None ):##{{{
		"""
		XSBCK.XZarr.sel_along_time
		==========================
		
		Arguments
		---------
		time:
			The time values to extract
		zc:
			The chunk identifier, given by XSBCK.XZarr.iter_zchunks. If None,
			all spatial values are returned.
		
		Returns
		-------
		A chunked xarray.DataArray
		 
		"""
		
		time_fmt = self.time.sel( time = time ).values.tolist() ## Ensure time and self.time have the save format
		if not type(time_fmt) is list: time_fmt = [time_fmt]
		idx      = self._fmatch( time_fmt , self.time.values.tolist() )
		
		dask_chunks = { "time" : -1 , "cvar" : -1 }
		if zc is None:
			sel    = (idx,slice(None),slice(None),slice(None))
			coords = [self.time[idx]] + self.coords[1:]
			if self.dask_chunks == -1:
				dask_chunks[self.dims[1]] = -1
				dask_chunks[self.dims[2]] = -1
			else:
				dask_chunks[self.dims[1]] = int( self.shape[1] / np.sqrt(self.dask_chunks) )
				dask_chunks[self.dims[2]] = int( self.shape[2] / np.sqrt(self.dask_chunks) )
		else:
			zc_y,zc_x = zc
			i0y =  zc_y    * self.data.chunks[1]
			i1y = (zc_y+1) * self.data.chunks[1]
			i0x =  zc_x    * self.data.chunks[2]
			i1x = (zc_x+1) * self.data.chunks[2]
			sel = (idx,slice(i0y,i1y,1),slice(i0x,i1x),slice(None))
			coords = [self.time[idx]] + [self.coords[1][i0y:i1y]] + [self.coords[2][i0x:i1x]] + [self.coords[3]]
			if self.dask_chunks == -1:
				dask_chunks[self.dims[1]] = -1
				dask_chunks[self.dims[2]] = -1
			else:
				dask_chunks[self.dims[1]] = int( (i1y-i0y+1) / np.sqrt(self.dask_chunks) )
				dask_chunks[self.dims[2]] = int( (i1x-i0x+1) / np.sqrt(self.dask_chunks) )
		
		X = xr.DataArray( self.data.get_orthogonal_selection(sel) ,
		                  dims = self.dims ,
		                  coords = coords,
		                ).chunk(
		                  dask_chunks
		                ).astype(self.dtype)
		
		return X
	##}}}
	
	def sel_cvar_along_time( self , time , cvar , zc = None ):##{{{
		"""
		XSBCK.XZarr.sel_cvar_along_time
		===============================
		
		To select only on cvar
		
		Arguments
		---------
		time:
			The time values to extract
		cvar:
			The climate variable selected
		zc:
			The chunk identifier, given by XSBCK.XZarr.iter_zchunks. If None,
			all spatial values are returned.
		
		Returns
		-------
		A NOT chunked xarray.DataArray
		 
		"""
		
		time_fmt = self.time.sel( time = time ).values.tolist() ## Ensure time and self.time have the save format
		if not type(time_fmt) is list: time_fmt = [time_fmt]
		idx      = self._fmatch( time_fmt , self.time.values.tolist() )
		icvar    = self.cvars.index(cvar)
		
		
		if zc is None:
			sel    = (idx,slice(None),slice(None),icvar)
			coords = [self.time[idx]] + self.coords[1:-1]
		else:
			zc_y,zc_x = zc
			i0y =  zc_y    * self.data.chunks[1]
			i1y = (zc_y+1) * self.data.chunks[1]
			i0x =  zc_x    * self.data.chunks[2]
			i1x = (zc_x+1) * self.data.chunks[2]
			sel = (idx,slice(i0y,i1y),slice(i0x,i1x),icvar)
			coords = [self.time[idx]] + [self.coords[1][i0y:i1y]] + [self.coords[2][i0x:i1x]]
		
		X = xr.DataArray( self.data.get_orthogonal_selection(sel).squeeze() ,
		                  dims = self.dims[:-1] ,
		                  coords = coords
		                ).chunk(
		                  { "time" : -1 , **{ d : -1 for d in self.dims[1:-1] } }
		                )
		
		return X
	##}}}
	
	def set_along_time( self , X , time = None , zc = None ):##{{{
		"""
		XSBCK.XZarr.set_along_time
		==========================
		
		To set X at time values in the zarr file
		
		Arguments
		---------
		X:
			A data array
		time:
			The time values to set
		zc:
			The chunk identifier, given by XSBCK.XZarr.iter_zchunks. If None,
			all spatial values are set.
		 
		"""
		
		time     = time if time is not None else X.time
		time_fmt = self.time.sel( time = time ).values.tolist() ## Ensure time and self.time have the save format
		if not type(time_fmt) is list: time_fmt = [time_fmt]
		idx      = self._fmatch( time_fmt , self.time.values.tolist() )
		
		if zc is None:
			sel    = (idx,slice(None),slice(None),slice(None))
		else:
			zc_y,zc_x = zc
			i0y =  zc_y    * self.data.chunks[1]
			i1y = (zc_y+1) * self.data.chunks[1]
			i0x =  zc_x    * self.data.chunks[2]
			i1x = (zc_x+1) * self.data.chunks[2]
			sel = (idx,slice(i0y,i1y,1),slice(i0x,i1x),slice(None))
		
		self.data.set_orthogonal_selection( sel , X.values )
		
	##}}}
	
	## Properties {{{
	@property
	def time(self):
		return self.coords[0]
	
	@property
	def cvars(self):
		return self.coords[-1]
	##}}}
	
##}}}


##============================================================================##
##                                                                            ##
##  Function to read data and create zarr files                               ##
##                                                                            ##
##============================================================================##

## load_data ##{{{
@log_start_end(logger)
def load_data( kwargs : dict ):
	"""
	XSBCK.load_data
	===============
	Function used to read data and copy in a temporary zarr file
	
	Arguments
	---------
	kwargs:
		dict of all parameters of XSBCK
	
	Returns
	-------
	zX:
		XZarr file of the biased dataset
	zY:
		XZarr file of the reference dataset
	zZ:
		XZarr file of the corrected dataset (empty)
	"""
	
	##
	time_axis = "time" ##kwargs["time_axis"]
	cvarsX = kwargs["cvarsX"]
	cvarsY = kwargs["cvarsY"]
	cvarsZ = kwargs["cvarsZ"]
	
	## Open with xarray to decode all variables and time axis
	xX = xr.open_mfdataset( kwargs["input_biased"]    , data_vars = "minimal" , coords = "minimal" , compat = "override" , combine_attrs = "drop" )
	xY = xr.open_mfdataset( kwargs["input_reference"] , data_vars = "minimal" , coords = "minimal" , compat = "override" , combine_attrs = "drop" )
	
	## Init cvars
	logger.info("Check cvars")
	if (cvarsX is not None and cvarsY is None) or (cvarsY is not None and cvarsX is None):
		raise Exception( "If cvars is given for the ref (or the biased), cvars must be given for the biased (or the ref)" )
	check_cvarsXY_is_same = False
	if cvarsX is None:
		cvarsX  = [key for key in xX.data_vars]
		cvarsX.sort()
		check_cvarsXY_is_same = True
	else:
		cvarsX = cvarsX.split(",")
	if cvarsY is None:
		cvarsY  = [key for key in xY.data_vars]
		cvarsY.sort()
		check_cvarsXY_is_same = True
	else:
		cvarsY = cvarsY.split(",")
	if check_cvarsXY_is_same:
		if ( not all([cvar in cvarsX for cvar in cvarsY]) ) or (not all([cvar in cvarsY for cvar in cvarsX])):
			raise Exception( "Variables from ref or biased differs" )
	if cvarsZ is None:
		cvarsZ = cvarsX
	else:
		cvarsZ = cvarsZ.split(",")
	
	kwargs["cvarsX"] = cvarsX
	kwargs["cvarsY"] = cvarsY
	kwargs["cvarsZ"] = cvarsZ
	
	logger.info( f" * cvarsX: {cvarsX}" )
	logger.info( f" * cvarsY: {cvarsY}" )
	logger.info( f" * cvarsZ: {cvarsZ}" )
	
	## Remove of the dataset all variables without time axis
	keys_to_del = [key for key in xX.data_vars if time_axis not in xX[key].dims]
	for key in keys_to_del:
		del xX[key]
	keys_to_del = [key for key in xY.data_vars if time_axis not in xY[key].dims]
	for key in keys_to_del:
		del xY[key]
	
	## Find spatial memory available
	total_memory       = kwargs["total_memory"].o
	frac_mem_per_array = kwargs["frac_memory_per_array"]
	max_mem_per_chunk  = SizeOf( f"{int(frac_mem_per_array * total_memory)}o" )
	max_time           = 365 * max( int(kwargs["calibration"][1]) - int(kwargs["calibration"][0]) + 1 , sum(kwargs['window']) )
	max_cvar           = len(cvarsZ)
	avail_spatial_mem  = SizeOf( f"{int( max_mem_per_chunk.o / ( max_time * max_cvar * (np.finfo('float32').bits // max_mem_per_chunk.bits_per_octet) ) )}o" )
	
	logger.info( "Check memory:" )
	logger.info( f" * Max mem. per chunk: {max_mem_per_chunk.o}o" )
	logger.info( f" * Max time step     : {max_time}" )
	logger.info( f" * Max cvar          : {max_cvar}" )
	logger.info( f" * Avail Spat. Mem.  : {avail_spatial_mem.o}o" )
	
	## Find chunks
	dask_chunks = kwargs["chunks"]
	if dask_chunks == -1:
		dask_chunks = kwargs["n_workers"] * kwargs["threads_per_worker"]
	
	ny = xX[xX[cvarsX[0]].dims[1]].size
	nx = xX[xX[cvarsX[0]].dims[2]].size
	avail_spatial_numb = int( avail_spatial_mem.o / (np.finfo(xX[cvarsX[0]].dtype).bits // max_mem_per_chunk.bits_per_octet) )
	zch_ny = max( int(ny / np.sqrt(avail_spatial_numb)) , 1 )
	zch_nx = max( int(nx / np.sqrt(avail_spatial_numb)) , 1 )
	zarr_chunks = [ None , ny // zch_ny , nx // zch_nx , 1 ]
	
	logger.info( "Chunks found:" )
	logger.info( f" * dask_chunks: {dask_chunks}" )
	logger.info( f" * zarr_chunks: {zarr_chunks}" )
	
	## Now zarr file
	logger.info( "Create biased zarr file..." )
	zX = XZarr.from_dataset( os.path.join( kwargs["tmp"] , "X.zarr" ) , xX , ifiles = kwargs["input_biased"]    , xcvars = cvarsX , zcvars = cvarsZ , dask_chunks = dask_chunks , zarr_chunks = zarr_chunks , time_axis = time_axis )
	logger.info( "Create reference zarr file..." )
	zY = XZarr.from_dataset( os.path.join( kwargs["tmp"] , "Y.zarr" ) , xY , ifiles = kwargs["input_reference"] , xcvars = cvarsY , zcvars = cvarsZ , dask_chunks = dask_chunks , zarr_chunks = zarr_chunks , time_axis = time_axis )
	logger.info( "Create corrected (empty) zarr file..." )
	zZ = zX.copy( os.path.join( kwargs["tmp"] , "Z.zarr" ) , np.nan )
	
	logger.info( f"About biased data:" )
	logger.info( f" * shape  : {str(zX.shape)}" )
	logger.info( f" * zchunks: {str(zX.zarr_chunks)}" )
	logger.info( f"About reference data:" )
	logger.info( f" * shape  : {str(zY.shape)}" )
	logger.info( f" * zchunks: {str(zY.zarr_chunks)}" )
	
	## Free memory
	del xX
	del xY
	gc.collect()
	
	return zX,zY,zZ
##}}}


##============================================================================##
##                                                                            ##
##  Function to save data                                                     ##
##                                                                            ##
##============================================================================##

## build_reference ##{{{
def build_reference( method : str ):
	"""
	XSBCK.build_reference
	=====================
	Function used to build a string of the reference article of the method.
	
	Arguments
	---------
	method:
		str
	
	Returns
	-------
	str
	"""
	
	ref = ""
	if "CDFt" in method:
		ref = "Michelangeli, P.-A., Vrac, M., and Loukos, H.: Probabilistic downscaling approaches: Application to wind cumulative distribution functions, Geophys. Res. Lett., 36, L11708, doi:10.1029/2009GL038401, 2009."
	
	if "R2D2" in method:
		ref = "Vrac, M. et S. Thao (2020). “R2 D2 v2.0 : accounting for temporal dependences in multivariate bias correction via analogue rank resampling”. In : Geosci. Model Dev. 13.11, p. 5367-5387. doi :10.5194/gmd-13-5367-2020."
	
	return ref
##}}}

## save_data ##{{{ 

@log_start_end(logger)
def save_data( zX : XZarr , zZ : XZarr , kwargs : dict ):
	"""
	XSBCK.save_data
	===============
	Function used to read the XZarr file of the corrected dataset and rewrite
	in netcdf.
	
	Arguments
	---------
	zX:
		XZarr file of the biased dataset
	zZ:
		XZarr file of the corrected dataset
	kwargs:
		dict of all parameters of XSBCK
	
	Returns
	-------
	None
	"""
	
	## Build mapping between cvarsX and cvarsZ
	cvarsX = kwargs["cvarsX"]
	cvarsZ = kwargs["cvarsZ"]
	mcvars = { x : z for x,z in zip(cvarsX,cvarsZ) }
	
	time_axis  = "time"
	time_chunk = zZ.zarr_chunks[0]
	tstart = kwargs["start_year"]
	tend   = kwargs["end_year"]
	
	## Loop on input files
	for ifile in kwargs["input_biased"]:
		
		logger.info( f"File {os.path.basename(ifile)}" )
		
		with netCDF4.Dataset( ifile , mode = "r" ) as incfile:
			
			## Read the time axis
			nctime = incfile.variables[time_axis]
			time   = cftime.num2date( nctime[:] , nctime.units , nctime.calendar )
			time   = xr.DataArray( time , dims = [time_axis] , coords = [time] )
			
			## Sub select the time
			if time.dt.year[-1] < int(tstart):
				logger.info( f" * End year {int(time.dt.year[-1])} < start year {tstart}, skip." )
				continue
			if time.dt.year[0] > int(tend):
				logger.info( f" * Start year {int(time.dt.year[0])} < end year {tend}, skip." )
				continue
			otime = time.sel( time = slice(tstart,tend) )
			
			## Find the variable
			cvarX = list(set(cvarsX) & set(incfile.variables))[0]
			icvar = cvarsX.index(cvarX)
			cvarZ = cvarsZ[icvar]
			logger.info( f" * cvarX: {cvarX}, cvarZ: {cvarZ}" )
			
			## The output file
			bifile = os.path.basename(ifile)
			prefix = f"{cvarZ}_{kwargs['method']}"
			if cvarX in bifile:
				ofile = bifile.replace(cvarX,prefix)
			else:
				ofile = f"{prefix}_{bifile}"
			ofile = os.path.join( kwargs["output_dir"] , ofile )
			logger.info( f" * ofile: {ofile}" )
			
			## Now write the output file
			with netCDF4.Dataset( ofile , mode = "w" ) as oncfile:
				
				oncfile.set_fill_off()
				
				## Copy main attributes
				for name in incfile.ncattrs():
					try:
						oncfile.setncattr( name , incfile.getncattr(name) )
					except AttributeError:
						pass
				
				## Add BC attributes
				oncfile.setncattr( "bc_creation_date" , str(dt.datetime.utcnow())[:19] + " (UTC)" )
				oncfile.setncattr( "bc_method"        , kwargs["method"] )
				oncfile.setncattr( "bc_period_calibration" , "/".join( [str(x) for x in kwargs["calibration"]] ) )
				oncfile.setncattr( "bc_window"        , ",".join( [str(x) for x in kwargs["window"]] ) )
				oncfile.setncattr( "bc_reference"     , build_reference(kwargs["method"]) )
				oncfile.setncattr( "bc_pkgs_versions" , ", ".join( [f"XSBCK:{version}"] + [f"{name}:{pkg.__version__}" for name,pkg in zip(["SBCK","numpy","xarray","dask","distributed","zarr","netCDF4"],[SBCK,np,xr,dask,distributed,zarr,netCDF4]) ] ) )
				
				## Start with dimensions
				dims   = [d for d in incfile.dimensions if not d == time_axis ]
				ncdims = { d : oncfile.createDimension( d  , incfile.dimensions[d].size )  for d in dims }
				ncdims[time_axis] = oncfile.createDimension( time_axis  , None )
				
				## Define variables of dimensions
				ncv_dims = { d : oncfile.createVariable( d , incfile.variables[d].dtype , (d,)  , shuffle = False , compression = "zlib" , complevel = 5 , chunksizes = incfile.variables[d].chunking() ) for d in dims if d in incfile.variables }
				ncv_dims[time_axis] = oncfile.createVariable( time_axis , incfile.variables[time_axis].dtype , (time_axis,)  , shuffle = False , compression = "zlib" , complevel = 5 , chunksizes = (1,) )
				
				## Copy attributes of the dimensions
				for d in ncv_dims:
					for name in incfile.variables[d].ncattrs():
						try:
							ncv_dims[d].setncattr( name , incfile.variables[d].getncattr(name) )
						except AttributeError:
							pass
				
				## And fill dimensions
				for d in list(set(dims) & set([k for k in ncv_dims])):
					ncv_dims[d][:] = incfile.variables[d][:]
				ncv_dims[time_axis][:] = cftime.date2num( otime.values , incfile.variables[time_axis].units , incfile.variables[time_axis].calendar )
				dims.append(time_axis)
				
				## Continue with all variables, except the "main" variable
				variables = [v for v in incfile.variables if v not in dims + [cvarX]]
				ncvars = {}
				for v in variables:
					
					## Create the variable
					ncvars[v] = oncfile.createVariable( v , incfile.variables[v].dtype , incfile.variables[v].dimensions , shuffle = False , compression = "zlib" , complevel = 5 , chunksizes = incfile.variables[v].chunking() )
					
					## Copy attributes
					for name in incfile.variables[v].ncattrs():
						try:
							ncvars[v].setncattr( name , incfile.variables[v].getncattr(name) )
						except AttributeError:
							pass
					
					## And Fill it
					if len(incfile.variables[v].shape) == 0: ## Scalar variable
						ncvars[v].assignValue(incfile.variables[v].getValue())
					else:
						if time_axis in incfile.variables[v].dimensions:
							
							## Tuple of selection
							idx = cftime.date2index( otime , incfile.variables[time_axis] , incfile.variables[time_axis].calendar )
							sel = [slice(None) for _ in range(len(incfile.variables[v].dimensions))]
							sel[list(incfile.variables[v].dimensions).index(time_axis)] = idx
							sel = tuple(sel)
							
							## And copy var
							ncvars[v][:] = incfile.variables[v][sel]
						else:
							ncvars[v][:] = incfile.variables[v][:]
				
				## Find fill and missing value
				try:
					fill_value = incfile.variables[cvarX].getncattr("_FillValue")
				except AttributeError:
					fill_value = np.nan
				try:
					missing_value = incfile.variables[cvarX].getncattr("missing_value")
				except AttributeError:
					missing_value = None
				
				## Create the main variable
				S = SizeOf( f"{np.prod(incfile.variables[cvarX].shape) * (np.finfo(incfile.variables[cvarX].dtype).bits // SizeOf('1o').bits_per_byte)}o" )
				if S.o < SizeOf("5Go").o:
					ncvar = oncfile.createVariable( cvarZ , incfile.variables[cvarX].dtype , incfile.variables[cvarX].dimensions , shuffle = False , fill_value = fill_value , compression = "zlib" , complevel = 5 , chunksizes = incfile.variables[cvarX].chunking() )
				else:
					ncvar = oncfile.createVariable( cvarZ , incfile.variables[cvarX].dtype , incfile.variables[cvarX].dimensions , shuffle = False , fill_value = fill_value )
				if missing_value is not None:
					ncvar.setncattr( "missing_value" , missing_value )
				
				## Copy attributes
				for name in incfile.variables[cvarX].ncattrs():
					try:
						if name == "long_name":
							ncvar.setncattr( name , "Bias Corrected " + incfile.variables[cvarX].getncattr(name) )
						else:
							ncvar.setncattr( name , incfile.variables[cvarX].getncattr(name) )
					except AttributeError:
						pass
				
				## And fill
				num_otime = cftime.date2num( otime , incfile.variables[time_axis].units , incfile.variables[time_axis].calendar )
				num_time  = cftime.date2num(  time , incfile.variables[time_axis].units , incfile.variables[time_axis].calendar )
				t0,t1 = num_time[:2]
				idx   = np.array( np.ceil( (num_otime - t0) / (t1 - t0 ) ) , int ).tolist()
				
				## Now loop on zchunks
				for zc in zZ.iter_zchunks():
					
					zc_y,zc_x = zc
					i0y =  zc_y    * zZ.data.chunks[1]
					i1y = (zc_y+1) * zZ.data.chunks[1]
					i0x =  zc_x    * zZ.data.chunks[2]
					i1x = (zc_x+1) * zZ.data.chunks[2]
					
					## And loop on time chunk to limit memory used
					for it in range(0,len(idx),time_chunk):
						it0 = it
						it1 = min( it + time_chunk , len(idx) )
						sel = (idx[it0:it1],slice(i0y,i1y),slice(i0x,i1x),icvar)
						ncvar[it0:it1,i0y:i1y,i0x:i1x] = zZ.data.get_orthogonal_selection(sel)
	
##}}}


##============================================================================##
##                                                                            ##
##  Function and classes to remove                                            ##
##                                                                            ##
##============================================================================##

#class CoordinatesSAVE:##{{{
#	"""
#	XSBCK.Coordinates
#	=================
#	
#	Class used to contain informations about dimensions and coordinates of the
#	dataset.
#	
#	"""
#	
#	def __init__( self , dX , dY , cvarsX = None , cvarsY = None , cvarsZ = None ):##{{{
#		"""
#		XSBCK.Coordinates.__init__
#		==========================
#		
#		Arguments
#		---------
#		dX: [xarray.Dataset]
#			The biased data
#		dY: [xarray.Dataset]
#			The reference data
#		cvarsX: [list]
#			List of variables from X
#		cvarsY: [list]
#			List of variables from Y
#		cvarsZ: [list]
#			List of name of output variables
#		
#		Notes
#		-----
#		cvarsX, cvarsY and cvarsZ must be in the same order. Used when the
#		reference and the biased data don't use the same name.
#		
#		"""
#		
#		## Check the two dataset have the same coordinates
#		coordsX = [key for key in dX.coords]
#		coordsY = [key for key in dY.coords]
#		coordsX.sort()
#		coordsY.sort()
#		
#		if "height" in coordsX:
#			del coordsX[coordsX.index("height")]
#		if "height" in coordsY:
#			del coordsY[coordsY.index("height")]
#		
#		if not all( [key in coordsX for key in coordsY] + [key in coordsY for key in coordsX] ):
#			raise Exception( f"Coordinates of input are differents: ref : {coordsY}, biased : {coordsX}" )
#		
#		self.coords = coordsX
#		
#		## Check that time is in coordinates
#		if not "time" in self.coords:
#			raise Exception( "No time axis detected" )
#		self.time = dX.time
#		
#		## Check the spatial coordinates
#		if not ( any([ s in self.coords for s in ["lat","latitude"]]) and any([ s in self.coords for s in ["lon","longitude"]]) ):
#			raise Exception( "Latitude or longitude are not given" )
#		
#		## Now the variables
#		if (cvarsX is not None and cvarsY is None) or (cvarsY is not None and cvarsX is None):
#			raise Exception( "If cvars is given for the ref (or the biased), cvars must be given for the biased (or the ref)" )
#		
#		if cvarsX is None:
#			self.cvarsX  = [key for key in dX.data_vars if len(dX[key].dims) == 3]
#			self.cvarsX.sort()
#		else:
#			self.cvarsX = cvarsX.split(",")
#		self.dimsX = dX[self.cvarsX[0]].dims
#		self.ny    = dX[self.dimsX[1]].size
#		self.nx    = dX[self.dimsX[2]].size
#		
#		if cvarsY is None:
#			self.cvarsY  = [key for key in dY.data_vars if len(dY[key].dims) == 3]
#			self.cvarsY.sort()
#		else:
#			self.cvarsY = cvarsY.split(",")
#		
#		if cvarsZ is None:
#			self.cvarsZ = list(self.cvarsX)
#		else:
#			self.cvarsZ = cvarsZ.split(",")
#		
#		if not len(self.cvarsY) == len(self.cvarsX) == len(self.cvarsZ):
#			raise Exception( "Different numbers of variables!" )
#		
#		if cvarsX is None and cvarsY is None:
#			if ( not all([cvar in self.cvarsX for cvar in self.cvarsY]) ) or (not all([cvar in self.cvarsY for cvar in self.cvarsX])):
#				raise Exception( "Variables from ref or biased differs" )
#		
#		self.ncvar = len(self.cvarsX)
#		
#		## Now check if a grid mapping exists
#		self.mapping = None
#		if "grid_mapping" in dX[self.cvarsX[0]].attrs:
#			self.mapping = dX[self.cvarsX[0]].attrs["grid_mapping"]
#		
#	##}}}
#	
#	def delete_mapping( self , *args ):##{{{
#		"""
#		XSBCK.Coordinates.delete_mapping
#		================================
#		
#		Used to delete the mapping variables if necessary
#		
#		Arguments
#		---------
#		*args: [xarray.Dataset]
#		
#		Returns
#		-------
#		*args
#		
#		"""
#		if self.mapping is None:
#			return args
#		
#		out = []
#		for a in args:
#			del a[self.mapping]
#			out.append(a)
#		
#		return tuple(out)
#	##}}}
#	
#	def rename_cvars( self , dX , dY ):##{{{
#		"""
#		XSBCK.Coordinates.rename_cvars
#		==============================
#		
#		Rename the cvars of dX and dY such that the name will be cvarsZ
#		
#		Arguments
#		---------
#		dX: [xarray.Dataset]
#		dY: [xarray.Dataset]
#		
#		Returns
#		-------
#		dX: [xarray.Dataset]
#		dY: [xarray.Dataset]
#		
#		"""
#		for cvarX,cvarY,cvarZ in self.cvars:
#			dY = dY.rename( **{ cvarY : cvarZ } )
#			dX = dX.rename( **{ cvarX : cvarZ } )
#		
#		return dX,dY
#	##}}}
#	
#	def summary(self):##{{{
#		"""
#		XSBCK.Coordinates.summary
#		=========================
#		
#		Return a summary of the Coordinates
#		
#		"""
#		lstr = []
#		lstr.append( f"Coordinates: {self.dimsX}" )
#		lstr = lstr + [ f" * {c}" for c in self.coords ]
#		if self.mapping is not None:
#			lstr.append( f"Mapping: {self.mapping}" )
#		lstr.append( "Reference variables:" )
#		lstr = lstr + [ f" * {cvarY}" for cvarY in self.cvarsY ]
#		lstr.append( "Biased variables:" )
#		lstr = lstr + [ f" * {cvarX}" for cvarX in self.cvarsX ]
#		lstr.append( "Output variables:" )
#		lstr = lstr + [ f" * {cvarZ}" for cvarZ in self.cvarsZ ]
#		
#		return "\n".join(lstr)
#	##}}}
#	
#	## Properties ##{{{
#	
#	@property
#	def cvars(self):
#		return [ (cvarX,cvarY,cvarZ) for cvarX,cvarY,cvarZ in zip(self.cvarsX,self.cvarsY,self.cvarsZ) ]
#	
#	##}}}
#	
###}}}
#
#class XZarrSAVE:##{{{
#	"""
#	XSBCK.XZarr
#	===========
#	
#	Class managing a zarr file, to access with the time axis
#	
#	"""
#	
#	def __init__( self , fzarr , dX = None , shape = None , dims = None , coords = None , xchunks = None , persist = False , cvars = None , dtype = "float32" , avail_spatial_mem = None ):##{{{
#		"""
#		XSBCK.XZarr.__init__
#		====================
#		
#		Arguments
#		---------
#		fzarr:
#			The file used as zarr file
#		dX: [xarray.Dataset]
#			Data to copy in the zarr file. If None, a zero file is init with
#			the shape, dims and coords parameters
#		shape:
#			Only used if dX is None
#		dims:
#			Only used if dX is None
#		coords:
#			Only used if dX is None
#		chunks:
#			The chunk of the data
#		persist:
#			If False, the fzarr file is removed when the object is deleted.
#		cvars:
#			List of cvar
#		dtype:
#			The data type, default is float32
#		avail_spatial_mem:
#			Memory limit in octet of a single map (no time and variables). Used
#			to spatially cut the dataset
#		 
#		"""
#		
#		self.fzarr   = fzarr
#		self.persist = persist
#		self.shape   = None
#		self.dims    = None
#		self.coords  = None
#		self.dtype   = dtype
#		self.avail_spatial_mem = avail_spatial_mem
#		self._fmatch = lambda a,b: [ b.index(x) if x in b else None for x in a ]
#		
#		if dX is not None:
#			self._init_from_dX( dX , cvars )
#		else:
#			self._init_from_args( shape , dims , coords )
#		
#		## Find time chunk
#		time_chunk = 4 * 365 + 1
#		try:
#			if isinstance(dX.time.values[0],cftime.DatetimeNoLeap):
#				time_chunk = 365
#			if isinstance(dX.time.values[0],cftime.Datetime360Day):
#				time_chunk = 360
#		except:
#			pass
#		
#		## Find zchunk
#		if avail_spatial_mem is None:
#			self.zchunks = ( time_chunk , len(self.coords[1]) , len(self.coords[2]) , 1 )
#		else:
#			ny = len(self.coords[1])
#			nx = len(self.coords[2])
#			avail_spatial_numb = int( self.avail_spatial_mem / (np.finfo(self.dtype).bits // 8) )
#			zch_ny = int(ny / np.sqrt(avail_spatial_numb)) + 1
#			zch_nx = int(nx / np.sqrt(avail_spatial_numb)) + 1
#			self.zchunks = ( time_chunk , ny // zch_ny , nx // zch_nx , 1 )
#		
#		## Find xchunk
#		self.xchunks = xchunks
#		if xchunks is None:
#			self.xchunks = -1
#		
#		self.data = zarr.open( self.fzarr , mode = "a" , shape = self.shape , chunks = self.zchunks , dtype = self.dtype )
#		if dX is not None:
#			cc = { d : zc for d,zc in zip(dX[self.cvars[0]].dims,self.zchunks) }
#			dX = dX.chunk(cc)
#			for icvar,cvar in enumerate(self.cvars):
#				self.data[:,:,:,icvar] = dX[cvar].values
#	##}}}
#	
#	def _init_from_dX( self , dX , cvars ):##{{{
#		
#		if cvars is None:
#			cvars = [key for key in dX.data_vars if dX[key].ndim > 0]
#		
#		ndim = dX[cvars[0]].ndim
#		for cvar in cvars:
#			if not dX[cvar].ndim == ndim:
#				raise Exception
#		
#		self.shape  = dX[cvars[0]].shape  + (len(cvars),)
#		self.dims   = dX[cvars[0]].dims   + ("cvar",)
#		self.coords = [dX[c] for c in self.dims[:-1]] + [cvars,]
#		if not len(self.shape) == ndim + 1:
#			raise Exception
#		if not len(self.dims) == ndim + 1:
#			raise Exception
#		if not len(self.coords) == ndim + 1:
#			raise Exception
#		
#		##
#		dtypes = list(set([str(dX.dtypes[cvar]) for cvar in self.cvars]))
#		if not len(dtypes) == 1:
#			raise Exception
#		self.dtype = dtypes[0]
#		
#	##}}}
#	
#	def _init_from_args( self , shape , dims , coords ):##{{{
#		
#		if shape is None or dims is None or coords is None:
#			raise Exception("If dX is None, shape, dims and coords must be set!")
#		self.shape  = shape
#		self.dims   = dims
#		self.coords = coords
#		
#	##}}}
#	
#	def copy( self , fzarr , value = np.nan ):##{{{
#		
#		copy = XZarr( fzarr , shape = self.shape , dims = self.dims , coords = self.coords ,
#		              xchunks = self.xchunks , persist = self.persist , dtype = self.dtype ,
#		              avail_spatial_mem = self.avail_spatial_mem
#		            )
#		copy.data[:] = value
#		
#		return copy
#	##}}}
#	
#	def __del__( self ):##{{{
#		if not self.persist:
#			self.clean()
#	##}}}
#	
#	def __str__(self):##{{{
#		
#		out = []
#		out.append( "<XSBCK.XZarr>" )
#		out.append( "Dimensions: (" + ", ".join( [f"{d}: {s}" for d,s in zip(self.dims,self.shape)] ) + ")" )
#		out.append( "zchunks: (" + ", ".join( [f"{d}: {s}" for d,s in zip(self.dims,self.zchunks)] ) + "; " +  "x".join( [str(x) for x in self.data.cdata_shape] ) + " )" )
##		out.append( "xchunks: (" + ", ".join( [f"{d}: {s}" for d,s in zip(self.dims,self.xchunks)] ) + ")" )
#		out.append( "xchunks: " + self.xchunks )
#		out.append( "Coordinates:" )
#		for d,coord in zip(self.dims,self.coords):
#			line = str(coord).split("\n")[-1]
#			if "*" in line:
#				out.append(line)
#			else:
#				out.append( "  * {:{fill}{align}{n}}".format(d,fill=" ",align="<",n=9) )#+ f"({d}) str " + " ".join(coord) )
#		
#		return "\n".join(out)
#	##}}}
#	
#	def __repr__(self):##{{{
#		return self.__str__()
#	##}}}
#	
#	def iter_zchunks(self):##{{{
#		"""
#		Return a generator to iterate over spatial zarr chunks
#		"""
#		
#		return itt.product(range(self.data.cdata_shape[1]),range(self.data.cdata_shape[2]))
#	##}}}
#	
#	def sel_along_time( self , time , zc = None ):##{{{
#		"""
#		XSBCK.XZarr.sel_along_time
#		==========================
#		
#		Arguments
#		---------
#		time:
#			The time values to extract
#		zc:
#			The chunk identifier, given by XSBCK.XZarr.iter_zchunks. If None,
#			all spatial values are returned.
#		
#		Returns
#		-------
#		A chunked xarray.DataArray
#		 
#		"""
#		
#		time_fmt = self.time.sel( time = time ).values.tolist() ## Ensure time and self.time have the save format
#		if not type(time_fmt) is list: time_fmt = [time_fmt]
#		idx      = self._fmatch( time_fmt , self.time.values.tolist() )
#		
#		xchunk = { "time" : -1 , "cvar" : -1 }
#		if zc is None:
#			sel    = (idx,slice(None),slice(None),slice(None))
#			coords = [self.time[idx]] + self.coords[1:]
#			if self.xchunks == -1:
#				xchunk[self.dims[1]] = -1
#				xchunk[self.dims[2]] = -1
#			else:
#				xchunk[self.dims[1]] = int( self.shape[1] / np.sqrt(self.xchunks) )
#				xchunk[self.dims[2]] = int( self.shape[2] / np.sqrt(self.xchunks) )
#		else:
#			zc_y,zc_x = zc
#			i0y =  zc_y    * self.data.chunks[1]
#			i1y = (zc_y+1) * self.data.chunks[1]
#			i0x =  zc_x    * self.data.chunks[2]
#			i1x = (zc_x+1) * self.data.chunks[2]
#			sel = (idx,slice(i0y,i1y,1),slice(i0x,i1x),slice(None))
#			coords = [self.time[idx]] + [self.coords[1][i0y:i1y]] + [self.coords[2][i0x:i1x]] + [self.coords[3]]
#			if self.xchunks == -1:
#				xchunk[self.dims[1]] = -1
#				xchunk[self.dims[2]] = -1
#			else:
#				xchunk[self.dims[1]] = int( (i1y-i0y+1) / np.sqrt(self.xchunks) )
#				xchunk[self.dims[2]] = int( (i1x-i0x+1) / np.sqrt(self.xchunks) )
#		
#		X = xr.DataArray( self.data.get_orthogonal_selection(sel) ,
#		                  dims = self.dims ,
#		                  coords = coords,
#		                ).chunk(
#		                  xchunk
##		                  { "time" : -1 , **{ d : c for d,c in zip(self.dims[1:],self.xchunks[1:])} , "cvar" : -1 }
#		                ).astype(self.dtype)
#		
#		return X
#	##}}}
#	
#	def sel_cvar_along_time( self , time , cvar ):##{{{
#		"""
#		XSBCK.XZarr.sel_cvar_along_time
#		===============================
#		
#		To select only on cvar
#		
#		Arguments
#		---------
#		time:
#			The time values to extract
#		cvar:
#			The climate variable selected
#		
#		Returns
#		-------
#		A NOT chunked xarray.DataArray
#		 
#		"""
#		
#		time_fmt = self.time.sel( time = time ).values.tolist() ## Ensure time and self.time have the save format
#		if not type(time_fmt) is list: time_fmt = [time_fmt]
#		idx      = self._fmatch( time_fmt , self.time.values.tolist() )
#		icvar    = self.cvars.index(cvar)
#		
#		X = xr.DataArray( self.data.get_orthogonal_selection( (idx,slice(None),slice(None),icvar) ).squeeze() ,
#		                  dims = self.dims[:-1] ,
#		                  coords = [self.time[idx]] + self.coords[1:-1]
#		                ).chunk(
#		                  { "time" : -1 , **{ d : -1 for d in self.dims[1:-1] } }
#		                )
#		
#		return X
#	##}}}
#	
#	def set_along_time( self , X , time = None , zc = None ):##{{{
#		"""
#		XSBCK.XZarr.set_along_time
#		==========================
#		
#		To set X at time values in the zarr file
#		
#		Arguments
#		---------
#		X:
#			A data array
#		time:
#			The time values to set
#		zc:
#			The chunk identifier, given by XSBCK.XZarr.iter_zchunks. If None,
#			all spatial values are set.
#		 
#		"""
#		
#		time     = time if time is not None else X.time
#		time_fmt = self.time.sel( time = time ).values.tolist() ## Ensure time and self.time have the save format
#		if not type(time_fmt) is list: time_fmt = [time_fmt]
#		idx      = self._fmatch( time_fmt , self.time.values.tolist() )
#		
#		if zc is None:
#			sel    = (idx,slice(None),slice(None),slice(None))
#		else:
#			zc_y,zc_x = zc
#			i0y =  zc_y    * self.data.chunks[1]
#			i1y = (zc_y+1) * self.data.chunks[1]
#			i0x =  zc_x    * self.data.chunks[2]
#			i1x = (zc_x+1) * self.data.chunks[2]
#			sel = (idx,slice(i0y,i1y,1),slice(i0x,i1x),slice(None))
#		
#		self.data.set_orthogonal_selection( sel , X.values )
#		
#	##}}}
#	
#	def clean(self):##{{{
#		"""
#		XSBCK.XZarr.clean
#		=================
#		Remove the fzarr file
#		 
#		"""
#		if os.path.isdir(self.fzarr):
#			for f in os.listdir(self.fzarr):
#				os.remove( os.path.join( self.fzarr , f ) )
#			os.rmdir(self.fzarr)
#	##}}}
#	
#	## Properties {{{
#	@property
#	def time(self):
#		return self.coords[0]
#	
#	@property
#	def cvars(self):
#		return self.coords[-1]
#	##}}}
#	
###}}}
#
### load_data_SAVE ##{{{
#@log_start_end(logger)
#def load_data_SAVE( kwargs : dict ):
#	"""
#	XSBCK.load_data
#	===============
#	Function used to read data and copy in a temporary zarr file
#	
#	Arguments
#	---------
#	kwargs:
#		dict of all parameters of XSBCK
#	
#	Returns
#	-------
#	zX:
#		XZarr file of the biased dataset
#	zY:
#		XZarr file of the reference dataset
#	coords:
#		Coordinates class of the data
#	"""
#	
#	## Read the data
#	dX = xr.open_mfdataset( kwargs["input_biased"]    , data_vars = "minimal" , coords = "minimal" , compat = "override" , combine_attrs = "drop" )
#	dY = xr.open_mfdataset( kwargs["input_reference"] , data_vars = "minimal" , coords = "minimal" , compat = "override" , combine_attrs = "drop" )
#	
#	## Identify coordinates
#	coords = Coordinates( dX , dY , kwargs["cvarsX"] , kwargs["cvarsY"] , kwargs["cvarsZ"] )
#	dX,dY  = coords.delete_mapping(dX,dY)
#	dX,dY  = coords.rename_cvars(dX,dY)
#	logger.info(coords.summary())
#	
#	## Now find chunks
#	xchunks = kwargs["chunks"]
#	if xchunks == -1:
#		xchunks = kwargs["n_workers"] * kwargs["threads_per_worker"]
#		logger.info(f"xchunks found: {xchunks}")
#	
#	## Find spatial memory available
#	total_memory       = kwargs["total_memory"].o
#	frac_mem_per_array = kwargs["frac_memory_per_array"]
#	max_mem_per_chunk  = SizeOf( f"{int(frac_mem_per_array * total_memory)}o" )
#	max_time = 365 * max( int(kwargs["calibration"][1]) - int(kwargs["calibration"][0]) + 1 , sum(kwargs['window']) )
#	max_cvar = len(coords.cvarsZ)
#	avail_spatial_mem  = SizeOf( f"{int( max_mem_per_chunk.o / ( max_time * max_cvar * (np.finfo('float32').bits // 8) ) )}o" )
#	
#	logger.info( f" * Max mem. per chunk: {max_mem_per_chunk.o}o" )
#	logger.info( f" * Max time step     : {max_time}" )
#	logger.info( f" * Max cvar          : {max_cvar}" )
#	logger.info( f" * Avail Spat. Mem.  : {avail_spatial_mem.o}o" )
#	
#	## Init XZarr
#	zX = XZarr( os.path.join( kwargs["tmp"] , "X.zarr" ) , dX , xchunks = xchunks , cvars = coords.cvarsZ , avail_spatial_mem = avail_spatial_mem.o )
#	zY = XZarr( os.path.join( kwargs["tmp"] , "Y.zarr" ) , dY , xchunks = xchunks , cvars = coords.cvarsZ , avail_spatial_mem = avail_spatial_mem.o )
#	
#	logger.info( f"* About biased data:" )
#	logger.info( f"  => shape  : {str(zX.shape)}" )
#	logger.info( f"  => zchunks: {str(zX.zchunks)}" )
#	logger.info( f"* About reference data:" )
#	logger.info( f"  => shape  : {str(zY.shape)}" )
#	logger.info( f"  => zchunks: {str(zY.zchunks)}" )
#	
#	## Free memory
#	del dX
#	del dY
#	gc.collect()
#	
#	
#	
#	return zX,zY,coords
###}}}
#
### save_data_SAVE ##{{{ 
#
#@log_start_end(logger)
#def save_data_SAVE( zX : XZarr , zZ : XZarr , kwargs : dict ):
#	"""
#	XSBCK.save_data
#	===============
#	Function used to read the XZarr file of the corrected dataset and rewrite
#	in netcdf.
#	
#	Arguments
#	---------
#	zX:
#		XZarr file of the biased dataset
#	zZ:
#		XZarr file of the corrected dataset
#	kwargs:
#		dict of all parameters of XSBCK
#	
#	Returns
#	-------
#	None
#	"""
#	
#	## Build mapping between cvarsX and cvarsZ
#	mcvars = { x : z for x,z in zip(zX.cvars,zZ.cvars) }
#	
#	tstart = kwargs["start_year"]
#	tend   = kwargs["end_year"]
#	
#	## Loop on input files
#	for ifile in kwargs["input_biased"]:
#		
#		## ofile
#		ifile  = os.path.basename(f)
#		prefix = f"{avar}_{kwargs['method']}"
#		if cvarX in ifile:
#			ofile = ifile.replace(cvarX,prefix)
#		else:
#			ofile = f"{prefix}_{ifile}"
#		
#		
#		## 
#	
#	raise Exception("Hard stop for dev")
#	
#	
#	for f in kwargs["input_biased"]:
#		
#		logger.info( f" * {os.path.basename(f)}" )
#		
#		## Start by read the time units (xarray deletes it)
#		with netCDF4.Dataset( f , mode = "r" ) as ncfile:
#			attrs = { v : {a : ncfile.variables[v].getncattr(a) for a in ncfile.variables[v].ncattrs()} for v in ncfile.variables }
#			time_units = ncfile.variables["time"].units
#		
#		## Load data with xarray (easier to load attributes)
#		dX = xr.open_dataset(f)
#		
#		## Check year
#		if dX.time.dt.year[-1] < int(tstart):
#			logger.info( f"     => End year {int(dX.time.dt.year[-1])} < start year {tstart}, skip." )
#			continue
#		if dX.time.dt.year[0] > int(tend):
#			logger.info( f"     => Start year {int(dX.time.dt.year[0])} < end year {tend}, skip." )
#			continue
#		otime = dX.time.sel( time = slice(tstart,tend) )
#		
#		## Find the variable
#		for cvarX,_,cvarZ in coords.cvars:
#			if cvarX in dX: break
#		
#		## Build output
#		avar  = cvarZ + "Adjust"
#		odata = dX.sel( time = otime ).rename( { cvarX : avar } )
#		odata[avar] = dZ.sel_cvar_along_time( otime , cvarZ )
#		
#		## Add variables attributes
#		for v in attrs:
#			if v == cvarX:
#				odata[avar].attrs = attrs[v]
#			else:
#				odata[v].attrs = attrs[v]
#			if v == "time":
#				for k in ["units","calendar"]:
#					if k in odata[v].attrs:
#						del odata[v].attrs[k]
#		odata[avar].attrs["long_name"] = "Bias Adjusted " + odata[avar].attrs["long_name"]
#		
#		## Add BC attributes
#		odata.attrs["bc_creation_date"] = str(dt.datetime.utcnow())[:19] + " (UTC)"
#		odata.attrs["bc_method"]        = kwargs["method"]
#		odata.attrs["bc_period_calibration"] = "/".join( [str(x) for x in kwargs["calibration"]] )
#		odata.attrs["bc_window"]        = ",".join( [str(x) for x in kwargs["window"]] )
#		odata.attrs["bc_reference"]     = build_reference(kwargs["method"])
#		odata.attrs["bc_pkgs_versions"] = ", ".join( [f"XSBCK:{version}"] + [f"{name}:{pkg.__version__}" for name,pkg in zip(["SBCK","numpy","xarray","dask","zarr"],[SBCK,np,xr,dask,zarr]) ] )
#		
#		## The encoding
#		encoding = {}
#		for key in odata.variables:
#			if key == "time":
#				encoding[key] = { "dtype" : "double" , "zlib" : True , "complevel" : 5 , "chunksizes" : (1,) , "calendar" : attrs["time"]["calendar"] , "units" : attrs["time"]["units"] }
#			elif key == "time_bnds":
#				encoding[key] = { "dtype" : "double" }
#			elif key == avar:
#				encoding[key] = { "dtype" : str(odata[key].dtype) , "zlib" : True , "complevel" : 5 , "chunksizes" : (1,) + odata[key].shape[1:] }
#			else:
#				encoding[key] = { "dtype" : str(odata[key].dtype) , "zlib" : True , "complevel" : 5 , "chunksizes" : odata[key].shape }
#		
#		## ofile
#		ifile  = os.path.basename(f)
#		prefix = f"{avar}_{kwargs['method']}"
#		if cvarX in ifile:
#			ofile = ifile.replace(cvarX,prefix)
#		else:
#			ofile = f"{prefix}_{ifile}"
#		
#		## And save
#		logger.info( f"     => {ofile}" )
#		odata.to_netcdf( os.path.join( kwargs["output_dir"] , ofile ) , encoding = encoding )
###}}}
#
#
