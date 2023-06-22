
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

import itertools as itt
import logging

import netCDF4
import cftime
import zarr
import datetime as dt

import numpy as np
import xarray as xr

from .__utils import delete_hour_from_time_axis
from .__utils import CalendarInfos
from .__utils import time_match

## Init logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#############
## Classes ##
#############

class XZarr:
	"""
	XSBCK.XZarr
	===========
	
	Class managing a zarr file, to access with the time axis
	
	"""
	
	def __init__( self , fzarr , persist = False ):##{{{
		"""
		HDAR.XZarr.__init__
		===================
		
		Arguments
		---------
		fzarr:
			The file used as zarr file
		persist:
			If False, the fzarr file is removed when the object is deleted.
		"""
		
		self.fzarr   = fzarr
		self.persist = persist
		
		self.zarr_chunks = None
		self.n_cpus      = None
		
		self.shape  = None
		self.dims   = None
		self.coords = None
		
		self.dtype  = None
		self.data   = None
		self.ifiles = None
	##}}}
	
	## @staticmethod.open_files ##{{{
	@staticmethod
	def open_files( fzarr , ifiles , cvars , avail_spatial_mem , time_axis = "time" , persist = False , n_cpus = -1 ):
		
		## Init the xzarr
		xzarr = XZarr( fzarr = fzarr , persist = persist )
		xzarr.n_cpus = n_cpus
		
		## Loop on files to find parameters
		time     = []
		calendar = None
		y        = None
		x        = None
		dtype    = "float32"
		for ifile in ifiles:
			
			with netCDF4.Dataset( ifile , mode = "r" ) as ncfile:
				
				## Read the time axis
				ccalendar = ncfile.variables[time_axis].getncattr("calendar")
				units     = ncfile.variables[time_axis].getncattr("units")
				ctime     = cftime.num2date( ncfile.variables[time_axis] , units = units , calendar = ccalendar )
				
				if calendar is None:
					calendar = CalendarInfos( name = ccalendar )
				
				if not ccalendar in calendar.names:
					raise Exception( f"Incoherent time axis, two calendars detected: '{calendar.name}' and '{ccalendar}'" )
				
				time = time + ctime.tolist()
				
				## Now check the variable
				id_cvar = None
				for cvar in cvars:
					if cvar in ncfile.variables:
						id_cvar = cvar
				if id_cvar is None:
					raise Exception( f"The file {ifile} do not contains any know variables " + "{}".format( ",".join(ncfile.variables) ) )
				ncvar = ncfile.variables[id_cvar]
				dtype = str(ncvar.datatype)
				
				## Spatial coords
				id_y = np.array(ncfile.variables[ncvar.dimensions[1]])
				id_x = np.array(ncfile.variables[ncvar.dimensions[2]])
				
				if y is None: y = id_y
				if x is None: x = id_x
				
				if (not y.size == id_y.size) or (not x.size == id_x.size):
					raise Exception( f"Incoherent spatial size: {ifile}" )
		
		time  = list(set(time))
		time.sort()
		dtime = delete_hour_from_time_axis(time)
		dtime = list(set(dtime))
		dtime.sort()
		time  = xr.DataArray( dtime , dims = ["time"] , coords = [dtime] )
		time_chunk = calendar.chunk
		
		## Spatial zarr chunks
		avail_spatial_numb = int( avail_spatial_mem.o / (np.finfo(dtype).bits // avail_spatial_mem.bits_per_octet) )
		ny     = y.size
		nx     = x.size
		zch_ny = max( int(ny / np.sqrt(avail_spatial_numb)) , 1 )
		zch_nx = max( int(nx / np.sqrt(avail_spatial_numb)) , 1 )
		
		## Set the total zarr chunks
		xzarr.zarr_chunks = [ time_chunk , ny // zch_ny , nx // zch_nx , 1 ]
		
		## Init the dimensions and coordinates
		xzarr.dtype  = dtype
		xzarr.shape  = [time.size,y.size,x.size,len(cvars)]
		xzarr.dims   = ["time","y","x","cvar"]
		xzarr.coords = [time,y,x,cvars]
		
		## Init the zarr file
		xzarr.data  = zarr.open( xzarr.fzarr , mode = "w" , shape = xzarr.shape , chunks = xzarr.zarr_chunks , dtype = xzarr.dtype )
		
		## And copy netcdf file to zarr file
		xzarr.ifiles = ifiles
		for ifile in ifiles:
			with netCDF4.Dataset( ifile , "r" ) as ncfile:
				
				## Find the cvar
				for icvar,cvar in enumerate(cvars):
					if cvar in ncfile.variables:
						break
				
				## Time idx
				ftime = cftime.num2date( ncfile.variables[time_axis] , ncfile.variables[time_axis].units , ncfile.variables[time_axis].calendar )
				ftime = delete_hour_from_time_axis(ftime)
				idx   = time_match( ftime , time )
				
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
						M   = ncfile.variables[cvar][it:(it+time_chunk),i0y:i1y,i0x:i1x]
						
						try:
							M = np.where( ~M.mask , np.array(M) , np.nan )
						except:
							M = np.array(M)
							logger.warning( "Problem with the fill_value of the input netcdf" )
						xzarr.data.set_orthogonal_selection( sel , M )
		
		return xzarr
	##}}}
	
	def copy( self , fzarr , value = np.nan , persist = False ):##{{{
		
		## Init the copy
		copy = XZarr( fzarr = fzarr , persist = persist )
		
		## Copy the attributes
		copy.zarr_chunks = list(self.zarr_chunks)
		copy.n_cpus      = self.n_cpus
		
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
		HDAR.XZarr.clean
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
		
		size = SizeOf( n = int(np.prod(self.shape) * np.finfo(self.dtype).bits) , unit = "b" )
		
		out = []
		out.append( "<HDAR.XZarr>" )
		out.append( f"File: {self.fzarr}" )
		out.append( "Dimensions: (" + ", ".join( [f"{d}: {s}" for d,s in zip(self.dims,self.shape)] ) + ")" )
		out.append( f"Size: {size}" )
		out.append( f"Type: {self.dtype}" )
		out.append( f"zarr_chunks: {self.zarr_chunks}" )
		out.append( f"n_cpus: {self.n_cpus}" )
		out.append( "Coordinates:" )
		for d,coord in zip(self.dims,self.coords):
			line = str(coord).split("\n")[-1]
			if "*" in line:
				out.append(line)
			else:
				C = " ".join([str(x) for x in coord])
				if len(C) > 24:
					C = C.split(" ")[0] + " ... " + C.split(" ")[-1]
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
		HDAR.XZarr.sel_along_time
		==========================
		
		Arguments
		---------
		time:
			The time values to extract
		zc:
			The chunk identifier, given by HDAR.XZarr.iter_zchunks. If None,
			all spatial values are returned.
		
		Returns
		-------
		A chunked xarray.DataArray
		 
		"""
		
		idx = time_match( self.time.sel( time = time ) , self.time )
		dask_chunks = { "time" : -1 , "cvar" : -1 }
		if zc is None:
			sel    = (idx,slice(None),slice(None),slice(None))
			coords = [self.time[idx]] + self.coords[1:]
			if self.n_cpus < 2:
				dask_chunks[self.dims[1]] = -1
				dask_chunks[self.dims[2]] = -1
			else:
				dask_chunks[self.dims[1]] = int( self.shape[1] / np.sqrt(self.n_cpus) )
				dask_chunks[self.dims[2]] = int( self.shape[2] / np.sqrt(self.n_cpus) )
		else:
			zc_y,zc_x = zc
			i0y =  zc_y    * self.data.chunks[1]
			i1y = (zc_y+1) * self.data.chunks[1]
			i0x =  zc_x    * self.data.chunks[2]
			i1x = (zc_x+1) * self.data.chunks[2]
			sel = (idx,slice(i0y,i1y,1),slice(i0x,i1x),slice(None))
			coords = [self.time[idx]] + [self.coords[1][i0y:i1y]] + [self.coords[2][i0x:i1x]] + [self.coords[3]]
			if self.n_cpus < 2:
				dask_chunks[self.dims[1]] = -1
				dask_chunks[self.dims[2]] = -1
			else:
				dask_chunks[self.dims[1]] = int( (i1y-i0y+1) / np.sqrt(self.n_cpus) )
				dask_chunks[self.dims[2]] = int( (i1x-i0x+1) / np.sqrt(self.n_cpus) )
		
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
		HDAR.XZarr.sel_cvar_along_time
		===============================
		
		To select only on cvar
		
		Arguments
		---------
		time:
			The time values to extract
		cvar:
			The climate variable selected
		zc:
			The chunk identifier, given by HDAR.XZarr.iter_zchunks. If None,
			all spatial values are returned.
		
		Returns
		-------
		A NOT chunked xarray.DataArray
		 
		"""
		
#		time_fmt = self.time.sel( time = time ).values.tolist() ## Ensure time and self.time have the save format
#		if not type(time_fmt) is list: time_fmt = [time_fmt]
#		idx   = self._fmatch( time_fmt , self.time.values.tolist() )
		idx   = time_match( self.time.sel( time = time ) , self.time )
		icvar = self.cvars.index(cvar)
		
		
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
		HDAR.XZarr.set_along_time
		==========================
		
		To set X at time values in the zarr file
		
		Arguments
		---------
		X:
			A data array
		time:
			The time values to set
		zc:
			The chunk identifier, given by HDAR.XZarr.iter_zchunks. If None,
			all spatial values are set.
		 
		"""
		
		time = time if time is not None else X.time
		idx  = time_match( self.time.sel( time = time ) , self.time )
		
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

