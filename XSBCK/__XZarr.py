
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

import netCDF4
import cftime
import zarr

import numpy as np
import xarray as xr

from .__utils import delete_hour_from_time_axis


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
		self.dtime  = None
		
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
		xzarr.dtime  = delete_hour_from_time_axis(xzarr.coords[0])
		dtime        = xzarr.dtime
		xzarr.coords[0] = xr.DataArray( dtime , dims = [time_axis] , coords = { time_axis : dtime } )
		
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
				itime     = delete_hour_from_time_axis( itime )
				num_itime = cftime.date2num( itime , ncfile.variables[time_axis].units , ncfile.variables[time_axis].calendar )
				num_time  = cftime.date2num( dtime , ncfile.variables[time_axis].units , ncfile.variables[time_axis].calendar )
				if not np.unique(np.diff(num_itime)).size == 1 or  not np.unique(np.diff(num_time )).size == 1:
					raise Exception(f"Missing values or invalid time axis in the file {ifile}")
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
						M   = ncfile.variables[xcvar][it:(it+time_chunk),i0y:i1y,i0x:i1x]
						
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
		copy.dask_chunks = self.dask_chunks
		
		copy.shape  = list(self.shape)
		copy.dims   = list(self.dims)
		copy.coords = list(self.coords)
		copy.dtime  = self.dtime
		
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
	
