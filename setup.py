
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

import os

## Start by import release details
cpath = os.path.dirname(os.path.abspath(__file__)) ## current-path
with open( os.path.join( cpath , "XSBCK" , "__release.py" ) , "r" ) as f:
	lines = f.readlines()
exec("".join(lines))

## Required elements
author           = ", ".join(authors)
author_email     = ", ".join(authors_email)
package_dir      = { "XSBCK" : "XSBCK" }
requires         = [ "numpy" , "scipy" , "xarray" , "netCDF4(>=1.6.0)" , "cftime" , "SBCK(>=0.5.0a29)" , "dask(>=2022.11.0)" , "zarr" ]
scripts          = ["scripts/xsbck"]
keywords         = []
platforms        = ["linux","macosx"]
packages         = [
	"XSBCK"
	]

## Now the setup
from distutils.core import setup

setup(  name             = name,
		version          = version,
		description      = description,
		long_description = long_description,
		author           = author,
		author_email     = author_email,
		url              = src_url,
		packages         = packages,
		package_dir      = package_dir,
		requires         = requires,
		scripts          = scripts,
		license          = license,
		keywords         = keywords,
		platforms        = platforms,
		include_package_data = True
     )

