
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

from .__release import version
from .__release import long_description
from .__release import license
from .__release import license_txt
from .__release import src_url
from .__release import authors_doc

###############
## Variables ##
###############

doc = """\

XSBCK ({})
{}

{}

Input parameters
----------------
--log Nothing, 'debug', 'info', 'warning', 'error' or 'critical'
    Set the log level, default is 'warning'. If '--log' is passed without
    arguments, 'debug' is used.
--help
    Ask to see the documentation
--input-reference, -iref, -iY
    A list of netcdf input files, as reference
--input-biased, -ibias, -iX
    A list of netcdf input files, as model to correct
--output-corrected, -ocorr, -oZ
    A list of netcdf output files
--method
    Bias correction method
--n-workers [int]
    CPU numbers
--threads-per-worker [int]
    Threads numbers per CPU
--memory
    Memory per worker
--tmp-base [str, default= '/tmp/']
    Base path to build a random temporary folder. Not used if '--tmp' is set.
--tmp [str, default is None]
    Temporary folder used. Optional
--window [w_left,w_predict,w_right, default=5,10,5]
    The moving window for the correction: w_left + w_predict + w_right is the
    fit window, whereas w_predict is the correction part.
--calibration [default= "1976/2005]
    Calibration period, in the form 'year_start/year_end'.
--disable-dask
    Use this option to disable the dask client.
--cvarsX
    Model climate variables, in the form var0,var1,var2,...
--cvarsY" , default = None )
    Reference climate variables, in the form var0,var1,var2,...

Examples
--------
xsbck --log --input-obs in1o.nc in2o.nc in3o.nc --input-model in1m.nc in2m.nc in3m.nc --output out1.nc out2.nc out3.nc --method R2D2-L-NV-2L --n-workers 5 --threads-per-worker 2 --memory 3gb


License {}
{}
{}

Sources and author(s)
---------------------
Sources   : {}
Author(s) : {}
""".format( version , "=" * (12+len(version)) ,
            long_description,
            license , "-" * ( 8 + len(license) ) , license_txt ,
            src_url , authors_doc )

