
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
--log [loglevel,logfile]
    Set the log level, default is 'warning'. If '--log' is passed without
    arguments, 'debug' is used. The default output is the console, and the
    second argument is a file to redirect the logs.
--help
    Ask to see the documentation
--input-reference, -iref, -iY
    A list of netcdf input files, as reference
--input-biased, -ibias, -iX
    A list of netcdf input files, as model to correct
--output-dir, -odir, -oZ
    A path to save the corrected data. The name of input-biased is used.
--method
    Bias correction method
--n-workers [int]
    CPU numbers
--threads-per-worker [int]
    Threads numbers per CPU
--memory-per-worker
    Memory per worker, default is 'auto'
--frac-memory-per-array
    Fraction of the total memory used per array. 4 arrays are stored in the
    memory:
     * The calibration period, for reference and biased data,
     * The projection period, for biased data and corrected data
    By default, this parameter is 0.2 = 20%, i.e. at most 4*20% = 80% of the
    memory is used to store data, and 20% is used for intermediate operations.
    If a memory error occurs, try to decrease this parameter.
--total-memory
    Total available memory, used if '--memory-per-worker' is 'auto' and
    '--total-memory' is not 'auto'
--tmp [str, default is tempfile.gettempdir()]
    Base path to build a random temporary folder.
--window [w_left,w_predict,w_right, default=5,10,5]
    The moving window for the correction: w_left + w_predict + w_right is the
    fit window, whereas w_predict is the correction part.
--chunks [chunk_lat,chunk_lon]
    Spatial chunk of the dataset. If not given, default value is:
    chunk_lat = int(nlat / sqrt(n_threads))
    chunk_lon = int(nlon / sqrt(n_threads))
--calibration [default=1976/2005]
    Calibration period, in the form 'year_start/year_end'.
--start-year [default=First year of the input data]
    Starting year for the correction.
--cvarsX
    Model climate variables, in the form var0,var1,var2,...
--cvarsY
    Reference climate variables, in the form var0,var1,var2,...
--cvarsZ
    Name of output variables, in the form var0,var1,var2,... If not given,
    cvarsX is used.
--ppp
    A list of PrePostProcessing methods.
--disable-dask
    Use this option to disable the dask client.


Examples
--------
xsbck --log -iref $ipathY/*.nc -ibias $ipathX/*.nc -odir $opathZ/\\
   --method R2D2-L-NV-2L\\
   --n-workers 5 --threads-per-worker 2\\
   --memory 2GB\\
   --window 5,50,5\\
   --cvarsX tas,tasmin,tasmax,pr\\
   --cvarsY tas,tasmin,tasmax,prtot\\
   --cvarsZ tas,tasmin,tasmax,prtot\\
   --ppp '_all_,LogLin[cols=prtot+tasmin+tasmax]'\\
   --ppp prtot,SSR\\
   --ppp '_all_,DiffRef[ref=tas,upper=tasmax,lower=tasmin],NotFiniteAnalog[analog_var=tas+prtot,threshold=0.05]'


About ppp
---------
The ppp arguments are PrePostProcessing class, in the SBCK.ppp package. You can
use it in XSBCK with the following rule:
- ppp are applied as a function (so right to left) in the order of the
  arguments passed to xsbck.  For the behind example, before the correction,
  ppp are applied in this order:
  * NotFiniteAnalog
  * DiffRef
  * SSR
  * LogLin
  And in the reverse order after the correction.
- a ppp:
  * start with the variable to applied,
  * and after the ',', the list of SSR to apply is given.
  * You can use [] to pass arguments to the ppp,
  * An you can use _all_ as variable to indicates the ppp applied to all
  variables.
In the example behind, we use the ppp 'NotFiniteAnalog' to replace invalid
values in tasmin, tasmax by the closer analog defined by tas and prtot, if the
proportion is lower < 5%. Then, tasmax is corrected as tasmax-tas, and tasmin
as tas-tasmin. The SSR is used for prtot, and the LogLin function is applied to
SSR(prtot), tasmax-tas and tas-tasmin (because they are non zero).

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

