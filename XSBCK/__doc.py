
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
--help
    Ask to see the documentation
--log [loglevel,logfile]
    Set the log level, default is 'warning'. If '--log' is passed without
    arguments, 'debug' is used. The default output is the console, and the
    second argument is a file to redirect the logs.

--input-reference, -iref, -iY
    A list of netcdf input files, as reference
--input-biased, -ibias, -iX
    A list of netcdf input files, as model to correct
--output-dir, -odir, -oZ
    A path to save the corrected data. The name of input-biased is used.
--tmp [str, default is tempfile.gettempdir()]
    Base path to build a random temporary folder, used to store temporary data.
    Note that the disk containing this folder must be large enough to store
    biased, reference data and correction in an uncompressed zarr file.

--method
    Bias correction method. See methods section.
--ppp
    A list of PrePostProcessing methods. See ppp section.

--cvarsX
    Model climate variables, in the form var0,var1,var2,...
--cvarsY
    Reference climate variables, in the form var0,var1,var2,...
--cvarsZ
    Name of output variables, in the form var0,var1,var2,... If not given,
    cvarsX is used.

--window [w_left,w_predict,w_right, default=5,10,5]
    The moving window for the correction: w_left + w_predict + w_right is the
    fit window, whereas w_predict is the correction part.
--calibration [default=1976/2005]
    Calibration period, in the form 'year_start/year_end'.
--start-year [default=First year of the input data]
    Starting year for the correction.
--end-year [default=Last year of the input data]
    End year for the correction. Note that if the window is larger than
    'end-year - start-year', a larger period is corrected, but just the period
    between start and end year is saved.

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
    By default, this parameter is 0.15 = 15%, i.e. at most 4*15% = 60% of the
    memory is used to store data, and 40% is used for intermediate operations.
    If a memory error occurs, try to decrease this parameter.
--total-memory
    Total available memory, used if '--memory-per-worker' is 'auto' and
    '--total-memory' is not 'auto'
--disable-dask
    Use this option to disable the dask client.


Examples
--------
xsbck --log -iref $ipathY/*.nc -ibias $ipathX/*.nc -odir $opathZ/\\
   --method CDFt\\
   --n-workers 5 --threads-per-worker 2\\
   --memory-per-worker 2GB\\
   --window 5,10,5\\
   --cvarsX tas,tasmin,tasmax,pr\\
   --cvarsY tas,tasmin,tasmax,prtot\\
   --cvarsZ tas,tasmin,tasmax,prtot\\
   --ppp prtot,LogLin,SSR\\
   --ppp '_all_,PreserveOrder[cols=tasminAdjust+tasAdjust+tasmaxAdjust]'\\
   --ppp '_all_,NotFiniteAnalog[analog_var=tas+prtot,threshold=0.05]'


About netcdf files
------------------
- The input (for reference and biased data) netcdf files must follow the CF
  convention, i.e. read:
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html
- The output netcdf copy the names and metadata of the input biased files, and
  add the following global attributes:
    * bc_creation_date: bias correction date, iso format, UTC time
    * bc_method: method used,
    * bc_period_calibration: calibration period
    * bc_window: the window,
    * bc_reference: the reference if available
    * bc_pkgs_versions: version of packages


About methods
-------------
Note all bias correction method use the following nomenclature to describes its
properties `METHOD-X-YV-ZL`, where X, Y and Z are:
- X: possibles values are L (for Local) or S (for Spatial). In XSBCK, the
  implementation of methods force the value L.
- Y: 1 (univariate methid) or N (inter-variable method)
- Z: Length of the temporal correction.
Currently, two methods are available:
- CDFt [1], a quantile mapping based method, univariate and non-stationary. If
  you don't know what you do, use it. CDF-t is always CDFt-L-1V-0L.
- R2D2 [2], a multivariate extention of CDF-t, reshuffling the ranks w.r.t. to
  specifics variables to correct the inter-variables dependence. The
  conditionning variables can be set with the argument 'col_cond'. This method
  tends to degrate the spatial and temporal structure, so use it with caution.
  R2D2 can takes into account of the temporal structure, you can enable this
  option by modifying the lag. For example, R2R2-L-NV-2L[col_cond=tas] corrects
  2 lags, i.e. by 3-days block.


About ppp
---------
The ppp arguments are PrePostProcessing class, in the SBCK.ppp package. You can
use it in XSBCK with the following rule:
- ppp are applied as a function (so right to left) in the order of the
  arguments passed to xsbck.  For the behind example, before the correction,
  ppp are applied in this order:
  * NotFiniteAnalog
  * PerserveOrder
  * SSR
  * LogLin
  And in the reverse order after the correction.
- a ppp:
  * start with the variable to applied,
  * and after the ',', the list of SSR to apply is given.
  * You can use [] to pass arguments to the ppp,
  * An you can use _all_ as variable to indicate the ppp applied to all
  variables.
In the example behind, we use the ppp 'NotFiniteAnalog' to replace invalid
values in tasmin, tasmax by the closer analog defined by tas and prtot, if the
proportion is lower < 5%. Then, if tasmax < tas after the correction, the two
values are swapped (and the same for tasmin). The SSR is used for prtot, and
the LogLin function is applied to SSR(prtot).


References
----------
[1] Michelangeli, P.‐A. et al. (2009). “Probabilistic downscaling approaches :
    Application to wind cumulative distribution functions”. In : Geophys. Res.
    Lett. 36.11. DOI : 10.1029/2009GL038401.
[2] Vrac, M. et S. Thao (2020). “R2D2 v2.0 : accounting for temporal dependences
    in multivariate bias correction via analogue rank resampling”. In : Geosci.
    Model Dev. 13.11, p. 5367‐5387. DOI : 10.5194/gmd-13-5367- 2020.


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


