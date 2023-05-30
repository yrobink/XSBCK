
# XSBCK: Executable SBCK

## What is XSBCK ?

XSBCK is a command line interface to SBCK, to correct very large dataset. XSBCK
is designed to work:
- On a personnal computer,
- On a cluster or supercomputer.

XSBCK has been tested until a global correction (the entire world by 0.25°,
between 1850 and 2100), on a 16 cores and 32GB of RAMs (~3 days in this case).


## How to install it ?

Start by install all dependencies:
- numpy
- scipy
- xarray
- netCDF4(>=1.6.0)
- cftime
- dask(>=2022.11.0)
- zarr
- SBCK(>=1.0.0)

The [SBCK](https://github.com/yrobink/SBCK-python) dependency can be installed
from [sources](https://github.com/yrobink/SBCK-python) or from
[PyPI](https://pypi.org/project/SBCK):

~~~bash
pip3 install SBCK
~~~

You can now install XSBCK from PyPI:

~~~bash
pip3 install XSBCK
~~~

or from sources:

~~~bash
git clone https://github.com/yrobink/XSBCK.git
cd XSBCK
pip3 install .
~~~


## How to use it ?

Although coded in python, XSBCK is a command line tools. The first step is
(obviously) to read the documentation:

~~~bash
xsbck --help
~~~

In a nutshell, the following command:


~~~bash
xsbck --log -iref $ipathY/*.nc -ibias $ipathX/*.nc -odir $opathZ/\\
   --method CDFt\\
   --n-workers 5 --threads-per-worker 2\\
   --memory-per-worker 2GB\\
   --window 5,10,5\\
   --cvarsX tas,tasmin,tasmax,pr\\
   --cvarsY tas,tasmin,tasmax,prtot\\
   --cvarsZ tas,tasmin,tasmax,prtot\\
   --ppp prtot,LogLin,SSR\\
   --ppp '_all_,PreserveOrder[cols=tasminAdjust+tasAdjust+tasmaxAdjust]'\
   --ppp '_all_,NotFiniteAnalog[analog_var=tas+prtot,threshold=0.05]'
~~~

corrects all the biased file in the folder `$ipathX` with respect to the refence
stored in the folder `$ipathY`. Corrections are save in the folder `$opathZ`.
In addition:

- `--log` enable the log
- `--method` the method used
- `--n-workers` numbers of CPU
- `--threads-per-worker` Number of threads per worker
- `--memory-per-worker` Memory per worker
- `--window` The size of moving window. `5,10,5` corresponds to a window of length 5+10+5 years for the fit, and the central 10 years for the predict.
- And read the doc for others arguments.


## How to cite it ?

If you produced and published some corrected dataset with XSBCK, consider to
cite it with the following DOI:
- SBCK: [DOI:223192066](https://zenodo.org/badge/latestdoi/223192066)
- XSBCK: TODO

## License

Copyright(c) 2022, 2023 Yoann Robin

This file is part of XSBCK.

XSBCK is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

XSBCK is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with XSBCK.  If not, see <https://www.gnu.org/licenses/>.


