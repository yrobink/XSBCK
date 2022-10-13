# XSBCK
Executable SBCK

Currently in pre-pre-pre alpha version, so don't use. Ths goal is to have a command like this:

~~~bash
xsbck --log -iref $ipathY/*.nc -ibias $ipathX/*.nc -odir $opathZ --method R2D2-L-NV-2L --n-workers 40 --threads-per-worker 1 --memory 4GB --window 5,10,5
~~~

where:

- `--log` enable the log
- `-iref` is a list of reference files for correction
- `-ibias` is a list of biased files to correct
- `-odir` output directory for correction
- `--method` the method used
- `--n-workers` numbers of CPU
- `--threads-per-worker` Number of threads per worker
- `--memory` Memory per worker
- `--window` The size of moving window. `5,10,5` corresponds to a window of length 5+10+5 years for the fit, and the central 10 years for the predict.


