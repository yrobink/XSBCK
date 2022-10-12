
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

##############
## Packages ##
##############

import os
import random
import string


###############
## Functions ##
###############

def build_tmp_dir( **kwargs ):##{{{
	
	if kwargs["tmp"] is not None:
		return kwargs["tmp"]
	
	is_not_valid = True
	while is_not_valid:
		tmp = os.path.join( kwargs["tmp_base"] , "XSBCK_" + "".join( random.choices( string.ascii_uppercase + string.digits , k = 30 ) ) )
		is_not_valid = os.path.isdir(tmp)
	
	os.makedirs(tmp)
	return tmp
##}}}

def delete_tmp_dir( **kwargs ):##{{{
	tmp = kwargs["tmp"]
	for f in os.listdir(tmp):
		os.remove( os.path.join( tmp , f ) )
	
	if isinstance( kwargs["tmp_base"] , bool ) and not kwargs["tmp_base"]:
		os.rmdir(tmp)
	
##}}}

