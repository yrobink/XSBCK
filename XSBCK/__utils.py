
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

## Package
##########

import logging
from .__logs import log_start_end

## Init logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


## Functions
############

## Classes
##########

class SizeOf:##{{{
	"""
	XSBCK.SizeOf
	============
	
	Class to manage units size for memory
	
	1o == 1octet == 1Byte == 1B == 8 bits == 8b
	
	1kB = 1000^1B
	1MB = 1000^2B
	1GB = 1000^3B
	1TB = 1000^4B
	
	1kiB = 1024^1B
	1MiB = 1024^2B
	1GiB = 1024^3B
	1TiB = 1024^4B
	
	"""
	
	def __init__( self , s ):##{{{
		
		## Check is str
		if not isinstance(s,str):
			raise Exception
		
		s = s.replace(" ","")
		
		## Start with unit
		self.unit = s[-1]
		if not self.unit.lower() in ["b","o"]:
			raise Exception(f"Bad unit: {self.unit}")
		
		## Others values
		if s[-2] == "i":
			self.ibase = "i"
			self.base = 1024
			if s[-3].lower() in ["k","m","g","t"]:
				scale = s[-3]
				value = s[:-3]
			else:
				scale = ""
				value = s[:-2]
		else:
			self.ibase = ""
			self.base = 1000
			if s[-2].lower() in ["k","m","g","t"]:
				scale = s[-2]
				value = s[:-2]
			else:
				scale = ""
				value = s[:-1]
		
		## Check value
		try:
			try:
				value = int(value)
			except:
				value = float(value)
		except:
			raise Exception(f"Value error:{value}")
		
		self.scale = scale
		self.value = value
		
		self.bits = self.value * self.base**self.iscale
		if not self.unit == "b":
			self.bits = self.bits * self.bits_per_byte
		if not 10 * int(self.bits) == int(10*self.bits):
			raise Exception(f"Value is a subdivision of a bit, it is not possible! b = {self.bits}" )
		self.bits = int(self.bits)
	##}}}
	
	def __repr__( self ):##{{{
		return self.__str__()
	##}}}
	
	def __str__( self ):##{{{
		return f"{self.value}{self.scale}{self.ibase}{self.unit} == {self.bits}b"
#		out = []
#		out.append( f"{self.value}{self.scale}{self.ibase}{self.unit} == {self.bits}b" )
#		out.append( f" * unit : {self.unit}" )
#		out.append( f" * base : {self.base}" )
#		out.append( f" * scale: {self.scale}" )
#		out.append( f" * value: {self.value}" )
#		
#		return "\n".join(out)
	##}}}
	
	## Properties ##{{{
	
	@property
	def bits_per_byte(self):
		return 8
	
	def bits_per_octet(self):
		return 8
	
	##}}}
	
	## property.iscale ##{{{
	@property
	def iscale(self):
		if self.scale.lower() == "k":
			return 1
		if self.scale.lower() == "m":
			return 2
		if self.scale.lower() == "g":
			return 3
		if self.scale.lower() == "t":
			return 4
		return 0
	
	##}}}
	
	## Octet properties ##{{{
	
	@property
	def o( self ):
		return self.bits / self.bits_per_octet / 1000**0
	
	@property
	def ko( self ):
		return self.bits / self.bits_per_octet / 1000**1
	
	@property
	def Mo( self ):
		return self.bits / self.bits_per_octet / 1000**2
	
	@property
	def Go( self ):
		return self.bits / self.bits_per_octet / 1000**3
	
	@property
	def To( self ):
		return self.bits / self.bits_per_octet / 1000**4
	
	##}}}
	
	## iOctet properties ##{{{
	
	@property
	def io( self ):
		return self.bits / self.bits_per_octet / 1024**0
	
	@property
	def kio( self ):
		return self.bits / self.bits_per_octet / 1024**1
	
	@property
	def Mio( self ):
		return self.bits / self.bits_per_octet / 1024**2
	
	@property
	def Gio( self ):
		return self.bits / self.bits_per_octet / 1024**3
	
	@property
	def Tio( self ):
		return self.bits / self.bits_per_octet / 1024**4
	
	##}}}
	
	## Byte properties ##{{{
	
	@property
	def B( self ):
		return self.bits / self.bits_per_byte / 1000**0
	
	@property
	def kB( self ):
		return self.bits / self.bits_per_byte / 1000**1
	
	@property
	def MB( self ):
		return self.bits / self.bits_per_byte / 1000**2
	
	@property
	def GB( self ):
		return self.bits / self.bits_per_byte / 1000**3
	
	@property
	def TB( self ):
		return self.bits / self.bits_per_byte / 1000**4
	
	##}}}
	
	## iByte properties ##{{{
	
	@property
	def iB( self ):
		return self.bits / self.bits_per_byte / 1024**0
	
	@property
	def kiB( self ):
		return self.bits / self.bits_per_byte / 1024**1
	
	@property
	def MiB( self ):
		return self.bits / self.bits_per_byte / 1024**2
	
	@property
	def GiB( self ):
		return self.bits / self.bits_per_byte / 1024**3
	
	@property
	def TiB( self ):
		return self.bits / self.bits_per_byte / 1024**4
	
	##}}}
	
	## bits properties ##{{{
	
	@property
	def b( self ):
		return self.bits / 1000**0
	
	@property
	def kb( self ):
		return self.bits / 1000**1
	
	@property
	def Mb( self ):
		return self.bits / 1000**2
	
	@property
	def Gb( self ):
		return self.bits / 1000**3
	
	@property
	def Tb( self ):
		return self.bits / 1000**4
	
	##}}}
	
	## ibits properties ##{{{
	
	@property
	def ib( self ):
		return self.bits / 1024**0
	
	@property
	def kib( self ):
		return self.bits / 1024**1
	
	@property
	def Mib( self ):
		return self.bits / 1024**2
	
	@property
	def Gib( self ):
		return self.bits / 1024**3
	
	@property
	def Tib( self ):
		return self.bits / 1024**4
	
	##}}}
	
##}}}


