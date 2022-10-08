
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

try:
	import curses
except:
	pass

from .__doc import doc as txt_doc

###############
## Functions ##
###############

def txt2curses_doc(doc):##{{{
	"""
	XSBCK.txt2curses_doc
	====================
	
	Function which split a text in a list of lines, and add bold / underline
	style for headers of sections / sub-sections.
	
	"""
	
	lines = doc.split("\n")
	
	clines = []
	for line in lines:
		
		if line == ("=" * len(line)) and len(line) > 0:
			clines[-1][1] = [curses.A_BOLD | curses.A_UNDERLINE]
		elif line == ("-" * len(line)) and len(line) > 0:
			clines[-1][1] = [curses.A_UNDERLINE]
			clines.append( ("",[]) )
		else:
			clines.append( [line,[]] )
	
	return clines
##}}}

def print_curses_doc( screen , doc ):##{{{
	"""
	XSBCK.print_curses_doc
	======================
	
	Curses function to print documentation.
	
	"""
	
	cdoc = txt2curses_doc(doc)
	
	start = 0
	
	curses.init_pair( 1 , curses.COLOR_BLACK , curses.COLOR_WHITE )
	while True:
		screen.clear()
		
		## Print rows
		for i in range(start,min(start+curses.LINES - 2,len(cdoc)),1):
			screen.addstr( i - start , 0 , cdoc[i][0] , *cdoc[i][1] )
		screen.addstr( curses.LINES - 1 , 0 , "Press 'q' to quit" , curses.A_BOLD | curses.color_pair(1) )
		
		screen.refresh()
		
		c = screen.getch()
		
		if chr(c) == 'q':
			break
		if c == 259:
			start = max( 0 , start - 1 )
		if c == 258:
			start = min( max( len(cdoc) - 10 , 0 ) , start + 1 )
		if c == 339:
			start = max( 0 , start - int(curses.LINES / 2) )
		if c == 338:
			start = min( max( len(cdoc) - 10 , 0 ) , start + int(curses.LINES / 2) )
##}}}

def print_doc():##{{{
	"""
	XSBCK.print_doc
	===============
	
	Function which print the documentation. Try first with curses for interactive
	documentation, but if curses not available (e.g. for microsoft system), just
	print.
	
	"""
	
	try:
		curses.wrapper( print_curses_doc , txt_doc )
	except:
		print(txt_doc)
##}}}

