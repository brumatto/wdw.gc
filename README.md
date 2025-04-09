# wdw.gc
Program to solve Wheeler DeWitt equation for the Early Universe
using de Initial Value Problem (IVP) approach.

wdw.gc is a program to solve Wheele-DeWitt (2+1) equations using 
the Crank-Nicolson method to generate a Linear System and using 
the Conjugate Gradiente (CG) method to solve the system at each
time iteration.

It is parametized by INI file (default is wdw.ini).

It writes to default console medium values for several
parameters, and it writes to err console the CG iteration
at each time lapsed.

Usage:
wdw.gc <ini.file>
where ini.file is optional, by default it looks for wdw.ini

To compile from source code use GNU C++ compiler 17 or higher
with OpenMP

This program is protected by GNU General Public License v3.0

More informations can be seen on wdw.gc.manual.pdf
