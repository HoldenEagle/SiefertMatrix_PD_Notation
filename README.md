# SiefertMatrix_PD_Notation

The goal of this project is to calculate the Siefert Matrix of a knot using a given 
PD Notation of the knot. This file will contain the steps I took to calculate the Siefert Matrix 
and the knot invarient determinent, as well as subsections of code for these steps.

Given: Knot in PD_Notation  , Output: Siefert Matrix and the determinent of the knot.

First step: Find the incoming and outgoing edges at each crossing. The PD_notation contains 
the edges of the knot component numbered in a certain direction. By going to each crossing and
respecting the order of the path, we can find the incoming and outgoing edges at each of the given 
crossings. The path will start from 1 and go to the length of the number of edges, which will be the max value
of the PD Notation. The incoming and outgoing edges will be stored in a hashmap for each given crossing. A picture 
is given below of what is going on, as well as the code for this part.

![image](https://github.com/user-attachments/assets/12289547-182a-4438-b1b5-9fdd4c1acffa)





