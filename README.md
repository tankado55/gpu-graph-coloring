# Final project for the GPU computing course of the University of Milan

## Usage
usage: coloring input_file mode

Mode can be one integer of this list:
1 - Sequential Greedy Colorer (CPU)
2 - Random Priority Colorer (GPU)
3 - Largest Degree First	(GPU)
4 - Smallest Degree Last	(GPU)
5 - Saturation Colorer		(GPU)
6 - Incidence Colorer		(GPU)
						
example: coloring "/path/to/delaunay_n24.mtx" 2
