stencil: stencil.c
	mpiicc -O3 -xHOST -std=c99 -Wall $^ -o $@

