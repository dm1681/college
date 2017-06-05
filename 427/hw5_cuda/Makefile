#Makefile
#define variables
objects = main.o kernels.o
NVCC= NVCC
opt= -02
ARCH= -arch=sm_30
LIBS=
execname= main

#compile

$(execname): $(objects)
	$(NVCC) $(opt) -o $(execname) $(objects) $(LIBS)

kernels.o: kernels.cu
	$(NVCC) $(opt) $(ARCH) -c kernels.cu

main.o: main.cu
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c mian.cu


#clean Makefile
clean:
	rm $(objects)
#end of Makefile