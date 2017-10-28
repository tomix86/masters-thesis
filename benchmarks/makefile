ROOT_DIR=./
INT_DIR=obj/
SRC_DIR=$(ROOT_DIR)
OUT_DIR=$(ROOT_DIR)bin/

INC_DIR=$(SRC_DIR) /usr/local/cuda/include/
LIB_DIR=/usr/local/cuda/lib64/

NVCC=nvcc
#NVCCFLAGS=-std=c++11 -arch=sm_52 --source-in-ptx -lineinfo -use_fast_math
#http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#options-for-steering-gpu-code-generation
NVCCFLAGS=-std=c++11 -arch=compute_52 -code=compute_52,sm_52 -Xptxas --warn-on-spills -Xptxas -v
#NVCCFLAGS=-std=c++11 -arch=compute_30 -code=compute_30,sm_30,sm_52 -Xptxas --warn-on-spills

NVCCLIBS=-lcudart -lnvToolsExt

OBJECTS=main.o 

all: create_dirs cuda-bench

create_dirs:
	@mkdir -p $(OUT_DIR)
	@mkdir -p $(INT_DIR)

clean:
	@rm -rf $(OUT_DIR)
	@rm -rf $(INT_DIR)

kernels: create_dirs main.o

cuda-bench: $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $(patsubst %, $(INT_DIR)%, $<) $(NVCCLIBS)

#files compiled with the Nvidia C++ compiler
%.o: $(SRC_DIR)%.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $(INT_DIR)$@ $(patsubst %,-I%, $(INC_DIR))
