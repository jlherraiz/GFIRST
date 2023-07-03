nvcc GFIRST.cu -o GFIRST.x -m64 -O3 -use_fast_math -DUSING_CUDA  -I. -I/usr/local/cuda-11.1/include -I/usr/local/cuda-11.1/samples/common/inc -L/usr/lib/ -L/usr/local/cuda-11.1/lib64 -lz --ptxas-options=-v -gencode=arch=compute_86,code=sm_86

