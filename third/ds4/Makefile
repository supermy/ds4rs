CC ?= cc
CFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99
OBJCFLAGS ?= -O3 -ffast-math -mcpu=native -Wall -Wextra -fobjc-arc
NVCC ?= nvcc
NVCCFLAGS ?= -O3 -arch=native -Xcompiler -fPIC

LDLIBS ?= -lm -pthread
UNAME_S := $(shell uname -s)
NATIVE_LDLIBS := $(LDLIBS)
METAL_SRCS := $(wildcard metal/*.metal)
HAS_NVCC := $(shell which nvcc 2>/dev/null)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_metal.o
NATIVE_CORE_OBJS = ds4_native.o
else
CFLAGS += -DDS4_NO_METAL -DDS4_HAS_TILELANG -D_POSIX_C_SOURCE=200809L
CORE_OBJS = ds4.o ds4_tl_kernels.o ds4_tl_runtime.o
NATIVE_CORE_OBJS = ds4_native.o
METAL_LDLIBS := $(LDLIBS)
TL_LDLIBS = -lm -pthread -ldl
TVM_FFI_INC ?= /usr/local/lib/python3.12/dist-packages/tvm_ffi/include
TVM_FFI_LIB ?= /usr/local/lib/python3.12/dist-packages/tvm_ffi/lib
TVM_LIB ?= /usr/local/lib/python3.12/dist-packages/tilelang/lib
ifdef HAS_NVCC
CFLAGS += -DDS4_HAS_EXPERT_GPU -DDS4_HAS_GPU_WEIGHTS -DDS4_HAS_EXPERT_FP4 -DDS4_HAS_SPARSE_ATTN -DDS4_HAS_GPU_PIPELINE
CORE_OBJS += ds4_expert_gpu.o ds4_gpu_weights.o ds4_expert_fp4_gpu.o ds4_sparse_attn_gpu.o ds4_gpu_pipeline.o ds4_gpu_kv_cache.o ds4_gpu_kernels.o
CUDA_PREFIX ?= /usr/local/cuda
TL_LDLIBS += -L$(CUDA_PREFIX)/lib64 -lcudart -lstdc++ -ljson-c -L$(TVM_FFI_LIB) -ltvm_ffi -L$(TVM_LIB) -Wl,-rpath,$(TVM_FFI_LIB) -Wl,-rpath,$(TVM_LIB)
endif
endif

.PHONY: all clean test tl-kernels

all: ds4 ds4-server

ifeq ($(UNAME_S),Darwin)
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4_native: ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS) $(NATIVE_LDLIBS)
else
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(TL_LDLIBS)

ds4-server: ds4_server.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(TL_LDLIBS)

ds4_native: ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS) $(LDLIBS)
endif

ds4.o: ds4.c ds4.h ds4_metal.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_tl_kernels.o: ds4_tl_kernels.c ds4_tl_kernels.h
	$(CC) $(CFLAGS) -c -o $@ ds4_tl_kernels.c

ds4_tl_runtime.o: ds4_tl_runtime.c ds4_tl_runtime.h
	$(CC) $(CFLAGS) -I$(TVM_FFI_INC) -c -o $@ ds4_tl_runtime.c

ds4_expert_gpu.o: ds4_expert_gpu.cu ds4_expert_gpu.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

ds4_gpu_weights.o: ds4_gpu_weights.cu ds4_gpu_weights.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

ds4_expert_fp4_gpu.o: ds4_expert_fp4_gpu.cu ds4_expert_fp4_gpu.h
	$(NVCC) $(NVCCFLAGS) -I$(TVM_FFI_INC) -c -o $@ $<

ds4_sparse_attn_gpu.o: ds4_sparse_attn_gpu.cu ds4_sparse_attn_gpu.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

ds4_gpu_pipeline.o: ds4_gpu_pipeline.cu ds4_gpu_pipeline.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

ds4_gpu_kv_cache.o: ds4_gpu_kv_cache.cu ds4_gpu_kv_cache.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

ds4_gpu_kernels.o: ds4_gpu_kernels.cu ds4_gpu_kernels.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

ds4_cli.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_server.o: ds4_server.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_native.o: ds4.c ds4.h ds4_metal.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -c -o $@ ds4.c

ds4_cli_native.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -c -o $@ ds4_cli.c

ds4_metal.o: ds4_metal.m ds4_metal.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_test: ds4_test.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_test.o $(CORE_OBJS) $(METAL_LDLIBS)

test: ds4_test
	./ds4_test

tl-kernels:
	python3 tilelang/compile_kernels.py all

clean:
	rm -f ds4 ds4-server ds4_native ds4_server_test ds4_test *.o
