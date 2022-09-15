CXX = ${OPTEE_DIR}/toolchains/aarch64/bin/aarch64-linux-gnu-g++
AR = aarch64-linux-gnu-ar
CXXFLAGS = -Os -g3 -nostdinc -isystem ${OPTEE_DIR}/toolchains/aarch64/bin/../lib/gcc/aarch64-none-linux-gnu/10.2.1/include -fpic  -I${OPTEE_DIR}/optee_os/out/arm/export-ta_arm64/include -I./include
SRC = $(wildcaarch64-linux-gnu-ar ard *.cp=p)

FC_OBJS = expf.o fabsf.o fcrun_wrapper.o funcs.o gemm_prepacked_int8.o gemm_s8.o gemv_arm_int8.o packed_sgemm.o round.o sgemm.o sgemv.o
SOFTMAX_OBJS = softmax.o softmax_wrapper.o

LIBNAME = libmath_backend.a

all: $(LIBNAME)

$(LIBNAME): $(FC_OBJS) $(SOFTMAX_OBJS)
	$(AR) crv $@ $(FC_OBJS) $(SOFTMAX_OBJS)

.PHONY: install
install:
	cp $(LIBNAME) ${OPTEE_DIR}/optee_os/out/arm/export-ta_arm64/lib/
	echo "$(LIBNAME) installed"

.PHONY: clean
clean:
	rm -f *.o $(LIBNAME)
	rm -f ${OPTEE_DIR}/optee_os/out/arm/export-ta_arm64/lib/$(LIBNAME)

