#ifndef FCRUN_WRAPPER_H
#define FCRUN_WRAPPER_H

#include <sgemm.h>
#include <sgemv.h>
#include <gemm_s8.h>
#include <gemv_arm_int8.h>
#include <context.h>
#include "math.h"
#include "funcs.h"

extern "C" void fc_run_ff(int M, int N, int K, const float* A, const float* B, \
    float* C, const float* bias, bool flag_act, bool flag_gemm);
extern "C" void fc_run_if(int M, int N, int K, const int8_t* A, const int8_t* B, \
    float* C, const float* bias, const float* scale, bool flag_act, bool flag_gemm);
extern "C" void fc_run_ii(int M, int N, int K, const int8_t* A, const int8_t* B, \
    int8_t* C, const float* bias, const float* scale, bool flag_act, bool flag_gemm);
#endif //FCRUN_WRAPPER_H