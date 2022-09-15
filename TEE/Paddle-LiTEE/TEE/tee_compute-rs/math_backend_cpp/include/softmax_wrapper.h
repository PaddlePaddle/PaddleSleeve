#ifndef SOFTMAX_WRAPPER_H
#define SOFTMAX_WRAPPER_H

#include "math.h"
#include "funcs.h"
#include <softmax.h>

extern "C" void softmax_run(
    const float* din, float* dout, int outer_num, int inner_num, int axis_size);

#endif // _SOFTMAX_WRAPPER_H
