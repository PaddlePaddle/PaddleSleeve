#include "softmax_wrapper.h"

extern "C" void trace_printf(const char* fn, int line, int level, bool level_ok, const char *fmt, ...);

#define IMSG(...)   trace_printf(__func__, __LINE__, 2, true, __VA_ARGS__);

extern "C" void softmax_run(const float* din, float* dout, int outer_num, int inner_num, int axis_size){
    using namespace paddle::lite::arm_trustzone;
    if (inner_num == 1) {
        if (axis_size > 4) {
            IMSG("invoke softmax_inner1_large_axis");
            math::softmax_inner1_large_axis(
                din, dout, outer_num, axis_size);
        } else {
            IMSG("invoke softmax_inner1_small_axis");
            math::softmax_inner1_small_axis(
                din, dout, outer_num, axis_size);
        }
    } else {
        if (axis_size == 4 && inner_num % 8 == 0) {
            IMSG("invoke softmax_inner8_axis4");
            math::softmax_inner8_axis4(
                din, dout, axis_size, inner_num, outer_num);
        } else if (axis_size == 4 && inner_num % 4 == 0) {
            IMSG("invoke softmax_inner4_axis4");
            math::softmax_inner4_axis4(
                din, dout, axis_size, inner_num, outer_num);
        } else {
            if (inner_num % 8 == 0) {
            IMSG("invoke softmax_inner8");
                math::softmax_inner8(
                    din, dout, axis_size, inner_num, outer_num);
            } else if (inner_num % 4 == 0) {
            IMSG("invoke softmax_inner4");
                math::softmax_inner4(
                    din, dout, axis_size, inner_num, outer_num);
            } else {
            IMSG("invoke softmax_basic");
                math::softmax_basic(
                    din, dout, axis_size, inner_num, outer_num);
            }
        }
  }
}