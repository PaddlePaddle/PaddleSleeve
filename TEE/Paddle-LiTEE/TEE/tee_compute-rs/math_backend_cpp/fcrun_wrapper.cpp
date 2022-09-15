#include "fcrun_wrapper.h"

extern "C" void trace_printf(const char* fn, int line, int level, bool level_ok, const char *fmt, ...);

#define IMSG(...)   trace_printf(__func__, __LINE__, 2, true, __VA_ARGS__);

extern "C" void fc_run_ff(int M, int N, int K, const float* A, \
        const float* B, float* C, const float* bias, bool flag_act, bool flag_gemm){
    using namespace paddle::lite;
    ARMTrustZoneContext ctx;
    paddle::lite_api::ActivationType act;
    if (flag_act) act = paddle::lite_api::ActivationType::kRelu;

    operators::ActivationParam act_param;
    act_param.has_active = false;
    if (flag_gemm) {
        arm_trustzone::math::sgemm(false, 
            false, 
            M, 
            N, 
            K, 
            1.f, 
            A, 
            K, 
            B, 
            N, 
            0.f, 
            C, 
            N, 
            nullptr, 
            false, 
            act_param, 
            &ctx);

        if (bias) {
            arm_trustzone::math::fill_bias_fc(C, bias, M, N, flag_act);
        }
    } 
    else {
        for (int i = 0; i < M; ++i) {
            auto* i_data_batch = A + i * K;
            auto* o_data_batch = C + i * N;
            arm_trustzone::math::sgemv(B,
                i_data_batch,
                o_data_batch,
                false,
                N,
                K,
                0.f,
                bias != nullptr,
                bias,
                flag_act,
                act,
                &ctx);
        }
    }
}

extern "C" void fc_run_if(int M, int N, int K, const int8_t* A,\
        const int8_t* B, float* C, const float* bias, const float* scale, bool flag_act, bool flag_gemm) {
    using namespace paddle::lite;
    ARMTrustZoneContext ctx;
    paddle::lite_api::ActivationType act;
    if (flag_act) act = paddle::lite_api::ActivationType::kRelu;

    operators::ActivationParam act_param;
    act_param.has_active = false;
    if (flag_gemm) {
        arm_trustzone::math::gemm_s8(false, 
            false, 
            M, 
            N, 
            K, 
            A, 
            B, 
            C, 
            nullptr, 
            false, 
            scale, 
            act_param, 
            &ctx);
        
        if (bias) {
            arm_trustzone::math::fill_bias_fc(C, bias, M, N, flag_act);
        }
    } 
    else {
        for (int i = 0; i < M; ++i) {
            auto* i_data_batch = A + i * K;
            auto* o_data_batch = C + i * N;

            arm_trustzone::math::gemv_int8(B,
                i_data_batch,
                o_data_batch,
                false,
                N,
                K,
                scale, 
                bias != nullptr,
                bias,
                act_param,
                &ctx);
        }
    }
}

extern "C" void fc_run_ii(int M, int N, int K, const int8_t* A, \
        const int8_t* B, int8_t* C, const float* bias, const float* scale, bool flag_act, bool flag_gemm) {
    using namespace paddle::lite;
    ARMTrustZoneContext ctx;
    paddle::lite_api::ActivationType act;
    operators::ActivationParam act_param;
    act_param.has_active = false;
    if (flag_act) {
        act = paddle::lite_api::ActivationType::kRelu;
        act_param.has_active = true;
        act_param.active_type =paddle::lite_api::ActivationType::kRelu;
    }
    if (flag_gemm) {
        arm_trustzone::math::gemm_s8(false, 
            false, 
            M, 
            N, 
            K, 
            A, 
            B, 
            C, 
            nullptr, 
            false, 
            scale, 
            act_param, 
            &ctx);
    } 
    else {
        for (int i = 0; i < M; ++i) {
            auto* i_data_batch = A + i * K;
            auto* o_data_batch = C + i * N;

            arm_trustzone::math::gemv_int8(B,
                i_data_batch,
                o_data_batch,
                false,
                N,
                K,
                scale, 
                bias != nullptr,
                bias,
                act_param,
                &ctx);
        }
    }
}