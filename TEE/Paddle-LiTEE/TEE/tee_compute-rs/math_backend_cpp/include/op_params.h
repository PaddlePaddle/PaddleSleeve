#ifndef _OP_PARAMS_H_
#define _OP_PARAMS_H_
#include "tensor.h"
#include "paddle_place.h"

namespace paddle {
namespace lite {
namespace operators {

struct ActivationParam {
  char padding[0x38];
  lite_api::ActivationType active_type{lite_api::ActivationType::kIndentity};
  bool has_active{false};
  float Leaky_relu_alpha{0.f};   // leaky_relu param
  float Relu_clipped_coef{6.f};  // relu_clipped param

  // relu6
  float threshold{6.0f};
};



}
}
}
#endif
