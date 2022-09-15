#ifndef _CONTEXT_H_
#define _CONTEXT_H_
#include "device_info.h"
#include "paddle_place.h"

typedef unsigned long size_t;

using TargetType = paddle::lite_api::TargetType;

namespace paddle {
namespace lite {

template <TargetType Type>
class Context;

template <>
class Context<TargetType::kARMTrustZone> {
  public:
  int threads() const { return 1; }
  ARMArch arch() const { return kARMArch_UNKOWN; }
  int llc_size() const { return 1; }
  bool has_dot() const { return false; }

  template <typename T>
  T* workspace_data() {
    return nullptr;
  }

  bool ExtendWorkspace(size_t size) {
	  return true;
  }
};

using ARMTrustZoneContext = Context<TargetType::kARMTrustZone>;

} // namespace paddle
} // namespace paddle
#endif
