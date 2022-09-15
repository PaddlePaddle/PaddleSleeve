#ifndef _TENSOR_H_
#define _TENSOR_H_

extern "C" {
#include <stdint.h>
}

namespace paddle {
namespace lite {

class TensorLite {
public:
  template <typename T, typename R = T>
  const R *data() const {
    return nullptr;
  }
  void Resize() {  }
  int64_t numel() const { return 1; }

  template <typename T, typename R = T>
  R *mutable_data() {
    return nullptr;
  }
};

} // namespace lite
} // namespace paddle
#endif
