#ifndef _TARGET_WRAPPER_H_
#define _TARGET_WRAPPER_H_
#include "device_info.h"
#include <arm_neon.h>
extern "C" {
#include <stdlib.h>
#include <stdio.h>
}

namespace paddle {
namespace lite {

const int MALLOC_ALIGN = 64;
class TargetWrapper {
public:
static void* Malloc(size_t size) {
  size_t offset = sizeof(void*) + MALLOC_ALIGN - 1;
  char* p = static_cast<char*>(malloc(offset + size));
  
  void* r = reinterpret_cast<void*>(reinterpret_cast<size_t>(p + offset) &
                                    (~(MALLOC_ALIGN - 1)));
  static_cast<void**>(r)[-1] = p;
  return r;
}

static void Free(void* ptr) {
  if (ptr) {
    free(static_cast<void**>(ptr)[-1]);
  }
}

static void MemcpySync(void* dst, const void* src, size_t size) {
  if (size > 0) {
    if (!dst) printf("Error: the destination of MemcpySync can not be nullptr.\n");
    if (!src) printf("Error: the source of MemcpySync can not be nullptr.\n");
    memcpy(dst, src, size);
  }
}

};

using TargetWrapperHost = TargetWrapper;


} // namespace lite
} // namespace paddle
#endif
