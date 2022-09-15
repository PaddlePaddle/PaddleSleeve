#ifndef _DEVICE_INFO_H_
#define _DEVICE_INFO_H_
namespace paddle {
namespace lite {

typedef enum {
  kAPPLE = 0,
  kA35 = 35,
  kA53 = 53,
  kA55 = 55,
  kA57 = 57,
  kA72 = 72,
  kA73 = 73,
  kA75 = 75,
  kA76 = 76,
  kA77 = 77,
  kA78 = 78,
  kARMArch_UNKOWN = -1
} ARMArch;

} // namespace lite
} // namespace paddle

#endif
