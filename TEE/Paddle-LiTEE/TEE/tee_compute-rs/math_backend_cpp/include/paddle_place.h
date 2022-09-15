#ifndef _PADDLE_PLACE_H_
#define _PADDLE_PLACE_H_
namespace paddle {
namespace lite_api {

enum class ActivationType : int {
  kIndentity = 0,
  kRelu = 1,
  kRelu6 = 2,
  kPRelu = 3,
  kLeakyRelu = 4,
  kSigmoid = 5,
  kTanh = 6,
  kSwish = 7,
  kExp = 8,
  kAbs = 9,
  kHardSwish = 10,
  kReciprocal = 11,
  kThresholdedRelu = 12,
  kElu = 13,
  kHardSigmoid = 14,
  kLog = 15,
  kSigmoid_v2 = 16,
  kTanh_v2 = 17,
  kGelu = 18,
  kErf = 19,
  kSign = 20,
  kSoftPlus = 21,
  kMish = 22,
  NUM = 23,
};

enum class TargetType : int {
  kUnk = 0,
  kHost = 1,
  kX86 = 2,
  kCUDA = 3,
  kARM = 4,
  kOpenCL = 5,
  kAny = 6,  // any target
  kFPGA = 7,
  kNPU = 8,
  kXPU = 9,
  kBM = 10,
  kMLU = 11,
  kRKNPU = 12,
  kAPU = 13,
  kHuaweiAscendNPU = 14,
  kImaginationNNA = 15,
  kIntelFPGA = 16,
  kARMTrustZone = 17,
  NUM = 18,  // number of fields.
};


}
}
#endif
