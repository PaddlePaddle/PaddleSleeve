#include "paddle_api.h"
#include <arm_neon.h>
#include <limits>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
#include <fstream>

const int CPU_THREAD_NUM = 2;
const paddle::lite_api::PowerMode CPU_POWER_MODE =
    paddle::lite_api::PowerMode::LITE_POWER_HIGH;


int main(int argc, char **argv) {
  std::string config_path;
  if (argc != 3) {
	config_path = "";
  } else {
	config_path = argv[2];
  }

  std::string model_path = argv[1];

  // Set MobileConfig
  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(model_path);
  config.set_config_from_file(config_path);
  config.set_threads(CPU_THREAD_NUM);
  config.set_power_mode(CPU_POWER_MODE);

  // Create PaddlePredictor by MobileConfig
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(config);

  return 0;
}
