#pragma once

#include "all_tuning_configs.h"

namespace ap {

struct DefaultConfig {
  static constexpr int kConfigId = 16;
  static constexpr int kSwizzleFactor = 1;
  static constexpr bool kBatched = false;
};

} // namespace ap
