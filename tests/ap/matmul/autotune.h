#pragma once

#include "all_tuning_configs.h"
#include "default_config_id.h"
#include "profile.h"

namespace ap {

template <typename Runner, SwizzleType ST, int... Is>
auto GenerateFuncList(std::integer_sequence<int, Is...>) {
  using FuncPtr = decltype(&Runner::template Apply<0, ST>);
  return std::vector<FuncPtr>{&Runner::template Apply<Is, ST>...};
}

template <typename T, typename Runner, bool EnableStreamK, typename... Args>
int RunWithAutotune(cudaStream_t stream, int config_id, Args &&...args) {
#if AP_ENABLE_AUTOTUNE
  int selected_config_id = config_id;

  using FuncPtr = decltype(&Runner::template Apply<0, SwizzleType::kCommon>);
  constexpr int N = ap::ConfigsInfo<T>::kNumTotals;

  static std::vector<FuncPtr> matmul_functions;
  static std::vector<FuncPtr> streamk_functions;

  if (matmul_functions.empty()) {
    matmul_functions = GenerateFuncList<Runner, SwizzleType::kCommon>(
        std::make_integer_sequence<int, N>{});
  }

  if constexpr (EnableStreamK) {
    if (streamk_functions.empty()) {
      streamk_functions = GenerateFuncList<Runner, SwizzleType::kStreamK>(
          std::make_integer_sequence<int, N>{});
    }
  }

  if (selected_config_id == -1) {
    selected_config_id = ap::ProfileBestConfig(matmul_functions, stream,
                                               std::forward<Args>(args)...);
    if constexpr (EnableStreamK) {
      std::vector<FuncPtr> mixed_functions = {
          matmul_functions[selected_config_id],
          streamk_functions[selected_config_id]};
      int mixed_config_id = ap::ProfileBestConfig(mixed_functions, stream,
                                                  std::forward<Args>(args)...);
      selected_config_id = (mixed_config_id == 0) ? selected_config_id
                                                  : (selected_config_id + N);
    }
  } else {
    if constexpr (EnableStreamK) {
      if (selected_config_id < N) {
        matmul_functions[selected_config_id](std::forward<Args>(args)...);
      } else {
        streamk_functions[selected_config_id - N](std::forward<Args>(args)...);
      }
    } else {
      matmul_functions[selected_config_id](std::forward<Args>(args)...);
    }
  }

  return selected_config_id;
#else
  Runner::template Apply<DefaultConfig::kConfigId, SwizzleType::kCommon>(
      std::forward<Args>(args)...);
  return -1;
#endif
}

} // namespace ap
