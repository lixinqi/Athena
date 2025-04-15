#pragma once

#include "all_tuning_configs.h"
#include "default_config_id.h"
#include "profile.h"

namespace ap {

template <typename Wrapper, int... Is>
auto GenerateFuncList(std::integer_sequence<int, Is...>) {
  using FuncPtr = decltype(&Wrapper::template Apply<0>);
  return std::vector<FuncPtr>{&Wrapper::template Apply<Is>...};
}

template <typename T, typename Wrapper, typename... Args>
int RunWithAutotune(cudaStream_t stream, int config_id, Args &&...args) {
  int selected_config_id = config_id;

#if AP_ENABLE_AUTOTUNE
  using FuncPtr = decltype(&Wrapper::template Apply<0>);
  constexpr int N = ap::ConfigsInfo<T>::kNumTotals;
  static std::vector<FuncPtr> matmul_functions =
      GenerateFuncList<Wrapper>(std::make_integer_sequence<int, N>{});
  if (selected_config_id == -1) {
    selected_config_id = ap::ProfileBestConfig(matmul_functions, stream,
                                               std::forward<Args>(args)...);
  } else {
    matmul_functions[selected_config_id](std::forward<Args>(args)...);
  }
#else
  func<DefaultConfig::kConfigId>(std::forward<Args>(args)...);
#endif

  return selected_config_id;
}

} // namespace ap
