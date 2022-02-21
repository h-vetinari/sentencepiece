// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.!

#ifndef SENTENCEPIECE_FLAG_H_
#define SENTENCEPIECE_FLAG_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sentencepiece {
namespace internal {
struct FlagFunc;

void RegisterFlag(const std::string &name, std::shared_ptr<FlagFunc> func);

}  // namespace internal

template <typename T>
class Flag {
 public:
  Flag(const char *name, const char *type, const char *help,
       const T &defautl_value);
  virtual ~Flag();
  const T &value() const;
  void set_value(const T &value);
  void set_value_as_str(const std::string &value_as_str);

 private:
  T value_;
  std::shared_ptr<internal::FlagFunc> func_;
};

template <typename T>
const T &GetFlag(const Flag<T> &flag) {
  return flag.value();
}

template <typename T, typename V>
void SetFlag(Flag<T> *flag, const V &v) {
  const T value(v);
  flag->set_value(value);
}

#define HAS_STPC_CLEANUP_FLAGS

void CleanupFlags();

}  // namespace sentencepiece

#define STPC_FLAG(Type, name, default_value, help) \
  sentencepiece::Flag<Type> FLAGS_##name(#name, #Type, help, default_value);

#define STPC_DECLARE_FLAG(Type, name) extern sentencepiece::Flag<Type> FLAGS_##name;

#endif  // SENTENCEPIECE_FLAG_H_
