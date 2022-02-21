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

#include "init.h"

#include "common.h"
#include "glue/flags/flag.h"
#include "testharness.h"

STPC_FLAG(int32, int32_f, 10, "int32_flags");
STPC_FLAG(bool, bool_f, false, "bool_flags");
STPC_FLAG(int64, int64_f, 9223372036854775807LL, "int64_flags");
STPC_FLAG(uint64, uint64_f, 18446744073709551615ULL, "uint64_flags");
STPC_FLAG(double, double_f, 40.0, "double_flags");
STPC_FLAG(std::string, string_f, "str", "string_flags");

STPC_DECLARE_FLAG(bool, help);
STPC_DECLARE_FLAG(bool, version);

using sentencepiece::ParseCommandLineFlags;

namespace absl {
TEST(FlagsTest, DefaultValueTest) {
  EXPECT_EQ(10, sentencepiece::GetFlag(FLAGS_int32_f));
  EXPECT_EQ(false, sentencepiece::GetFlag(FLAGS_bool_f));
  EXPECT_EQ(9223372036854775807LL, sentencepiece::GetFlag(FLAGS_int64_f));
  EXPECT_EQ(18446744073709551615ULL, sentencepiece::GetFlag(FLAGS_uint64_f));
  EXPECT_EQ(40.0, sentencepiece::GetFlag(FLAGS_double_f));
  EXPECT_EQ("str", sentencepiece::GetFlag(FLAGS_string_f));
}

TEST(FlagsTest, ParseCommandLineFlagsTest) {
  const char *kFlags[] = {"program",        "--int32_f=100",  "other1",
                          "--bool_f=true",  "--int64_f=200",  "--uint64_f=300",
                          "--double_f=400", "--string_f=foo", "other2",
                          "other3"};
  int argc = arraysize(kFlags);
  char **argv = const_cast<char **>(kFlags);
  ParseCommandLineFlags(kFlags[0], &argc, &argv);

  EXPECT_EQ(100, sentencepiece::GetFlag(FLAGS_int32_f));
  EXPECT_EQ(true, sentencepiece::GetFlag(FLAGS_bool_f));
  EXPECT_EQ(200, sentencepiece::GetFlag(FLAGS_int64_f));
  EXPECT_EQ(300, sentencepiece::GetFlag(FLAGS_uint64_f));
  EXPECT_EQ(400.0, sentencepiece::GetFlag(FLAGS_double_f));
  EXPECT_EQ("foo", sentencepiece::GetFlag(FLAGS_string_f));
  EXPECT_EQ(4, argc);
  EXPECT_EQ("program", std::string(argv[0]));
  EXPECT_EQ("other1", std::string(argv[1]));
  EXPECT_EQ("other2", std::string(argv[2]));
  EXPECT_EQ("other3", std::string(argv[3]));
}

TEST(FlagsTest, ParseCommandLineFlagsTest2) {
  const char *kFlags[] = {"program",       "--int32_f", "500",
                          "-int64_f=600",  "-uint64_f", "700",
                          "--bool_f=FALSE"};
  int argc = arraysize(kFlags);
  char **argv = const_cast<char **>(kFlags);
  ParseCommandLineFlags(kFlags[0], &argc, &argv);

  EXPECT_EQ(500, sentencepiece::GetFlag(FLAGS_int32_f));
  EXPECT_EQ(600, sentencepiece::GetFlag(FLAGS_int64_f));
  EXPECT_EQ(700, sentencepiece::GetFlag(FLAGS_uint64_f));
  EXPECT_FALSE(sentencepiece::GetFlag(FLAGS_bool_f));
  EXPECT_EQ(1, argc);
}

TEST(FlagsTest, ParseCommandLineFlagsTest3) {
  const char *kFlags[] = {"program", "--bool_f", "--int32_f", "800"};

  int argc = arraysize(kFlags);
  char **argv = const_cast<char **>(kFlags);
  ParseCommandLineFlags(kFlags[0], &argc, &argv);
  EXPECT_TRUE(sentencepiece::GetFlag(FLAGS_bool_f));
  EXPECT_EQ(800, sentencepiece::GetFlag(FLAGS_int32_f));
  EXPECT_EQ(1, argc);
}

#ifndef _USE_EXTERNAL_ABSL

TEST(FlagsTest, ParseCommandLineFlagsHelpTest) {
  const char *kFlags[] = {"program", "--help"};
  int argc = arraysize(kFlags);
  char **argv = const_cast<char **>(kFlags);
  EXPECT_DEATH(ParseCommandLineFlags(kFlags[0], &argc, &argv), "");
  sentencepiece::SetFlag(&FLAGS_help, false);
}

TEST(FlagsTest, ParseCommandLineFlagsVersionTest) {
  const char *kFlags[] = {"program", "--version"};
  int argc = arraysize(kFlags);
  char **argv = const_cast<char **>(kFlags);
  EXPECT_DEATH(ParseCommandLineFlags(kFlags[0], &argc, &argv), "");
  sentencepiece::SetFlag(&FLAGS_version, false);
}

TEST(FlagsTest, ParseCommandLineFlagsUnknownTest) {
  const char *kFlags[] = {"program", "--foo"};
  int argc = arraysize(kFlags);
  char **argv = const_cast<char **>(kFlags);
  EXPECT_DEATH(ParseCommandLineFlags(kFlags[0], &argc, &argv), "");
}

TEST(FlagsTest, ParseCommandLineFlagsInvalidBoolTest) {
  const char *kFlags[] = {"program", "--bool_f=X"};
  int argc = arraysize(kFlags);
  char **argv = const_cast<char **>(kFlags);
  EXPECT_DEATH(ParseCommandLineFlags(kFlags[0], &argc, &argv), "");
}

TEST(FlagsTest, ParseCommandLineFlagsEmptyStringArgs) {
  const char *kFlags[] = {"program", "--string_f="};
  int argc = arraysize(kFlags);
  char **argv = const_cast<char **>(kFlags);
  ParseCommandLineFlags(kFlags[0], &argc, &argv);
  EXPECT_EQ(1, argc);
  EXPECT_EQ("", sentencepiece::GetFlag(FLAGS_string_f));
}

TEST(FlagsTest, ParseCommandLineFlagsEmptyBoolArgs) {
  const char *kFlags[] = {"program", "--bool_f"};
  int argc = arraysize(kFlags);
  char **argv = const_cast<char **>(kFlags);
  ParseCommandLineFlags(kFlags[0], &argc, &argv);
  EXPECT_EQ(1, argc);
  EXPECT_TRUE(sentencepiece::GetFlag(FLAGS_bool_f));
}

TEST(FlagsTest, ParseCommandLineFlagsEmptyIntArgs) {
  const char *kFlags[] = {"program", "--int32_f"};
  int argc = arraysize(kFlags);
  char **argv = const_cast<char **>(kFlags);
  EXPECT_DEATH(ParseCommandLineFlags(kFlags[0], &argc, &argv), );
}
#endif  // _USE_EXTERNAL_ABSL
}  // namespace absl
