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

#include <map>

#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"

#include "filesystem.h"
#include "glue/flags/flag.h"
#include "init.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_trainer.h"
#include "util.h"

using sentencepiece::NormalizerSpec;
using sentencepiece::TrainerSpec;

namespace {
static sentencepiece::TrainerSpec kDefaultTrainerSpec;
static sentencepiece::NormalizerSpec kDefaultNormalizerSpec;
}  // namespace

STPC_FLAG(std::string, input, "", "comma separated list of input sentences");
STPC_FLAG(std::string, input_format, kDefaultTrainerSpec.input_format(),
          "Input format. Supported format is `text` or `tsv`.");
STPC_FLAG(std::string, model_prefix, "", "output model prefix");
STPC_FLAG(std::string, model_type, "unigram",
          "model algorithm: unigram, bpe, word or char");
STPC_FLAG(int32, vocab_size, kDefaultTrainerSpec.vocab_size(),
          "vocabulary size");
STPC_FLAG(std::string, accept_language, "",
          "comma-separated list of languages this model can accept");
STPC_FLAG(int32, self_test_sample_size,
          kDefaultTrainerSpec.self_test_sample_size(),
          "the size of self test samples");
STPC_FLAG(double, character_coverage, kDefaultTrainerSpec.character_coverage(),
          "character coverage to determine the minimum symbols");
STPC_FLAG(std::uint64_t, input_sentence_size,
          kDefaultTrainerSpec.input_sentence_size(),
          "maximum size of sentences the trainer loads");
STPC_FLAG(bool, shuffle_input_sentence,
          kDefaultTrainerSpec.shuffle_input_sentence(),
          "Randomly sample input sentences in advance. Valid when "
          "--input_sentence_size > 0");
STPC_FLAG(int32, seed_sentencepiece_size,
          kDefaultTrainerSpec.seed_sentencepiece_size(),
          "the size of seed sentencepieces");
STPC_FLAG(double, shrinking_factor, kDefaultTrainerSpec.shrinking_factor(),
          "Keeps top shrinking_factor pieces with respect to the loss");
STPC_FLAG(int32, num_threads, kDefaultTrainerSpec.num_threads(),
          "number of threads for training");
STPC_FLAG(int32, num_sub_iterations, kDefaultTrainerSpec.num_sub_iterations(),
          "number of EM sub-iterations");
STPC_FLAG(int32, max_sentencepiece_length,
          kDefaultTrainerSpec.max_sentencepiece_length(),
          "maximum length of sentence piece");
STPC_FLAG(int32, max_sentence_length, kDefaultTrainerSpec.max_sentence_length(),
          "maximum length of sentence in byte");
STPC_FLAG(bool, split_by_unicode_script,
          kDefaultTrainerSpec.split_by_unicode_script(),
          "use Unicode script to split sentence pieces");
STPC_FLAG(bool, split_by_number, kDefaultTrainerSpec.split_by_number(),
          "split tokens by numbers (0-9)");
STPC_FLAG(bool, split_by_whitespace, kDefaultTrainerSpec.split_by_whitespace(),
          "use a white space to split sentence pieces");
STPC_FLAG(bool, split_digits, kDefaultTrainerSpec.split_digits(),
          "split all digits (0-9) into separate pieces");
STPC_FLAG(std::string, pretokenization_delimiter,
          kDefaultTrainerSpec.pretokenization_delimiter(),
          "specifies the delimiter of pre-tokenization");
STPC_FLAG(bool, treat_whitespace_as_suffix,
          kDefaultTrainerSpec.treat_whitespace_as_suffix(),
          "treat whitespace marker as suffix instead of prefix.");
STPC_FLAG(bool, allow_whitespace_only_pieces,
          kDefaultTrainerSpec.allow_whitespace_only_pieces(),
          "allow pieces that only contain (consecutive) whitespace tokens");
STPC_FLAG(std::string, control_symbols, "",
          "comma separated list of control symbols");
STPC_FLAG(std::string, control_symbols_file, "",
          "load control_symbols from file.");
STPC_FLAG(std::string, user_defined_symbols, "",
          "comma separated list of user defined symbols");
STPC_FLAG(std::string, user_defined_symbols_file, "",
          "load user_defined_symbols from file.");
STPC_FLAG(std::string, required_chars, "",
          "UTF8 characters in this flag are always used in the character "
          "set regardless of --character_coverage");
STPC_FLAG(std::string, required_chars_file, "",
          "load required_chars from file.");
STPC_FLAG(bool, byte_fallback, kDefaultTrainerSpec.byte_fallback(),
          "decompose unknown pieces into UTF-8 byte pieces");
STPC_FLAG(bool, vocabulary_output_piece_score,
          kDefaultTrainerSpec.vocabulary_output_piece_score(),
          "Define score in vocab file");
STPC_FLAG(std::string, normalization_rule_name, "nmt_nfkc",
          "Normalization rule name. "
          "Choose from nfkc or identity");
STPC_FLAG(std::string, normalization_rule_tsv, "",
          "Normalization rule TSV file. ");
STPC_FLAG(std::string, denormalization_rule_tsv, "",
          "Denormalization rule TSV file.");
STPC_FLAG(bool, add_dummy_prefix, kDefaultNormalizerSpec.add_dummy_prefix(),
          "Add dummy whitespace at the beginning of text");
STPC_FLAG(bool, remove_extra_whitespaces,
          kDefaultNormalizerSpec.remove_extra_whitespaces(),
          "Removes leading, trailing, and "
          "duplicate internal whitespace");
STPC_FLAG(bool, hard_vocab_limit, kDefaultTrainerSpec.hard_vocab_limit(),
          "If set to false, --vocab_size is considered as a soft limit.");
STPC_FLAG(bool, use_all_vocab, kDefaultTrainerSpec.use_all_vocab(),
          "If set to true, use all tokens as vocab. "
          "Valid for word/char models.");
STPC_FLAG(int32, unk_id, kDefaultTrainerSpec.unk_id(),
          "Override UNK (<unk>) id.");
STPC_FLAG(int32, bos_id, kDefaultTrainerSpec.bos_id(),
          "Override BOS (<s>) id. Set -1 to disable BOS.");
STPC_FLAG(int32, eos_id, kDefaultTrainerSpec.eos_id(),
          "Override EOS (</s>) id. Set -1 to disable EOS.");
STPC_FLAG(int32, pad_id, kDefaultTrainerSpec.pad_id(),
          "Override PAD (<pad>) id. Set -1 to disable PAD.");
STPC_FLAG(std::string, unk_piece, kDefaultTrainerSpec.unk_piece(),
          "Override UNK (<unk>) piece.");
STPC_FLAG(std::string, bos_piece, kDefaultTrainerSpec.bos_piece(),
          "Override BOS (<s>) piece.");
STPC_FLAG(std::string, eos_piece, kDefaultTrainerSpec.eos_piece(),
          "Override EOS (</s>) piece.");
STPC_FLAG(std::string, pad_piece, kDefaultTrainerSpec.pad_piece(),
          "Override PAD (<pad>) piece.");
STPC_FLAG(std::string, unk_surface, kDefaultTrainerSpec.unk_surface(),
          "Dummy surface string for <unk>. In decoding <unk> is decoded to "
          "`unk_surface`.");
STPC_FLAG(bool, train_extremely_large_corpus,
          kDefaultTrainerSpec.train_extremely_large_corpus(),
          "Increase bit depth for unigram tokenization.");
STPC_FLAG(uint32, random_seed, static_cast<uint32>(-1),
          "Seed value for random generator.");

// DP related.
STPC_FLAG(bool, enable_differential_privacy, false,
          "Whether to add DP while training. Currently supported only by "
          "UNIGRAM model.");

STPC_FLAG(float, differential_privacy_noise_level, 0.0f,
          "Amount of noise to add for"
          " DP");
STPC_FLAG(std::uint64_t, differential_privacy_clipping_threshold, 0,
          "Threshold for"
          " clipping the counts for DP");

int main(int argc, char *argv[]) {
  sentencepiece::ScopedResourceDestructor cleaner;
  sentencepiece::ParseCommandLineFlags(argv[0], &argc, &argv, true);

  sentencepiece::TrainerSpec trainer_spec;
  sentencepiece::NormalizerSpec normalizer_spec;
  NormalizerSpec denormalizer_spec;

  CHECK(!sentencepiece::GetFlag(FLAGS_input).empty());
  CHECK(!sentencepiece::GetFlag(FLAGS_model_prefix).empty());

  if (sentencepiece::GetFlag(FLAGS_random_seed) != -1) {
    sentencepiece::SetRandomGeneratorSeed(sentencepiece::GetFlag(FLAGS_random_seed));
  }

  auto load_lines = [](absl::string_view filename) {
    std::vector<std::string> lines;
    auto input = sentencepiece::filesystem::NewReadableFile(filename);
    CHECK_OK(input->status());
    std::string line;
    while (input->ReadLine(&line)) lines.emplace_back(line);
    return lines;
  };

// Populates the value from flags to spec.
#define SetTrainerSpecFromFlag(name) \
  trainer_spec.set_##name(sentencepiece::GetFlag(FLAGS_##name));

#define SetNormalizerSpecFromFlag(name) \
  normalizer_spec.set_##name(sentencepiece::GetFlag(FLAGS_##name));

#define SetTrainerSpecFromFile(name)                                   \
  if (!sentencepiece::GetFlag(FLAGS_##name##_file).empty()) {          \
    const auto lines = load_lines(sentencepiece::GetFlag(FLAGS_##name##_file)); \
    trainer_spec.set_##name(absl::StrJoin(lines, ""));                 \
  }

#define SetRepeatedTrainerSpecFromFlag(name)                                \
  if (!sentencepiece::GetFlag(FLAGS_##name).empty()) {                      \
    for (const auto &v :                                                    \
         sentencepiece::util::StrSplitAsCSV(sentencepiece::GetFlag(FLAGS_##name))) { \
      trainer_spec.add_##name(v);                                           \
    }                                                                       \
  }

#define SetRepeatedTrainerSpecFromFile(name)                               \
  if (!sentencepiece::GetFlag(FLAGS_##name##_file).empty()) {              \
    for (const auto &v : load_lines(sentencepiece::GetFlag(FLAGS_##name##_file))) { \
      trainer_spec.add_##name(v);                                          \
    }                                                                      \
  }

  SetRepeatedTrainerSpecFromFlag(input);

  SetTrainerSpecFromFlag(input_format);
  SetTrainerSpecFromFlag(model_prefix);
  SetTrainerSpecFromFlag(vocab_size);
  SetTrainerSpecFromFlag(self_test_sample_size);
  SetTrainerSpecFromFlag(character_coverage);
  SetTrainerSpecFromFlag(input_sentence_size);
  SetTrainerSpecFromFlag(shuffle_input_sentence);
  SetTrainerSpecFromFlag(seed_sentencepiece_size);
  SetTrainerSpecFromFlag(shrinking_factor);
  SetTrainerSpecFromFlag(num_threads);
  SetTrainerSpecFromFlag(num_sub_iterations);
  SetTrainerSpecFromFlag(max_sentencepiece_length);
  SetTrainerSpecFromFlag(max_sentence_length);
  SetTrainerSpecFromFlag(split_by_unicode_script);
  SetTrainerSpecFromFlag(split_by_whitespace);
  SetTrainerSpecFromFlag(split_by_number);
  SetTrainerSpecFromFlag(split_digits);
  SetTrainerSpecFromFlag(pretokenization_delimiter);
  SetTrainerSpecFromFlag(byte_fallback);
  SetTrainerSpecFromFlag(treat_whitespace_as_suffix);
  SetTrainerSpecFromFlag(allow_whitespace_only_pieces);
  SetTrainerSpecFromFlag(hard_vocab_limit);
  SetTrainerSpecFromFlag(use_all_vocab);
  SetTrainerSpecFromFlag(unk_id);
  SetTrainerSpecFromFlag(bos_id);
  SetTrainerSpecFromFlag(eos_id);
  SetTrainerSpecFromFlag(pad_id);
  SetTrainerSpecFromFlag(unk_piece);
  SetTrainerSpecFromFlag(bos_piece);
  SetTrainerSpecFromFlag(eos_piece);
  SetTrainerSpecFromFlag(pad_piece);
  SetTrainerSpecFromFlag(unk_surface);
  SetTrainerSpecFromFlag(required_chars);
  SetTrainerSpecFromFile(required_chars);
  SetTrainerSpecFromFlag(vocabulary_output_piece_score);
  SetRepeatedTrainerSpecFromFlag(accept_language);
  SetRepeatedTrainerSpecFromFlag(control_symbols);
  SetRepeatedTrainerSpecFromFlag(user_defined_symbols);
  SetTrainerSpecFromFlag(train_extremely_large_corpus);
  // DP related.
  SetTrainerSpecFromFlag(enable_differential_privacy);
  SetTrainerSpecFromFlag(differential_privacy_noise_level);
  SetTrainerSpecFromFlag(differential_privacy_clipping_threshold);

  SetRepeatedTrainerSpecFromFile(control_symbols);
  SetRepeatedTrainerSpecFromFile(user_defined_symbols);

  normalizer_spec.set_name(sentencepiece::GetFlag(FLAGS_normalization_rule_name));
  SetNormalizerSpecFromFlag(normalization_rule_tsv);
  SetNormalizerSpecFromFlag(add_dummy_prefix);
  SetNormalizerSpecFromFlag(remove_extra_whitespaces);

  if (!sentencepiece::GetFlag(FLAGS_denormalization_rule_tsv).empty()) {
    denormalizer_spec.set_normalization_rule_tsv(
        sentencepiece::GetFlag(FLAGS_denormalization_rule_tsv));
    denormalizer_spec.set_add_dummy_prefix(false);
    denormalizer_spec.set_remove_extra_whitespaces(false);
    denormalizer_spec.set_escape_whitespaces(false);
  }

  CHECK_OK(sentencepiece::SentencePieceTrainer::PopulateModelTypeFromString(
      sentencepiece::GetFlag(FLAGS_model_type), &trainer_spec));

  CHECK_OK(sentencepiece::SentencePieceTrainer::Train(
      trainer_spec, normalizer_spec, denormalizer_spec));

  return 0;
}
