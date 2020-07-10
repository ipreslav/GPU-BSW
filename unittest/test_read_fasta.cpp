#include "doctest.h"
#include <gpu_bsw/read_fasta.hpp>

#include <stdexcept>

TEST_CASE("No fasta file"){
  CHECK_THROWS_AS(ReadFasta("not-a-file"), std::runtime_error);
}

//TODO: Test for modifiers reading correctly
TEST_CASE("Read Fasta"){
  FastaInput fasta;
  CHECK_NOTHROW(fasta=ReadFasta("../test-data/dna-reference.fasta"));

  CHECK(fasta.sequences.size()==30'000);
  CHECK(fasta.modifiers.size()==30'000);
  CHECK(fasta.headers.size()==30'000);
  CHECK(fasta.maximum_sequence_length==579);
  CHECK(fasta.sequence_count()==30'000);
  CHECK(fasta.sequences.at(5)=="CGCACAAATCAGAAGCTCCGGGTGGCAAACACAGCTAAATAGTTGTAATTATGGAATATAGAAAAATGTTCGATTGTCGTTATGAGGATTATGAGCGCCTCAAAGCCCCCCCACCGCAAAAAGGCCCTGTGTTCGCCCCTCTCCACCCATCCATCGCATGGCCCAACGAAGCGGATATCGCTCCGGAATCCTCCTACGAAAAACTTCTGTAAAAAGAACAAAACCGGAAATCCACTTGGGAACGCGAAACCCCAGCTTCGCATATTGACCCAGAAGATCAACAGTAGAATTTGTGGCAACGGAACAACGTCCCGGAACTTCTCCTGAACCAAAACAACTTCACTGTTCGATTCCCCGCACCATTACATGATGCAGCGTTCCCGGTGTGTCAAGTCTCGCTCCTCGTGGCATATGGCTCTCTTGTCTTTTGCTTTTCAAAAGCTGCCTGCACAAATCGTTTATTCCTCACTGCAAAATACAATTTTCTACGCTATTGCACTGCGTCCCCTCAGGCTCACTCTCAGGCTCAATAATGACAGAAAATTCAGCGGTAAATGGATGGAATCATACGTATGTGAA");
}

TEST_CASE("Read Pair"){
  const auto input_data = ReadFastaQueryTargetPair("../test-data/dna-reference.fasta", "../test-data/dna-query.fasta");
  CHECK(input_data.sequence_count()==30'000);
}