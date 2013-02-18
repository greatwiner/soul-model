/******************************************************************
 Structure OUtput Layer (SOUL) Language Model Toolkit
 (C) Copyright 2009 - 2012 Hai-Son LE LIMSI-CNRS

 Contains default definitions and parameters
 *******************************************************************/

#ifndef SOULCONFIG_H_
#define SOULCONFIG_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <set>
#include <vector>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

using namespace std;

/*******************************************************************
 Change parameters below as you want
 *******************************************************************/

// Vocabulary
// UNK is for unknown word
// Use <UNK> as in-house LIMSI toolkit

// srilm ...: usually <unk>
#define SS "<s>"
#define ES "</s>"
#define UNK "<UNK>"

// Maximal ngram number  while resampling
#define RESAMPLING_NGRAM_NUMBER      20000000
// For context grouping, read how many n-grams for each block
#define BLOCK_NGRAM_NUMBER           10000000
// Maximal example number for function model
#define FUNC_NGRAM_NUMBER              500000

/*******************************************************************
 Normally, you should not modify these parameters below unless you
 really understand what you are supposed to do
 ********************************************************************/

// Interval for initializing parameters
// For LookupTable: [LKT_INIT0, LKT_INIT1]
#define LKT_INIT0     0.0
#define LKT_INIT1     0.1
// For weights, biases of Neural Layers: [LINEAR_INIT0, LINEAR_INIT1]
#define LINEAR_INIT0 -0.1
#define LINEAR_INIT1  0.1

// Change these prefixes if in your vocabulary, you have words
// starting with them. Need to modify also in some script files:
// voc_createDisWordTuple.py
// voc_createOutVoc.py
// Prefix used when training step 3 (with out-of-shortlist word)
#define PREFIX_OUT "prefix."
// Prefix used when training tuple models (for source words)
#define PREFIX_SOURCE "src."

// When probability is zero, changing to PROBZERO, don't be afraid
// Use zero as zero
#define  PROBZERO 0

// Model name type
// c: classical, n: ngram, ov: one vector, lbl: log bilinear,
// r: recurrent, rank: ranking model (similar to Collobert's model)

#define CN  "cn" // classical n-gram (of Bengio, Schwenk)
#define OVN  "ovn" // one vector n-gram
#define OVNB "ovnb" // one vector n-gram with Bayesian inference
// Two names below for translation models are kepth to work with old
// models
// dwt: discriminative word tuple => word translation model
// dt: discriminative tuple => word translation model
#define WTOVN  "dwtovn" // word translation one vector n-gram
#define PTOVN  "dtovn" // phrase translation one vector n-gram
// All models below are not useful
#define RANKOVN  "rankovn" // ranking one vector n-gram
#define LBL  "lbl" // log bilinear
#define ROVN  "rovn" // pseudo recurrent one vector n-gram
#define MAXOVN  "maxovn" // max (at the hidden layer) one vector n-gram
#define OVR  "ovr" // one vector recurrent
#define COVR  "covr" // continuous one vector recurrent, take
// into account words of the previous sentences
#define RANKCN  "rankcn" // ranking classical n-gram

// Activation type
#define TANH   "t"
#define SIGM   "s"
#define LINEAR "l"

// Compute and training

#define DEFAULT_BLOCK_SIZE 1

// Two ways of using learning rate
// (1) NORMAL, as Bengio, Schwenk suggest
// currentLearningRate = learningRate / (1 + exampleNumbers * learningRateDecay);
#define LEARNINGRATE_NORMAL "n"

// (2) DOWN, not a good name though, similar to Mikolov's way
// currentLearningRate is fixed and is equal to learningRate at each epoch
// If entropy increases more than MUL_LOGLKLHOOD (1.002) after one epoch,
// learningRate = learningRate / learningRateDecay
#define LEARNINGRATE_DOWN   "d"
#define MUL_LOGLKLHOOD 1.002
// Stop training after dividing MAX_DIVIDE times
// MAX_DIVIDE = 100: Never stopping
#define MAX_DIVIDE 100

// Format
#define BINARY    "b"
#define TEXT      "t"

// Commands for zip file
#define GZIP_CMD          "exec gzip -c"
#define GUNZIP_CMD        "exec gunzip -cf"

// Text

// Maximal number of words per sentence
#define MAX_WORD_PER_SENTENCE            5000
// Maximal characters per sentence
#define MAX_CHAR_PER_SENTENCE          100000

// Internal parameters
#define ID_END_NGRAM                       -1
#define SIGN_NOT_WORD                      -1

// For vocabulary
#define ID_UNK               -1
#define ID_ROOT              -2
#define ID_INIT              -3
#define VOCAB_TABLE_SIZE  22721

// For verbosity
#define PRINT_DEBUG      1
// Print when CONSTPRINT parts of data were done
#define CONSTPRINT    0.05
// Print when NLINEPRINT lines of data were done
#define NLINEPRINT 1000000

#endif /* SOULCONFIG_H_ */
