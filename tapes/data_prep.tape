task bpe_data : subword_nmt
  > TRAIN_SRC
  > TRAIN_TRG
  > DEV_SRC
  > DEV_TRG
  > SRC_VOCAB
  > TRG_VOCAB
  :: SRC=(Lang: deen="de" ruen="ru" zhen="zh" soen="so" swen="sw" tlen="tl")
  :: TRG=(Lang: deen="en" ruen="en" zhen="en" soen="en" swen="en" tlen="en")
  :: bpe_size=(BPE: 30000 2000 50000)
  :: pyenv=@ .submitter=@ .resource_flags=$resource_flags_decode .action_flags=@ {

  # TODO: set locations of tokenized data based on $SRC language
  tok_train_src="TODO" 
  tok_train_trg="TODO" 
  tok_dev_src="TODO"
  tok_dev_trg="TODO"

  # Learn the BPE vocab from the training data
  subword-nmt learn-bpe -s $bpe_size < $tok_train_src > $SRC_VOCAB
  subword-nmt learn-bpe -s $bpe_size < $tok_train_trg > $TRG_VOCAB

  # Apply it to both train and dev
  subword-nmt apply-bpe -c $SRC_VOCAB < $tok_train_src > $TRAIN_SRC
  subword-nmt apply-bpe -c $TRG_VOCAB < $tok_train_trg > $TRAIN_TRG

  subword-nmt apply-bpe -c $SRC_VOCAB < $tok_dev_src > $DEV_SRC
  subword-nmt apply-bpe -c $TRG_VOCAB < $tok_dev_trg > $DEV_TRG
}

task build_vocab
  < train_src=$TRAIN_SRC@bpe_data
  < train_trg=$TRAIN_TRG@bpe_data
  > SOCKEYE_VOCAB_SRC
  > SOCKEYE_VOCAB_TRG
  :: pyenv=@ .submitter=@ .resource_flags=$resource_flags_decode .action_flags=@ {
  # Builds a sockeye-compatible vocab from the whole training data
  python -m sockeye.vocab -i $train_src -o $SOCKEYE_VOCAB_SRC
  python -m sockeye.vocab -i $train_trg -o $SOCKEYE_VOCAB_TRG
}

task subset_data
  < train_src=$TRAIN_SRC@bpe_data
  < train_trg=$TRAIN_TRG@bpe_data
  > TRAIN_SUBSET_SRC
  > TRAIN_SUBSET_TRG
  :: seed=(ShuffleSeed: 42 1 2 3 4)
  :: data_percent=(DataPercent: 100 95 90 80 70 60 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025 0.0125)
  :: pyenv=@ {

  lines=$(wc -l $train_src | awk '{print $1}')
  subset_size=$(python -c "print(int($lines * $data_percent / 100))")

  # See https://www.gnu.org/software/coreutils/manual/html_node/Random-sources.html#Random-sources
  get_seeded_random()
  {
    openssl enc -aes-256-ctr -pass pass:"$1" -nosalt </dev/zero 2>/dev/null
  }

  # Wrap the shuf in a ( . || true) to keep it from pipefailing the script when head stops accepting lines
  paste $train_src $train_trg | (shuf --random-source=<(get_seeded_random "$seed $data_percent") || true) | head -n $subset_size > subset
  cut -f1 subset > $TRAIN_SUBSET_SRC
  cut -f2 subset > $TRAIN_SUBSET_TRG
  rm subset
}


task subset_stats : scripts : subword_nmt
  < subset_src=$TRAIN_SUBSET_SRC@subset_data
  < subset_trg=$TRAIN_SUBSET_TRG@subset_data
  < dev_src=$DEV_SRC@bpe_data
  < dev_trg=$DEV_TRG@bpe_data
  < src_vocab=$SRC_VOCAB@bpe_data
  < trg_vocab=$TRG_VOCAB@bpe_data
  > src_dev_coverage trg_dev_coverage 
  > src_vocab_overlap trg_vocab_overlap 
  :: bpe_size=$bpe_size@bpe_data
  :: pyenv=@ .submitter=@ .resource_flags=$resource_flags_decode .action_flags=@ {
  
  python $scripts/coverage.py $subset_src $dev_src | awk '{print $1}' > $src_dev_coverage
  python $scripts/coverage.py $subset_trg $dev_trg | awk '{print $1}' > $trg_dev_coverage

  cat $subset_src | sed -r 's/@@( |$)//g' > de_bpe_subset_src
  cat $subset_trg | sed -r 's/@@( |$)//g' > de_bpe_subset_trg
  subword-nmt learn-bpe -s $bpe_size < de_bpe_subset_src > subset_src_vocab
  subword-nmt learn-bpe -s $bpe_size < de_bpe_subset_trg > subset_trg_vocab

  python $scripts/coverage.py subset_src_vocab $src_vocab | awk '{print $1}' > $src_vocab_overlap
  python $scripts/coverage.py subset_trg_vocab $trg_vocab | awk '{print $1}' > $trg_vocab_overlap
}

task word_count : scripts
  < subset_trg=$TRAIN_SUBSET_TRG@subset_data
  < subset_src=$TRAIN_SUBSET_SRC@subset_data # NOTE: need this for the summary
  > unique_words
  > seen_five
  :: pyenv=@ {
  python $scripts/word_count.py $subset_trg > tmp
  cat tmp | awk '{print $1}' > $unique_words
  cat tmp | awk '{print $2}' > $seen_five
  rm tmp
}