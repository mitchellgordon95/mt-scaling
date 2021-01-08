task bpe_data : subword_nmt
  > TRAIN_SRC
  > TRAIN_TRG
  > DEV_SRC
  > DEV_TRG
  > SRC_VOCAB
  > TRG_VOCAB
  :: SRC=(Lang: deen="de" ruen="ru" zhen="zh" soen="so" swen="sw" tlen="tl")
  :: TRG=(Lang: deen="en" ruen="en" zhen="en" soen="en" swen="en" tlen="en")
  :: bpe_size=(BPE: 30000 2000)
  # We use a GPU here even though we don't need it. See https://git.io/JIKDI
  :: pyenv=@ .submitter=@ .resource_flags=$resource_flags_train .action_flags=@ {

  if [[ $SRC == "so" ]] || [[ $SRC == "sw" ]] || [[ $SRC == "tl" ]]; then
    tok_train_src="/exp/kduh/data/material/bitext/$SRC-$TRG/BUILD.$SRC"
    tok_train_trg="/exp/kduh/data/material/bitext/$SRC-$TRG/BUILD.$TRG"

    tok_dev_src="/exp/kduh/data/material/bitext/$SRC-$TRG/ANALYSIS2-text.$SRC"
    tok_dev_trg="/exp/kduh/data/material/bitext/$SRC-$TRG/ANALYSIS2-text.$TRG"
  else
    tokcln="tok"
    if [[ $SRC == "zh" ]]; then tokcln="cln"; fi
    tok_train_src="/exp/scale18/mt/data/$SRC-$TRG/generaldomain/generaldomain.train.${tokcln}.$SRC" 
    tok_train_trg="/exp/scale18/mt/data/$SRC-$TRG/generaldomain/generaldomain.train.${tokcln}.$TRG" 

    tok_dev_src="/exp/scale18/mt/data/$SRC-$TRG/generaldomain/generaldomain.dev.tok.$SRC"
    tok_dev_trg="/exp/scale18/mt/data/$SRC-$TRG/generaldomain/generaldomain.dev.tok.$TRG"
  fi

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
  :: pyenv=@ .submitter=@ .resource_flags=$resource_flags_train .action_flags=@ {
  # Builds a sockeye-compatible vocab from the whole training data
  python -m sockeye.vocab -i $train_src -o $SOCKEYE_VOCAB_SRC
  python -m sockeye.vocab -i $train_trg -o $SOCKEYE_VOCAB_TRG
}

task subset_data
  < train_src=$TRAIN_SRC@bpe_data
  < train_trg=$TRAIN_TRG@bpe_data
  > TRAIN_SUBSET_SRC
  > TRAIN_SUBSET_TRG
  :: seed=(ShuffleSeed: 42 1 2)
  :: data_percent=(DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025 0.0125)
  :: pyenv=@ .submitter=@ .resource_flags=$resource_flags_train .action_flags=@ {

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