task train : scripts
    < train_src=$TRAIN_SUBSET_SRC@subset_data
    < train_trg=$TRAIN_SUBSET_TRG@subset_data
    < src_vocab=$SOCKEYE_VOCAB_SRC@build_vocab
    < trg_vocab=$SOCKEYE_VOCAB_TRG@build_vocab
    < dev_src=$DEV_SRC@bpe_data
    < dev_trg=$DEV_TRG@bpe_data
    > model
    > learning_rate
    :: num_layers=(Layers: 6 3 1)
    :: ff_size=(FeedForward: 2048 2496 1024 512)
    :: model_size=(ModelSize: 512 624 256 128)
    :: attn_heads=(AttnHeads: 8)
    :: checkpoint_freq=(CheckpointFreq: 5000 1000 500)
    :: batch_size=(BatchSize: 4096 500)
    :: label_smoothing=(LabelSmoothing: 0.1 0)
    :: .resource_flags=(Layers: 6=$resource_flags_v100 3=$resource_flags_train 1=$resource_flags_train)
    :: pyenv=@
    :: .submitter=@ .action_flags=@ {

    # Use a learning rate depending on the model size
    # lr=$(python -c "import math; print(0.003239 - 0.00013952 * math.log(2 * $num_layers * (4 * $model_size**2 + 2 * $model_size * $ff_size)))")
    lr=0.0002
    echo "Using LR $lr"
    echo $lr > $learning_rate

    $scripts/train_transformer.sh -o $model \
    --source-vocab $src_vocab \
    --target-vocab $trg_vocab \
    --initial-learning-rate=$lr \
    --source $train_src \
    --target $train_trg \
    --validation-source $dev_src \
    --validation-target $dev_trg \
    --num-layers=$num_layers \
    --transformer-feed-forward-num-hidden=$ff_size \
    --transformer-model-size=$model_size \
    --transformer-attention-heads=$attn_heads \
    --checkpoint-frequency=$checkpoint_freq \
    --batch-size=$batch_size \
    --label-smoothing=$label_smoothing
}

task continue_train : scripts
    < train_src=$TRAIN_SUBSET_SRC@subset_data
    < train_trg=$TRAIN_SUBSET_TRG@subset_data
    < src_vocab=$SOCKEYE_VOCAB_SRC@build_vocab
    < trg_vocab=$SOCKEYE_VOCAB_TRG@build_vocab
    < dev_src=$DEV_SRC@bpe_data
    < dev_trg=$DEV_TRG@bpe_data
    < model=$model@train
    :: num_layers=$num_layers@train
    :: ff_size=$ff_size@train
    :: model_size=$model_size@train
    :: attn_heads=$attn_heads@train
    :: checkpoint_freq=$checkpoint_freq@train
    :: .resource_flags=(Layers: 6=$resource_flags_v100 3=$resource_flags_train 1=$resource_flags_train)
    :: pyenv=@
    :: .submitter=@ .action_flags=@ {

    # Use a learning rate depending on the model size
    lr=0.0002

    set +e
    $scripts/train_transformer.sh -o $model \
    --source-vocab $src_vocab \
    --target-vocab $trg_vocab \
    --initial-learning-rate=$lr \
    --source $train_src \
    --target $train_trg \
    --validation-source $dev_src \
    --validation-target $dev_trg \
    --num-layers=$num_layers \
    --transformer-feed-forward-num-hidden=$ff_size \
    --transformer-model-size=$model_size \
    --transformer-attention-heads=$attn_heads \
    --checkpoint-frequency=$checkpoint_freq 2> tmp_err > tmp_out

    status=$?

   if [[ $(head -n 1 tmp_err | awk '{print $1}') == "Refusing" ]]; then
     echo 0 > exitcode
     exit 0
   fi

   if [[ $status == 0 ]]; then
     cat tmp_err >> $(dirname $model)/job.out
     echo 0 > exitcode
     exit 0
   fi

   echo $status > exitcode
   exit $status
}

task bleu_dev : scripts
    < src=$DEV_SRC@bpe_data
    < trg=$DEV_TRG@bpe_data
    < model=$model@train
    > out out_log out_scores bleu
    ::  pyenv=@
    :: .submitter=@ :: .action_flags=@ :: .resource_flags=$resource_flags_decode {

  python3 -m sockeye.translate \
    -m $model \
    $($scripts/device.sh) \
    -i $src \
    -o out.all \
    --output-type translation_with_score \
    --beam-size 12 \
    --batch-size 8 \
    --disable-device-locking

    cat out.all | cut -f 1 > $out_scores
    cat out.all | cut -f 2 > $out
    mv out.all.log $out_log

    # TODO: use something else so we don't get yelled at by Matt?
    ~/mosesdecoder/scripts/generic/multi-bleu.perl -lc $trg < $out > $bleu
}


task score_dev : scripts
    < src=$DEV_SRC@bpe_data
    < trg=$DEV_TRG@bpe_data
    < model=$model@train
    > scores
    > avgscore
    ::  pyenv=@
    :: .submitter=@ :: .action_flags=@ :: .resource_flags=$resource_flags_decode {

  python3 -m sockeye.score \
    -m $model \
    $($scripts/device.sh) \
    --source $src \
    --target $trg \
    --batch-size 100 \
    --disable-device-locking \
    > scores

  cat scores | awk '{sum+=$1;count++} END {print sum/count}' > avgscore
}