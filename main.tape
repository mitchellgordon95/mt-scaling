import "tapes/submitters.tape"
import "tapes/versioners.tape"

## Tasks start here
import "tapes/data_prep.tape"
import "tapes/train.tape"

plan data_scaling {
  # WMT
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: *) * (Layers: 6) * (ModelSize: 512) * (FeedForward: 2048)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: *) * (Layers: 6) * (ModelSize: 624) * (FeedForward: 2496)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: *) * (Layers: 3) * (ModelSize: 512) * (FeedForward: 2048)
  # reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: *) * (Layers: 3) * (ModelSize: 256) * (FeedForward: 1024)
  # reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: *) * (Layers: 3) * (ModelSize: 128) * (FeedForward: 512)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: *) * (Layers: 1) * (ModelSize: 512) * (FeedForward: 2048)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: *) * (Layers: 1) * (ModelSize: 256) * (FeedForward: 1024)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: *) * (Layers: 1) * (ModelSize: 128) * (FeedForward: 512)
}

plan continue {
  # WMT
  reach continue_train via (Lang: deen ruen zhen) * (DataPercent: *) * (Layers: 6) * (ModelSize: 512) * (FeedForward: 2048)
  reach continue_train via (Lang: deen ruen zhen) * (DataPercent: *) * (Layers: 6) * (ModelSize: 624) * (FeedForward: 2496)
  reach continue_train via (Lang: deen ruen zhen) * (DataPercent: *) * (Layers: 3) * (ModelSize: 512) * (FeedForward: 2048)
  reach continue_train via (Lang: deen ruen zhen) * (DataPercent: *) * (Layers: 1) * (ModelSize: 512) * (FeedForward: 2048)
  reach continue_train via (Lang: deen ruen zhen) * (DataPercent: *) * (Layers: 1) * (ModelSize: 256) * (FeedForward: 1024)
  reach continue_train via (Lang: deen ruen zhen) * (DataPercent: *) * (Layers: 1) * (ModelSize: 128) * (FeedForward: 512)
}

plan low_resource {
  reach bleu_dev, score_dev via (Lang: swen soen tlen) * (DataPercent: 100 50 25 12.5 6.25) * (Layers: 6) * (ModelSize: 512) * (FeedForward: 2048) * (CheckpointFreq: 1000)
}

summary data_scaling {
  of train
    # > best_checkpoint
    > data_bytes {
    # tail -n 1 $(dirname $model)/job.out | awk '{for(i=1; i<=NF; i++) {if($i=="checkpoint:") print $(i+1)}}' > $best_checkpoint
    src_size=$(du $train_src | awk '{print $1}')
    trg_size=$(du $train_trg | awk '{print $1}')
    python -c "print($src_size + $trg_size)" > $data_bytes
  }
  of bleu_dev > bleu_dev {
    cat $bleu | awk '{print substr($3, 1, length($3) - 1)}' > $bleu_dev
  }
  of score_dev > ent_dev {
    cat $avgscore > $ent_dev
  }

}

# Nuts and bolts:
global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}

