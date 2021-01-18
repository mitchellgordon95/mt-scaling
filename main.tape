import "tapes/submitters.tape"
import "tapes/versioners.tape"

## Tasks start here
import "tapes/data_prep.tape"
import "tapes/train.tape"

plan data_scaling {
  # WMT
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 6) * (ModelSize: 512) * (FeedForward: 2048)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 6) * (ModelSize: 624) * (FeedForward: 2496)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 3) * (ModelSize: 512) * (FeedForward: 2048)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 1) * (ModelSize: 512) * (FeedForward: 2048)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 1) * (ModelSize: 256) * (FeedForward: 1024)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 1) * (ModelSize: 128) * (FeedForward: 512)

  # Low-resource settings need a smaller checkpoint freq
  # reach bleu_dev, score_dev via (Lang: deen) * (DataPercent: 0.39 0.2 0.1 0.05) * (Layers: *) * (ModelSize: 512) * (FeedForward: 2048) * (CheckpointFreq: 500) * (ShuffleSeed: *)
  # reach bleu_dev, score_dev via (Lang: ruen) * (DataPercent: 0.1 0.05 0.025 0.0125) * (Layers: *) * (ModelSize: 512) * (FeedForward: 2048) * (CheckpointFreq: 500) * (ShuffleSeed: *)
  # reach bleu_dev, score_dev via (Lang: zhen) * (DataPercent: 0.2 0.1 0.05 0.025) * (Layers: *) * (ModelSize: 512) * (FeedForward: 2048) * (CheckpointFreq: 500) * (ShuffleSeed: *)
}

plan smaller_bpe {
  # WMT
  reach bleu_dev, score_dev via (Lang: deen) * (DataPercent: 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 6) * (ModelSize: 624) * (FeedForward: 2496) * (BPE: 2000) * (CheckpointFreq: 500)
  reach bleu_dev, score_dev via (Lang: deen) * (DataPercent: 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 1) * (ModelSize: 512) * (FeedForward: 2048) * (BPE: 2000) * (CheckpointFreq: 500)
  reach bleu_dev, score_dev via (Lang: deen) * (DataPercent: 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 1) * (ModelSize: 256) * (FeedForward: 1024) * (BPE: 2000) * (CheckpointFreq: 500)
  reach bleu_dev, score_dev via (Lang: deen) * (DataPercent: 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 1) * (ModelSize: 128) * (FeedForward: 512) * (BPE: 2000) * (CheckpointFreq: 500)
}

plan random_baseline {
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 0.2) * (Layers: 3) * (ModelSize: 512) * (FeedForward: 2048) * (CheckpointFreq: 500) * (MaxUpdates: 0)
}

plan low_resource {
  reach bleu_dev, score_dev via (Lang: swen soen tlen) * (DataPercent: 100 50 25 12.5) * (Layers: 3 6) * (ModelSize: 512) * (FeedForward: 2048) * (BPE: 2000) * (ShuffleSeed: *) * (BatchSize: 500) * (CheckpointFreq: 500)
}

plan subset_stats {
  reach subset_stats via (Lang: deen ruen zhen) * (DataPercent: *)
  reach subset_stats via (Lang: deen) * (DataPercent: *) * (BPE: 2000)
}

summary data_scaling {
  of train
    # > best_checkpoint
    > data_bytes {
    # tail -n 1 $(dirname $model)/job.out | awk '{for(i=1; i<=NF; i++) {if($i=="checkpoint:") print $(i+1)}}' > $best_checkpoint
    src_size=$(du $train_src | awk '{print $1}')
    trg_size=$(du $train_trg | awk '{print $1}')
    python -c "print(($src_size + $trg_size) * 512)" > $data_bytes
  }
  of bleu_dev > bleu_dev {
    cat $bleu | awk '{print substr($3, 1, length($3) - 1)}' > $bleu_dev
  }
  of score_dev > ent_dev {
    cat $avgscore > $ent_dev
  }
  of subset_stats > src_coverage > trg_coverage > vocab_overlap_src > vocab_overlap_trg
  > data_bytes > line_count { # Note: data_bytes is summarized twice, depending on which plan we're using (data_scaling vs. subset_stats).

    cat $src_dev_coverage > $src_coverage
    cat $trg_dev_coverage > $trg_coverage
    cat $src_vocab_overlap > $vocab_overlap_src
    cat $trg_vocab_overlap > $vocab_overlap_trg
    src_size=$(du $subset_src | awk '{print $1}')
    trg_size=$(du $subset_trg | awk '{print $1}')
    python -c "print(($src_size + $trg_size) * 512)" > $data_bytes
    wc -l $subset_src | awk '{print $1}' > $line_count
  }
}


# Nuts and bolts:
global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}

