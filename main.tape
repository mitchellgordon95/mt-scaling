import "tapes/submitters.tape"
import "tapes/versioners.tape"

## Tasks start here
import "tapes/data_prep.tape"
import "tapes/train.tape"

plan data_scaling {
  # WMT
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 6) * (ModelSize: 512) * (FeedForward: 2048) * (BleuType: *)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 6) * (ModelSize: 624) * (FeedForward: 2496) * (BleuType: *)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 3) * (ModelSize: 512) * (FeedForward: 2048) * (BleuType: *)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 1) * (ModelSize: 512) * (FeedForward: 2048) * (BleuType: *)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 1) * (ModelSize: 256) * (FeedForward: 1024) * (BleuType: *)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 1) * (ModelSize: 128) * (FeedForward: 512) * (BleuType: *)
}

plan sacrebleu {
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 6) * (ModelSize: 512) * (FeedForward: 2048) * (BleuType: detok)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 6) * (ModelSize: 624) * (FeedForward: 2496) * (BleuType: detok)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 3) * (ModelSize: 512) * (FeedForward: 2048) * (BleuType: detok)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 1) * (ModelSize: 512) * (FeedForward: 2048) * (BleuType: detok)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 1) * (ModelSize: 256) * (FeedForward: 1024) * (BleuType: detok)
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 100 50 25 12.5 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 1) * (ModelSize: 128) * (FeedForward: 512) * (BleuType: detok)
}

plan smaller_bpe {
  reach bleu_dev, score_dev via (Lang: deen) * (DataPercent: 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 6) * (ModelSize: 624) * (FeedForward: 2496) * (BPE: 2000) * (CheckpointFreq: 500)
  reach bleu_dev, score_dev via (Lang: deen) * (DataPercent: 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 1) * (ModelSize: 512) * (FeedForward: 2048) * (BPE: 2000) * (CheckpointFreq: 500)
  reach bleu_dev, score_dev via (Lang: deen) * (DataPercent: 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 1) * (ModelSize: 256) * (FeedForward: 1024) * (BPE: 2000) * (CheckpointFreq: 500)
  reach bleu_dev, score_dev via (Lang: deen) * (DataPercent: 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025) * (Layers: 1) * (ModelSize: 128) * (FeedForward: 512) * (BPE: 2000) * (CheckpointFreq: 500)

  reach bleu_dev, score_dev via (Lang: ruen zhen) * (DataPercent: 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025 0.0125) * (Layers: 6) * (ModelSize: 624) * (FeedForward: 2496) * (BPE: 2000) * (CheckpointFreq: 500)
  reach bleu_dev, score_dev via (Lang: ruen zhen) * (DataPercent: 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025 0.0125) * (Layers: 1) * (ModelSize: 512) * (FeedForward: 2048) * (BPE: 2000) * (CheckpointFreq: 500)
  reach bleu_dev, score_dev via (Lang: ruen zhen) * (DataPercent: 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025 0.0125) * (Layers: 1) * (ModelSize: 256) * (FeedForward: 1024) * (BPE: 2000) * (CheckpointFreq: 500)
  reach bleu_dev, score_dev via (Lang: ruen zhen) * (DataPercent: 6.25 3.125 1.56 0.78 0.39 0.2 0.1 0.05 0.025 0.0125) * (Layers: 1) * (ModelSize: 128) * (FeedForward: 512) * (BPE: 2000) * (CheckpointFreq: 500)
}

plan random_baseline {
  reach bleu_dev, score_dev via (Lang: deen ruen zhen) * (DataPercent: 0.2) * (Layers: 3) * (ModelSize: 512) * (FeedForward: 2048) * (CheckpointFreq: 500) * (MaxUpdates: 0)
}

plan low_resource {
  reach bleu_dev, score_dev via (Lang: swen soen tlen) * (DataPercent: 100 50 25 12.5) * (Layers: 1 3 6) * (ModelSize: 512) * (FeedForward: 2048) * (BPE: 2000) * (ShuffleSeed: 1) * (BatchSize: 500) * (CheckpointFreq: 500)
}

plan subset_stats {
  reach subset_stats via (Lang: deen ruen zhen) * (DataPercent: *)
  reach subset_stats via (Lang: deen) * (DataPercent: *) * (BPE: 2000)
}

plan language_model {
  reach language_model via (Lang: deen) * (DataPercent: 25 12.5 1.56) * (BPE: 30000 2000 50000)
  reach language_model via (Lang: ruen zhen) * (DataPercent: 25 12.5 1.56) * (BPE: 30000 2000)
  reach language_model via (Lang: tlen soen swen) * (DataPercent: 100) * (BPE: 2000)
}

plan word_count {
  reach word_count via (Lang: deen ruen zhen) * (DataPercent: *) * (BPE: 30000 2000)
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
    if [[ $bleu_type == bpe ]]; then
      cat $bleu | awk '{print substr($3, 1, length($3) - 1)}' > $bleu_dev
    else
      cat $bleu | awk '{print $3}' > $bleu_dev
    fi
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
  of language_model > lm_score {
    cat $avgscore > $lm_score
  }
  of word_count > unique_word_count {
    cat $unique_words > $unique_word_count
  }
}


# Nuts and bolts:
global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}

