import "tapes/submitters.tape"
import "tapes/versioners.tape"

## Tasks start here
import "tapes/data_prep.tape"
import "tapes/train.tape"

plan data_scaling {
  reach bleu_dev, score_dev via (Lang: deen) * (DataPercent: *)
}

summary data_scaling {
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

