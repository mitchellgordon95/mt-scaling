  python -m sockeye.train \
    $($(dirname $0)/device.sh) \
    --encoder=transformer \
    --decoder=transformer \
    --optimizer=adam \
    --learning-rate-reduce-num-not-improved=8 \
    --learning-rate-reduce-factor=0.9 \
    --learning-rate-scheduler-type=plateau-reduce \
    --learning-rate-decay-optimizer-states-reset=best \
    --learning-rate-decay-param-reset \
    --transformer-dropout-attention=0.1 \
    --transformer-dropout-act=0.1 \
    --transformer-dropout-prepost=0.1 \
    --label-smoothing=0.1 \
    --max-num-checkpoint-not-improved 10 \
    --batch-type=word \
    --batch-size=4096 \
    --decode-and-evaluate=0 \
    --checkpoint-frequency=5000 \
    --disable-device-locking \
    --keep-last-params=10 \
    --max-updates=250000 \
    "$@"
    # Defaults
    # --transformer-positional-embedding-type=fixed \
    # --transformer-preprocess=n \
    # --transformer-postprocess=dr \
    # --weight-init=xavier \
    # --weight-init-scale=3.0 \
    # --weight-init-xavier-factor-type=avg \
    # --optimized-metric=perplexity \
    # --gradient-clipping-threshold=1 \
