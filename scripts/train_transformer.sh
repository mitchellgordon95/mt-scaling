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
    --checkpoint-frequency=5000 \
    --decode-and-evaluate=-1 \
    --decode-and-evaluate-use-cpu \
    --disable-device-locking \
    --keep-last-params=10 \
    --max-num-epochs=100 \
    --max-updates=300000 \
    "$@"
    # These are default
    # --transformer-positional-embedding-type=learned \
    # --transformer-preprocess=n \
    # --transformer-postprocess=dr \
    # --weight-init=xavier \
    # --weight-init-scale=3.0 \
    # --weight-init-xavier-factor-type=avg \
    # --optimized-metric=perplexity \
    # --gradient-clipping-threshold=1 \