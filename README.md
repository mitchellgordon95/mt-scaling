# Training Details

Data pre-processing steps can be found in tapes/data_prep.tape. We use
[subword-nmt](https://github.com/rsennrich/subword-nmt) to construct BPE vocabularies and assume the data has already
been tokenized. Models in this repository are trained using the
[sockeye](https://github.com/awslabs/sockeye) framework. See requirements.txt
for the pinned version. Training arguments are split between scripts/train.sh
and tapes/train.tape.

# Raw Results

The raw results used to create the plots in our paper can be found in in the
summaries folder. These are tab-delimited csv files. Example usage of these
csv's can be found in plots/main.py. Beware: older summaries may be out of date
with the current version of main.tape.

# Replicating Experiments

These experiments are managed using the [ducttape framework](https://github.com/jhclark/ducttape), version
[0.4](https://github.com/ExperimentWith/ducttape/files/1725708/ducttape.v0.4.binary.zip). We recommend
reading the [introduction](http://mitchgordon.me/ml/2021/02/09/ducttape.html) and/or the
[tutorial](https://github.com/jhclark/ducttape/blob/master/tutorial/TUTORIAL.md)
before attempting to work with this repository. Before running any tasks, there are a few steps you should take:

1. Update main.tconf, point the "scripts" path to the scripts folder in this directory. 

2. If your cluster management system is not Univa Grid Engine, you must do two things:
- Update the SGE submitter in tapes/submitters.tape to work with your cluster.
- Update the flags that should be used in main.tconf for different types of jobs (training, decoding, etc.).

3. In data_prep.tape, point tok_train_src, tok_train_trg, tok_dev_src, and tok_dev_trg to the locations of the tokenized data on your machine.

If you have done these steps, you should be able to run

`ducttape main.tape -C main.tconf -p [plan_name]`

to replicate our results. See main.tape for the different plan names available.
