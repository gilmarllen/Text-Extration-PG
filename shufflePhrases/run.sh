# See what problems, models, and hyperparameter sets are available.
# You can easily swap between them (and add new ones).
# t2t-trainer --registry_help

PROBLEM=shuffle_problem
MODEL=transformer
HPARAMS=transformer_base

USR_DIR=./shuffle_problem
DATA_DIR=./data
TMP_DIR=./tmp
TRAIN_DIR=./train/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
#t2t-datagen \
#  --t2t_usr_dir=$USR_DIR \
#  --data_dir=$DATA_DIR \
#  --tmp_dir=$TMP_DIR \
#  --problem=$PROBLEM

#echo "FINISHED GEN DATA"

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
#t2t-trainer \
#  --t2t_usr_dir=$USR_DIR \
#  --data_dir=$DATA_DIR \
#  --problem=$PROBLEM \
#  --model=$MODEL \
#  --hparams_set=$HPARAMS \
#  --keep_checkpoint_max=2 \
#  --output_dir=$TRAIN_DIR

#echo "TRAIN FINISH"

# Decode

#DECODE_FILE=$DATA_DIR/decode_this.txt
DECODE_FILE=$TMP_DIR/shuffle_in_test.txt
#echo "This a is training of test the" >> $DECODE_FILE
#echo 'This is a test of the training' > ref-translation.txt

BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DECODE_FILE \
  --decode_to_file=translated.txt

# Evaluate the BLEU score
# Note: Report this BLEU score in papers, not the internal approx_bleu metric.
t2t-bleu --translation=translated.txt --reference=$TMP_DIR/shuffle_out_test.txt

