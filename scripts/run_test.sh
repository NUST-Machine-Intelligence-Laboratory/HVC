gpus='0'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "Using gpus ${gpus}"
now=$(date +"%T")
echo "Current time : $now"

MODEL=$1
# # echo "Set '$RUN_ID' not recognised. Should be one of [your run id]."

# MODEL_DIR=$2
# # echo "Set '$MODEL_DIR' not recognised. Should be one of [your model path]."

DS=$2
case $DS in
ytvos)
  echo "Test dataset: YouTube-VOS 2018 (val)"
  FILELIST=filelists/val_ytvos2018_test
  ;;
davis17)
  echo "Test dataset: DAVIS-2017 (val)"
  FILELIST=filelists/val_davis2017_val
  ;;
davis17dev)
  echo "Test dataset: DAVIS-2017 (dev)"
  FILELIST=filelists/val_davis2017_dev
  ;;
davis16val)
  echo "Test dataset: DAVIS-2016 (val)"
  FILELIST=filelists/val_davis2016_val
  ;;
*)
  echo "Dataset '$DS' not recognised. Should be one of [ytvos|davis17val|davis17dev|davis16val]."
  exit 1
  ;;
esac

OUTPUT_DIR=./results
LISTNAME=`basename $FILELIST .txt`
SAVE_DIR=$OUTPUT_DIR/$MODEL/$LISTNAME
LOG_FILE=$OUTPUT_DIR/$MODEL/${LISTNAME}.log
NUM_THREADS=32
export OMP_NUM_THREADS=$NUM_THREADS
export MKL_NUM_THREADS=$NUM_THREADS
CMD="python infer_vos.py   --infer-list $FILELIST \
                           --mask-output-dir $SAVE_DIR
                           --test-model $MODEL"
if [ ! -d $SAVE_DIR ]; then
  echo "Creating directory: $SAVE_DIR"
  mkdir -p $SAVE_DIR
else
  echo "Saving to: $SAVE_DIR"
fi

echo $CMD > ${SAVE_DIR}.cmd
$CMD
