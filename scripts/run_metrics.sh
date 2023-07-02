MODEL=$1

# test for davis16
CMD1="python davis-evaluation/eval.py -i ./results/${MODEL}/val_davis2016_val/hvc_vos -o ./results/${MODEL}/val_davis2016_test/hvc_vos/hvc_16.yaml --year 2016 --single-object --phase val"

# test for davis17
CMD2="python davis-evaluation/eval.py -i ./results/${MODEL}/val_davis2017_val/hvc_vos -o ./results/${MODEL}/val_davis2017_val/hvc_vos/hvc_17.yaml --year 2017 --phase val"

DS=$2
case $DS in
davis17)
  echo "Test dataset: DAVIS-2017 (val)"
  $CMD2
  ;;
davis16)
  echo "Test dataset: DAVIS-2016 (val)"
  $CMD1
  ;;
*)
  echo "Dataset '$DS' not recognised. Should be one of [davis16|davis17]."
  exit 1
  ;;
esac
