dataset_prefix='/data/datasets/ribfrac'
history_path='/data/PyTorch_model/FracNet/history/'
train_image_dir=$dataset_prefix/'ribfrac-train-images/train_image'
train_label_dir=$dataset_prefix/'ribfrac-train-images/train_label'
val_image_dir=$dataset_prefix/'ribfrac-val-images/val_image'
val_label_dir=$dataset_prefix/'ribfrac-val-images/val_label'
test_image_dir=$dataset_prefix/'ribfrac-test-images'

#train
#python -m main --train_image_dir /data/datasets/ribfrac/ribfrac-train-images/train_image --train_label_dir /data/datasets/ribfrac/ribfrac-train-images/train_label --val_image_dir /data/datasets/ribfrac/ribfrac-val-images/val_image --val_label_dir /data/datasets/ribfrac/ribfrac-val-images/val_label --save_model True
python -m main --train_image_dir "$train_image_dir" --train_label_dir "$train_label_dir" --val_image_dir "$val_image_dir" --val_label_dir "$val_label_dir" --save_model True
if [[ $? -ne 0 ]]; then
  echo 'training error';
  exit;
fi

cur_type='test';
if [ "$cur_type" = "test" ];then
  image_dir="$test_image_dir";
elif [ "$cur_type" = "val" ]; then
  image_dir="$val_image_dir";
elif [ "$cur_type" = "train" ]; then
  image_dir="$train_image_dir";
fi


#model_dir='/data/PyTorch_model/FracNet/history/1641882725.147945';
model_dir=$(realpath "$history_path/"`ls "$history_path"| sort -r | head -1`)
echo $model_dir;
mkdir -p "$model_dir/$cur_type/prediction_directory"

# test
python -m predict --image_dir "$image_dir" --pred_dir "$model_dir/$cur_type/prediction_directory" --model_path "$model_dir/model_weights.pth"
if [[ $? -ne 0 ]]; then
  echo 'testing error';
  exit;
fi
pre=`pwd`
cd "$model_dir/$cur_type"
zip -r prediction_directory.zip prediction_directory
cd "$pre";
#python -m predict --image_dir /data/datasets/ribfrac/ribfrac-test-images --pred_dir /data/PyTorch_model/FracNet/history/1641717302.3811588/test/prediction_directory --model_path /data/PyTorch_model/FracNet/history/1641717302.3811588/model_weights.pth
