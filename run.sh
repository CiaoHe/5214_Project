
echo "Continual Learning Experiments"
CUDA_VISIBLE_DEVICES=4 python run_naive.py --lr 1e-4 --epochs 10 --log_f_name naive_lr4

# CUDA_VISIBLE_DEVICES=7 python run_gem.py --only_mlp --custom_name 50 --patterns_per_exp 50 --epochs 2 --memory_strength 0.2

# CUDA_VISIBLE_DEVICES=4 python run_ewc.py --only_mlp --ewc_lambda 0.4 --custom_name lambda0.4



# OUT_PATH=/home/nayeon/Course_5214_Project/save

# # CUDA_VISIBLE_DEVICES=4 python run_tran.py \
# #     --do_eval \
# #     --train_file data/train_clean.json \
# #     --validation_file data/test_clean.json \
# #     --model_name_or_path '/home/nayeon/Course_5214_Project/save/LR.2e-5.BSZ.8.ACC.4' \
# #     --output_dir $OUT_PATH 


# OUT_PATH=/home/nayeon/Course_5214_Project/save
# # TRAIN_PATH=data/train_clean.json
# # TEST_PATH=data/test_clean.json
# TRAIN_PATH=data/train_split.json
# TEST_PATH=data/test_split.json

# CUDA=4
# LR=2e-5
# BSZ=8
# ACC=4
# MODEL_NAME=LR.$LR.BSZ.$BSZ.ACC.$ACC
# CUDA_VISIBLE_DEVICES=$CUDA python run_tran.py \
#     --do_train \
#     --do_eval \
#     --train_file $TRAIN_PATH \
#     --validation_file $TEST_PATH \
#     --model_name_or_path bert-base-uncased \
#     --output_dir $OUT_PATH/$MODEL_NAME \
#     --num_train_epochs 10 \
#     --learning_rate $LR \
#     --per_device_train_batch_size $BSZ \
#     --gradient_accumulation_steps $ACC \
#     --per_device_eval_batch_size $BSZ \
#     --overwrite_output_dir




# OUT_PATH=/home/nayeon/Course_5214_Project/save
# TRAIN_PATH=data/train_clean.json
# TEST_PATH=data/test_clean.json
# # TRAIN_PATH=data/train_split.json
# # TEST_PATH=data/test_split.json

# CUDA=1
# LR=2e-4
# BSZ=8
# ACC=4
# MODEL_NAME=LR.$LR.BSZ.$BSZ.ACC.$ACC
# CUDA_VISIBLE_DEVICES=$CUDA python run_tran.py \
#     --do_train \
#     --do_eval \
#     --train_file $TRAIN_PATH \
#     --validation_file $TEST_PATH \
#     --model_name_or_path bert-base-uncased \
#     --output_dir $OUT_PATH/$MODEL_NAME \
#     --num_train_epochs 5 \
#     --learning_rate $LR \
#     --per_device_train_batch_size $BSZ \
#     --gradient_accumulation_steps $ACC \
#     --per_device_eval_batch_size $BSZ \
#     --overwrite_output_dir