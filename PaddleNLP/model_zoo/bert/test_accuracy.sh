unset CUDA_VISIBLE_DEVICES
python3 -m paddle.distributed.launch run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name SST2 \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 500 \
    --output_dir ./tmp_1222/ \
    --device cpu \
    --use_amp False \
    --s4_model_path ../../../../paddle/models/bert-base/model.bin
#    --s4_model_path ../../../../paddle/model-builder/compile_output/bert-base-uncased/model.bin
