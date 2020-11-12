from datasets import load_dataset
import os

if not os.path.exists('data'):
    os.makedirs('data')

NUM_EPOCHS = 0.2

CKPT_DIR=f'/scratch/ua388/ckpts'
if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)

def main(dataset_name, dataset_subname=None, key='text', val_name='validation', key2=None):
    if dataset_subname is not None:
        fname = dataset_name + '_' + dataset_subname
        train, val = load_dataset(dataset_name, dataset_subname, split=['train', val_name])
    else:
        fname = dataset_name
        train, val = load_dataset(dataset_name, split=['train', val_name])

    print(f"Processing dataset {fname}...")

    if key2 is None:
        train_str = " <|endoftext|>\n".join(train[key])
        val_str = " <|endoftext|>\n".join(val[key])
    else:
        train_str = " <|endoftext|>\n".join(train[key])
        train_str += " <|endoftext|>\n".join(train[key2])
        val_str = " <|endoftext|>\n".join(val[key])
        val_str += " <|endoftext|>\n".join(val[key2])

    fname_train = f'data/{fname}_train.txt'
    fname_val = f'data/{fname}_val.txt'
    with open (fname_train, 'w') as f:
        f.write(train_str)
    with open (fname_val, 'w') as f:
        f.write(val_str)

    print("Running fine-tuning...")

    cmd = 'python run_language_modeling.py ' + \
    f'--train_data_file {fname_train} ' + \
    f'--eval_data_file {fname_val} ' + \
    f'--output_dir {CKPT_DIR}/gpt2-{fname}-{NUM_EPOCHS} ' + \
    '--model_type gpt2 ' + \
    '--model_name_or_path gpt2 ' + \
    '--save_total_limit 1 ' + \
    f'--num_train_epochs {NUM_EPOCHS} ' + \
    '--do_train \
    --evaluate_during_training \
    --logging_steps 500 \
    --save_steps 500 \
    --do_eval \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --line_by_line \
    --gradient_accumulation_steps 1'

    os.system(cmd)

if __name__ == '__main__':
    main(dataset_name='glue', dataset_subname='sst2', key='sentence')
    # main(dataset_name='yelp_polarity', val_name='test')
    #main(dataset_name='glue', dataset_subname='mnli', key='premise', key2='hypothesis', val_name='validation_mismatched')
