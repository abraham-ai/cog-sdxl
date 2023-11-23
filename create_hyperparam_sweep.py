import random, os, ast
from itertools import product
import time

random.seed(int(1000*time.time()))

def hamming_distance(dict1, dict2):
    distance = 0
    for key in dict1.keys():
        if dict1[key] != dict2.get(key, None):
            distance += 1
    return distance

#######################################################################################
#######################################################################################

"""


Faces:
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_2.zip
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_5.zip
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_best.zip
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip

Objects:
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/banny_all.zip
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/banny_best.zip
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/koji_color.zip
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/plantoid_imgs.zip

Styles:
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/does.zip
https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_200.zip



"""


# Setup the base experiment config:
#lora_training_urls    = "https://minio.aws.abraham.fun/creations-stg/d6f8446d13a82bc159f4b26aadca90a888493e92cf0bab1e510cb5354fb065a7.zip|https://minio.aws.abraham.fun/creations-stg/991d70ba870022aef6c893b8335fee53ed9a32e8f998e23ec9dcf2adc0ee3f76.zip|https://minio.aws.abraham.fun/creations-stg/6b25015c2f655915c90c41fc35cc5f42f8a877307c2a8affc2d47ed812cf23c3.zip|https://minio.aws.abraham.fun/creations-stg/fbdc59246ee841bb8303787155a6a0c5cae56d7545a9bd0d5d077a9d8193baff.zip"
run_name             = "xander"
caption_prefix       = ""  # "" to activate chatgpt
mask_target_prompts  = ""  # "" to activate chatgpt
n_exp                = 300  # how many random experiment settings to generate
min_hamming_distance = 1   # min_n_params that have to be different from any previous experiment to be scheduled

# Define training hyperparameters and their possible values
# The params are sampled stochastically, so if you want to use a specific value more often, just put it in multiple times

if 0:
    hyperparameters = {
        'lora_training_urls': [
            "https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/does.zip",
            "https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/clipx_200.zip",
                                ],
        'concept_mode': ['object'],
        'left_right_flip_augmentation': ['True', 'False'],
        'augment_imgs_up_to_n': ['0', '30'],
        'resolution': [1024],
        'hard_pivot': ['False', 'True'],
        'ti_lr': ['1e-3', '4e-3'],
        'lora_weight_decay': ['0.001', '0.005'],
        'ti_weight_decay': ['1e-4'],
        'lora_rank': ['4', '8'],
        'checkpointing_steps': ['1200'],
        'prodigy_d_coef': ['0.4', '1.0'],
        'max_train_steps': ['1200'],
        'train_batch_size': ['2'],
        'seed': ['0'],
        'debug': ['True']   # avoid sending the entire .rar file back after each training run (takes a long time)
    }
else:
    hyperparameters = {
        'lora_training_urls': [
            "https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_2.zip",
            "https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/xander_5.zip",
            "https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/steel.zip",
            "https://storage.googleapis.com/public-assets-xander/A_workbox/lora_training_sets/egon.zip",
        ],
        'concept_mode': ['face'],
        'left_right_flip_augmentation': ['False'],
        'augment_imgs_up_to_n': ['0', '20'],
        'resolution': [1024],
        'hard_pivot': ['False', 'True'],
        'ti_lr': ['2e-3'],
        'lora_weight_decay': ['0.002'],
        'ti_weight_decay': ['1e-4'],
        'lora_rank': ['6'],
        'checkpointing_steps': ['1200'],
        'prodigy_d_coef': ['0.5'],
        'max_train_steps': ['1200'],
        'train_batch_size': ['2'],
        'seed': ['0'],
        'debug': ['True']   # avoid sending the entire .rar file back after each training run (takes a long time)
        }

#######################################################################################
#######################################################################################

output_filename = f"training_exp_{run_name}.sh"

# Create a set to hold the combinations that have already been run
scheduled_experiments = set()

# if output_filename exists, remove it:
os.remove(output_filename) if os.path.exists(output_filename) else None

# Open the shell script file
try_sampling_n_times = 500
with open(output_filename, "w") as f:
    for exp_index in range(n_exp):  # number of combinations you want to generate
        resamples, combination = 0, None

        while resamples < try_sampling_n_times:
            experiment_settings = {name: random.choice(values) for name, values in hyperparameters.items()}
            resamples += 1

            min_distance = float('inf')
            for str_experiment_settings in scheduled_experiments:
                existing_experiment_settings = dict(sorted(ast.literal_eval(str_experiment_settings)))
                distance = hamming_distance(experiment_settings, existing_experiment_settings)
                min_distance = min(min_distance, distance)

            if min_distance >= min_hamming_distance:
                str_experiment_settings = str(sorted(experiment_settings.items()))
                scheduled_experiments.add(str_experiment_settings)
                # Write the experiment to the shell script file, if necessary.
                break

        if resamples == try_sampling_n_times:
            print(f"\nCould not find a new experiment_setting after random sampling {try_sampling_n_times} times, dumping all experiment_settings to .sh script")
            break

        # extract the dataset name:
        for name, value in sorted(experiment_settings.items()): 
            if name == "lora_training_urls":
                dataset_name = value.split('/')[-1][:30]

        # add experiment_settings to run_name:
        run_name_exp = f"{run_name}_{dataset_name}_{exp_index:03d}_{int(time.time()*1000)%10000}"

        f.write(f'cog predict \\\n')
        f.write(f'    -i run_name="{run_name_exp}" -i caption_prefix="{caption_prefix}" \\\n')
        f.write(f'    -i mask_target_prompts="{mask_target_prompts}" \\\n')

        for name, value in sorted(experiment_settings.items()): 
            if type(value) == str:
                f.write(f'    -i {name}="{value}" \\\n')
            else:
                f.write(f'    -i {name}={value} \\\n')
        
        # Remove the last backslash and add a new line
        f.seek(f.tell() - 3)
        f.write('  \n\n')

print(f"\n\n---> Saved {len(scheduled_experiments)} experiment commands to {output_filename}")
print("To run this experiment:")
print(f"sudo sh {output_filename}")
