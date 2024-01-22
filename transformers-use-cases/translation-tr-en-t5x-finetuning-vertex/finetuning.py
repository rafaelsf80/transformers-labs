from datetime import datetime
from google.cloud import aiplatform as vertex_ai
import utils

PROJECT_ID = 'argolis-rafaelsanchez-ml-dev' 
REGION = 'europe-west4' 
BUCKET = 'argolis-t5x-vertex-new' 
TFDS_DATA_DIR = 'gs://argolis-t5x-vertex-new/datasets'
TENSORBOARD_NAME = 'tb-t5x-vertex'
IMAGE_URI = f'gcr.io/{PROJECT_ID}/t5x-base'

TRAIN_PATH = 'gs://argolis-t5x-vertex-new/datasets/train.tfrecords'
VALIDATION_PATH = 'gs://argolis-t5x-vertex-new/datasets/eval.tfrecords'

EXPERIMENT_NAME = 'experiment-t5x-tr-en-vertex'
EXPERIMENT_WORKSPACE = f'gs://{BUCKET}/experiments/{EXPERIMENT_NAME}'
EXPERIMENT_RUNS = f'{EXPERIMENT_WORKSPACE}/runs'

JOB_GIN_FILE = 'configs/finetune_t511_base_tr_en.gin'

GIN_FILES = [JOB_GIN_FILE]  
GIN_OVERWRITES = [
        'USE_CACHED_TASKS=False',
        f'TRAIN_PATH="{TRAIN_PATH}"',
        f'VALIDATION_PATH="{VALIDATION_PATH}"',
    ]


vertex_ai.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=EXPERIMENT_WORKSPACE,
    experiment=EXPERIMENT_NAME
)

# Custom job constants
MACHINE_TYPE = 'cloud-tpu'
ACCELERATOR_TYPE = 'TPU_V2'
ACCELERATOR_COUNT = 8
RUN_NAME = f't5-tr-en-finetune-v2-8' 
RUN_ID = f'{EXPERIMENT_NAME}-{RUN_NAME}-{datetime.now().strftime("%Y%m%d%H%M")}'
RUN_DIR = f'{EXPERIMENT_RUNS}/{RUN_ID}'
RUN_MODE = 'train'

job = utils.create_t5x_custom_job(
    display_name=RUN_ID,
    machine_type=MACHINE_TYPE,
    accelerator_type=ACCELERATOR_TYPE,
    accelerator_count=ACCELERATOR_COUNT,
    image_uri=IMAGE_URI,
    run_mode=RUN_MODE,
    gin_files=GIN_FILES,
    model_dir=RUN_DIR,
    tfds_data_dir=TFDS_DATA_DIR,
    gin_overwrites=GIN_OVERWRITES
)

utils.submit_and_track_t5x_vertex_job(
    custom_job=job,
    job_display_name=RUN_ID,
    run_name=RUN_ID,
    experiment_name=EXPERIMENT_NAME,
    execution_name=RUN_ID,
    tfds_data_dir=TFDS_DATA_DIR,
    model_dir=RUN_DIR,
    vertex_ai=vertex_ai,
    run_mode=RUN_MODE
)

