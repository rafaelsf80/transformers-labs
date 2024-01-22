from datetime import datetime
from google.cloud import aiplatform as vertex_ai
import utils

PROJECT_ID = 'argolis-rafaelsanchez-ml-dev' 
REGION = 'europe-west4' 
BUCKET = 'argolis-t5x-vertex-new' 
TFDS_DATA_DIR = 'gs://argolis-t5x-vertex-new/datasets'
TENSORBOARD_NAME = 'tb-t5x-vertex'
IMAGE_URI = f'gcr.io/{PROJECT_ID}/t5x-base'

EXPERIMENT_NAME = 'experiment-t5x-tr-en-vertex-new'
EXPERIMENT_WORKSPACE = f'gs://{BUCKET}/experiments/{EXPERIMENT_NAME}'
EXPERIMENT_RUNS = f'{EXPERIMENT_WORKSPACE}/runs'

GIN_FILES = ['configs/infer_t511_base_tr_en.gin'] 


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
INFER_RUN_NAME = f't5-tr-en-infer-v2-8'
INFER_RUN_ID = f'{EXPERIMENT_NAME}-{INFER_RUN_NAME}-{datetime.now().strftime("%Y%m%d%H%M")}'
INFER_RUN_DIR = f'{EXPERIMENT_RUNS}/{INFER_RUN_ID}'
RUN_MODE = 'infer'
CHECKPOINT_PATH = 'gs://argolis-t5x-vertex-new/experiments/experiment-t5x-tr-en-vertex/runs/experiment-t5x-tr-en-vertex-t5-tr-en-finetune-v2-8-202209081834/checkpoint_1001000/' 
GIN_OVERWRITES = [
    'USE_CACHED_TASKS=False',
    f'CHECKPOINT_PATH="{CHECKPOINT_PATH}"',
    f'INFER_OUTPUT_DIR="{INFER_RUN_DIR}"',
    f"TF_EXAMPLE_FILE_PATHS=['gs://argolis-t5x-vertex-new/datasets/eval.tfrecords']"
]
job = utils.create_t5x_custom_job(
    display_name=INFER_RUN_ID,
    machine_type=MACHINE_TYPE,
    accelerator_type=ACCELERATOR_TYPE,
    accelerator_count=ACCELERATOR_COUNT,
    image_uri=IMAGE_URI,
    run_mode=RUN_MODE,
    gin_files=GIN_FILES,
    model_dir=CHECKPOINT_PATH,
    tfds_data_dir=TFDS_DATA_DIR,
    gin_overwrites=GIN_OVERWRITES
)

utils.submit_and_track_t5x_vertex_job(
    custom_job=job,
    job_display_name=INFER_RUN_ID,
    run_name=INFER_RUN_ID,
    experiment_name=EXPERIMENT_NAME,
    execution_name=INFER_RUN_ID,
    tfds_data_dir=TFDS_DATA_DIR,
    model_dir=INFER_RUN_DIR,
    vertex_ai=vertex_ai,
    run_mode=RUN_MODE
)

