# Custom training pipeline, with script located at 'trainer.py"'

from google.cloud import aiplatform

BUCKET = 'gs://argolis-vertex-europewest4'
PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
LOCATION = 'europe-west4'
SERVICE_ACCOUNT = 'tensorboard-sa@argolis-rafaelsanchez-ml-dev.iam.gserviceaccount.com'
TENSORBOARD_RESOURCE_NAME = 'projects/989788194604/locations/europe-west4/tensorboards/8884581718011412480'

# Initialize the *client* for Vertex
aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET, location=LOCATION)
                
# Launch Training pipeline, a type of Vertex Training Job.
# A Training pipeline integrates three steps into one job: Accessing a Managed Dataset (not used here), Training, and Model Upload. 
job = aiplatform.CustomTrainingJob(
    display_name="bert_imdb_finetuning_gpu",
    script_path="trainer.py",
    requirements=["py7zr==0.20.4",
                  "nltk==3.7",
                  "evaluate==0.4.0",
                  "rouge_score==0.1.2", 
                  "transformers==4.25.1",
                  "tensorboard==2.11.2",
                  "datasets==2.9.0"],
    container_uri="europe-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-10:latest",
    model_serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-10:latest",
)

model = job.run(
    model_display_name='bert-imdb-finetuning-gpu',
    replica_count=1,
    service_account = SERVICE_ACCOUNT,
    tensorboard = TENSORBOARD_RESOURCE_NAME,
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_V100",
    accelerator_count = 1,
)
print(model)

# Deploy endpoint
endpoint = model.deploy(machine_type='n1-standard-4',
    accelerator_type= "NVIDIA_TESLA_T4",
    accelerator_count = 1)
print(endpoint.resource_name)


