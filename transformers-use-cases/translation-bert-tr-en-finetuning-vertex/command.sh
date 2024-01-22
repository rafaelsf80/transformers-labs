gcloud ai custom-jobs create \
--region=us-central1 \
--display-name=fine_tune_bert_tr_en \
--args=--job_dir=argolis-vertex-uscentral1 \
--worker-pool-spec=machine-type=n1-standard-4,replica-count=1,accelerator-type=NVIDIA_TESLA_V100,executor-image-uri="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-7:latest",local-package-path='autopackage',python-module=trainer.task