from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.amazon.aws.operators.emr import EmrCreateJobFlowOperator, EmrTerminateJobFlowOperator
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerStartPipelineExecutionOperator

# Configuration
S3_BUCKET = "your-mlops-bucket"
S3_KEY = "input/data-{{ ds }}.csv"
SAGEMAKER_PIPELINE_NAME = "Financial-Model-Pipeline"
EMR_CLUSTER_NAME = "MLOps-Processing-Cluster"

default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    's3_event_sagemaker_mlops',
    default_args=default_args,
    schedule_interval=None, # Triggered by S3 event/Sensor
    catchup=False
) as dag:

    # 1. S3 Event Trigger (Sensor)
    wait_for_data = S3KeySensor(
        task_id='wait_for_s3_data',
        bucket_name=S3_BUCKET,
        bucket_key=S3_KEY,
        wildcard_match=True,
        timeout=60 * 60, # 1 hour
        poke_interval=60 # Check every minute
    )

    # 2. EMR Processing Step
    # (Simplified EMR Config - in a real scenario, use a detailed JSON config)
    job_flow_definition = {
        "Name": EMR_CLUSTER_NAME,
        "ReleaseLabel": "emr-6.10.0",
        "Instances": {
            "InstanceGroups": [
                {"Name": "Master node", "Market": "ON_DEMAND", "InstanceRole": "MASTER", "InstanceType": "m5.xlarge", "InstanceCount": 1},
                {"Name": "Core nodes", "Market": "ON_DEMAND", "InstanceRole": "CORE", "InstanceType": "m5.xlarge", "InstanceCount": 2},
            ],
            "KeepJobFlowAliveWhenNoSteps": False,
        },
        "Steps": [{
            "Name": "Spark-Data-Processing",
            "ActionOnFailure": "TERMINATE_JOB_FLOW",
            "HadoopJarStep": {
                "Jar": "command-runner.jar",
                "Args": ["spark-submit", f"s3://{S3_BUCKET}/scripts/process.py", "--input", f"s3://{S3_BUCKET}/{S3_KEY}"]
            }
        }]
    }

    process_data_emr = EmrCreateJobFlowOperator(
        task_id='process_data_with_emr',
        job_flow_overrides=job_flow_definition,
        aws_conn_id='aws_default'
    )

    # 3. Trigger SageMaker Pipeline
    # This ensures the execution appears in the SageMaker "Pipelines" UI
    run_sagemaker_pipeline = SageMakerStartPipelineExecutionOperator(
        task_id='run_sagemaker_ml_pipeline',
        pipeline_name=SAGEMAKER_PIPELINE_NAME,
        pipeline_params={
            "InputDataS3": f"s3://{S3_BUCKET}/processed/{{ ds }}/",
            "ModelApprovalStatus": "PendingManualApproval"
        },
        display_name=f"Airflow-Trigger-{{ ds }}"
    )

    # Task Dependencies
    wait_for_data >> process_data_emr >> run_sagemaker_pipeline
