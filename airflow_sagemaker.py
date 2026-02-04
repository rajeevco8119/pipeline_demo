from datetime import datetime
from airflow import DAG
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.amazon.aws.operators.emr import EmrCreateJobFlowOperator, EmrTerminateJobFlowOperator
from airflow.providers.amazon.aws.operators.sagemaker import (
    SageMakerProcessingOperator,
    SageMakerTrainingOperator,
    SageMakerTuningOperator,
    SageMakerModelOperator,
    SageMakerTransformOperator,
    SageMakerEndpointConfigOperator,
    SageMakerEndpointOperator,
    SageMakerDeleteModelOperator
)

# Constants & Configurations
S3_BUCKET = "asfew3"
ROLE_ARN = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole"
IMAGE_URI = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.1-cpu-py38"

default_args = {
    'owner': 'rajeev_mlops',
    'start_date': datetime(2025, 5, 17),
    'depends_on_past': False,
}

with DAG(
    dag_id='end_to_end_sagemaker_bank_pipeline',
    default_args=default_args,
    schedule_interval=None, # Triggered via S3 Event/Lambda or Manual
    catchup=False
) as dag:

    # 1. S3 EVENT SENSOR: Wait for raw data arrival
    wait_for_data = S3KeySensor(
        task_id='wait_for_s3_data',
        bucket_name=S3_BUCKET,
        bucket_key='input/raw_data.csv',
        poke_interval=60
    )

    # 2. EMR STAGE: Heavy Data Engineering (Big Data scale)
    emr_processing = EmrCreateJobFlowOperator(
        task_id='emr_big_data_prep',
        job_flow_overrides={
            "Name": "ML-Data-ETL",
            "ReleaseLabel": "emr-6.x",
            "Instances": {"InstanceCount": 3, "KeepJobFlowAliveWhenNoSteps": False}
        }
    )

    # 3. SAGEMAKER PROCESSING: ML-Specific Prep (Featurization)
    sm_processing = SageMakerProcessingOperator(
        task_id='sagemaker_processing',
        config={
            'ProcessingJobName': 'job-{{ ds_nodash }}',
            'ProcessingResources': {'ClusterConfig': {'InstanceCount': 1, 'InstanceType': 'ml.m5.xlarge', 'VolumeSizeInGB': 30}},
            'AppSpecification': {'ImageUri': IMAGE_URI, 'ContainerEntrypoint': ['python3', '/opt/ml/processing/input/code/script.py']},
            'RoleArn': ROLE_ARN,
        }
    )

    # 4. SAGEMAKER TUNING (HPO): Find best hyperparameters
    sm_tuning = SageMakerTuningOperator(
        task_id='sagemaker_hpo',
        config={
            'HyperParameterTuningJobName': 'hpo-{{ ds_nodash }}',
            'HyperParameterTuningJobConfig': {
                'Strategy': 'Bayesian',
                'HyperParameterTuningJobObjective': {'Type': 'Minimize', 'MetricName': 'validation:error'},
                'ResourceLimits': {'MaxNumberOfTrainingJobs': 5, 'MaxParallelTrainingJobs': 2},
                'ParameterRanges': {'ContinuousParameterRanges': [{'Name': 'learning_rate', 'MinValue': '0.01', 'MaxValue': '0.1'}]}
            },
            'TrainingJobDefinition': {
                'AlgorithmSpecification': {'TrainingImage': IMAGE_URI, 'TrainingInputMode': 'File'},
                'OutputDataConfig': {'S3OutputPath': f's3://{S3_BUCKET}/output/'},
                'ResourceConfig': {'InstanceCount': 1, 'InstanceType': 'ml.m5.xlarge', 'VolumeSizeInGB': 30},
                'RoleArn': ROLE_ARN,
                'StaticHyperParameters': {'epochs': '10'}
            }
        }
    )

    # 5. SAGEMAKER TRAINING: Final training with best params
    sm_training = SageMakerTrainingOperator(
        task_id='sagemaker_training',
        config={
            'TrainingJobName': 'train-{{ ds_nodash }}',
            'AlgorithmSpecification': {'TrainingImage': IMAGE_URI, 'TrainingInputMode': 'File'},
            'RoleArn': ROLE_ARN,
            'OutputDataConfig': {'S3OutputPath': f's3://{S3_BUCKET}/models/'},
            'ResourceConfig': {'InstanceCount': 1, 'InstanceType': 'ml.p3.2xlarge', 'VolumeSizeInGB': 30},
        }
    )

    # 6. SAGEMAKER MODEL CREATION: Register artifact in SageMaker
    sm_create_model = SageMakerModelOperator(
        task_id='sagemaker_create_model',
        config={
            'ModelName': 'model-{{ ds_nodash }}',
            'PrimaryContainer': {'Image': IMAGE_URI, 'ModelDataUrl': "{{ task_instance.xcom_pull(task_ids='sagemaker_training')['ModelArtifacts']['S3ModelArtifacts'] }}"},
            'ExecutionRoleArn': ROLE_ARN
        }
    )

    # 7. SAGEMAKER BATCH TRANSFORM: Offline Inference on test data
    sm_batch_transform = SageMakerTransformOperator(
        task_id='sagemaker_batch_transform',
        config={
            'TransformJobName': 'batch-{{ ds_nodash }}',
            'ModelName': 'model-{{ ds_nodash }}',
            'TransformInput': {'DataSource': {'S3DataSource': {'S3DataType': 'S3Prefix', 'S3Uri': f's3://{S3_BUCKET}/test-data/'}}},
            'TransformOutput': {'S3OutputPath': f's3://{S3_BUCKET}/predictions/'},
            'TransformResources': {'InstanceCount': 1, 'InstanceType': 'ml.m5.xlarge'}
        }
    )

    # 8. ENDPOINT DEPLOYMENT: Deploy for Real-time API
    sm_endpoint_config = SageMakerEndpointConfigOperator(
        task_id='sagemaker_endpoint_config',
        config={
            'EndpointConfigName': 'config-{{ ds_nodash }}',
            'ProductionVariants': [{
                'VariantName': 'AllTraffic', 'ModelName': 'model-{{ ds_nodash }}',
                'InitialInstanceCount': 1, 'InstanceType': 'ml.t2.medium'
            }]
        }
    )

    sm_endpoint_deploy = SageMakerEndpointOperator(
        task_id='sagemaker_endpoint_deploy',
        config={
            'EndpointConfigName': 'config-{{ ds_nodash }}',
            'EndpointName': 'banking-risk-prod-endpoint'
        }
    )

    # 9. CLEANUP: Delete old model version if necessary
    sm_delete_old_model = SageMakerDeleteModelOperator(
        task_id='sagemaker_cleanup',
        model_name='old-model-version-to-delete'
    )

    # Workflow Dependency Tree
    wait_for_data >> emr_processing >> sm_processing >> sm_tuning >> sm_training >> sm_create_model 
    sm_create_model >> [sm_batch_transform, sm_endpoint_config]
    sm_endpoint_config >> sm_endpoint_deploy >> sm_delete_old_model
