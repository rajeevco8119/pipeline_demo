import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
pipeline_session = PipelineSession()
bucket = sagemaker_session.default_bucket()

instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")

sku_processor = SKLearnProcessor(
    framework_version="1.2-1",
    instance_type=instance_type,
    instance_count=1,
    base_job_name="data-prep-job",
    role=role,
    sagemaker_session=pipeline_session,
)

step_process = ProcessingStep(
    name="PreprocessData",
    processor=sku_processor,
    inputs=[sagemaker.processing.ProcessingInput(source=f"s3://{bucket}/raw-data", destination="/opt/ml/processing/input")],
    outputs=[
        sagemaker.processing.ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        sagemaker.processing.ProcessingOutput(output_name="test", source="/opt/ml/processing/test")
    ],
    code="preprocess.py", # Local script
)

image_uri = sagemaker.image_uris.retrieve(framework="xgboost", region="us-east-1", version="1.7-1")
xgb_train = Estimator(
    image_uri=image_uri,
    instance_type=instance_type,
    instance_count=1,
    role=role,
    sagemaker_session=pipeline_session,
)

step_train = TrainingStep(
    name="TrainModel",
    estimator=xgb_train,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv"
        )
    },
)

model = sagemaker.model.Model(
    image_uri=image_uri,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    sagemaker_session=pipeline_session,
)

register_model_step = ModelStep(
    name="RegisterModel",
    step_args=model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name="MyModelPackageGroup",
        approval_status=model_approval_status,
    ),
)

pipeline = Pipeline(
    name="MLOps-End-to-End-Pipeline",
    parameters=[instance_type, model_approval_status],
    steps=[step_process, step_train, register_model_step],
    sagemaker_session=pipeline_session,
)

pipeline.upsert(role_arn=role)

execution = pipeline.start()
print(f"Pipeline Execution Started: {execution.arn}")
