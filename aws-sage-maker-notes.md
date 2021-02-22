# AWS Sage Maker Notes

This curse notes are taken from the Udacity Machine Learning Engineer curse and they do not represent the totality of the course.

I recomment doing the course to complement this notes.

## Table of Contents
1. [AWS Sage Maker](#1)
2. [Initialization](#1-1)
3. [Storing data to S3](#1-2)
4. [Training and building a model](#1-3)
5. [Testing the model](#1-4)
6. [Deploying a model](#1-5)
7. [How to use a deployed model](#1-6)
8. [Hyper parameter tuning](#1-7)
9. [Model A/B testing](#1-8)
10. [Updating an endpoint](#1-9)

## AWS Sage Maker<a name="1" />

- [Amazon SageMaker Python SDK](https://sagemaker.readthedocs.io/en/latest/index.html#)
- [Udacity notebooks repository](https://github.com/udacity/sagemaker-deployment)

### Initialization<a name="1-1" />

Note that the following code uses SageMaker 1.x and it is run inside a SageMaker notebook instance.

```bash
pip install sagemaker==1.72.0
```

This is an object that represents the **SageMaker session** that we are currently operating in. 
This object contains some useful information that we will need to access later such as our region.

```python
import sagemaker
session = sagemaker.Session()
```

This is an object that represents the **IAM role** that we are currently assigned. 
When we construct and launch the training job later we will need to tell it what IAM role it should have. 
Since our use case is relatively simple we will simply assign the training job the role we currently have.

```python
from sagemaker import get_execution_role
role = get_execution_role()
```

### Storing data to S3<a name="1-2" />

Since we are currently running inside of a SageMaker session, we can use the object which represents this session to upload our data to the 'default' S3 bucket. Note that it is good practice to provide a custom prefix (essentially an S3 folder) to make sure that you don't accidentally interfere with data uploaded from some other notebook or project.

```python
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)
```

**NOTE**: The documentation for the XGBoost algorithm in SageMaker requires that the saved datasets should contain **no headers or index** and that for the training and validation data, the **label should occur first** for each sample.

### Training and building a model<a name="1-3" />

Sage Maker trains models with a training job and creates the model artifacts.

#### High level API

Now that we have the training and validation data uploaded to S3, we can construct our **XGBoost model** and train it. We will be making use of the **high level SageMaker API** to do this which will make the resulting code a little easier to read at the cost of some flexibility.

To construct an **estimator**, the object which we wish to train, we need to provide the location of a **container** which contains the training code. Since we are using a **built in algorithm** this container is provided by Amazon. However, the full name of the container is a bit lengthy and depends on the region that we are operating in. Fortunately, SageMaker provides a useful utility method called **get_image_uri** that constructs the image name for us.

To use the get_image_uri method we need to provide it with our current region, which can be obtained from the session object, and the name of the algorithm we wish to use. In this notebook we will be using XGBoost however you could try another algorithm if you wish. The list of built in algorithms can be found in the list of [Common Parameters](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-algo-docker-registry-paths.html).

```python
container = get_image_uri(session.boto_region_name, 'xgboost')

xgb = sagemaker.estimator.Estimator(container, 
                                    role,
                                    train_instance_count=1, 
                                    rain_instance_type='ml.m4.xlarge',                                   
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                    sagemaker_session=session)
```

Before asking SageMaker to begin the **training job**, we should probably set any model specific **hyperparameters**. There are quite a few that can be set when using the XGBoost algorithm, below are just a few of them. If you would like to change the hyperparameters below or modify additional ones you can find additional information on the [XGBoost hyperparameter page](https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html).

```python
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='reg:linear',
                        early_stopping_rounds=10,
                        num_round=200)
```

Now that we have our estimator object completely set up, it is time to **train** it. To do this we make sure that SageMaker knows our **input data** is in csv format and then execute the fit method.

```python
s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_input_val = sagemaker.s3_input(s3_data=val_location, content_type='csv')

xgb.fit({'train': s3_input_train, 'validation': s3_input_val})
```

**Note**: the fit method does not create the Model per se. It creates the **model artifacts**. With the high level API, the model is created when needed. For exsmple with a transformer is created or the model is deployed.

#### Low level API

Create training job ([API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTrainingJob.html))
```python
container = get_image_uri(session.boto_region_name, 'xgboost')

training_params = {}

# We need to specify the permissions that this training job will have. For our purposes we can use the same permissions that our current SageMaker session has.
training_params['RoleArn'] = role

# Here we describe the algorithm we wish to use. The most important part is the container which contains the training code.
training_params['AlgorithmSpecification'] = {
    "TrainingImage": container,
    "TrainingInputMode": "File"
}

# We also need to say where we would like the resulting model artifacts stored.
training_params['OutputDataConfig'] = {
    "S3OutputPath": "s3://" + session.default_bucket() + "/" + prefix + "/output"
}

# We also need to set some parameters for the training job itself. Namely we need to describe what sort of
# compute instance we wish to use along with a stopping condition to handle the case that there is
# some sort of error and the training script doesn't terminate.
training_params['ResourceConfig'] = {
    "InstanceCount": 1,
    "InstanceType": "ml.m4.xlarge",
    "VolumeSizeInGB": 5
}
    
training_params['StoppingCondition'] = {
    "MaxRuntimeInSeconds": 86400
}

# Next we set the algorithm specific hyperparameters. You may wish to change these to see what effect
# there is on the resulting model.
training_params['HyperParameters'] = {
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "subsample": "0.8",
    "objective": "reg:linear",
    "early_stopping_rounds": "10",
    "num_round": "200"
}

# Now we need to tell SageMaker where the data should be retrieved from.
training_params['InputDataConfig'] = [
    {
        "ChannelName": "train",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": train_location,
                "S3DataDistributionType": "FullyReplicated"
            }
        },
        "ContentType": "csv",
        "CompressionType": "None"
    },
    {
        "ChannelName": "validation",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": val_location,
                "S3DataDistributionType": "FullyReplicated"
            }
        },
        "ContentType": "csv",
        "CompressionType": "None"
    }
]

training_job = session.sagemaker_client.create_training_job(**training_params)

session.logs_for_job(training_job_name, wait=True)
```

Creating a model with the model artifacts

```python
# We begin by asking SageMaker to describe for us the results of the training job. The data structure returned contains a lot more information than we currently need, try checking it out yourself in more detail.

training_job_info = session.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)

model_artifacts = training_job_info['ModelArtifacts']['S3ModelArtifacts']

# Just like when we created a training job, the model name must be unique
model_name = training_job_name + "-model"

# We also need to tell SageMaker which container should be used for inference and where it should retrieve the model artifacts from. In our case, the xgboost container that we used for training can also be used for inference.
primary_container = {
    "Image": container,
    "ModelDataUrl": model_artifacts
}

# And lastly we construct the SageMaker model
model_info = session.sagemaker_client.create_model(
                                ModelName = model_name,
                                ExecutionRoleArn = role,
                                PrimaryContainer = primary_container)
```

### Testing the model<a name="1-4" />

Testing the model can be done making predictions on the test dataset with a batch transform job. It can also be done by deploying the model, but the deployed enpoint has a cost while it is running.

#### High level

Now that we have fit our model to the training data, using the validation data to avoid overfitting, we can test our model. To do this we will make use of **SageMaker's Batch Transform** functionality. To start with, we need to build a **transformer object** from our fit model.

```python
xgb_transformer = xgb.transformer(instance_count = 1, instance_type = 'ml.m4.xlarge')
```

Next we ask SageMaker to begin a batch **transform job** using our trained model and applying it to the test data we previously stored in S3. We need to make sure to provide SageMaker with the type of data that we are providing to our model, in our case text/csv, so that it knows how to serialize our data. In addition, we need to make sure to let SageMaker know how to split our data up into chunks if the entire data set happens to be too large to send to our model all at once.

Note that when we ask SageMaker to do this it will execute the batch transform job in the **background**. Since we need to wait for the results of this job before we can continue, we use the `wait() method. An added benefit of this is that we get some output from our batch transform job which lets us know if anything went wrong.

```python
xgb_transformer.transform(test_location, content_type='text/csv', split_type='Line')

xgb_transformer.wait()
```
#### Low level

Create batch transform ([API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateTransformJob.html))

```python
# Just like in each of the previous steps, we need to make sure to name our job and the name should be unique.
transform_job_name = 'boston-xgboost-batch-transform-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# Now we construct the data structure which will describe the batch transform job.
transform_request = \
{
    "TransformJobName": transform_job_name,
    
    # This is the name of the model that we created earlier.
    "ModelName": model_name,
    
    # This describes how many compute instances should be used at once. If you happen to be doing a very large
    # batch transform job it may be worth running multiple compute instances at once.
    "MaxConcurrentTransforms": 1,
    
    # This says how big each individual request sent to the model should be, at most. One of the things that
    # SageMaker does in the background is to split our data up into chunks so that each chunks stays under
    # this size limit.
    "MaxPayloadInMB": 6,
    
    # Sometimes we may want to send only a single sample to our endpoint at a time, however in this case each of
    # the chunks that we send should contain multiple samples of our input data.
    "BatchStrategy": "MultiRecord",
    
    # This next object describes where the output data should be stored. Some of the more advanced options which
    # we don't cover here also describe how SageMaker should collect output from various batches.
    "TransformOutput": {
        "S3OutputPath": "s3://{}/{}/batch-bransform/".format(session.default_bucket(),prefix)
    },
    
    # Here we describe our input data. Of course, we need to tell SageMaker where on S3 our input data is stored, in
    # addition we need to detail the characteristics of our input data. In particular, since SageMaker may need to
    # split our data up into chunks, it needs to know how the individual samples in our data file appear. In our
    # case each line is its own sample and so we set the split type to 'line'. We also need to tell SageMaker what
    # type of data is being sent, in this case csv, so that it can properly serialize the data.
    "TransformInput": {
        "ContentType": "text/csv",
        "SplitType": "Line",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": test_location,
            }
        }
    },
    
    # And lastly we tell SageMaker what sort of compute instance we would like it to use.
    "TransformResources": {
            "InstanceType": "ml.m4.xlarge",
            "InstanceCount": 1
    }
}

transform_response = session.sagemaker_client.create_transform_job(**transform_request)

transform_desc = session.wait_for_transform_job(transform_job_name)
```

### Deploying a model<a name="1-5" />

**NOTE**: When deploying a model you are asking SageMaker to launch a compute instance that will wait for data to be sent to it. As a result, this compute instance will continue to run until you shut it down. This is important to know since **the cost of a deployed endpoint depends on how long it has been running for**.

#### High level

Create an enpoint to make inference with the deployed model.

```python
# This creates the enpoint
xgb_predictor = xgb.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

# We need to tell the endpoint what format the data we are sending is in
xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer

Y_pred = xgb_predictor.predict(X_test.values).decode('utf-8')
# predictions is currently a comma delimited string and so we would like to break it up
# as a numpy array. This may change depending on the type of model used.
Y_pred = np.fromstring(Y_pred, sep=',')

xgb_predictor.delete_endpoint()
```

#### Low level

Endpoint configuration

```python
# As before, we need to give our endpoint configuration a name which should be unique
endpoint_config_name = "boston-xgboost-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we ask SageMaker to construct the endpoint configuration
endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = endpoint_config_name,
                            ProductionVariants = [{
                                "InstanceType": "ml.m4.xlarge",
                                "InitialVariantWeight": 1,
                                "InitialInstanceCount": 1,
                                "ModelName": model_name,
                                "VariantName": "AllTraffic"
                            }])

```

Create endpoint

```python
# Again, we need a unique name for our endpoint
endpoint_name = "boston-xgboost-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we can deploy our endpoint
endpoint_info = session.sagemaker_client.create_endpoint(
                    EndpointName = endpoint_name,
                    EndpointConfigName = endpoint_config_name)

endpoint_dec = session.wait_for_endpoint(endpoint_name)
```

Use the deployed model through the endpoint

```python
# First we need to serialize the input data. In this case we want to send the test data as a csv and so we manually do this. 
# Of course, there are many other ways to do this.
payload = [[str(entry) for entry in row] for row in X_test.values]
payload = '\n'.join([','.join(row) for row in payload])

# This time we use the sagemaker runtime client rather than the sagemaker client so that we can invoke the endpoint that we created.
response = session.sagemaker_runtime_client.invoke_endpoint(
                                                EndpointName = endpoint_name,
                                                ContentType = 'text/csv',
                                                Body = payload)

# We need to make sure that we deserialize the result of our endpoint call.
result = response['Body'].read().decode("utf-8")
Y_pred = np.fromstring(result, sep=',')
```

To delete the endpoint

```python
session.sagemaker_client.delete_endpoint(EndpointName = endpoint_name)
```

### How to use a deployed model<a name="1-6" />

The two main issues when using a deployed model are:

- **Security**: the endpoint is secured, meaning that only entities that are authenticated with AWS can send or receive data from the deployed model. We will solve this with **AWS API Gateway**
- **Data processing**: the deployed model expects a processed input. To process the input raw data sent to the endpoint we will use **AWS Lambda**

#### Creating a Lambda function
Inside the Lambda function, we can access a SageMaker session using boto3.

```python
runtime = boto3.Session().client('sagemaker-runtime')
```

**Part A: Create an IAM Role for the Lambda function**
Since we want the Lambda function to call a SageMaker endpoint, we need to make sure that it has permission to do so. To do this, we will construct a role that we can later give the Lambda function.

1. Using the AWS Console, navigate to the IAM page and click on Roles. Then, click on Create role. Make sure that the AWS service is the type of trusted entity selected and choose Lambda as the service that will use this role, then click Next: Permissions.
2. In the search box type sagemaker and select the check box next to the AmazonSageMakerFullAccess policy. Then, click on Next: Review.
3. Lastly, give this role a name. Make sure you use a name that you will remember later on, for example LambdaSageMakerRole. Then, click on Create role.

**Part B: Create a Lambda function**
To start, using the AWS Console, navigate to the AWS Lambda page and click on Create a function. When you get to the next page, make sure that **Author from scratch** is selected. Now, name your Lambda function, using a name that you will remember later on, for example sentiment_analysis_xgboost_func. Make sure that the Python 3.6 runtime is selected and then choose the **role** that you created in the previous part. Then, click on Create Function.

#### Setting up API Gateway

Now that our Lambda function is set up, it is time to create a new API using API Gateway that will trigger the Lambda function we have just created.

Using AWS Console, navigate to Amazon API Gateway and then click on Get started.
On the next page, make sure that New API is selected and give the new api a name, for example, sentiment_analysis_web_app. Then, click on Create API.
Now we have created an API, however it doesn't currently do anything. What we want it to do is to trigger the Lambda function that we created earlier.
Select the Actions dropdown menu and click Create Method. A new blank method will be created, select its dropdown menu and select POST, then click on the check mark beside it.

For the integration point, make sure that Lambda Function is selected and click on the Use Lambda Proxy integration. This option makes sure that the data that is sent to the API is then sent directly to the Lambda function with no processing. It also means that the return value must be a proper response object as it will also not be processed by API Gateway.
Type the name of the Lambda function you created earlier into the Lambda Function text entry box and then click on Save. Click on OK in the pop-up box that then appears, giving permission to API Gateway to invoke the Lambda function you created.

The last step in creating the API Gateway is to select the Actions dropdown and click on Deploy API. You will need to create a new Deployment stage and name it anything you like, for example prod.

You have now successfully set up a public API to access your SageMaker model. Make sure to copy or write down the URL provided to invoke your newly created public API as this will be needed in the next step. This URL can be found at the top of the page, highlighted in blue next to the text Invoke URL.

### Hyper parameter tuning<a name="1-7" />

After creating an estimator object and setting the default hyperparameters, we can run a hyper parameter tuning job with the following code.

Note that we can create an Estimator object attaching it to an already finished training job.

#### High level

```python
from sagemaker.tuner import IntegerParameter, ContinuousParameter, HyperparameterTuner

xgb_hyperparameter_tuner = HyperparameterTuner(
estimator = xgb, 
objective_metric_name = 'validation:rmse', 
objective_type = 'Minimize', 
max_jobs = 20, 
max_parallel_jobs = 3, 
hyperparameter_ranges = { 
    'max_depth': IntegerParameter(3, 12),
    'eta'      : ContinuousParameter(0.05, 0.5),
    'min_child_weight': IntegerParameter(2, 8),
    'subsample': ContinuousParameter(0.5, 0.9),
    'gamma': ContinuousParameter(0, 10),
})

s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data=val_location, content_type='csv')

xgb_hyperparameter_tuner.fit({'train': s3_input_train, 'validation': s3_input_validation})

xgb_hyperparameter_tuner.wait()

xgb_hyperparameter_tuner.best_training_job()

xgb_attached = sagemaker.estimator.Estimator.attach(xgb_hyperparameter_tuner.best_training_job())
```

#### Low level
On the definition of the training parameters, it’s all similar to the [training low level](https://docs.google.com/document/d/1hxyjz2xntah_xwNJRUIAxOJUthv-T1ZU7lVVLszj7ro/edit#heading=h.vf5vprgt7u5h), except we specify only the **static** hyper parameters.

```python
training_params['StaticHyperParameters'] = {
    "gamma": "4",
    "subsample": "0.8",
    "objective": "reg:linear",
    "early_stopping_rounds": "10",
    "num_round": "200"
}
```

Then, to set up the training job we specify the configuration we want.

```python
tuning_job_config = {
    # First we specify which hyperparameters we want SageMaker to be able to vary,
    # and we specify the type and range of the hyperparameters.
    "ParameterRanges": {
    "CategoricalParameterRanges": [],
    "ContinuousParameterRanges": [
        {
            "MaxValue": "0.5",
            "MinValue": "0.05",
            "Name": "eta"
        },
    ],
    "IntegerParameterRanges": [
        {
            "MaxValue": "12",
            "MinValue": "3",
            "Name": "max_depth"
        },
        {
            "MaxValue": "8",
            "MinValue": "2",
            "Name": "min_child_weight"
        }
    ]},
    # We also need to specify how many models should be fit and how many can be fit in parallel
    "ResourceLimits": {
        "MaxNumberOfTrainingJobs": 20,
        "MaxParallelTrainingJobs": 3
    },
    # Here we specify how SageMaker should update the hyperparameters as new models are fit
    "Strategy": "Bayesian",
    # And lastly we need to specify how we'd like to determine which models are better or worse
    "HyperParameterTuningJobObjective": {
        "MetricName": "validation:rmse",
        "Type": "Minimize"
    }
  }
```

After this we create and execute the **tuning job**. Note that the naming should be **32 characters long** (and 62 characters for the training jobs).

```python
tuning_job_name = "tuning-job" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

session.sagemaker_client.create_hyper_parameter_tuning_job(
                            HyperParameterTuningJobName = tuning_job_name,
                            HyperParameterTuningJobConfig = tuning_job_config,
                            TrainingJobDefinition = training_params)

session.wait_for_tuning_job(tuning_job_name)
```

Now, the best training job is chosen.

```python
tuning_job_info = session.sagemaker_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)

best_training_job_name = tuning_job_info['BestTrainingJob']['TrainingJobName']
training_job_info = session.sagemaker_client.describe_training_job(TrainingJobName=best_training_job_name)

model_artifacts = training_job_info['ModelArtifacts']['S3ModelArtifacts']

model_name = best_training_job_name + "-model"

# We also need to tell SageMaker which container should be used for inference and where it should retrieve the model artifacts from. In our case, the xgboost container that we used for training can also be used for inference.
primary_container = {
    "Image": container,
    "ModelDataUrl": model_artifacts
}

# And lastly we construct the SageMaker model
model_info = session.sagemaker_client.create_model(
                                ModelName = model_name,
                                ExecutionRoleArn = role,
                                PrimaryContainer = primary_container)

```
### Model A/B testing<a name="1-8" />

We can do so by configuring an endpoint with multiple models.

```python
combined_endpoint_config_name = "boston-combined-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we ask SageMaker to construct the endpoint configuration
combined_endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = combined_endpoint_config_name,
                            ProductionVariants = [
                                { # First we include the linear model
                                    "InstanceType": "ml.m4.xlarge",
                                    "InitialVariantWeight": 1,
                                    "InitialInstanceCount": 1,
                                    "ModelName": linear_model_name,
                                    "VariantName": "Linear-Model"
                                }, { # And next we include the xgb model
                                    "InstanceType": "ml.m4.xlarge",
                                    "InitialVariantWeight": 1,
                                    "InitialInstanceCount": 1,
                                    "ModelName": xgb_model_name,
                                    "VariantName": "XGB-Model"
                                }])

endpoint_name = "boston-update-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# And then we can deploy our endpoint
endpoint_info = session.sagemaker_client.create_endpoint(
                    EndpointName = endpoint_name,
                    EndpointConfigName = combined_endpoint_config_name)

endpoint_dec = session.wait_for_endpoint(endpoint_name)
```

### Updating an endpoint<a name="1-9" />

We can update an endpoint with a new configuration. This can be useful for updating the production model with a newly trained one or to set a test A/B.

```python
session.sagemaker_client.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=linear_endpoint_config_name)

endpoint_dec = session.wait_for_endpoint(endpoint_name)

pprint(session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name))
```
