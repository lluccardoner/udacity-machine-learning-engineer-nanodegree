# ML Case Studies Notes

## Table of contents
1. [Population Segmentation](#1)

Udacity [repository](https://github.com/udacity/ML_SageMaker_Studies)

## Population Segmentation<a name="1" />

### PCA

Creating the PCA model
```python
from sagemaker import PCA

# this is current features - 1
# you'll select only a portion of these to use, later
N_COMPONENTS=33

pca_SM = PCA(role=role,
             train_instance_count=1,
             train_instance_type='ml.c4.xlarge',
             output_path=output_s3_path,
             num_components=N_COMPONENTS, 
             sagemaker_session=session)
```

Training the PCA model. The required format for the training input data is **RecordSet**.

```python
# Convert tu numpy array
train_data_np = training_data_df.values.astype('float32')

# convert to RecordSet format
formatted_train_data = pca_SM.record_set(train_data_np)

# train the PCA mode on the formatted data
pca_SM.fit(formatted_train_data)
```
