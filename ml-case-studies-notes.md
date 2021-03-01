# ML Case Studies Notes

## Table of contents
1. [PCA](#1)
2. [K-Means](#2)
3. [Linear Model](#3)
4. [Custom PyTorch model](#4)

Udacity [repository](https://github.com/udacity/ML_SageMaker_Studies)

## PCA<a name="1" />

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

Unzip the Model Details
Model artifacts are stored in S3 as a TAR file; a compressed file in the output path we specified + 'output/model.tar.gz'. The artifacts stored here can be used to deploy a trained model.

```python
training_job_name=pca_SM.latest_training_job.job_name

# where the model is saved, by default
model_key = os.path.join(prefix, training_job_name, 'output/model.tar.gz')

# download and unzip model
boto3.resource('s3').Bucket(bucket_name).download_file(model_key, 'model.tar.gz')

# unzipping as model_algo-1
os.system('tar -zxvf model.tar.gz')
os.system('unzip model_algo-1')
```
Many of the Amazon SageMaker algorithms use MXNet for computational speed, including PCA, and so the model artifacts are stored as an array. After the model is unzipped and decompressed, we can load the array using MXNet.

```python
import mxnet as mx

pca_model_params = mx.ndarray.load('model_algo-1')

s=pd.DataFrame(pca_model_params['s'].asnumpy())
v=pd.DataFrame(pca_model_params['v'].asnumpy())
```

Three types of model attributes are contained within the PCA model.

* **mean**: The mean that was subtracted from a component in order to center it.
* **v**: The makeup of the principal components; (same as ‘components_’ in an sklearn PCA model).
* **s**: The singular values of the components for the PCA transformation. This does not exactly give the % variance from the original feature space, but can give the % variance from the projected feature space.

From s, we can get an approximation of the data variance that is covered in the first `n` principal components. The approximate explained variance is given by the formula: the sum of squared s values for all top n components over the sum over squared s values for _all_ components:

\begin{equation*}
\frac{\sum_{n}^{ } s_n^2}{\sum s^2}
\end{equation*}

From v, we can learn more about the combinations of original features that make up each principal component.

**Note**: The top principal components, with the largest s values, are actually at the end of the s DataFrame. Let's print out the s values for the top n, principal components.

```python
# looking at top 5 components
n_principal_components = 5

start_idx = N_COMPONENTS - n_principal_components  # 33-n

# print a selection of s
print(s.iloc[start_idx:, :])
```

To find the explained variance of the top N components we can use the following method:
```python
def explained_variance(s, n_top_components):
    '''Calculates the approx. data variance that n_top_components captures.
       :param s: A dataframe of singular values for top components; 
           the top value is in the last row.
       :param n_top_components: An integer, the number of top components to use.
       :return: The expected data variance covered by the n_top_components.'''
    
    start_idx = N_COMPONENTS - n_top_components

    exp_variance = np.square(s.iloc[start_idx:, :]).sum() / np.square(s).sum()
    return exp_variance[0]
```

We can now examine the makeup of each PCA component based on the weightings of the original features that are included in the component. The following code shows the feature-level makeup of the first component.
```python
import seaborn as sns

def display_component(v, features_list, component_num, n_weights=10):
    
    # get index of component (last row - component_num)
    row_idx = N_COMPONENTS-component_num

    # get the list of weights from a row in v, dataframe
    v_1_row = v.iloc[:, row_idx]
    v_1 = np.squeeze(v_1_row.values)

    # match weights to features in counties_scaled dataframe, using list comporehension
    comps = pd.DataFrame(list(zip(v_1, features_list)), 
                         columns=['weights', 'features'])

    # we'll want to sort by the largest n_weights
    # weights can be neg/pos and we'll sort by magnitude
    comps['abs_weights']=comps['weights'].apply(lambda x: np.abs(x))
    sorted_weight_data = comps.sort_values('abs_weights', ascending=False).head(n_weights)

    # display using seaborn
    ax=plt.subplots(figsize=(10,6))
    ax=sns.barplot(data=sorted_weight_data, 
                   x="weights", 
                   y="features", 
                   palette="Blues_d")
    ax.set_title("PCA Component Makeup, Component #" + str(component_num))
    plt.show()
    
display_component(v, train_df.columns.values, component_num=1, n_weights=10)
```
Deploying the PCA model. This creates an endpoint for making inferences. The endpoint has a cost as long as it is running.
```python
pca_predictor = pca_SM.deploy(initial_instance_count=1, 
                              instance_type='ml.t2.medium')
```
We can pass the original, numpy dataset to the model and transform the data using the model we created. Then we can take the largest n components to reduce the dimensionality of our data.
```python
train_pca = pca_predictor.predict(train_data_np)
```

Now we can create a data frame with the top N components values for each data point.
```python
# create dimensionality-reduced data
def create_transformed_df(train_pca, counties_scaled, n_top_components):
    ''' Return a dataframe of data points with component features. 
        The dataframe should be indexed by State-County and contain component values.
        :param train_pca: A list of pca training data, returned by a PCA model.
        :param counties_scaled: A dataframe of normalized, original features.
        :param n_top_components: An integer, the number of top components to use.
        :return: A dataframe, indexed by State-County, with n_top_component values as columns.        
     '''
    counties_transformed = pd.DataFrame()
    
    for data in train_pca:
        components = data.label["projection"].float32_tensor.values
        counties_transformed = counties_transformed.append([list(components)])
        
    counties_transformed.index = counties_scaled.index
    
    start_idx = N_COMPONENTS - n_top_components
    counties_transformed = counties_transformed.iloc[:,start_idx:]
    counties_transformed = counties_transformed.iloc[:, ::-1]
    
    counties_transformed.columns = ["c_" + str(i) for i in range(1,n_top_components+1)]
    
    return counties_transformed
```

## K-Means<a name="2" />

Define the estimator
```python
kmeans = sagemaker.KMeans(role, 
                          instance_count=1, 
                          instance_type='ml.c4.xlarge', 
                          k=8, 
                          output_path=output_path,
                          sagemaker_session=session
                         )
```

Fit the estimator
```python
# Convert tu numpy array
counties_transformed_np = counties_transformed.values.astype('float32')

# convert to RecordSet format
counties_transformed_formatted = kmeans.record_set(counties_transformed_np)

kmeans.fit(counties_transformed_formatted)
```

Deploy the model
```python
kmeans_predictor = kmeans.deploy(
    initial_instance_count=1, 
    instance_type='ml.t2.medium'
)
```

After deploying the model, you can pass in the k-means training data, as a numpy array, and get resultant, predicted cluster labels for each data point.
```python
cluster_info=kmeans_predictor.predict(counties_transformed_np)

# print cluster info for first data point
data_idx = 0

print('County is: ', counties_transformed.index[data_idx])
print()
print(cluster_info[data_idx])
```

Visualize the distribution of data over clusters
```python
cluster_labels = [c.label['closest_cluster'].float32_tensor.values[0] for c in cluster_info]

cluster_df = pd.DataFrame(cluster_labels)[0].value_counts()

print(cluster_df)
```

To access the models attributes (centroids) we have to download the model artifacts.
```python
# download and unzip the kmeans model file
# use the name model_algo-1
training_job_name=kmeans.latest_training_job.job_name

# where the model is saved, by default
model_key = os.path.join(prefix, training_job_name, 'output/model.tar.gz')

# download and unzip model
boto3.resource('s3').Bucket(bucket_name).download_file(model_key, 'model.tar.gz')

# unzipping as model_algo-1
os.system('tar -zxvf model.tar.gz')
os.system('unzip model_algo-1')

# print the parameters
import mxnet as mx

kmeans_model_params = mx.ndarray.load('model_algo-1')

print(kmeans_model_params)

# get all the centroids
cluster_centroids=pd.DataFrame(kmeans_model_params[0].asnumpy())
cluster_centroids.columns=counties_transformed.columns

display(cluster_centroids)

# generate a heatmap in component space, using the seaborn library
plt.figure(figsize = (12,9))
ax = sns.heatmap(cluster_centroids.T, cmap = 'YlGnBu')
ax.set_xlabel("Cluster")
plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)
ax.set_title("Attribute Value by Centroid")
plt.show()
```

Add the cluster label to the original training dataframe.
```python
# add a 'labels' column to the dataframe
counties_transformed['labels']=list(map(int, cluster_labels))

# sort by cluster label 0-6
sorted_counties = counties_transformed.sort_values('labels', ascending=False)
# view some pts in cluster 0
sorted_counties.head(20)
```

## Linear Model<a name="3" />

SageMaker Linear Model has a way to set for a specific target: accuracy, precision or recall. Also it automatically handles inbalanced data.
```python
linear_balanced = LinearLearner(role=role,
                              train_instance_count=1, 
                              train_instance_type='ml.c4.xlarge',
                              predictor_type='binary_classifier',
                              output_path=output_path,
                              sagemaker_session=sagemaker_session,
                              epochs=15,
                              binary_classifier_model_selection_criteria='precision_at_target_recall', # target recall
                              target_recall=0.9, # 90% recall
                              positive_example_weight_mult='balanced' ) 
```

A way to evaluate the model and see the confusion matrix:
```python
# code to evaluate the endpoint on test data
# returns a variety of model metrics
def evaluate(predictor, test_features, test_labels, verbose=True):
    """
    Evaluate a model on a test set given the prediction endpoint.  
    Return binary classification metrics.
    :param predictor: A prediction endpoint
    :param test_features: Test features
    :param test_labels: Class labels for test data
    :param verbose: If True, prints a table of all performance metrics
    :return: A dictionary of performance metrics.
    """
    
    # We have a lot of test data, so we'll split it into batches of 100
    # split the test data set into batches and evaluate using prediction endpoint    
    prediction_batches = [predictor.predict(batch) for batch in np.array_split(test_features, 100)]
    
    # LinearLearner produces a `predicted_label` for each data point in a batch
    # get the 'predicted_label' for every point in a batch
    test_preds = np.concatenate([np.array([x.label['predicted_label'].float32_tensor.values[0] for x in batch]) 
                                 for batch in prediction_batches])
    
    # calculate true positives, false positives, true negatives, false negatives
    tp = np.logical_and(test_labels, test_preds).sum()
    fp = np.logical_and(1-test_labels, test_preds).sum()
    tn = np.logical_and(1-test_labels, 1-test_preds).sum()
    fn = np.logical_and(test_labels, 1-test_preds).sum()
    
    # calculate binary classification metrics
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    
    # printing a table of metrics
    if verbose:
        print(pd.crosstab(test_labels, test_preds, rownames=['actual (row)'], colnames=['prediction (col)']))
        print("\n{:<11} {:.3f}".format('Recall:', recall))
        print("{:<11} {:.3f}".format('Precision:', precision))
        print("{:<11} {:.3f}".format('Accuracy:', accuracy))
        print()
        
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 
            'Precision': precision, 'Recall': recall, 'Accuracy': accuracy}
```
## Custom PyTorch model<a name="4" />

* [code for a 3-layer MLP](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/mnist-mlp/mnist_mlp_solution.ipynb)
* [torch.nn](https://pytorch.org/docs/stable/nn.html#torch-nn)

Training
We instantiate a PyTorch estimator giving the [train.py](https://github.com/udacity/ML_SageMaker_Studies/blob/master/Moon_Data/source_solution/train.py) script as entry point and the hyperparameters that this script accepts. In case of a PyTorch model the model is defined in the [model.py](https://github.com/udacity/ML_SageMaker_Studies/blob/master/Moon_Data/source_solution/model.py) file.
```python
from sagemaker.pytorch import PyTorch
output_path = f"s3://{bucket}/{prefix}"

# instantiate a pytorch estimator
estimator = PyTorch(
    role=role,
    sagemaker_session=sagemaker_session,
    instance_count=1,
    instance_type='ml.c4.xlarge',
    output_path=output_path,
    entry_point='train.py',
    source_dir='source',
    framework_version='1.6',
    py_version='py36',
    hyperparameters={
        'input_dim': 2,
        'hidden_dim': 20,
        'output_dim': 1,
        'epochs': 100
    }
)

```
