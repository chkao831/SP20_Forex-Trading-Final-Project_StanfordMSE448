SUPPORTED_SETTINGS = ['simple_binary', 'simple_regression']

ALL_SETTINGS = {
    'simple_regression': {'dataset': 'SingleFXDatasetRegression',
                          'type': 'regression'},
    'simple_binary': {'dataset': 'SingleFXDatasetBinaryClassification',
                          'type': 'binary_classification'}
}