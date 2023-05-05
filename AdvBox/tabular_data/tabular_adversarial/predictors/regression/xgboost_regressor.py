import xgboost as xgb
from xgboost import Booster

class XGBoostRegressor(object):
    '''
    Regressor using xgboost model.
    '''

    def __init__(
        self, 
        model, 
        preprocessing=None, 
        postprocessing=None, 
        nb_features=None,
    ):
        '''
        Create a Regressor from a XGBoost model.

        Args:
            model: XGBoost model.
            preprocessing: Preprocessing methods for input samples. Default: None, if None, the input samples must be data that does not need to be processed.
            postprocessing: Postprocessing methods for model predictions. Default: None, if None, the model predictions must be data that does not need to be processed.
            nb_features: The number of features in the training data.
        '''

        if not isinstance(model, Booster):
            raise TypeError("Model must be of type xgboost.Booster.")

        self.model = model
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.input_shape = (nb_features, )

    def predict(self, x):
        '''
        Perform prediction for a batch of inputs.

        Args:
            x (numpy.ndarray): Input samples.

        Returns:
            y_pred (numpy.ndarray): Array of predictions of shape (nb_inputs, ).
        '''
        
        x_preprocessed = self.apply_preprocessing(data=x)

        xgb_data = xgb.DMatrix(x_preprocessed, label=None)

        y_pred = self.model.predict(xgb_data)

        y_pred = self.apply_postprocessing(preds=y_pred)

        return y_pred

    def apply_preprocessing(self, data):
        '''
        Apply all preprocessing operations on the inputs `data`. This function has to be applied to all raw inputs `data` provided to the estimator.

        Args:
            data (numpy.ndarray): Input samples.

        Returns:
            data : preprocessed data, format as expected by the `model`.
        '''

        # No processing is needed to directly return input samples.
        if self.preprocessing is None:
            return data

        # The input samples are processed using preprocessing.
        else:
            return self.preprocessing(data)

    def apply_postprocessing(self, preds):
        '''
        Apply all postprocessing on model predictions.

        Args:
            preds: Model output to be post-processed.

        Returns:
            post_preds: Post-processed model predictions.
        '''

        # No processing is needed to directly return model predictions.
        if self.postprocessing is None:
            return preds

        # The model predictions are processed using postprocessing.
        else:
            return self.postprocessing(preds)
