import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

def get_onehot_encoder(data):
    '''
    Fit a OneHotEncoder by input data.

    Args:
        data (numpy.ndarray): The data that used to fit OneHotEncoder. Shape (n_samples, 1).

    Returns:
        onehot_encoder (OneHotEncoder): A fitted OneHotEncoder.
    '''

    # Fit a OneHotEncoder
    onehot_encoder = OneHotEncoder()
    onehot_encoder.fit(data)

    return onehot_encoder

def get_onehot_encoders(raw_data, deal_type=str):
    '''
    Get onehot encoders for all fields.

    Args:
        raw_data (numpy.ndarray): raw_data, shape (n_samples, n_fields).
        deal_type (type | list of types): a type or list of types that needs to be converted to onehot.

    Returns:
        onehot_encoders_list (list): List of onehot_encoders. The value is `None` or `onehot_encoder`. If type of field is not in deal_type, value is None, else value is onehot_encoder.
    '''

    onehot_encoders_list = []
    for i in range(raw_data.shape[1]):
        # Determines if the field is in the fields to be processed.
        if isinstance(raw_data[0, i], deal_type):
            onehot_encoder = get_onehot_encoder(raw_data[:, i].reshape(-1, 1))
            onehot_encoders_list.append(onehot_encoder)

        # Field that do not need to be processed
        else:
            onehot_encoders_list.append(None)

    return onehot_encoders_list

def raw_to_onehot(raw_data, onehot_encoders_list):
    '''
    Transform raw data to onehot data by onehot_encoders_list.

    Args:
        raw_data (numpy.ndarray): raw data. Shape (n_samples, n_fields).
        onehot_encoders_list (list): List of onehot_encoders. The value is `None` or `onehot_encoder`. If type of field is not in deal_type, value is None, else value is onehot_encoder.

    Returns:
        onehot_data (numpy.ndarray): converted onehot data. Shape (nb_samples, nb_features).
    '''

    onehot_data = []

    for i, onehot_encoder in enumerate(onehot_encoders_list):

        # Determines if the field is in the fields to be processed.
        if not onehot_encoder is None: 
            # Transform raw data to onehot embedding.
            onehot_encoder_x = onehot_encoder.transform(raw_data[:, i].reshape(-1, 1)).toarray()
            onehot_data.append(onehot_encoder_x)

        # Field that do not need to be processed
        else:
            onehot_data.append(raw_data[:, i].reshape(-1, 1))

    onehot_data = np.concatenate(onehot_data, axis=1)

    return onehot_data


def onehot_to_raw(onehot_data, onehot_encoders_list):
    '''
    Transform onehot data to raw data by onehot_encoders_list.

    Args:
        onehot_data (numpy.ndarray): onehot data. Shape (n_samples, n_features)
        onehot_encoders_list (list): List of onehot_encoders. The value is `None` or `onehot_encoder`. If type of field is not in deal_type, value is None, else value is onehot_encoder.

    Returns:
        raw_data (numpy.ndarray): raw data. Shape (n_samples, n_fields).
    '''
    idx = 0
    raw_data = []

    for onehot_encoder in onehot_encoders_list:
        if not onehot_encoder is None:
            onehot_embedding_len = len(onehot_encoder.get_feature_names_out())
            onehot_embedding = onehot_data[:, idx: idx + onehot_embedding_len].reshape(-1, onehot_embedding_len)
            field_data = onehot_encoder.inverse_transform(onehot_embedding)
            raw_data.append(field_data)
            idx += onehot_embedding_len

        else:
            raw_data.append(onehot_data[:, idx].reshape(-1, 1))
            idx += 1
            
    raw_data = np.concatenate(raw_data, axis=1)

    return raw_data
            

def get_label_encoder(data):
    '''
    Fit a LabelEncoder by input data.
    
    Args:
        data ((numpy.ndarray): The data that used to fit LabelEncoder. Shape (n_samples, 1).

    Returns:
        label_encoder (LabelEncoder): A fitted LabelEncoder. 
    '''

    # Fit a LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(data)

    return label_encoder

def get_label_encoders(raw_data, deal_type=str):
    '''
    Get label encoders for all fields.

    Args:
        raw_data (numpy.ndarray): raw_data, shape (n_samples, n_fields).
        deal_type (type | list of types): a type or list of types that needs to be converted to onehot.

    Returns:
        label_encoders_list (list): List of label_encoders. The value is `None` or `label_encoder`. If type of field is not in deal_type, value is None, else value is label_encoder.
    '''

    label_encoders_list = []
    for i in range(raw_data.shape[1]):
        # Determines if the field is in the fields to be processed.
        if isinstance(raw_data[0, i], deal_type):
            label_encoder = get_label_encoder(raw_data[:, i].reshape(-1, 1))
            label_encoders_list.append(label_encoder)

        # Field that do not need to be processed
        else:
            label_encoders_list.append(None)

    return label_encoders_list


def raw_to_label(raw_data, label_encoders_list):
    '''
    Transform raw data to label data and return label encoders.

    Args:
        raw_data (numpy.ndarray): raw data.
        label_encoders_list (list): List of label_encoders. The value is `None` or `label_encoder`. If type of field is not in deal_type, value is None, else value is label_encoder.

    Returns:
        label_data (numpy.ndarray): converted onehot data.
    '''

    label_data = []

    for i, label_encoder in enumerate(label_encoders_list):
        # Determines if the field is in the fields to be processed.
        if not label_encoder is None:
            # Transform raw data to label embedding.
            label_encoder_x = label_encoder.transform(raw_data[:, i]).reshape(-1, 1)
            label_data.append(label_encoder_x)
 
        # Field that do not need to be processed
        else:
            label_data.append(raw_data[:, i].reshape(-1, 1))

    label_data = np.concatenate(label_data, axis=1)

    return label_data


def label_to_raw(label_data, label_encoders_list):
    '''
    Transform label data to raw data by label_encoders_list.

    Args:
        label_data (numpy.ndarray): label data. Shape (n_samples, n_features)
        label_encoders_list (list): List of label_encoders. The value is `None` or `label_encoder`. If type of field is not in deal_type, value is None, else value is label_encoder.

    Returns:
        raw_data (numpy.ndarray): raw data. Shape (n_samples, n_fields).
    '''

    raw_data = []

    for i, label_encoder in enumerate(label_encoders_list):
        if not label_encoder is None:
            label_embedding = label_data[:, i].reshape(-1, 1)
            field_data = label_encoder.inverse_transform(label_embedding).reshape(-1, 1)
            raw_data.append(field_data)

        else:
            raw_data.append(label_data[:, i].reshape(-1, 1))

    raw_data = np.concatenate(raw_data, axis=1)

    return raw_data


def vector_transform_by_onehot_info(field_vector, onehot_encoders_list):
    '''
    Transform field-level vectors to feature-level vectors.

    Args:
        field_vector (list | numpy.ndarray): field-level vectors. Shape (n_samples, n_fields).
        onehot_encoders_list (list): List of onehot_encoders. The value is `None` or `onehot_encoder`. If type of field is not in deal_type, value is None, else value is onehot_encoder.

    Returns:
        feature_vector (numpy.ndarray): feature-level vectors.
    '''

    feature_vector = []

    for i, onehot_encoder in enumerate(onehot_encoders_list):
        # Determines if the field is onehot.
        if not onehot_encoder is None:
            onehot_embedding_len = len(onehot_encoder.get_feature_names_out())
            
            feature_vector.append(np.full(onehot_embedding_len, field_vector[i]))

        # Not the onehot field.
        else:
            feature_vector.append(np.array(field_vector[i]))

    feature_vector = np.hstack(feature_vector)
    return feature_vector


def check_and_transform_label_format(labels, nb_classes):
    '''
    Check label format and transform to one-hot-encoded labels if necessary.

    Args:
        labels: An array of integer labels of shape `(nb_samples,)`, `(nb_samples, 1)` or `(nb_samples, nb_classes)`.
        nb_classes: The number of classes. If None the number of classes is determined automatically.
        return_one_hot: True if returning one-hot encoded labels, False if returning index labels.

    Returns:
        formatted_labels: Formatted labels with shape `(nb_samples, )` (index).
    '''

    # one-hot encoded.
    if len(labels.shape) == 2 and labels.shape[1] > 1:
        formatted_labels = np.argmax(labels, axis=1)
        #formatted_labels = np.expand_dims(formatted_labels, axis=1)
    # index shape (nb_samples, )
    elif len(labels.shape) == 1:
        #formatted_labels = np.expand_dims(labels, axis=1)
        formatted_labels = labels
    # index shape (nb_samples, 1)
    elif len(labels.shape) == 2 and labels.shape[1] == 1:
        formatted_labels = labels.reshape(-1)
    else:
        raise ValueError(f'Not supports shape: {labels.shape}')

    return formatted_labels


def get_labels_np_array(preds):
    '''
    Returns the label of the most probable class given a array of class confidences.

    Args:
        preds: Array of class confidences, nb of instances as first dimension.

    Return: 
        y: Labels. Shape (nb_samples, ).
    '''

    # Shape of preds (nb_samples, nb_classes).
    if len(preds.shape) == 2 and preds.shape[1] > 1:
        y = np.argmax(preds, axis=1)
        # y = np.expand_dims(y, axis=1)
    # Shape of preds (nb_samples, 1)
    elif len(preds.shape) == 2 and preds.shape[1] == 1:
        y = np.round(preds).reshape(-1)
    # Shape of preds (nb_samples, )
    elif len(preds.shape) == 1:
        y = np.round(preds)
        # y = np.expand_dims(y, axis=1)
    else:
        raise ValueError(f'Shape of preds {preds.shape} is not supported.')

    return y


def to_onehot(data, nb_features):
    '''
    Transform input `data` to onehot by `nb_features`.

    Args:
        data: Data that needs to be transformed. Shape (nb_samples, ).
        nb_features: Number of features.

    Returns:
        onehot_data: The transformed onehot data. Shape (nb_samples, nb_features).
    '''

    onehot_data = np.zeros([data.shape[0], nb_features])
    onehot_data[np.array(range(data.shape[0])), data] = 1.

    return onehot_data

class DataProcessor(object):
    '''
    Data processor for converting raw data to encoded data and encoded data to raw data.
    '''
    def __init__(self, onehot_encoders_list, scaler=None, corrector=None):
        '''
        Initialize.

        Args:
            onehot_encoders_list (list): List of onehot_encoders. The value is `None` or `onehot_encoder`. If type of field is not in deal_type, value is None, else value is onehot_encoder.
            scaler: The scaler used to scale values. Must has transform and inverse_transform functions.
            corrector: The corrector used to correct the data features based on the data types of each field.
        '''

        self.onehot_encoders_list = onehot_encoders_list
        self.scaler = scaler
        self.corrector = corrector

    def transform(self, raw_data):
        '''
        Transform raw data to embedding.

        Args:
            raw_data: The raw data to be transformed.

        Returns:
            onehot_data: The transformed onehot data.
        '''

        onehot_data = raw_to_onehot(raw_data, self.onehot_encoders_list)
        if not self.scaler is None:
            onehot_data = self.scaler.transform(onehot_data)
        
        return onehot_data

    def inverse_transform(self, onehot_data, clip_values=False):
        '''
        Inverse transform embedding data to raw data.

        Args:
            onehot_data: The onehot data to be inverse transformed.

        Returns:
            raw_data: The inverse transformed raw data.
        '''

        if not self.scaler is None:
            onehot_data = self.scaler.inverse_transform(onehot_data)

        if not self.corrector is None:
            onehot_data = self.corrector.transform(onehot_data, clip_values=clip_values)

        raw_data = onehot_to_raw(onehot_data, self.onehot_encoders_list)

        return raw_data

class DataCorrector(object):
    '''
    Correct the data features based on the data types of each field.
    '''
    def __init__(self, fields_type_list, onehot_encoders_list, fields_clip_list, fields_min_list, fields_max_list):
        '''
        Initialize.

        Args:
            fields_type_list (list): A list of individual field data types. Type must in [`Boolean`, `Integer`, `Positive Integer`, `Positive Float`, `String`, `Integer more 1`, None].
            onehot_encoders_list (list): List of onehot_encoders. The value is `None` or `onehot_encoder`. If type of field is not in deal_type, value is None, else value is onehot_encoder.
            fields_clip_list (list): A list of flags that fields need to be cilpped. Note, `String` is unsupported clip.
            fields_min_list (list): A list of minimum values for fields. Note, `String` values is ignored.
            fields_max_list (list): A list of maximum values for fields. Note, `String` values is ignored.

        '''

        self.fields_type_list = fields_type_list
        self.onehot_encoders_list = onehot_encoders_list
        self.fields_clip_list = fields_clip_list
        self.fields_min_list = fields_min_list
        self.fields_max_list = fields_max_list
        
    def transform(self, data, clip_values=False):
        '''
        Correct the input `data` by `fields_type_list` and `onehot_encoders_list`

        Args:
            data: The data to be correct. Shape (nb_samples, nb_features).

        Returns:
            corrected_data: The corrected data. Shape (nb_samples, nb_features).
        '''

        idx = 0
        corrected_data = []

        for i in range(len(self.fields_type_list)):
            # Get field info.
            field_type = self.fields_type_list[i]
            onehot_encoder = self.onehot_encoders_list[i]
            # Get field value clip info.
            clip_value = self.fields_clip_list[i]
            min_value = self.fields_min_list[i]
            max_value = self.fields_max_list[i]

            # Check `clip_value` and `field_type`.
            if field_type == 'String' and clip_value == True:
                print(f'Warning: Field type {field_type} not supported value clip.')
                clip_value = False

            # Field is processed by onehot.
            if not onehot_encoder is None:
                onehot_embedding_len = len(onehot_encoder.get_feature_names_out())
                field_data = data[:, idx: idx + onehot_embedding_len].reshape(-1, onehot_embedding_len)
                idx += onehot_embedding_len
            # Field is not processed by onehot.
            else:
                field_data = data[:, idx].reshape(-1, 1)
                idx += 1

            
            corrected_data.append(self.correct(field_data, field_type, clip_value, min_value, max_value))

        corrected_data = np.concatenate(corrected_data, axis=1)

        return corrected_data

    def correct(self, field_data, field_type, clip_value=False, min_value=None, max_value=None):
        '''
        Correct the data based on the field data type.

        Args:
            field_data: The data to be corrected for a field. Shape (nb_samples, nb_features).
            field_type: Type of the data to be corrected.

        Returns:
            corrected_field_data: Corrected field data. Shape (nb_samples, nb_features).
        '''

        if field_type == 'Boolean':
            corrected_field_data = field_data > 0.5

        elif field_type == 'Integer':
            corrected_field_data = field_data.astype(np.float32)
            corrected_field_data = np.round(corrected_field_data)

        elif field_type == 'Positive Integer':
            corrected_field_data = field_data.astype(np.float32)
            corrected_field_data = np.where(corrected_field_data > 0, np.round(corrected_field_data), 0)

        elif field_type == 'Positive Float':
            corrected_field_data = field_data.astype(np.float32)
            corrected_field_data = np.where(corrected_field_data > 0., corrected_field_data, 0.)

        elif field_type == 'String':
            onehot_len = len(field_data)
            max_value = np.max(field_data, axis=1, keepdims=True)
            corrected_field_data = field_data == max_value
            corrected_field_data = corrected_field_data.astype(np.uint8)

        elif field_type == 'Integer more 1':
            corrected_field_data = field_data.astype(np.float32)
            corrected_field_data = np.round(corrected_field_data)
            corrected_field_data = np.where(corrected_field_data > 1., corrected_field_data, 1.)
        elif field_type is None:
            corrected_field_data = field_data
        else:
            raise ValueError(f'Not support field type: {field_type}.')

        if clip_value:
            corrected_field_data = np.clip(corrected_field_data, min_value, max_value)

        return corrected_field_data


def get_target_scores(scores, direction, target_score_change, target_score_type):
    '''
    The target score is calculated for the adversarial attack of the regression task.

    Args:
        scores: The original scores.
        direction: The direction of the change in the scores, `increase` or `decrease`.
        target_score_change: The target value of change scores.
        target_score_type: The type of `target_score_change`, `absolute` or `relative`.

    Returns:
        target_scores: The target scores for adversarial attack of the regression task.
    '''

    if target_score_type == 'absolute':
        score_change_values = np.full(scores.shape, target_score_change)
    elif target_score_type == 'relative':
        score_change_values = scores * target_score_change
    else:
        raise ValueError(f'Error: The `target_score_type` {target_score_type} is nonsupport.')

    if direction == 'increase':
        target_scores = scores + score_change_values
    elif direction == 'decrease':
        target_scores = scores - score_change_values
    else:
        raise ValueError(f'Error: The `direction` {direction} is nonsupport.')

    return target_scores
