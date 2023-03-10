import abc

import numpy as np

class BaseNorm(abc.ABC):
    @abc.abstractmethod
    def __call__(self, adv_samples, ori_samples):
        '''
        Calculate the norm.

        Args:
            adv_samples: Adversarial samples.
            ori_samples: Original samples.

        Returns:
            norm_values: The calculated norm values.
        '''

        raise NotImplementedError


class PNorm(BaseNorm):
    '''
    P-Norm.
    '''
    def __init__(self, norm_type='l2'):
        '''
        Initialize p-norm.

        Args:
            norm_type:  Type of p-norm, only `l2` and `inf` is supported.
        '''

        assert norm_type in ['l2', 'inf'], f'The norm {norm} is not supported. Only `l2` and `inf` is supported.'

        self.norm = norm_type

    def __call__(self, adv_samples, ori_samples):
        '''
        Calculate the value of the norm according to its type.
        
        Args:
            adv_samples: Adversarial samples.
            ori_samples: Original samples.

        Returns:
            norm_values: The calculated norm values.
        '''
        perturbation = adv_samples - ori_samples
        if self.norm_type == 'l2':
            norm_values = np.linalg.norm(perturbation, ord=2, axis=1)
        elif self.norm_type == 'inf':
            norm_values = np.linalg.norm(new_p, ord=np.inf, axis=1)

        return norm_values


class CheckAndImportanceNorm(BaseNorm):
    '''
    The norm is calculated according to whether the field is checked and importance.

    n = p-norm(perturbation * (alpha * check_vector + beta * [(1 - check_vector) * (1 - importance_vector) + check_vector * importance_vector]))
    '''
    def __init__(self, check_vector, importance_vector, alpha=1, beta=1, norm_type='l2'):
        '''
        Initialize.

        Args:
            check_vector (numpy.ndarray): The feature-level mark whether each location will be checked, `1` check and `0` not check. Shape (nb_features, ).
            importance_vector (numpy.ndarray): The feature-level importance vector, with values ranging from 0 to 1. Shape (nb_features, ).
            alpha: Value of alpha. 
            beta: Value of beta.
            norm_type: The type of method used to calculate the norm
, only `l2` and `inf` is supported.
        '''
        self.check_vector = check_vector
        self.importance_vector = importance_vector
        self.alpha = alpha
        self.beta = beta
        self.norm_type = norm_type

    def __call__(self, adv_samples, ori_samples):
        '''
        Calculate the value of the norm according to whether the field is checked and importance and type of norm.
        
        Args:
            adv_samples: Adversarial samples.
            ori_samples: Original samples.

        Returns:
            norm_values: The calculated norm values.
        '''

        assert self.check_vector.shape[0] == self.importance_vector.shape[0] == adv_samples.shape[1] == ori_samples.shape[1], 'Vectors and features are not of the same length.'

        perturbation = adv_samples - ori_samples

        scale = self.alpha * self.check_vector
        scale += self.beta * ((1 - self.check_vector) * (1 - self.importance_vector) + self.check_vector * self.importance_vector)
        perturbation = perturbation * scale

        if self.norm_type == 'l2':
            norm_values = np.linalg.norm(perturbation, ord=2, axis=1)
        elif self.norm_type == 'inf':
            norm_values = np.linalg.norm(new_p, ord=np.inf, axis=1)

        return norm_values
