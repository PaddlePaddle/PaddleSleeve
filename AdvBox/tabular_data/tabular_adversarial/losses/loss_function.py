import abc

import numpy as np

from tabular_adversarial.utils.data_utils import to_onehot

class BaseLoss(abc.ABC):
    def __call__(self, scores, targets):
        '''
        Calculate the value of the adversarial loss function.

        Args:
            scores (numpy.ndarray): The scores of the model output. Shape (nb_samples, nb_classes) or Shape (nb_samples, 1) or Shape (nb_samples, ).
            targets: The labels of the original labels or the target labels, Shape (nb_samples, ), or the scores of the target scores.

        Returns:
            loss_values: Values of loss function.
        '''

        raise NotImplementedError

class CustomizeThresholdAdversarialLoss(BaseLoss):
    '''
    Loss function for the model trained for the unbalanced use cases. Set a custom threshold for the category being attacked.
    Note: not supported targeted attack.
    '''
    def __init__(self, attacked_label, threshold, margin=0.):
        '''
        Initialize the loss function.

        Args:
            attacked_label: The label of the category to be attacked.
            threshold: Customized threshold, the score of the attacked category is below the threshold for successful attack.
            margin: The margin used to adjust the attack intensity. The value of margin is greater than zero.
        '''

        self.attacked_label = attacked_label
        self.threshold = threshold
        self.margin = margin

    def __call__(self, scores, targets):
        '''
        Calculate the value of the adversarial loss function.

        Args:
            scores (numpy.ndarray): The scores of the model output. Shape (nb_samples, nb_classes).
            targets: The labels of the original labels or the target labels. Shape (nb_samples, ).

        Returns:
            loss_values: Values of loss function.
        '''

        loss_values = np.maximum(scores[:, self.attacked_label] - self.threshold + self.margin, 0.)

        return loss_values

class CWLoss(BaseLoss):
    '''
    The loss function in C&W
    '''
    def __init__(self, targeted=False, margin=0.):
        '''
        Initialize the loss function.

        Args:
            targeted: Is it a targeted attack.
            margin: The margin used to adjust the attack intensity. The value of margin is greater than zero.
        '''

        self.targeted = targeted
        self.margin = margin

    def __call__(self, scores, targets):
        '''
        Calculate the value of the adversarial loss function.

        Args:
            scores (numpy.ndarray): The scores of the model output. Shape (nb_samples, nb_classes).
            targets: The labels of the original labels or the target labels. Shape (nb_samples, ).

        Returns:
            loss_values: Values of loss function.
        '''

        onehot_labels = to_onehot(targets, scores.shape[1])

        # Get the maximum scores for each category except labels.
        max_other_scores = np.max(
            scores * (1 - onehot_labels) + (np.min(scores, axis=1) - 1)[:, np.newaxis] * onehot_labels,
            axis=1
        )

        target_socres = np.max(scores * onehot_labels, axis=1)

        # Targeted attack
        if self.targeted:
            loss_values = np.maximum(max_other_scores - target_socres + self.margin, 0.)

        # Untarget attack
        else:
            loss_values = np.maximum(target_socres - max_other_scores + self.margin, 0.)

        return loss_values


class RegressionAttackLoss(BaseLoss):
    '''
    The loss function for attack regression task.
    '''

    def __init__(self, direction, margin=0.):
        '''
        Initialize the RegressionAttackLoss.

        Args:
            direction: The direction of the change in the scores, `increase` or `decrease`.
            margin: The margin used to adjust the attack intensity. The value of margin is greater than zero.
        '''

        self.direction = direction
        assert self.direction in ['increase', 'decrease'], f'Error: The direction `{self.direction}` nonsupport.'
        self.margin = margin

    def __call__(self, scores, targets):
        '''
        Calculate the value of the adversarial loss function.

        Args:
            scores (numpy.ndarray): The scores of the model output. Shape (nb_samples, 1) or (nb_samples, ).
            targets (numpy.ndarray): The target scores. Shape (nb_samples, ).

        Returns:
            loss_values: Values of loss function.
        '''
        scores = scores.reshape(-1)
        targets = targets.reshape(-1)

        if self.direction == 'increase':
            loss_values = np.maximum(targets - scores + self.margin, 0.)
        else:
            loss_values = np.maximum(scores - targets + self.margin, 0.)

        return loss_values

