import abc

class BaseAttackSuccessDiscriminator(abc.ABC):
    def __call__(self, adv_preds, targets):
        '''
        Determine whether the attack was successful.

        Args:
            adv_preds: The preds of the adversarial samples. Shape (nb_samples, 1).
            targets: The target labels or scores. Shape (nb_samples, 1).
        '''
        raise NotImplementedError
    
class ClassificationAttackSuccessDiscriminator(BaseAttackSuccessDiscriminator):
    '''
    Determine whether the attack was successful for classification task.
    '''
    def __init__(self, targeted):
        '''
        Initialize.

        Args:
            targeted: Whether targeted attack.
        '''

        self.targeted = targeted

    def __call__(self, adv_preds, targets):
        '''
        Determine whether the attack was successful.

        Args:
            adv_preds: The preds of the adversarial samples. Shape (nb_samples, 1).
            targets: The target labels or scores. Shape (nb_samples, 1).

        Returns:
            success_masks: The mask of the successful attack sample. Shape (nb_samples, 1)
        '''

        assert adv_preds.shape == targets.shape, f'Shape of `adv_preds` {adv_preds.shape} is not same shape of `targets` {targets.shape}.'

        # Targeted attack.
        if self.targeted:
            return adv_preds == targets

        # Untargeted attack.
        else:
            return adv_preds != targets


class RegressionAttackSuccessDiscriminator(BaseAttackSuccessDiscriminator):
    def __init__(self, direction):
        '''
        Initialize.

        Args:
            direction: The direction of the change in the scores, `increase` or `decrease`.
        '''

        self.direction = direction

        assert self.direction in ['increase', 'decrease'], f'Error: The direction `{self.direction}` nonsupport.'

    def __call__(self, adv_preds, targets):
        '''
        Determine whether the attack was successful.

        Args:
            adv_preds: The preds of the adversarial samples. Shape (nb_samples, 1).
            targets: The target labels or scores. Shape (nb_samples, 1).

        Returns:
            success_masks: The mask of the successful attack sample. Shape (nb_samples, 1)
        '''
        
        if self.direction == 'increase':
            return adv_preds >= targets

        else:
            return adv_preds <= targets
