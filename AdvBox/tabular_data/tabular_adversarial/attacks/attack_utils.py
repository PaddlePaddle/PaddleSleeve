
def judge_attack_success(adv_labels, targets, targeted=False):
    '''
    Determine whether the attack was successful.
    
    Args:
        adv_labels: The label of the adversarial samples. Shape (nb_samples, 1)
        targets: Targets. Shape (nb_samples, 1)
        targeted: Whether target attack.

    Returns:
        success_masks: The mask of the successful attack sample. Shape (nb_samples, 1)
    '''

    assert adv_labels.shape == targets.shape, f'Shape of `adv_labels` {adv_labels.shape} is not same shape of `targets` {targets.shape}.'

    # Targeted attack.
    if targeted:
        return adv_labels == targets

    # Untargeted attack.
    else:
        return adv_labels != targets
