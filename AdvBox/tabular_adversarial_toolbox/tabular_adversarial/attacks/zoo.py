import numpy as np
from tabular_adversarial.utils.data_utils import check_and_transform_label_format, get_labels_np_array
from tabular_adversarial.attacks.attack_utils import judge_attack_success

class ZooAttack(object):
    '''
    Zeroth-order optimization attack Modified against Structured Data adversarial attacks.

    The black-box zeroth-order optimization attack from Pin-Yu Chen et al. (2018). 
    This attack is a variant of the C&W attack which uses ADAM coordinate descent to perform numerical estimation of gradients. 
    Paper link: https://arxiv.org/abs/1708.03999
    '''

    def __init__(
        self, 
        predictor,
        norm_func,
        loss_func,
        targeted=False,
        learning_rate=1.0,
        max_iter=1000,
        const_binary_search_steps=1,
        initial_const=1.0, 
        allowed_vector=None, 
        nb_parallel=1,
        variable_h=0.1,
        adam_beta1=0.9, 
        adam_beta2=0.999, 
        processor=None,
        corrector=None, 
    ):
        '''
        Create a ZOO attack instance.

        Args:
            predictor: A trained predictor.
            norm_func: Function of calculate the distortion norm.
            loss_func: Function of calculate adversarial loss.
            targeted: Should the attack target one specific class.
            learning_rate: The initial learning rate for the attack algorithm.
            max_iter: The maximum number of iterations.
            const_binary_search_steps: Number of times to adjust constant with binary search (positive value).
            initial_const: The initial trade-off constant `c` to use to tune the relative importance of distance and confidence. If `binary_search_steps` is large, the initial constant is not important, as discussed in Carlini and Wagner (2016).
            allowed_vector: A vector representing the editability of individual fields, `1` means editable and `0` means uneditable.
            nb_parallel: Number of coordinate updates to run in parallel.
            variable_h (int|list|numpy.ndarray): Step size for numerical estimation of derivatives.
            processor: 
            corrector: Correct the data features based on the data types of each field.
        '''            

        self.predictor = predictor
        self.norm_func = norm_func
        self.loss_func = loss_func
        self.targeted = targeted
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.const_binary_search_steps = const_binary_search_steps
        self.initial_const = initial_const
        self.allowed_vector = allowed_vector
        self.nb_parallel = nb_parallel
        self.variable_h = variable_h
        self.processor = processor
        self.corrector = corrector

        self.adam_mean = None
        self.adam_var = None
        self.adam_epochs = None
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2

    def generate(self, raw_data, target_labels=None):
        '''
        Generate adversarial samples.

        Args:
            raw_data: Raw data to be attacked. Shape (nb_samples, nb_fields)
            target_labels: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape (nb_samples, ).

        Returns:
            o_best_distortion_norms: Global best distortion norms.
            o_best_adversarial_losses: Global best adversarial losses.
            o_best_labels: Global best attack labels.
            o_best_attacks: Global best attack samples.
            o_success_indices: Indices of attack success samples.
        '''

        # Check for targeted attacks.
        if self.targeted:
            # Check `target_label` is provided for targeted attacks.
            # `target_label` is not provided.
            if target_label is None:
                raise ValueError('Target labels `target_labels` need to be provided for a targeted attack.')
            # `target_label` is provided.
            else:
                # Check and transform format for `target_label`.
                # Return shape (nb_samples, )
                y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)

            # Check the loss function supports targeted attacks.
            if not getattr(self.loss_func, 'targeted', False):
                raise ValueError('The loss function not supports targeted attacks.')
        # Check for untargeted attacks.
        else:
            # Unatargeted not use target_labels.
            if not target_labels is None:
                raise Warning("Untarget `target_labels` not used.")
            # Untargeted use model prediction as correct class.
            # Return shape (nb_samples, ).
            y = get_labels_np_array(self.predictor.predict(raw_data))

        # Transform raw_data to embedding data.
        ori_data = self.processor.transform(raw_data)

        # Get number of features.
        nb_features = ori_data.shape[1]

        # Build allowed_vector.
        if self.allowed_vector is None:
            self.allowed_vector = np.ones(nb_features)

        # Judge shape of allowed_vector
        if self.allowed_vector.shape[0] != nb_features:
            raise ValueError('Shape of `allowed_vector` must same `nb_features`.')

        # Judge and transform shape of variable_h to nb_features. 
        # Judge variable_h type and transform to np.ndarray.
        if not isinstance(self.variable_h, np.ndarray):
            self.variable_h = np.array(self.variable_h).reshape(-1)

        # Judge shape of variable_h with nb_features.
        if self.variable_h.shape[0] != nb_features:
            if self.variable_h.shape[0] == 1:
                self.variable_h = np.repeat(self.variable_h, ori_data.shape[1])
            else:
                raise ValueError('Shape of variable_h must `1` or `nb_features`.')
            

        # Initialize binary search for trade-off constant.
        c_current = self.initial_const * np.ones(ori_data.shape[0])
        c_lower_bound = np.zeros(ori_data.shape[0])
        c_upper_bound = 1e10 * np.ones(ori_data.shape[0])

        # Initialize indices of successful adversarial samples.
        o_success_indices = np.full(ori_data.shape[0], False, dtype=bool)

        # Initialize best globally
        o_best_distortion_norms = np.inf * np.ones(ori_data.shape[0])
        o_best_adversarial_losses = np.inf * np.ones(ori_data.shape[0])
        o_best_labels = -np.inf * np.ones(ori_data.shape[0])
        o_best_attacks = ori_data.copy()

        # Start with a binary search
        for const_binary_search_step in range(self.const_binary_search_steps):
            # Run with 1 specific binary search step
            best_distortion_norms, best_adversarial_losses, best_labels, best_attacks, success_indices = self.generate_bss(ori_data, y, c_current)

            # Update best results so far
            for i in range(ori_data.shape[0]):
                # The attacks have been successful before
                if o_success_indices[i]:
                    # The distortion norms are smaller and the attack is successful
                    if best_distortion_norms[i] < o_best_distortion_norms[i] and success_indices[i]:
                        o_best_distortion_norms[i] = best_distortion_norms[i]
                        o_best_adversarial_losses[i] = best_adversarial_losses[i]
                        o_best_labels[i] = best_labels[i]
                        o_best_attacks[i] = best_attacks[i].copy()

                # The attacks have not been successful before
                else:
                    # The attack was successful
                    if success_indices[i]:
                        o_best_distortion_norms[i] = best_distortion_norms[i]
                        o_best_adversarial_losses[i] = best_adversarial_losses[i]
                        o_best_labels[i] = best_labels[i]
                        o_best_attacks[i] = best_attacks[i].copy()
                        o_success_indices[i] = success_indices[i]
                    # The attack was unsuccessful
                    else:
                        if best_adversarial_losses[i] < o_best_adversarial_losses[i]:
                            o_best_distortion_norms[i] = best_distortion_norms[i]
                            o_best_adversarial_losses[i] = best_adversarial_losses[i]
                            o_best_labels[i] = best_labels[i]
                            o_best_attacks[i] = best_attacks[i].copy()
                 

            # Adjust the constant as needed
            c_current, c_lower_bound, c_upper_bound = self.update_const(
                success_indices, c_current, c_lower_bound, c_upper_bound
            )

        return o_best_distortion_norms, o_best_adversarial_losses, o_best_labels, o_best_attacks, o_success_indices

    def generate_bss(self, ori_data, y, constants):
        '''
        Generate adversarial examples for input with a specific of constant.

        Args:
            ori_data: Original data to be attacked. Shape (nb_samples, nb_features).
            y: Target labels or original labels. Shape (nb_samples, ).
            constants: Trade-off constants. Shape (nb_samples, ).

        Returns:
            best_distortion_norms: Local best distortions.
            best_adversarial_losses: Local best adversarial losses.
            best_labels: Local best changed labels.
            best_attacks: Local best adversarial samples.
            success_indices: Indices of successful adversarial samples.
        '''
        x_ori = ori_data.copy()

        fine_tuning = np.full(x_ori.shape[0], False, dtype=bool)
        prev_loss = 1e6 * np.ones(x_ori.shape[0])
        prev_l2dist = np.zeros(x_ori.shape[0])

        self.reset_adam(ori_data.shape)

        self.current_noise = np.zeros(x_ori.shape, dtype=np.float32)

        x_adv = x_ori.copy()

        # Initialize best distortions, best adversarial losses, changed labels and best attacks for local.
        best_distortion_norms = np.inf * np.ones(x_adv.shape[0])
        best_adversarial_losses = np.inf * np.ones(x_adv.shape[0])
        best_labels = -np.inf * np.ones(x_adv.shape[0])
        best_attacks = x_adv.copy()

        for _iter in range(self.max_iter):

            print(f'Attack iter: {_iter}.') 
            x_adv = self.optimizer(x_ori, y, constants)

            # Correct perturbed data and inverse transform to field-level
            # Correct perturbed data
            if not self.corrector is None:
                corrected_x_adv = self.corrector.transform(x_adv)
            else:
                corrected_x_adv = x_adv

            # Inverse transform to field-level
            if not self.processor is None:
                inverse_transformed_x_adv = inverse_transformed_data = self.processor.inverse_transform(corrected_x_adv)
            else:
                inverse_transformed_x_adv = corrected_x_adv

            adv_preds = self.predictor.predict(inverse_transformed_x_adv)

            # Calculate the loss function
            # Calculate the distortion norms using x_ori and corrected_x_adv
            distortion_norms = self.norm_func(corrected_x_adv, x_ori)
            # Calculate the adversarial losses
            adversarial_losses = self.loss_func(adv_preds, y)

            # Trade-off the distortion norm and the adversarial loss
            losses = constants * adversarial_losses + distortion_norms

            # Get the label of the adversarial sample.
            adv_labels = get_labels_np_array(adv_preds)

            # Determine whether the attack is successful
            success_masks = judge_attack_success(adv_labels, y, self.targeted)
            success_masks = success_masks.reshape(-1)

            # Reset Adam if a success adversarial example has been found
            mask_fine_tune = (~fine_tuning) & (success_masks)
            self.reset_adam(
                self.adam_mean.shape, 
                mask_fine_tune
            )
             
            for i in range(x_ori.shape[0]):
                # Judging success
                success_flag = success_masks[i]
                # The attacks have been successful before
                if fine_tuning[i] == True: 
                    if distortion_norms[i] < best_distortion_norms[i] and mask_success[i]:
                        best_distortion_norms[i] = distortion_norms[i]
                        best_adversarial_losses[i] = adversarial_losses[i]
                        best_labels[i] = adv_labels[i]
                        best_attacks[i] = x_adv[i].copy()
                # The attacks not have been successful before
                else:
                    # The attack was successful
                    if success_masks[i]:
                        best_distortion_norms[i] = distortion_norms[i]
                        best_adversarial_losses[i] = adversarial_losses[i]
                        best_labels[i] = adv_labels[i]
                        best_attacks[i] = x_adv[i].copy()
                    # The attack was unsuccessful
                    else:
                        if adversarial_losses[i] < best_adversarial_losses[i]:
                            best_distortion_norms[i] = distortion_norms[i]
                            best_adversarial_losses[i] = adversarial_losses[i]
                            best_labels[i] = adv_labels[i]
                            best_attacks[i] = x_adv[i].copy()
                
            fine_tuning[mask_fine_tune] = True

        success_indices = fine_tuning

        return best_distortion_norms, best_adversarial_losses, best_labels, best_attacks, success_indices

    def update_const(self, success_indices, c_current, c_lower_bound, c_upper_bound):
        '''
        Update the constant that characterizes the trade-off between attack strength and amount of noise introduced.

        Args:
            success_indices: Indices of successful adversarial samples.
            c_current: The current trade-off constant.
            c_lower_bound: The current lower bound of the tradeoff constant.
            c_upper_bound: The current upper bound of the tradeoff constant.

        Returns:
            c_current: The updated trade-off constant.
            c_lower_bound: The updated lower bound of the tradeoff constant.
            c_upper_bound: The updated upper bound of the tradeoff constant.
        '''
        for i, success in enumerate(success_indices):
            # Successful attack
            if success:
                c_upper_bound[i] = min(c_upper_bound[i], c_current[i])
                c_current[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2
            # Failure attack
            else:
                c_lower_bound[i] = max(c_lower_bound[i], c_current[i])
                if c_upper_bound[i] < 1e9:
                    c_current[i] = (c_lower_bound[i] + c_upper_bound[i]) / 2
                else:
                    c_current[i] = c_current[i] * 10
        return c_current, c_lower_bound, c_upper_bound
            
    def reset_adam(self, var_shape, sample_idxs=None):
        '''
        Reset the variable values in ADAM.

        Args:
            var_shape: Shape of variables.
            sample_idxs: Indices of samples. Shape (nb_samples, )
        '''
        # If variables are already there and at the right size, reset values
        if self.adam_mean is not None and self.adam_mean.shape == var_shape:
            if sample_idxs is None:
                self.adam_mean.fill(0)
                self.adam_var.fill(0)
                self.adam_epochs.fill(1)
            else:
                self.adam_mean[sample_idxs] = 0
                self.adam_var[sample_idxs] = 0  # type: ignore
                self.adam_epochs[sample_idxs] = 1  # type: ignore

        else:
            self.adam_mean = np.zeros(var_shape, dtype=np.float32)
            self.adam_var = np.zeros(var_shape, dtype=np.float32)
            self.adam_epochs = np.ones(var_shape, dtype=int)

    def optimizer(self, x_ori, y, constants):
        '''
        Variation of input for computing loss, updates current_noise and return current adversarial examples.

        Args:
            ori_data: Original data to be attacked. Shape (nb_samples, nb_features).
            y: Target labels or original labels. Shape (nb_samples, )
            constants: Trade-off constants. Shape (nb_samples, )
        Returns:
            x_adv: adversarial examples.
        '''
        if self.nb_parallel > x_ori.shape[-1]:
            raise ValueError(
                "Too many samples are requested for the random coordinate. Try to reduce the number of parallel coordinate updates `nb_parallel` to < `nb_features`."
            )

        if not self.allowed_vector is None and self.nb_parallel > sum(self.allowed_vector):
            raise ValueError(
                "Too many samples are requested for the random coordinate. Try to reduce the number of parallel coordinate updates `nb_parallel` to < `sum(allowed_vector)`."
            )

        # Variation of noise for computing loss.
        # Shape of coord_batch (nb_samples * 2 * nb_parallel, nb_features).
        coord_batch = np.repeat(self.current_noise, 2 * self.nb_parallel, axis=0)

        # Sample indices of exploration for optimization.
        indices = []
        for i in range(x_ori.shape[0]):
            indices.append(
                np.random.choice(
                    x_ori.shape[-1],
                    self.nb_parallel,
                    replace=False,
                    p=self.allowed_vector / sum(self.allowed_vector)
                )
            )
        # Shape of indices (nb_samples * nb_parallel, )
        indices = np.concatenate(indices)

        # Create the batch of modifications to run
        for i in range(self.nb_parallel * self.current_noise.shape[0]):
            coord_batch[2 * i, indices[i]] += self.variable_h[indices[i]]
            coord_batch[2 * i + 1, indices[i]] -= self.variable_h[indices[i]]


        # Compute loss for all samples and coordinates, then optimize
        # Repeat for all coordinates
        expanded_x = np.repeat(x_ori, 2 * self.nb_parallel, axis=0)
        expanded_y = np.repeat(y, 2 * self.nb_parallel, axis=0)
        expanded_c = np.repeat(constants, 2 * self.nb_parallel)

        # Add noise to original feature-level data
        expanded_x_perturbed = expanded_x + coord_batch.reshape(expanded_x.shape)

        # Correct perturbed data and inverse transform to field-level
        # Correct perturbed data
        if not self.corrector is None:
            correct_perturbed_data = self.corrector.transform(expanded_x_perturbed)
        else:
            correct_perturbed_data = expanded_x_perturbed
        # Inverse transform to field-level
        if not self.processor is None:
            inverse_transformed_perturbed_data = self.processor.inverse_transform(correct_perturbed_data)
        else:
            inverse_transformed_perturbed_data = correct_perturbed_data

        perturbed_preds = self.predictor.predict(inverse_transformed_perturbed_data)

        # Calculate the loss function
        # Calculate the distortion norms.
        distortion_norms = self.norm_func(correct_perturbed_data, expanded_x).reshape(-1)
        # Calculate the adversarial losses
        adversarial_losses = self.loss_func(perturbed_preds, expanded_y).reshape(-1)
        # Trade-off the distortion norm and the adversarial loss.
        losses = expanded_c * adversarial_losses + distortion_norms
        
        # Convert to shape (nb_samples, 2 * nb_parallel)
        losses = losses.reshape(x_ori.shape[0], 2 * self.nb_parallel)

        if self.adam_mean is not None and self.adam_var is not None and self.adam_epochs is not None:
            self.optimizer_adam_coordinate(
                losses,
                indices,
            )
        else:
            raise ValueError("Unexpected `None` in `adam_mean`, `adam_var` or `adam_epochs` detected.")

        return x_ori + self.current_noise

    def optimizer_adam_coordinate(self, losses, indices):
        '''
        Implementation of the ADAM optimizer for coordinate descent, updates noise.

        Args:
            losses: Overall loss. Shape (nb_samples, 2 * nb_parallel)
            indices: Indices of the coordinates to update. Shape (nb_samples * nb_parallel, )
        '''

        # Estimate grads from loss variation, shape (nb_samples, nb_parallel)
        grads = losses[:, 2 * np.array(range(self.nb_parallel))] - losses[:, 2 * np.array(range(self.nb_parallel)) + 1]
        # Convert shape of indices from (nb_samples * nb_parallel, ) to (nb_samples, nb_parallel)
        indices = indices.reshape(grads.shape)
        grads = grads / 2 * self.variable_h[indices]

        # ADAM update
        for i in range(grads.shape[0]):
            # Update adam_mean and adam_var
            self.adam_mean[i][indices[i]] = self.adam_beta1 * self.adam_mean[i][indices[i]] + (1 - self.adam_beta1) * grads[i]
            self.adam_var[i][indices[i]] = self.adam_beta2 * self.adam_var[i][indices[i]] + (1 - self.adam_beta2) * grads[i] ** 2

            # Correct adam_mean and adam_var
            adam_mean_corr = self.adam_mean[i][indices[i]] / (1 - np.power(self.adam_beta1, self.adam_epochs[i][indices[i]]))
            adam_var_corr = self.adam_var[i][indices[i]] / (1 - np.power(self.adam_beta2, self.adam_epochs[i][indices[i]]))

            # Update noise
            self.current_noise[i][indices[i]] -= self.learning_rate * adam_mean_corr / (np.sqrt(adam_var_corr) + 1e-8)

            # Update epoch
            self.adam_epochs[i][indices[i]] += 1


