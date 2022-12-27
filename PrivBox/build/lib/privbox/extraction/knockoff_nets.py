# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement of Knockoff Nets extraction attack
ref paper: https://arxiv.org/pdf/1812.02766.pdf
"""

import os

import abc
import random
from .extraction_attack import ExtractionAttack
from privbox.metrics import Metric
import paddle

from typing import List
from paddle import Tensor
import paddle.nn.functional as F
import numpy as np


class ActionBasedSampler:
    """
    Define an Action based sampler that randomly
    sampling data based input action (i.e., label)
    """
    def __init__(self, dataset, nlabels):
        """
        self.idx store action based data idx,
        i.e., self.idx[action][:] is the data idx for the same label
        """
        self.idx = [[]] * nlabels
        # shuffle data idx
        rand_idx = np.random.choice(np.arange(0, len(dataset)), size=len(dataset), replace=False)

        for id in rand_idx:
            label = dataset[id][1].reshape([-1])[0]
            self.idx[label].append(id)

    def get_idx(self, action):
        """
        radomly get data idx based on action
        """
        idx = self.idx[action][np.random.choice(len(self.idx[action]))]
        return idx


class KnockoffExtractionAttack(ExtractionAttack):
    """ 
    Implement Knockoff model extraction attack class
    """

    """
    Params:
        policy(str): Sampling policy. One can choose "random" or "adaptive" policy.
            "random" sampling policy randomly samples input data to query victim model,
            while "adaptive" sampling policy samples input data based on its feedback of rewards.
        has_label(bool): Whether input data has label, for "adaptive" policy, input data must have label
        reward(str): Reward strategy, only for "adaptive" policy. One can choose "certainty",
            "diversity", "loss" and "all". "certainty" reward is used margin-based certainty measure,
            "diversity" is used for preventing the degenerate case of image exploitation,
            "loss" strategy reward high loss images, and "all" strategy uses all three reward strategies.
        num_labels(int): The number of labels for input data
        num_queries(int): The number of queries is allowed for adversary. i.e., the budget for attack.
        knockoff_batch_size(int): The batch size for training and predicting knockoff model
        knockoff_epochs(int): The epochs for training knockoff model
        knockoff_lr(float): The learning rate for training knockoff model
    """
    params = ["policy", "has_label", "reward", "num_labels", "num_queries",
              "knockoff_batch_size", "knockoff_epochs", "knockoff_lr"]

    def __init__(self, query_functor, knockoff_net=None):
        """
        Construct funtion
        Args:
            query_functor(Callable object): A Callable query functor that accepts input of data X,
                and output of its label. input X is a Tensor with shape of (batch_size, ),
                output Y is a Tensor with shape(batch_size, num_labels)
            knockoff_net(paddle.nn.Layer): User-defined knockoff network
        """
        self.query_functor = query_functor
        self.knockoff_net = knockoff_net
        if knockoff_net is None:
            # use resnet34 for default
            self.knockoff_net = paddle.vision.resnet34()

    def set_params(self, **kwargs):
        """
        Set parameters for attacks

        Args:
            kwargs(dict): Parameters of dictionary type
        """
        super().set_params(**kwargs)
        self.__check_params()

    def extract(self, data, **kwargs) -> paddle.nn.Layer:
        """
        Extract target models

        Args:
            data(Dataset): input dataset that used for model extraction

        Returns:
            (Layer): extracted model
        """
        knockoff_model = paddle.Model(self.knockoff_net)
        knockoff_model.prepare(optimizer=paddle.optimizer.Adam(parameters=knockoff_model.parameters(),
                                                               learning_rate=self.knockoff_lr),
                               loss=paddle.nn.CrossEntropyLoss(),
                               metrics=paddle.metric.Accuracy())

        # construct transfer set
        transfer_set = []
        if self.policy == "random":
            transfer_set = self._transfer_set_random(data, self.num_queries, self.query_functor)
        else:
            # "adaptive" samling policy
            transfer_set = self._transfer_set_adaptive(data, self.num_queries, self.query_functor,
                                                  self.reward, self.num_labels, knockoff_model)
        
        transfer_dataset = paddle.io.TensorDataset([transfer_set[0], transfer_set[1]])
        
        # train knockoff model
        knockoff_model.fit(transfer_dataset, batch_size=self.knockoff_batch_size, epochs=self.knockoff_epochs)

        return self.knockoff_net

    def evaluate(self, target, result, metric_list, **kwargs) -> List[float]:
        """
        Evaluate target and result model using dataset for metrics

        Args:
            target: Attack target model, can be set None if no target model
            result: Attack result model, i.e., extracted knockoff_net
            metrics(List[Metric]): Metric list
            kwargs: must contain "test_dataset" variabel,
                i.g., kwargs = {"test_dataset": test_dataset}

        Returns:
            (List[float]): Evaluate result,
                ret[0] is target evaluate result ret[1] is knockoff evaluate result
        """
        if not ("test_dataset" in kwargs):
            raise ValueError("""must contain "test_dataset" variabel in kwargs,
                i.g., kwargs = {"test_dataset": test_dataset}""")

        test_dataset = kwargs["test_dataset"]
        
        for metric in metric_list:
            if not isinstance(metric, Metric):
                raise ValueError("input metrics type error.")

        test_dataloader = paddle.io.DataLoader(test_dataset, batch_size=self.knockoff_batch_size)

        expected_label = []
        actual_label = []
        target_label =[]

        for d in test_dataloader():
            actual_label.append(result(d[0]))
            expected_label.append(d[1])

            if target is not None:
                target_label.append(target(d[0]))

        actual_ret = []
        target_ret = []

        expected_label = paddle.concat(expected_label)
        actual_label = paddle.concat(actual_label)
        if target is not None:
            target_label = paddle.concat(target_label)

        for metric in metric_list:
            actual_ret.append(metric.compute(actual_label, expected_label))
            if target is not None:
                target_ret.append(metric.compute(target_label, expected_label))
        return [target_ret, actual_ret]

    def _transfer_set_random(self, dataset, nqueries, query_func):
        """
        Get transfer set randomly
        """
        chosen_data = []
        chosen_label = []
        chosen_idx = random.choices(range(0, len(dataset)), k=nqueries)

        for id in chosen_idx:
            label = query_func(paddle.to_tensor(dataset[id][0]))
            chosen_data.append(dataset[id][0])
            chosen_label.append(np.argmax(label))

        return [paddle.to_tensor(chosen_data), paddle.to_tensor(chosen_label)]

    def _transfer_set_adaptive(self, dataset, nqueries, query_func,
                               reward_strategy, nlabels,
                               knockoff_model):
        """
        Get transfer set based on adaptive policy
        """
        chosen_data = []
        chosen_label = []
        if reward_strategy == "div" or reward_strategy == "all":
            self.avg_query_label = paddle.zeros([1, nlabels])

        if reward_strategy == "all":
            self.avg_reward_t = paddle.zeros([3, 1])
            self.var_reward_t = paddle.zeros([3, 1])

        h_estimation = paddle.zeros([nlabels])
        
        learning_rate = paddle.zeros([nlabels])

        sampler = ActionBasedSampler(dataset, nlabels)

        time_counter = 1

        avg_reward = paddle.zeros([1])

        for i in range(nqueries):

            action_prob = self._calc_action_prob(h_estimation)

            # choose action
            action = int(np.random.choice(np.arange(0, nlabels), p=action_prob.numpy()))

            # sample data based on action
            sample_idx = sampler.get_idx(action)
            sample_data = dataset[sample_idx]

            # query to victim's model
            query_label = query_func(paddle.to_tensor(sample_data[0]))

            # train knockoff model for evaluating sample reward
            dataset_ = paddle.io.TensorDataset([paddle.to_tensor([sample_data[0]]),
                                                    paddle.to_tensor(sample_data[1])])
            knockoff_model.fit(dataset_, epochs=1, verbose=0)
            knockoff_label = knockoff_model.network(paddle.to_tensor([sample_data[0]]))
            
            # get reward for sample
            reward = self._get_reward(reward_strategy, knockoff_label, query_label, time_counter)
            avg_reward = avg_reward + (1.0 / time_counter) * (reward - avg_reward)

            # update estimation for H, using gradient bandit algorithm
            learning_rate[action] += 1
            h_estimation = self._update_estimation(reward, learning_rate,
                                avg_reward, action_prob, action, h_estimation)

            chosen_data.append([sample_data[0]])
            chosen_label.append(sample_data[1])

            time_counter += 1

        chosen_dataset = [paddle.to_tensor(chosen_data), paddle.to_tensor(chosen_label)]

        return chosen_dataset

    def _calc_action_prob(self, estimation):
        """
        calc action probability, using softmax op
        """
        prob = F.softmax(estimation)
        return prob

    def _get_reward(self, strategy, knockoff_label, query_label, time_counter):
        """
        Get reward for sample
        """
        if strategy == "certainty":
            return self._certainty_reward(query_label)
        elif strategy == "diversity":
            return self._diversity_reward(query_label, time_counter)
        elif strategy == "loss":
            return self._loss_reward(knockoff_label, query_label)
        else:
            return self._all_reward(knockoff_label, query_label, time_counter)


    def _certainty_reward(self, query_label):
        """
        certainty reward strategy based on margin-based measure
        """
        largests = query_label.topk(2, axis=-1)[0][0]
        reward = largests[1] - largests[0]
        return reward

    def _diversity_reward(self, query_label, time_counter):
        """
        diversity reward strategy
        to prevent the degenerate case of image exploitation over a single label
        """
        self.avg_query_label = self.avg_query_label + (1.0 / time_counter) * \
                               (query_label - self.avg_query_label)

        reward = paddle.zeros(query_label.shape).maximum(query_label - self.avg_query_label)
        return reward.sum()

    def _loss_reward(self, query_label, label):
        """
        loss based reward strategy that reward high loss
        """
        q_probs = F.softmax(query_label)
        l_probs = F.softmax(label)

        reward = - q_probs * paddle.log(l_probs)

        return reward.sum()

    def _all_reward(self, query_label, label, time_counter):
        """
        for all three reward strategies
        """
        cert_reward = self._certainty_reward(query_label)
        div_reward = self._diversity_reward(query_label, time_counter)
        loss_reward = self._loss_reward(query_label, label)

        reward = paddle.to_tensor([cert_reward.numpy(), div_reward.numpy(), loss_reward.numpy()])

        # rewards standardization
        self.avg_reward_t = self.avg_reward_t + (1.0 / time_counter) * (reward - self.avg_reward_t)
        self.var_reward_t = self.var_reward_t + (1.0 / time_counter) * ((reward - self.avg_reward_t)**2 - self.var_reward_t)

        if time_counter > 1:
            reward = (reward - self.avg_reward_t) / paddle.sqrt(self.var_reward_t)
        else:
            reward = paddle.clip(reward, min=0, max=1)

        return paddle.mean(reward)

    def _update_estimation(self, reward, learning_rate, avg_reward,
                           action_prob, action, h_estimation):
        """
        Gradient bandit algorithm.
        Construct an estimation function H that uses for
        selecting an optimizing action
        """
        estimation = h_estimation.numpy()
        for i in range(self.num_labels):
            if i != action:
                estimation[i] = estimation[i] - 1.0 / learning_rate[action] * \
                                       (reward - avg_reward) * action_prob[i]
            else:
                estimation[i] = estimation[i] + 1.0 / learning_rate[action] * \
                                       (reward - avg_reward) * (1 - action_prob[i])

        return paddle.to_tensor(estimation)

    def __check_params(self) -> None:
        """
        Check params
        """
        if not isinstance(self.knockoff_batch_size, int) or self.knockoff_batch_size <= 0:
            raise ValueError("The parameter of knockoff_batch_size must be a positive int value.")

        if not isinstance(self.knockoff_epochs, int) or self.knockoff_epochs <= 0:
            raise ValueError("The parameter of knockoff_epochs must be a positive int value.")

        if not isinstance(self.num_queries, int) or self.num_queries <= 0:
            raise ValueError("The parameter of num_queries must be a positive int value.")

        if self.policy not in ["random", "adaptive"]:
            raise ValueError("The parameter of policy must be 'random' or 'adaptive'.")

        if not isinstance(self.knockoff_lr, float):
            raise ValueError("The parameter of knockoff_lr must be a float value.")

        if not callable(self.query_functor):
            raise ValueError("The constructing parameter of query_functor must be callable.")

        if not isinstance(self.knockoff_net, paddle.nn.Layer):
            raise ValueError("The constructing parameter of knockoff_net must be paddle.nn.Layer type.")

        if self.policy == "adaptive":
            if not (isinstance(self.has_label, bool) and self.has_label == True):
                raise ValueError("The parameter of has_label must be set to True for 'adaptive' policy."
                                 " And must input label within data for extract.")
            if self.reward not in ["cert", "div", "loss", "all"]:
                raise ValueError("The parameter of reward must be 'cert', 'div', 'loss' or 'all'.")
