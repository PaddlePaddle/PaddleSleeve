#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import enum

class DefenseStrategies(enum.Enum):
    """R
    Defenses for attacks
    """
    REGULARIZATION = 1
    CONFIDENCEMASKING = 2
    KNOWLEDGEDISTILLATION = 3
    DIFFERENTIALPRIVACY = 4
    

class DefenseDesc(object):
    """R
    Detail description for defenses
    """
    desc_ = {
        DefenseStrategies.CONFIDENCEMASKING: \
            "Confidence Masking. Confidence score masking method aims to hide the true confidence score returned "\
                    "by the target model and thus mitigates the effectiveness of membership inference attack. "\
                    "The defense methods belonging to this stratety includes restricting the prediction vector to top k classes, "\
                    "rounding prediction vector to small decimals, only returning labels, "\
                    "or adding crafted noise to prediciton vector.",
        DefenseStrategies.REGULARIZATION:\
            "Regularization. Overfitting is the main factor that contributes to membership inference attack. "\
                    "Therefore, regularization techniques that can reduce the overfitting of ML models can be leveraged to defend against the attack. "\
                    "Regularization techniques including L2-norm regularization, dropout, data argumentation, model stacking, early stopping, "\
                    "label smoothing and adversarial regularization can be used as defense methods for defenses.",
        DefenseStrategies.KNOWLEDGEDISTILLATION:\
            "Knowledge Distillation. Knowledge distillation uses the outputs of a large teacher model to train a smaller one, "\
                    "in order to transfer knowledge from the large model to the small one. "\
                    "This strategy is to restrict the protected classifier’s direct access to the private training dataset, "\
                    "thus significantly reduces the membership information leakage.",
        DefenseStrategies.DIFFERENTIALPRIVACY:\
            "Differential Privacy. A model is trained in a differentially private manner, the learned model does not learn or remember any specific user’s details. "\
                    "Thus, differential privacy naturally counteracts membership inference."
    }


    def get_desc(self, defense):
        """R
        get detal description of defense
        """
        return self.desc_[defense]


class RiskLevel(str, enum.Enum):
    """R
    The level of threat risk
    """
    HIGH = "HIGH"
    MIDDLE = "MIDDLE"
    LOW = "LOW"


def mem_inf_risk_level(auc):
    """R
    get membership inference risk level according attack's auc
    """
    if auc < 0.55:
        return RiskLevel.LOW
    elif auc < 0.8:
        return RiskLevel.MIDDLE
    else:
        return RiskLevel.HIGH
