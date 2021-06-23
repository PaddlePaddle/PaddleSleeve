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
Implement of rule based membership inference attacks
ref paper: https://arxiv.org/pdf/1709.01604.pdf
"""

import sys

import abc
import paddle

from typing import List
from paddle import Tensor
from .membership_inference_attack import MembershipInferenceAttack

class BaselineMembershipInferenceAttack(MembershipInferenceAttack):
    """ 
    Baseline membership inference attack class which is based on the rule
    that infer an instance as member if its predict result is correct.
    """

    def infer(self, data, **kwargs) -> paddle.Tensor:
        """
        Infer whether data is in training set

        Args:
            data(List[Tensor]): input ([predict result, label]) to infer its membership (whether in training set)

        Returns:
            (Tensor): infer result
        """
        result = (data[0] == data[1]).astype("int32")
        return result
