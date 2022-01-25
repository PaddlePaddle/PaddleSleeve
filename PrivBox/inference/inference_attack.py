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
Abstract base class for inference attacks
"""

import abc
from privbox.attack import Attack
import paddle

from typing import List
from paddle import Tensor

class InferenceAttack(Attack):
    """ 
    Abstract model inference attack class
    """

    @abc.abstractmethod
    def infer(self, data, **kwargs) -> paddle.Tensor:
        """
        Infer data's relationship with training set

        Args:
            data(Tensor|List[Tensor]): input data to infer its relationship (whether in training set)

        Returns:
            (Tensor): infer result
        """
        raise NotImplementedError
