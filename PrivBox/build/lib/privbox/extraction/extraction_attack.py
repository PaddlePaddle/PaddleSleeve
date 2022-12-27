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
Abstract base class for model extraction attacks
"""


import abc
from privbox.attack import Attack
import paddle

from typing import List
from paddle import Tensor

class ExtractionAttack(Attack):
    """ 
    Abstract model extraction attack class
    """

    @abc.abstractmethod
    def extract(self, data, **kwargs) -> paddle.nn.Layer:
        """
        Extract target models

        Args:
            data(Tensor): input data that used for model extraction

        Returns:
            (Layer): extracted model
        """
        raise NotImplementedError
