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
Abstract base class for model inversion attacks
"""


import abc
from privbox.attack import Attack
import paddle

from typing import List
from paddle import Tensor

class InversionAttack(Attack):
    """ 
    Abstract model inversion attack class
    """

    @abc.abstractmethod
    def reconstruct(self, **kwargs) -> List[Tensor]:
        """
        Reconstruct target trained data from InversionAttack

        Returns:
            (Tensor): reconstructed data
        """
        raise NotImplementedError
