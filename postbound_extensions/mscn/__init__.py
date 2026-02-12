"""MSCN is a supervised cardinality estimator using a multi-set convolutional neural network architecture.

The MIT License

Copyright (c) 2019, Andreas Kipf, 2026 Rico Bergmann

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from ._estimator import MscnEstimator
from ._featurizer import FeaturizedQuery, MscnFeaturizer
from ._model import SetConv
from ._util import normalize_labels, qerror_loss, unnormalize_labels

__all__ = [
    "FeaturizedQuery",
    "MscnEstimator",
    "MscnFeaturizer",
    "SetConv",
    "normalize_labels",
    "qerror_loss",
    "unnormalize_labels",
]
