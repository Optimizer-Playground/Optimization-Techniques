"""
MSCN model architecture
    provided in the original MSCN respository

The MIT License

Copyright (c) 2019 Andreas Kipf, 2026 Rico Bergmann

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Original author: Andreas Kipf
Original source: https://github.com/andreaskipf/learnedcardinalities
Modified by: Rico Bergmann
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from postbound.util import Logger

from ..util import wrap_logger


class SetConv(nn.Module):
    def __init__(
        self,
        *,
        n_tables: int,
        n_columns: int,
        n_operators: int,
        n_joins: int,
        hidden_units: int = 256,
        verbose: bool | Logger = False,
    ):
        super(SetConv, self).__init__()
        log = wrap_logger(verbose)

        sample_feats = n_tables
        predicate_feats = n_columns + n_operators + 1
        join_feats = n_joins
        self.hid_units = hidden_units

        self.sample_mlp1 = nn.Linear(sample_feats, self.hid_units)
        self.sample_mlp2 = nn.Linear(self.hid_units, self.hid_units)
        self.predicate_mlp1 = nn.Linear(predicate_feats, self.hid_units)
        self.predicate_mlp2 = nn.Linear(self.hid_units, self.hid_units)
        self.join_mlp1 = nn.Linear(join_feats, self.hid_units)
        self.join_mlp2 = nn.Linear(self.hid_units, self.hid_units)
        self.out_mlp1 = nn.Linear(self.hid_units * 3, self.hid_units)
        self.out_mlp2 = nn.Linear(self.hid_units, 1)

        log(
            f"Initialized SetConv with {self.hid_units} hidden units "
            f"({sample_feats} table feats, {predicate_feats} predicate feats, {join_feats} join feats)"
        )

    def forward(
        self,
        samples: torch.FloatTensor,
        joins: torch.FloatTensor,
        predicates: torch.FloatTensor,
        sample_mask: torch.FloatTensor,
        join_mask: torch.FloatTensor,
        predicate_mask: torch.FloatTensor,
    ):
        # samples has shape [batch_size x num_joins+1 x sample_feats]
        # predicates has shape [batch_size x num_predicates x predicate_feats]
        # joins has shape [batch_size x num_joins x join_feats]

        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask  # Mask
        hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
        sample_norm = sample_mask.sum(1, keepdim=False)
        sample_norm[sample_norm == 0] = float("-inf")
        hid_sample = (
            hid_sample / sample_norm
        )  # Calculate average only over non-masked parts

        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = hid_predicate * predicate_mask
        hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
        predicate_norm = predicate_mask.sum(1, keepdim=False)
        predicate_norm[predicate_norm == 0] = float("-inf")
        hid_predicate = hid_predicate / predicate_norm

        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        hid_join = hid_join * join_mask
        hid_join = torch.sum(hid_join, dim=1, keepdim=False)
        join_norm = join_mask.sum(1, keepdim=False)
        join_norm[join_norm == 0] = float("-inf")
        hid_join = hid_join / join_norm

        hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))

        return out
