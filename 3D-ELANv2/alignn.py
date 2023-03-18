import dgl
import dgl.function as fn
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
import dgl.function as fn
from dgl.nn.pytorch import AvgPooling

class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""
    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale = None,
    ):
        """Register th parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", th.linspace(self.vmin, self.vmax, self.bins)
        )
        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale
        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: th.Tensor) -> th.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        return th.exp(
            -self.gamma * (distance.unsqueeze(1) - self.centers) ** 2
        )

class ALIGNN(nn.Module):
    """Atomistic Line graph network.
    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """
    def __init__(self,config):
        """Initialize class with number of input features, conv layers."""
        super().__init__()

        self. angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1,vmax=1.0,bins=config.triplet_input_features,),
            )

    def forward(self, g):
        """ALIGNN : start with `atom_features`
        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)
        """
        # 2.1 获得对应的线图
        g, lg = g
        lg = lg.local_var()

        # 2.1 线图中每一个三元组，取出夹角特征 然后用RBF转化成特征向量 angle features (fixed)
        z = self.angle_embedding(lg.edata.pop("h"))

        return z
