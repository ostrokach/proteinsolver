import torch_geometric
import torch_scatter


def scatter_(name, src, index, out=None, dim=0, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).

    Note:
        This method was copied from torch-geometric v1.3.0 to maintain
        backwards-compatibility.

    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    out = torch_scatter.scatter(src, index, out=None, dim=dim, dim_size=dim_size, reduce=name)
    return out


try:
    from torch_geometric.utils import scatter_  # noqa
except ImportError:
    torch_geometric.utils.scatter_ = scatter_
