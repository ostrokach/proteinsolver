import heapq
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data


def get_node_proba(net, x, edge_index, edge_attr, num_categories=20):
    raise Exception("Use get_node_outputs instead!")


def get_node_value(net, x, edge_index, edge_attr, num_categories=20):
    raise Exception("Use get_node_outputs instead!")


@torch.no_grad()
def get_node_outputs(
    net: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_categories: int = 20,
    output_transform: Optional[str] = None,
    oneshot: bool = False,
) -> torch.Tensor:
    """Return network output for each node in the reference sequence.

    Args:
        net: The network to use for making predictions.
        x: Node attributes for the target sequence.
        edge_index: Edge indices of the target sequence.
        edge_attr: Edge attributes of the target sequence.
        num_categories: The number of categories to which the network assigns individual nodes
            (e.g. the number of amino acids for the protein design problem).
        output_transform: Transformation to apply to network outputs.
            - `None` - No transformation.
            - `proba` - Apply the softmax transformation.
            - `logproba` - Apply the softmax transformation and log the results.
        oneshot: Whether predictions should be made using a single pass through the network,
            or incrementally, by making a single prediction at a time.

    Returns:
        A tensor of network predictions for each node in `x`.
    """
    assert output_transform in [None, "proba", "logproba"]

    x_ref = x
    x = torch.ones_like(x_ref) * num_categories
    x_proba = torch.zeros_like(x_ref).to(torch.float)
    index_array_ref = torch.arange(x_ref.size(0))
    mask = x == num_categories
    while mask.any():
        output = net(x, edge_index, edge_attr)
        if output_transform == "proba":
            output = torch.softmax(output, dim=1)
        elif output_transform == "logproba":
            output = torch.softmax(output, dim=1).log()

        output_for_x = output.gather(1, x_ref.view(-1, 1))

        if oneshot:
            return output_for_x.data.cpu()

        output_for_x = output_for_x[mask]
        index_array = index_array_ref[mask]
        max_proba, max_proba_position = output_for_x.max(dim=0)

        assert x[index_array[max_proba_position]] == num_categories
        assert x_proba[index_array[max_proba_position]] == 0
        correct_amino_acid = x_ref[index_array[max_proba_position]].item()
        x[index_array[max_proba_position]] = correct_amino_acid
        assert output[index_array[max_proba_position], correct_amino_acid] == max_proba
        x_proba[index_array[max_proba_position]] = max_proba
        mask = x == num_categories
    return x_proba.data.cpu()


@torch.no_grad()
def scan_with_mask(
    net: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_categories: int = 20,
    output_transform: Optional[str] = None,
) -> torch.Tensor:
    """Generate an output for each node in the sequence by masking one node at a time."""
    assert output_transform in [None, "proba", "logproba"]

    x_ref = x
    output_for_mask = torch.zeros_like(x_ref).to(torch.float)
    for i in range(x_ref.size(0)):
        x = x_ref.clone()
        x[i] = num_categories
        output = net(x, edge_index, edge_attr)
        if output_transform == "proba":
            output = torch.softmax(output, dim=1)
        elif output_transform == "logproba":
            output = torch.softmax(output, dim=1).log()
        output_for_x = output.gather(1, x_ref.view(-1, 1))
        output_for_mask[i] = output_for_x[i]
    return output_for_mask.data.cpu()


# === Protein design ===


@torch.no_grad()
def design_sequence(
    net: nn.Module,
    data: Data,
    random_position: bool = False,
    value_selection_strategy: str = "map",
    num_categories: int = None,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate new sequences.

    Args:
        net: A trained neural network to use for designing sequences.
        data: The data on which to base new sequences.
        random_position: Whether the next position to explore should be selected at random
            or by selecting the position for which we have the most confident predictions.
        value_selection_strategy: Controls the strategy for generating new sequences:
            - "map" - Select the most probable residue each time.
            - "multinomial" - Sample residues according to the probability assigned
                by the network.
            - "ref" - Select the residue provided by the `data.x` reference.
        num_categories: The number of categories possible.
            If `None`, assume that the number of categories corresponds to the maximum value
            in `data.x`.

    Returns:
        A torch tensor of designed sequences.
    """
    assert value_selection_strategy in ("map", "multinomial", "ref")

    if num_categories is None:
        num_categories = data.x.max().item()

    if hasattr(data, "batch"):
        batch_size = data.batch.max().item() + 1
    else:
        batch_size = 1

    x_ref = data.y if hasattr(data, "y") and data.y is not None else data.x
    x = torch.ones_like(data.x) * num_categories
    x_proba = torch.zeros_like(x).to(torch.float)

    # First, gather probabilities for pre-assigned residues
    mask_filled = (x_ref != num_categories) & (x == num_categories)
    while mask_filled.any():
        for (
            max_proba_index,
            chosen_category,
            chosen_category_proba,
        ) in _select_residue_for_position(
            net,
            x,
            x_ref,
            data,
            batch_size,
            mask_filled,
            random_position,
            "ref",
            temperature=temperature,
        ):
            assert chosen_category != num_categories
            assert x[max_proba_index] == num_categories
            assert x_proba[max_proba_index] == 0
            x[max_proba_index] = chosen_category
            x_proba[max_proba_index] = chosen_category_proba
        mask_filled = (x_ref != num_categories) & (x == num_categories)
    assert (x == x_ref).all().item()

    # Next, select residues for unassigned positions
    mask_empty = x == num_categories
    while mask_empty.any():
        for (
            max_proba_index,
            chosen_category,
            chosen_category_proba,
        ) in _select_residue_for_position(
            net,
            x,
            x_ref,
            data,
            batch_size,
            mask_empty,
            random_position,
            value_selection_strategy,
            temperature=temperature,
        ):
            assert chosen_category != num_categories
            assert x[max_proba_index] == num_categories
            assert x_proba[max_proba_index] == 0
            x[max_proba_index] = chosen_category
            x_proba[max_proba_index] = chosen_category_proba
        mask_empty = x == num_categories

    return x.cpu(), x_proba.cpu()


def _select_residue_for_position(
    net,
    x,
    x_ref,
    data,
    batch_size,
    mask_ref,
    random_position,
    value_selection_strategy,
    temperature=1.0,
):
    """Predict a new residue for an unassigned position for each batch in `batch_size`."""
    assert value_selection_strategy in ("map", "multinomial", "ref")

    output = net(x, data.edge_index, data.edge_attr)
    output = output / temperature
    output_proba_ref = torch.softmax(output, dim=1)
    output_proba_max_ref, _ = output_proba_ref.max(dim=1)
    index_array_ref = torch.arange(x.size(0))

    for i in range(batch_size):
        mask = mask_ref
        if batch_size > 1:
            mask = mask & (data.batch == i)

        index_array = index_array_ref[mask]
        max_probas = output_proba_max_ref[mask]

        if random_position:
            selected_residue_subindex = torch.randint(0, max_probas.size(0), (1,)).item()
            max_proba_index = index_array[selected_residue_subindex]
        else:
            selected_residue_subindex = max_probas.argmax().item()
            max_proba_index = index_array[selected_residue_subindex]

        category_probas = output_proba_ref[max_proba_index]

        if value_selection_strategy == "map":
            chosen_category_proba, chosen_category = category_probas.max(dim=0)
        elif value_selection_strategy == "multinomial":
            chosen_category = torch.multinomial(category_probas, 1).item()
            chosen_category_proba = category_probas[chosen_category]
        elif value_selection_strategy == "ref":
            chosen_category = x_ref[max_proba_index]
            chosen_category_proba = category_probas[chosen_category]

        yield max_proba_index, chosen_category, chosen_category_proba


# ASTAR approach


@torch.no_grad()
def get_descendents(net, x, x_proba, edge_index, edge_attr, cutoff):
    index_array = torch.arange(x.size(0))
    mask = x == 20

    output = net(x, edge_index, edge_attr)
    output = torch.softmax(output, dim=1)
    output = output[mask]
    index_array = index_array[mask]

    max_proba, max_index = output.max(dim=1)[0].max(dim=0)
    row_with_max_proba = output[max_index]

    sum_log_prob = x_proba.sum()
    assert sum_log_prob.item() <= 0, x_proba
    #     p_cutoff = min(torch.exp(sum_log_prob), row_with_max_proba.max()).item()

    children = []
    for i, p in enumerate(row_with_max_proba):
        #         if p < p_cutoff:
        #             continue
        x_clone = x.clone()
        x_proba_clone = x_proba.clone()
        assert x_clone[index_array[max_index]] == 20
        assert x_proba_clone[index_array[max_index]] == cutoff
        x_clone[index_array[max_index]] = i
        x_proba_clone[index_array[max_index]] = torch.log(p)
        children.append((x_clone, x_proba_clone))
    return children


@dataclass(order=True)
class PrioritizedItem:
    p: float
    x: Any = field(compare=False)
    x_proba: float = field(compare=False)


@torch.no_grad()
def design_protein(net, x, edge_index, edge_attr, results, cutoff):
    """Design protein sequences using a search strategy."""
    x_proba = torch.ones_like(x).to(torch.float) * cutoff
    heap = [PrioritizedItem(0, x, x_proba)]
    i = 0
    while heap:
        item = heapq.heappop(heap)
        if i % 1000 == 0:
            print(
                f"i: {i}; p: {item.p:.4f}; num missing: {(item.x == 20).sum()}; "
                f"heap size: {len(heap):7d}; results size: {len(results)}"
            )
        if not (item.x == 20).any():
            results.append(item)
        else:
            children = get_descendents(net, item.x, item.x_proba, edge_index, edge_attr, cutoff)
            for x, x_proba in children:
                heapq.heappush(heap, PrioritizedItem(-x_proba.sum(), x, x_proba))
        i += 1
        if len(heap) > 1_000_000:
            heap = heap[:700_000]
            heapq.heapify(heap)
    return results
