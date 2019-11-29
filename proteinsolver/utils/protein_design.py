import heapq
from dataclasses import dataclass, field
from typing import Any

import torch


@torch.no_grad()
def get_node_proba(net, x, edge_index, edge_attr, num_categories=20):
    x_ref = x
    x = torch.ones_like(x_ref) * num_categories
    x_proba = torch.zeros_like(x_ref).to(torch.float)
    index_array_ref = torch.arange(x_ref.size(0))
    mask = x == num_categories
    while mask.any():
        output = net(x, edge_index, edge_attr)
        output = torch.softmax(output, dim=1)
        output_for_x = output.gather(1, x_ref.view(-1, 1))

        output_for_x = output_for_x[mask]
        index_array = index_array_ref[mask]
        max_proba, max_proba_position = output_for_x.max(dim=0)

        assert x[index_array[max_proba_position]] == num_categories
        assert x_proba[index_array[max_proba_position]] == 0
        x[index_array[max_proba_position]] = x_ref[index_array[max_proba_position]]
        x_proba[index_array[max_proba_position]] = max_proba
        mask = x == num_categories
    return x_proba


@torch.no_grad()
def get_node_value(net, x, edge_index, edge_attr, num_categories=20):
    x_ref = x
    x = torch.ones_like(x_ref) * num_categories
    x_proba = torch.zeros_like(x_ref).to(torch.float)
    index_array_ref = torch.arange(x_ref.size(0))
    mask = x == num_categories
    while mask.any():
        output = net(x, edge_index, edge_attr)
        output_proba = torch.softmax(output, dim=1)
        output_for_x = output_proba.gather(1, x_ref.view(-1, 1))

        output_for_x = output_for_x[mask]
        index_array = index_array_ref[mask]
        max_proba, max_proba_position = output_for_x.max(dim=0)

        assert x[index_array[max_proba_position]] == num_categories
        assert x_proba[index_array[max_proba_position]] == 0
        correct_amino_acid = x_ref[index_array[max_proba_position]].item()
        x[index_array[max_proba_position]] = correct_amino_acid
        x_proba[index_array[max_proba_position]] = output[
            index_array[max_proba_position], correct_amino_acid
        ]
        mask = x == num_categories
    return x_proba


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
    x_proba: Any = field(compare=False)


@torch.no_grad()
def design_protein(net, x, edge_index, edge_attr, results, cutoff):
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
