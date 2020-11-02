from __future__ import annotations

import logging
import multiprocessing as mp
from queue import Queue
from typing import Union

import torch

from proteinsolver.dashboard.msa_view import MSASeq
from proteinsolver.utils import array_to_seq, design_sequence

ctx = mp.get_context("spawn")


class ProteinSolverProcess(ctx.Process):  # type: ignore
    def __init__(self, net_class, state_file, data, num_designs, temperature=1.0, net_kwargs=None):
        super().__init__(daemon=True)
        self.net_class = net_class
        self.state_file = state_file
        self.net_kwargs = {} if net_kwargs is None else net_kwargs
        self.output_queue: Queue[Union[Exception, MSASeq]] = ctx.Queue()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data = data
        self.num_designs = num_designs
        self.temperature = temperature
        self._cancel_event = ctx.Event()

    def run(self) -> None:
        logger = logging.getLogger(f"protein_solver_process.pid{self.pid}")  # noqa

        try:
            net = self._init_network()
            data = self.data.to(self.device)
        except RuntimeError as e:
            self.output_queue.put(e)
            return

        for i in range(self.num_designs):
            if self.cancelled():
                return

            x, x_proba = design_sequence(
                net,
                data,
                value_selection_strategy="multinomial",
                num_categories=20,
                temperature=self.temperature,
            )
            sum_proba = x_proba.mean().item()
            sum_logproba = x_proba.log().mean().item()
            seq = array_to_seq(x.data.numpy())
            design = MSASeq(i, f"gen-{i + 1:05d}", seq, proba=sum_proba, logproba=sum_logproba)
            self.output_queue.put(design)
            del x, x_proba

    def _init_network(self) -> torch.nn.Module:
        net = self.net_class(**self.net_kwargs)
        net = net.to(self.device)
        net.load_state_dict(torch.load(self.state_file, map_location=self.device))
        net = net.eval()
        return net

    def cancel(self) -> None:
        self._cancel_event.set()

    def cancelled(self) -> bool:
        return self._cancel_event.is_set()
