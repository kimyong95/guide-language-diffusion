"""SGLang 0.5.23 extended with group generation: requests sharing custom_params["group"] sample the same tokens, each
from Σ_k W_k log p_k over the group, and are scheduled together.

GroupLogitProcessor samples each group's token and gives it to all of its requests. GroupEngine's scheduler keeps a
group's requests in the same batches: GroupPrefillAdder admits them all or none, unchunked, and group_retract_decode
retracts them together.
"""

import functools

import torch
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


def group(req):
    """The group of an SGLang request from group_generate, None for one from generate."""
    return (req.sampling_params.custom_params or {}).get("group")


class GroupLogitProcessor(CustomLogitProcessor):
    """Samples each group's next token from Σ_k W_k log p_k at the group's temperature and leaves it the only finite
    logit in each of the group's rows.

    A row's custom_params hold its group, the group's size and its weight W_k. A group's weights sum to 1, so its
    weighted logits are its weighted log probabilities up to a constant.
    """

    def __call__(self, logits, custom_param_list):
        """
        Args:
            logits: (B, V), one row per request of a group, after SGLang's penalties and before its temperature
            custom_param_list: list (B) of dict, each row's custom_params, its request at "__req__"

        Returns:
            (B, V), modified in place

        Raises:
            RuntimeError: a group has other than its size of requests in the batch, or their outputs differ in length
        """
        groups = {}
        for params in custom_param_list:
            groups.setdefault(params["group"], []).append(params)
        if any(len(members) != members[0]["size"] for members in groups.values()):
            raise RuntimeError(f"groups split across batches, requests per group: { {group: len(members) for group, members in groups.items()} }")
        output_lengths = {group: [len(params["__req__"].output_ids) for params in members] for group, members in groups.items()}
        if any(len(set(lengths)) > 1 for lengths in output_lengths.values()):
            raise RuntimeError(f"groups out of sync, output lengths per group: {output_lengths}")

        group_numbers = {group: g for g, group in enumerate(groups)}
        rows = torch.arange(len(custom_param_list), device=logits.device)
        rows_group = torch.tensor([group_numbers[params["group"]] for params in custom_param_list], device=logits.device)
        weights = torch.zeros(len(groups), len(custom_param_list), device=logits.device)
        weights[rows_group, rows] = torch.tensor([params["weight"] for params in custom_param_list], device=logits.device)
        temperatures = torch.tensor([members[0]["__req__"].sampling_params.temperature for members in groups.values()], device=logits.device)
        # Every TP rank samples on its own, so this must be deterministic.
        logits_guided = weights @ logits.float() / temperatures[:, None]
        tokens = torch.multinomial(logits_guided.softmax(-1), 1)[:, 0]
        logits.fill_(float("-inf"))
        logits[rows, tokens[rows_group]] = 0
        return logits


class GroupPrefillAdder(PrefillAdder):
    """SGLang's PrefillAdder that adds all of a group's requests to the prefill batch, none of them chunked, or none.

    The scheduler offers the waiting queue's requests in order, a group's requests consecutive as FCFS keeps them. A
    group opens at its first request once all of it waits and the batch has rows for it; a request of it failing undoes
    the whole group. A group first in its batch may exceed the batch's token budgets, as a lone request may.
    """

    def __init__(self, *args, scheduler, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.members = []  # the open group's requests not yet added

    def budget_state(self):
        return AddReqResult.CONTINUE if self.members else super().budget_state()

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        if group(req) is None:
            return super().add_one_req(req, has_chunked_req, truncation_align_size)
        if not self.members:
            size = req.sampling_params.custom_params["size"]
            start = self.scheduler.waiting_queue.index(req)
            members = self.scheduler.waiting_queue[start:start + size]
            if len(members) < size or any(group(member) != group(req) for member in members):
                return AddReqResult.CONTINUE  # the rest of the group has yet to arrive
            num_held_rows = sum(r.kv.holds_kv for r in self.can_run_list)
            if len(self.can_run_list) + size > self.scheduler.get_num_allocatable_reqs(len(self.running_batch.reqs), running_batch=self.running_batch, num_held_rows=num_held_rows):
                return AddReqResult.NO_TOKEN
            self.snapshot = len(self.can_run_list), dict(vars(self)), dict(vars(self.memory_budget))
            if not self.can_run_list:
                tokens = sum(self.ceil_paged_tokens(len(member.origin_input_ids) + len(member.output_ids)) for member in members) + 1
                self.rem_input_tokens = max(self.rem_input_tokens, tokens)
                if self.rem_chunk_tokens is not None:
                    self.rem_chunk_tokens = max(self.rem_chunk_tokens, tokens)
            self.members = members
        assert req is self.members[0], "a group's requests are not consecutive in the waiting queue"

        result = super().add_one_req(req, has_chunked_req, truncation_align_size)
        if not self.can_run_list or self.can_run_list[-1] is not req or self.new_chunked_req is req:
            self.undo_group()
            if result == AddReqResult.NO_TOKEN and not self.can_run_list and not self.running_batch.reqs:
                raise RuntimeError(f"the KV cache cannot hold group {group(req)} even alone")
            return AddReqResult.OTHER if result == AddReqResult.CONTINUE else result
        self.members = self.members[1:]
        return self.budget_state()

    def undo_group(self):
        """Removes the open group's added requests and restores the budgets from before it, closing it."""
        num_added, attributes, memory_budget_attributes = self.snapshot
        for member in self.can_run_list[num_added:]:
            self.tree_cache.unlock(member.lock)
            member.lock = None
        del self.can_run_list[num_added:]
        vars(self).update(attributes)
        vars(self.memory_budget).update(memory_budget_attributes)


def group_decode_retraction_order(reqs, decode_retraction_order=ScheduleBatch._get_decode_retraction_order):
    """ScheduleBatch._get_decode_retraction_order with each group's requests moved up to its first."""
    indices = {}
    for i in decode_retraction_order(reqs):
        indices.setdefault(group(reqs[i]) or reqs[i].rid, []).append(i)
    return [i for group_indices in indices.values() for i in group_indices]


def group_retract_decode(batch, retract_decode=ScheduleBatch.retract_decode):
    """ScheduleBatch.retract_decode, then retracting the requests left of each group it retracted part of.

    Raises:
        RuntimeError: it aborted a group's request, the KV cache unable to hold the group even alone
    """
    retracted_reqs, new_token_ratio, reqs_to_abort = retract_decode(batch)
    if any(group(req) for req in reqs_to_abort):
        raise RuntimeError(f"the KV cache cannot hold groups {[group(req) for req in reqs_to_abort]} even alone")
    groups = {group(req) for req in retracted_reqs} - {None}
    keep_indices = [i for i, req in enumerate(batch.reqs) if group(req) not in groups]
    for i, req in enumerate(batch.reqs):
        if group(req) in groups:
            batch.release_req(i, len(keep_indices))
            retracted_reqs.append(req)
    batch.filter_batch(keep_indices=keep_indices)
    return retracted_reqs, new_token_ratio, reqs_to_abort


def run_group_scheduler_process(*args, **kwargs):
    """SGLang's run_scheduler_process, groups admitted by GroupPrefillAdder and retracted by group_retract_decode."""
    Scheduler.get_prefill_adder_class = lambda self: functools.partial(GroupPrefillAdder, scheduler=self)
    ScheduleBatch._get_decode_retraction_order = staticmethod(group_decode_retraction_order)
    ScheduleBatch.retract_decode = group_retract_decode
    return run_scheduler_process(*args, **kwargs)


class GroupEngine(Engine):
    """SGLang's Engine, its scheduler processes run by run_group_scheduler_process."""

    run_scheduler_process_func = staticmethod(run_group_scheduler_process)
