"""SGLang 0.5.23 extended with group generation: requests sharing custom_params["group"] sample the same tokens, each
from the group's guided log probabilities under their sampling parameters, and are scheduled together.

GroupLogitProcessor samples each group's token and gives it to all of its requests. GroupEngine's scheduler keeps a
group's requests in the same batches: GroupPrefillAdder prefills them through SGLang's chunked prefill, each up to its
last input token, then extends all of their last input tokens in one batch, and group_retract_decode retracts them
together.
"""

import functools
import math
import types

import torch
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.layers.sampler import Sampler
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.sampling.sampling_params import TOP_K_ALL


def group(req):
    """The group of an SGLang request from group_generate, None for one from generate."""
    return (req.sampling_params.custom_params or {}).get("group")


def sample_tokens(probs, all_sampling_params):
    """One token per row, by SGLang's own Sampler._sample_from_probs.

    Args:
        probs: (G, V), each row's softmax at its temperature
        all_sampling_params: list (G) of SamplingParams, each row's

    Returns:
        (G) int
    """
    top_ks = [sampling_params.top_k for sampling_params in all_sampling_params]
    top_ps = [sampling_params.top_p for sampling_params in all_sampling_params]
    min_ps = [sampling_params.min_p for sampling_params in all_sampling_params]
    need_top_k_sampling = any(top_k != TOP_K_ALL for top_k in top_ks)
    need_top_p_sampling = any(top_p != 1.0 for top_p in top_ps)
    need_min_p_sampling = any(min_p > 0 for min_p in min_ps)
    sampling_info = types.SimpleNamespace(
        sampling_mask_batch_indices=None,
        sampling_seed=None,
        top_ks=torch.tensor(top_ks, dtype=torch.int32, device=probs.device),
        top_ps=torch.tensor(top_ps, dtype=torch.float, device=probs.device),
        min_ps=torch.tensor(min_ps, dtype=torch.float, device=probs.device),
        need_top_k_sampling=need_top_k_sampling,
        need_top_p_sampling=need_top_p_sampling,
        need_min_p_sampling=need_min_p_sampling,
        max_top_k=max(top_ks),
    )
    simple_sampling_case = not (need_top_k_sampling or need_top_p_sampling or need_min_p_sampling)
    # _sample_from_probs uses no attribute of self.
    return Sampler._sample_from_probs(None, probs, sampling_info, None, simple_sampling_case)[0]


def weights(utilities):
    """The weights maximising Σ_k W_k u_k with Σ_k W_k = 1 and the mean W_k² at most 1; uniform when the utilities are
    all equal.

    Args:
        utilities: list (G) of float, finite

    Returns:
        list (G) of float, 1/G + √(G - 1/G) (u_k - ū) / √(Σ_j (u_j - ū)²), ū being the utilities' mean
    """
    G = len(utilities)
    if min(utilities) == max(utilities):
        return [1 / G] * G
    mean = sum(utilities) / G
    norm = math.sqrt(sum((u - mean) ** 2 for u in utilities))
    return [1 / G + math.sqrt(G - 1 / G) * (u - mean) / norm for u in utilities]


class GroupLogitProcessor(CustomLogitProcessor):
    """Samples each group's next token from its guided log probabilities under the group's sampling parameters, as
    SGLang's Sampler samples a request's, and leaves it the only finite logit in each of the group's rows.

    The guided log probabilities are log p_B + Σ_{g_k > 0} g_k Δ_k + min(Σ_{g_k < 0} g_k Δ_k, 0): p_k the model
    under the group's request k, the base log p_B = (1/G) Σ_k log p_k over its G requests, Δ_k = log p_k - log p_B and
    g_k = W_k - 1/G, W being weights of the requests' utilities. A row's custom_params hold its group, the group's size
    G and its request's utility.
    """

    def __call__(self, logits, custom_param_list):
        """
        Args:
            logits: (B, V), one row per request of a group, after SGLang's penalties and before its temperature
            custom_param_list: list (B) of dict, each row's custom_params, its request at "__req__"

        Returns:
            (B, V), modified in place; the row of a request with input left to prefill, whose token SGLang discards,
            as given

        Raises:
            RuntimeError: a group has other than its size of requests sampling in the batch
        """
        rows = [i for i, params in enumerate(custom_param_list) if params["__req__"].extend_end >= len(params["__req__"].full_untruncated_fill_ids)]
        if not rows:
            return logits
        custom_param_list = [custom_param_list[i] for i in rows]
        groups = {}
        for i, params in enumerate(custom_param_list):
            groups.setdefault(params["group"], []).append(i)
        if any(len(members) != custom_param_list[members[0]]["size"] for members in groups.values()):
            raise RuntimeError(f"groups split across batches, requests per group: { {group: len(members) for group, members in groups.items()} }")

        rows_group, guidance = [0] * len(rows), [0.0] * len(rows)
        for g, members in enumerate(groups.values()):
            for i, weight in zip(members, weights([custom_param_list[i]["utility"] for i in members])):
                rows_group[i], guidance[i] = g, weight - 1 / len(members)
        all_sampling_params = [custom_param_list[members[0]]["__req__"].sampling_params for members in groups.values()]
        temperatures = torch.tensor([sampling_params.temperature for sampling_params in all_sampling_params], device=logits.device)
        # Every TP rank samples on its own, so this must be deterministic.
        rows_group = torch.tensor(rows_group, device=logits.device)
        groups_rows = torch.nn.functional.one_hot(rows_group).T.float()
        guidance = groups_rows * torch.tensor(guidance, device=logits.device)
        log_p = logits[rows].float().log_softmax(-1)
        log_p_base = groups_rows @ log_p / groups_rows.sum(1, keepdim=True)
        # Σ_k g_k Δ_k as Σ_k g_k log p_k - (Σ_k g_k) log p_B, over the positive and over the negative g_k.
        guidance_positive, guidance_negative = (g @ log_p - g.sum(1, keepdim=True) * log_p_base for g in (guidance.clamp(min=0), guidance.clamp(max=0)))
        log_p_guided = (log_p_base + guidance_positive + guidance_negative.clamp(max=0)) / temperatures[:, None]
        tokens = sample_tokens(log_p_guided.softmax(-1), all_sampling_params)
        logits[rows] = torch.full_like(logits[rows], float("-inf")).scatter_(1, tokens[rows_group, None].long(), 0)
        return logits


class GroupPrefillAdder(PrefillAdder):
    """SGLang's PrefillAdder that has a group's requests sample their first token in the same batch.

    A group opens at a request of it once the batch has rows for all of its requests. Each of its requests is then
    prefilled as the scheduler's chunked request, up to its last input token, and parked in scheduler.parked_reqs;
    once all are, they extend their last input tokens in one batch. Other requests wait while a group is open.
    """

    def __init__(self, *args, scheduler, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        open_reqs = [r for r in (*self.scheduler.parked_reqs, self.scheduler.chunked_req) if r is not None and group(r)]
        open_group = group(open_reqs[0]) if open_reqs else None
        if open_group is None and group(req) is None:
            return super().add_one_req(req, has_chunked_req, truncation_align_size)
        if has_chunked_req:
            return AddReqResult.OTHER
        if open_group is not None and group(req) != open_group:
            return AddReqResult.CONTINUE
        if open_group is None:
            size = req.sampling_params.custom_params["size"]
            num_held_rows = sum(r.kv.holds_kv for r in self.can_run_list)
            if len(self.can_run_list) + size > self.scheduler.get_num_allocatable_reqs(len(self.running_batch.reqs), running_batch=self.running_batch, num_held_rows=num_held_rows):
                return AddReqResult.NO_TOKEN

        context_len = len(req.full_untruncated_fill_ids) - 1 - req.prefix_len
        self.rem_chunk_tokens = context_len if self.rem_chunk_tokens is None else min(self.rem_chunk_tokens, context_len)
        result = super().add_one_req(req, has_chunked_req, truncation_align_size)
        return AddReqResult.OTHER if result == AddReqResult.CONTINUE else result

    def add_chunked_req(self, req):
        if group(req) is None:
            return super().add_chunked_req(req)
        context_len = len(req.full_untruncated_fill_ids) - 1 - req.prefix_len
        if context_len > 0:
            self.chunked_req_limit = context_len if self.chunked_req_limit is None else min(self.chunked_req_limit, context_len)
            return super().add_chunked_req(req)

        if req not in self.scheduler.parked_reqs:
            self.scheduler.parked_reqs += (req,)
        parked_reqs = self.scheduler.parked_reqs
        if len(parked_reqs) < req.sampling_params.custom_params["size"]:
            return None
        num_added, attributes, memory_budget_attributes = len(self.can_run_list), dict(vars(self)), dict(vars(self.memory_budget))
        self.chunked_req_limit = None
        for parked_req in parked_reqs:
            parked_req.init_next_round_input()
            super().add_chunked_req(parked_req)
        if len(self.can_run_list) - num_added < len(parked_reqs):
            for parked_req in parked_reqs:
                parked_req.extend_end = parked_req.prefix_len
            del self.can_run_list[num_added:]
            vars(self).update(attributes)
            vars(self.memory_budget).update(memory_budget_attributes)
            if not self.running_batch.reqs:
                raise RuntimeError(f"the KV cache cannot hold group {group(req)} even alone")
            return req  # stays the chunked request, so that the next prefill retries
        self.scheduler.parked_reqs = ()
        return None


def group_decode_retraction_order(reqs, decode_retraction_order=ScheduleBatch._get_decode_retraction_order):
    """ScheduleBatch._get_decode_retraction_order with each group's requests moved up to its first."""
    indices = {}
    for i in decode_retraction_order(reqs):
        indices.setdefault(group(reqs[i]) or reqs[i].rid, []).append(i)
    return [i for group_indices in indices.values() for i in group_indices]


def group_retract_decode(batch, retract_decode=ScheduleBatch.retract_decode):
    """ScheduleBatch.retract_decode, then retracting the requests left of each group it retracted part of."""
    retracted_reqs, new_token_ratio, reqs_to_abort = retract_decode(batch)
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
    Scheduler.parked_reqs = ()
    Scheduler.get_prefill_adder_class = lambda self: functools.partial(GroupPrefillAdder, scheduler=self)
    ScheduleBatch._get_decode_retraction_order = staticmethod(group_decode_retraction_order)
    ScheduleBatch.retract_decode = group_retract_decode
    return run_scheduler_process(*args, **kwargs)


class GroupEngine(Engine):
    """SGLang's Engine, its scheduler processes run by run_group_scheduler_process."""

    run_scheduler_process_func = staticmethod(run_group_scheduler_process)
