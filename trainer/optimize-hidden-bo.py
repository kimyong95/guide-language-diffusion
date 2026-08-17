import math
import sys
from contextlib import ExitStack

import botorch
import einops
import gpytorch
import torch
from absl import flags
from accelerate.utils import broadcast, gather_object
from gpytorch.constraints import GreaterThan
from gpytorch.models.exact_prediction_strategies import LinearPredictionStrategy
from gpytorch.priors import LogNormalPrior
from linear_operator.operators import MatmulLinearOperator, RootLinearOperator
from ml_collections import config_flags
from torch.utils.data import DataLoader
from tqdm import tqdm

from base import BaseTrainer
from mixins import DistributedSubsampleDataset
from utils import concat

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/optimize-hidden-bo.py", "Training configuration.")


class SphericalLinearKernel(gpytorch.kernels.Kernel):
    r"""k(x, x') = b0 + b1 <x, x'>, for x already on the unit sphere (arxiv 2512.00170, eq. 2).

    The paper's kernel is b0 + b1 <P(z), P(z')> with z = x/(a*l) and P the inverse stereographic
    projection R^D -> S^D. Nearly all of that exists to manufacture a sphere out of a hypercube, and
    collapses on a domain that is one already:

    - P is dropped. It is the identity for unit-norm inputs (P(z) = [z, 0]), and its one extra
      coordinate (||z||^2 - 1)/(||z||^2 + 1) is a pure function of the magnitude ||z||, so on a
      fixed-radius domain it is a constant shared by every input that b0 already absorbs. Theorem 1's
      boundary-seeking pathology, the sole reason for P, does not arise on a sphere.
    - The global lengthscale a is dropped: it only rescales z ahead of a map that renormalizes, so on a
      fixed radius it cancels exactly and leaves a flat direction in the marginal likelihood.
    - The ARD lengthscales l are dropped: Lx*D of them against N ~ 10^2 observations are not
      identifiable, and a hidden state has no axis-aligned structure to justify them.

    What is kept is b = softmax(raw_coeffs), so b0 + b1 = 1 and there is no implicit outputscale, and
    the exact prediction strategy, which the Lx*D + 1 feature map makes available.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter("raw_coeffs", torch.nn.Parameter(torch.zeros(2)))

    @property
    def coeffs(self):
        """The coefficients b0, b1 for the constant and linear terms"""
        return torch.nn.functional.softmax(self.raw_coeffs, dim=-1)

    def features(self, x):
        """
        Args:
            x: (..., N, Lx*D) of unit norm

        Returns:
            (..., N, Lx*D+1) whose inner products are the kernel
        """
        b0, b1 = self.coeffs
        return torch.cat([x * b1.sqrt(), b0.sqrt().expand_as(x[..., :1])], dim=-1)

    def forward(self, x1, x2, diag=False, **params):
        features1, features2 = self.features(x1), self.features(x2)
        if diag:
            return (features1 * features2).sum(dim=-1)
        if torch.equal(x1, x2):
            return RootLinearOperator(features1)   # the rank Lx*D+1 always exceeds N, so the root form is the cheap one
        return MatmulLinearOperator(features1, features2.mT)

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return LinearPredictionStrategy(train_inputs, train_prior_dist, train_labels, likelihood)


class Trainer(BaseTrainer):
    """optimize-hidden with the intervention chosen by Bayesian optimization instead of Adam on the NFT loss.

    x is the same object as in optimize-hidden -- n_intervene hidden states, each on the sphere of radius
    sqrt(D) -- but the reward is now a black box: no loss, no backward pass through the model, no
    gradient checkpointing. Each epoch evaluates one x and hands the surrogate a single scalar.

    That scalar has to be a function of x rather than a sample of one, which is what fixes two things
    the gradient runs leave free. Generation is greedy (model.temperature = 0), and the prompt set is
    the whole sample task, subsampled once in the dataset's own __init__ and never again -- so k = 1,
    since greedy decoding makes repeats of a prompt bit-identical.

    Acquisition is exact Thompson sampling. A posterior function sample under this kernel is linear in
    x, so its maximizer over the product of Lx spheres is closed form and globally optimal (see
    thompson_sample); the paper's own argument for linear kernels. No inner optimizer is involved.

    n_intervene is the lever to reach for if the run stalls: the search space is n_intervene*D, which at
    the default 8 is 16384 dims. With a budget of a few hundred evaluations that leaves N << D, so the
    posterior is prior-dominated in the unexplored directions and early draws are near-random
    directions with a small data-informed tilt. That is past the regime the paper tested (6392 dims at
    N = 1000), and dropping n_intervene is what brings it back inside.
    """

    def __init__(self, config):
        super().__init__(config)

        # explicit, because setup_accelerator seeds device_specific and every rank has to land on the same x
        self.generator = torch.Generator().manual_seed(config.seed)
        self.x = self.random_x()
        self.x_best = self.x.clone()

        # the history is identical on every rank: Y comes off an all-gather and x off a broadcast
        self.X = torch.empty(0, config.model.n_intervene * self.pipeline.config.hidden_size, device=self.accelerator.device, dtype=torch.float64)
        self.Y = torch.empty(0, device=self.accelerator.device, dtype=torch.float64)

        # N = -1 with k = 1 is one pass over the whole task, and DistributedSubsampleDataset.subsample is never called again
        self.train_dataset = DistributedSubsampleDataset(all_data=self.sample_task.data,N=-1,G=self.accelerator.num_processes,N_batch_max=config.model.max_batch_size_per_device,k=1,base_seed=config.seed,)
        training_dataloader = DataLoader(self.train_dataset, batch_size=self.train_dataset.N_local_batch, shuffle=False)
        self.training_dataloader = self.accelerator.prepare(training_dataloader)

        self.val_dataset = DistributedSubsampleDataset(all_data=self.val_task.data,N=-1,G=self.accelerator.num_processes,N_batch_max=config.model.max_batch_size_per_device,k=config.val.k,base_seed=config.seed,)
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.val_dataset.N_local_batch, shuffle=False)
        self.val_dataloader = self.accelerator.prepare(val_dataloader)

    @staticmethod
    def project_to_sphere(x):
        return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True) * math.sqrt(x.shape[-1])

    @staticmethod
    def standardize(y):
        return (y - y.mean()) / y.std().clamp_min(1e-6)   # a run of identical rewards has std 0, which botorch's standardize turns into nan

    def randn(self, shape):
        """Draws off self.generator, so the sample is reproducible and independent of the rank-specific global seed."""
        return torch.randn(shape, generator=self.generator, dtype=torch.float64).to(self.accelerator.device)

    def random_x(self):
        """A draw from the uniform measure on the product of spheres, which is what replaces linear-bo's Sobol design."""
        x = torch.randn(self.config.model.n_intervene, self.pipeline.config.hidden_size, generator=self.generator)
        return self.project_to_sphere(x).to(self.accelerator.device, torch.float32)

    def unit_features(self, x):
        """
        Args:
            x: (Lx, D), each row of norm sqrt(D)

        Returns:
            (Lx*D,) float64 of unit norm
        """
        return einops.rearrange(x, "Lx D -> (Lx D)").double() / math.sqrt(x.numel())

    def fit_model(self):
        """Refits from scratch on the whole history, as linear-bo does at every iteration.

        Returns:
            (SingleTaskGP, float) the fitted model and its marginal log likelihood
        """
        noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)
        with botorch.settings.validate_input_scaling(False):   # unit-norm rows are not in the unit cube, and have no reason to be
            model = botorch.models.SingleTaskGP(
                train_X=self.X,
                train_Y=self.standardize(self.Y)[:, None],
                mean_module=gpytorch.means.ConstantMean(),
                covar_module=SphericalLinearKernel(),
                likelihood=gpytorch.likelihoods.GaussianLikelihood(noise_prior, noise_constraint=GreaterThan(1e-4, initial_value=noise_prior.mode)),
                outcome_transform=None,   # Y is standardized above, so the model's targets are the space thompson_sample reads
            )

        model.train()
        with ExitStack() as es:
            es.enter_context(gpytorch.settings.cholesky_max_tries(10))
            es.enter_context(gpytorch.settings.max_cholesky_size(float("inf")))
            es.enter_context(gpytorch.settings.fast_computations(log_prob=True, covar_root_decomposition=False, solves=False))   # woodbury, for a kernel this low-rank
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
            botorch.fit.fit_gpytorch_mll(mll, optimizer=botorch.fit.fit_gpytorch_mll_scipy)
            mll.train()   # fit_gpytorch_mll hands back an eval-mode mll, where model(train_X) is the posterior rather than the prior
            with torch.no_grad():
                mll_value = mll(model(*model.train_inputs), model.train_targets).item()
        model.eval()

        return model, mll_value

    @torch.no_grad()
    def thompson_sample(self, model):
        """Draws one posterior function and returns its exact maximizer over the product of Lx spheres.

        A sample under this kernel is f(x) = mean + sqrt(b0)*t0 + sqrt(b1)*theta @ u, with
        u = vec(x)/sqrt(Lx*D) -- linear in u, so the maximizer of each row is that row of theta scaled
        back onto the sphere, and neither the sqrt(b1) nor the two offsets can move it. Matheron's rule
        draws theta exactly through one (N, N) solve, never forming the (Lx*D, Lx*D) posterior covariance.

        Args:
            model: SingleTaskGP fitted on (self.X, standardize(self.Y))

        Returns:
            (Lx, D) float32, each row of norm sqrt(D)
        """
        b0, b1 = model.covar_module.coeffs
        noise = model.likelihood.noise.squeeze()
        U, y = self.X, self.standardize(self.Y)                                             # (N, Lx*D), (N,)

        theta = self.randn((U.shape[-1],))                                                  # (Lx*D,) prior weights
        prior_y = b1.sqrt() * (U @ theta) + b0.sqrt() * self.randn(())                      # (N,) the same draw seen through the data
        residual = y - model.mean_module.constant - prior_y - noise.sqrt() * self.randn((len(y),))
        covariance = b1 * (U @ U.mT) + b0 + noise * torch.eye(len(y), dtype=U.dtype, device=U.device)   # the bare b0 is the b0 * ones @ ones.T term
        theta = theta + b1.sqrt() * (U.mT @ torch.linalg.solve(covariance, residual))       # (Lx*D,) now a draw from the posterior

        return self.project_to_sphere(einops.rearrange(theta, "(Lx D) -> Lx D", Lx=self.config.model.n_intervene)).float()

    def run(self):
        self.validation_step(epoch=0)
        for epoch in tqdm(range(1, self.config.max_epochs + 1), desc="Epochs", position=0, disable=not self.accelerator.is_main_process):
            y = self.sampling_step(epoch)
            self.optimize_step(epoch=epoch, y=y)
            if epoch % self.config.val.every_n_epochs == 0:
                self.validation_step(epoch)

        self.accelerator.end_training()

    @torch.no_grad()
    def sampling_step(self, epoch):
        self.pipeline.model.eval()
        cfg = self.config.model

        sampling_data = []
        for data_ids in tqdm(self.training_dataloader, desc="Sampling", position=1, leave=False, disable=not self.accelerator.is_main_process):
            prompts = [self.sample_task.prompt(int(data_id)) for data_id in data_ids]
            prompt_tokens = self.pipeline.texts_to_tokens(prompts, system_prompt=self.sample_task.SYSTEM_PROMPT, enable_thinking=cfg.enable_thinking, n_intervene=cfg.n_intervene)  # 2D list (N_local_batch, Lp)
            x = einops.repeat(self.x, "Lx D -> N Lx D", N=len(prompt_tokens))                                                                                                      # (N_local_batch, Lx, D)
            generated_output = self.pipeline.generate(prompt_tokens, x, max_new_tokens=cfg.max_new_tokens, temperature=cfg.temperature)
            generated_texts = generated_output.texts
            rewards = torch.tensor([self.sample_task.evaluate(int(data_id), text) for data_id, text in zip(data_ids, generated_texts)], device=self.accelerator.device, dtype=torch.float32)
            entropies = generated_output.entropies   # (N_local_batch,) nats/token

            sampling_data.append({
                "generated_texts": generated_texts,   # N_local_batch x str
                "rewards": rewards,                   # (N_local_batch,)
                "entropies": entropies,               # (N_local_batch,)
            })

        sampling_data = {key: concat([batch[key] for batch in sampling_data]) for key in sampling_data[0]}

        gathered_rewards = self.accelerator.gather(sampling_data["rewards"])
        gathered_entropy = self.accelerator.gather(sampling_data["entropies"]).mean()
        gathered_texts = gather_object(sampling_data["generated_texts"])

        objective_evaluations = epoch * self.train_dataset.N
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="sampling", extra={"sampling/entropy": gathered_entropy.item()})
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_texts, stage="sampling")

        return gathered_rewards.mean().double()

    @torch.no_grad()
    def validation_step(self, epoch):
        self.pipeline.model.eval()
        cfg = self.config.model

        val_data = []
        for data_ids in tqdm(self.val_dataloader, desc="Validation", position=1, leave=False, disable=not self.accelerator.is_main_process):
            prompts = [self.val_task.prompt(int(data_id)) for data_id in data_ids]
            prompt_tokens = self.pipeline.texts_to_tokens(prompts, system_prompt=self.val_task.SYSTEM_PROMPT, enable_thinking=cfg.enable_thinking, n_intervene=cfg.n_intervene)  # 2D list (N_local_batch, Lp)
            x = einops.repeat(self.x_best, "Lx D -> N Lx D", N=len(prompt_tokens))   # (N_local_batch, Lx, D), the incumbent: a Thompson draw is exploratory, so its held-out score says little
            generated_output = self.pipeline.generate(prompt_tokens, x, max_new_tokens=cfg.max_new_tokens, temperature=self.config.val.temperature)   # sampled, not greedy, or pass@k collapses to k copies of one text
            generated_texts = generated_output.texts
            rewards = torch.tensor([self.val_task.evaluate(int(data_id), text) for data_id, text in zip(data_ids, generated_texts)], device=self.accelerator.device, dtype=torch.float32)
            entropies = generated_output.entropies   # (N_local_batch,) nats/token

            val_data.append({
                "data_ids": data_ids,                 # (N_local_batch,)
                "generated_texts": generated_texts,   # N_local_batch x str
                "rewards": rewards,                   # (N_local_batch,)
                "entropies": entropies,               # (N_local_batch,)
            })

        val_data = {key: concat([batch[key] for batch in val_data]) for key in val_data[0]}

        gathered_data_ids = self.accelerator.gather(val_data["data_ids"]).tolist()
        gathered_rewards = self.accelerator.gather(val_data["rewards"])
        gathered_entropy = self.accelerator.gather(val_data["entropies"]).mean()
        gathered_texts = gather_object(val_data["generated_texts"])
        pass_at_k = torch.stack([gathered_rewards[[i for i, x in enumerate(gathered_data_ids) if x == data_id]].max() for data_id in set(gathered_data_ids)]).mean()

        objective_evaluations = epoch * self.train_dataset.N
        self.log_rewards(objective_evaluations=objective_evaluations, rewards=gathered_rewards, stage="validation", extra={"validation/pass-at-k": pass_at_k.item(), "validation/entropy": gathered_entropy.item()})
        self.log_texts(objective_evaluations=objective_evaluations, rewards=gathered_rewards, texts=gathered_texts, stage="validation")

    def optimize_step(self, epoch, y):
        """Files the x that was just evaluated, then proposes the next one.

        Args:
            epoch: int
            y: () float64, the mean reward the current x earned over the whole task
        """
        self.X = torch.cat([self.X, self.unit_features(self.x)[None]])
        self.Y = torch.cat([self.Y, y.reshape(1)])
        if y >= self.Y.max():
            self.x_best = self.x.clone()

        model_log = {}
        if self.accelerator.is_main_process:
            if len(self.Y) < self.config.bo.n_init:
                self.x = self.random_x()
            else:
                model, mll_value = self.fit_model()
                b0, b1 = model.covar_module.coeffs
                self.x = self.thompson_sample(model)
                model_log = {"bo/b0": b0.item(), "bo/b1": b1.item(), "bo/noise": model.likelihood.noise.item(), "bo/mll": mll_value}
        self.x = broadcast(self.x)

        objective_evaluations = epoch * self.train_dataset.N
        self.accelerator.log({
            "objective-evaluations": objective_evaluations,
            "bo/observations": len(self.Y),
            "bo/y": y.item(),
            "bo/y-best": self.Y.max().item(),
            # against the 1/sqrt(D) a random direction scores, this is how far the posterior has pulled the proposal off the prior
            "bo/cosine-to-best": torch.nn.functional.cosine_similarity(self.x, self.x_best, dim=-1).mean().item(),
            **model_log,
        })


if __name__ == "__main__":
    FLAGS(sys.argv)
    trainer = Trainer(FLAGS.config)
    trainer.run()
