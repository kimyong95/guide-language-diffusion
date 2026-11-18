import asyncio
import concurrent.futures
import importlib
import inspect
import math
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import uuid
import numpy as np
import shapely
import torch
import trimesh
from baselines.problems.problem_utils import PROBLEMS_DIR, ProblemConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer


async def subprocess_run(args, **kwargs):
    """subprocess.run(args, check=True, **kwargs) on asyncio, killing the process if cancelled.

    Returns:
        bytes, the process's stdout if kwargs pipe it, else None

    Raises:
        subprocess.CalledProcessError: the process exits non-zero
    """
    process = await asyncio.create_subprocess_exec(*args, **kwargs)
    try:
        stdout, _ = await process.communicate()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, args)
    return stdout


class OpenFOAM:
    """Drag coefficient of a car, simulated with OpenFOAM v14 (Foundation) following its official drivaerFastback case.

    evaluate_one_async builds the car by running the response's bpy script in headless Blender and exporting the scene to STL,
    BLENDER_SCRIPT as execute-blender.py does before rendering. It then runs IMAGE, microfluidica's docker image of the
    Foundation's own packages, on a copy of the image's tutorials/incompressibleFluid/drivaerFastback. OPENFOAM_COMMAND changes
    only the entries tagged by their source: [DrivAerNet++] following Elrefaie et al. (NeurIPS 2024), [ours] otherwise.
    Everything else is the official case run by its own Allrun, MESH_SIZE being one of its mesh sizes: the domain,
    snappyHexMesh refinement, k-ω SST model, numerics, moving ground, 16 m/s inlet and 1000 iterations.

    The car keeps its scale and x position, so it must be built in the official frame: metres, length along x, front
    facing -x and front axle at x = 0. [ours] It is moved down onto the ground and sideways to be symmetric about y = 0,
    and its wheels do not rotate, being part of the car's one surface.
    """

    PROMPT_TEMPLATE = inspect.cleandoc("""
        Design a car with the lowest possible drag coefficient, evaluated with OpenFOAM.

        Write a Blender Python script using bpy to build the car, including its wheels, in metres.
        Place its length along x, its front facing -x, and its front axle at x = 0.
        The scene is exported to STL, centered along y and placed on the ground before simulation.
        The metric is the drag coefficient C_d; lower is better.

        Write the script within a ```python block.
    """)
    IN_CONTEXT_PROMPT_TEMPLATE = inspect.cleandoc("""
        {instruction}
        Here are some example solutions from previous attempts, please improve on them:

        {item}

        Please now provide your new, better solution.
    """)
    IN_CONTEXT_ITEM_TEMPLATE = inspect.cleandoc("""
        # Solution {index}:
        ```python
        {code}
        ```
        This solution, solution {index}, got a drag coefficient C_d of {metric}.
    """)
    CODE_BLOCK = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL)

    IMAGE = "microfluidica/openfoam:14"
    AVERAGE_LAST = 100
    N_CPUS = 6
    MESH_SIZE = "S"  # S, M, L, XL

    BLENDER_SCRIPT = inspect.cleandoc("""
        import runpy
        import bpy
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        runpy.run_path("car.py", run_name="__main__")
        bpy.ops.wm.stl_export(filepath="car.stl")
    """)

    OPENFOAM_COMMAND = inspect.cleandoc("""
        set -e
        {
            cp -r $FOAM_TUTORIALS/incompressibleFluid/drivaerFastback case && cd case
            rm constant/geometry/*.obj.gz && mv ../car.obj constant/geometry/body.obj
            foamDictionary system/snappyHexMeshDict.orig -entry geometry/frontWheels -remove
            foamDictionary system/snappyHexMeshDict.orig -entry geometry/rearWheels -remove
            foamDictionary system/snappyHexMeshDict.orig -entry "castellatedMeshControls/refinementSurfaces/.*Wheels" -remove
            foamDictionary system/forceCoeffsIncompressible -entry Aref -set $AREF
            ./Allrun -cores $N_CPUS -mesh $MESH_SIZE
        } >&2
        cat postProcessing/forceCoeffsIncompressible/0/forceCoeffs.dat
    """)

    def __init__(self, concurrency: int):
        """
        Args:
            concurrency: int, cars simulated at once, each in its own container of N_CPUS
        """
        self.concurrency = concurrency

    def prompt(self) -> str:
        """Returns the instruction for the first generation."""
        return self.PROMPT_TEMPLATE

    def prompts_with_rewards(self, responses: list[str], rewards: list[float]) -> str:
        """Returns the instruction with previous solutions and their drag coefficients.

        Args:
            responses: Responses containing scripts, in example order.
            rewards: Corresponding finite rewards, each equal to -C_d.

        Returns:
            One prompt requesting an improved solution.
        """
        in_contexts = []
        for i, (reward, response) in enumerate(zip(rewards, responses), 1):
            in_contexts.append(self.IN_CONTEXT_ITEM_TEMPLATE.format(index=i, metric=-reward, code=self.extract_code(response)))
        return self.IN_CONTEXT_PROMPT_TEMPLATE.format(instruction=self.prompt(), item="\n\n\n".join(in_contexts))

    def extract_code(self, response: str) -> str | None:
        """Returns the last code block stripped of its python tag, or None without code."""
        codes = self.CODE_BLOCK.findall(response)
        return (codes[-1].strip() or None) if codes else None

    def evaluate(self, responses: list[str]) -> list[float]:
        """Evaluates the responses with evaluate_one_async, concurrency at a time, blocking until all are done.

        Ctrl+C cancels every evaluate_one_async, each removing its container and temporary directory, then raises
        KeyboardInterrupt.

        Args:
            responses: list (N) of str

        Returns:
            list (N) of float, each response's -C_d, higher is better; -inf where it failed.
        """
        semaphore = asyncio.Semaphore(self.concurrency)

        async def evaluate_async():
            return await asyncio.gather(*(self.evaluate_one_async(response, semaphore) for response in responses))

        return asyncio.run(evaluate_async())

    async def evaluate_one_async(self, response: str, semaphore: asyncio.Semaphore) -> float:
        """Builds the response's car in Blender, simulates it in a docker container and reads its drag coefficient.

        Every file lives in one temporary directory, removed with the container on any exception or cancellation.

        Args:
            response: str, its last python code block a bpy script building the car in metres
            semaphore: asyncio.Semaphore, held from building the car to the end of its simulation

        Returns:
            Negated C_d averaged over the last AVERAGE_LAST iterations, using the car's frontal reference area;
            -inf without code, for a nonfinite metric or when any step raises, with exceptions printed to stderr.
        """
        script = self.extract_code(response)
        if not script:
            return -math.inf
        name = f"openfoam-{uuid.uuid4().hex}"
        async with semaphore:
            with tempfile.TemporaryDirectory() as home:
                try:
                    with open(f"{home}/car.py", "w") as f:
                        f.write(script)
                    await subprocess_run("blender -b --factory-startup --python-exit-code 1 --python-expr".split() + [self.BLENDER_SCRIPT], cwd=home, stdout=subprocess.DEVNULL)
                    mesh = trimesh.load_mesh(f"{home}/car.stl")
                    mesh.apply_translation([0, -mesh.bounds[:, 1].mean(), -mesh.bounds[0, 2]])  # [ours]
                    mesh.export(f"{home}/car.obj", include_normals=False, include_texture=False)
                    Aref = shapely.union_all(shapely.polygons(mesh.triangles[mesh.area_faces * np.abs(mesh.face_normals[:, 0]) > 1e-12][:, :, 1:])).area
                    lines = (await subprocess_run(
                        f"docker run --rm --name {name} --cpus {self.N_CPUS} --user {os.getuid()}:{os.getgid()} -v {home}:/home/openfoam -e AREF={Aref} -e N_CPUS={self.N_CPUS} -e MESH_SIZE={self.MESH_SIZE} {self.IMAGE} bash -c".split() + [self.OPENFOAM_COMMAND],
                        stdout=subprocess.PIPE,
                    )).decode().splitlines()
                    header = [line for line in lines if line.startswith("#")][-1].lstrip("#").split()
                    Cd = float(np.loadtxt(lines, comments="#")[-self.AVERAGE_LAST:, header.index("Cd")].mean())
                except Exception as exception:
                    print(f"OpenFOAM.evaluate_one_async failed: {exception!r}", file=sys.stderr)
                    return -math.inf
                finally:
                    subprocess.run(f"docker rm -f {name}".split(), capture_output=True)
        return -Cd if math.isfinite(Cd) else -math.inf


class ScsTask:
    """A problem of Gideoni et al. (2026)'s baselines, PROBLEM_NAME, which a subclass sets.

    The prompts are the paper's own: the problem's minimal domain-knowledge instruction, its time limit
    MAX_EXECUTION_TIME as in the paper's Table 1, alone for IID random sampling and the first generation of sequential
    conditioned sampling (SCS), followed by solutions from previous attempts for SCS's later generations. There is no
    seed program.

    A response's code is its last code block. The problem's evaluator runs the code's function in a fresh subprocess
    killed after MAX_EXECUTION_TIME, concurrency at a time, and validates what it returns. The metric is the
    problem's config's conf_metric_name; the reward is reward_sign * metric, so a higher reward is better whether the
    problem maximises or minimises its metric.

    Attributes:
        concurrency: int, codes evaluated at once
        problem: BaseProblem, its max_execution_time MAX_EXECUTION_TIME
        evaluator: BaseEvaluator
        instruction: str, the problem's instruction
        reward_sign: int, -1 when the config's lower_is_better, else 1
    """

    PROBLEM_NAME = None  # The problem's directory under PROBLEMS_DIR, e.g. "short/geometry/pack_circ26".
    MAX_EXECUTION_TIME = 300.0
    CODE_BLOCK = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL)
    IN_CONTEXT_PROMPT_TEMPLATE = inspect.cleandoc("""
        {instruction}
        Here are some example solutions from previous attempts, please improve on them:

        {item}

        Please now provide your new, better solution.
    """)
    IN_CONTEXT_ITEM_TEMPLATE = inspect.cleandoc("""
        # Solution {index}:
        ```python
        {code}
        ```
        This solution, solution {index}, got a {metric_name} of {metric}.
    """)

    def __init__(self, concurrency: int):
        """
        Args:
            concurrency: int, codes evaluated at once
        """
        self.concurrency = concurrency
        config = ProblemConfig(PROBLEMS_DIR / self.PROBLEM_NAME / "config.json")
        module_name = f"baselines.problems.{self.PROBLEM_NAME.replace('/', '.')}"
        self.problem = getattr(importlib.import_module(f"{module_name}.problem"), config["problem_class"])(config)
        self.evaluator = getattr(importlib.import_module(f"{module_name}.evaluator"), config["evaluator_class"])(config)
        self.problem.params["max_execution_time"] = self.MAX_EXECUTION_TIME
        self.instruction = self.problem.generate_instruction()
        self.reward_sign = -1 if config.get("lower_is_better", True) else 1

    def prompt(self) -> str:
        """The instruction alone, the prompt of IID random sampling and of SCS's first generation."""
        return self.instruction

    def prompts_with_rewards(self, responses: list[str], rewards: list[float]) -> str:
        """Every response's code as a solution of one IN_CONTEXT_PROMPT_TEMPLATE, each by IN_CONTEXT_ITEM_TEMPLATE
        with its metric, reward_sign * reward, numbered from 1 in the given order, as the paper's
        ProblemWithArchive.generate_instruction writes them for SCS's later generations.

        Args:
            responses: list (G) of str, G ≥ 1, each holding code
            rewards: list (G) of float, each response's evaluate, finite

        Returns:
            str
        """
        metric_name = self.problem.config["metric_name"]
        in_contexts = []
        for i, (reward, response) in enumerate(zip(rewards, responses), 1):
            in_contexts.append(self.IN_CONTEXT_ITEM_TEMPLATE.format(index=i, metric_name=metric_name, metric=self.reward_sign * reward, code=self.extract_code(response)))
        return self.IN_CONTEXT_PROMPT_TEMPLATE.format(instruction=self.instruction, item="\n\n\n".join(in_contexts))  # The paper leaves two blank lines between solutions.

    def extract_code(self, response: str) -> str | None:
        """The response's last code block, its python tag dropped, stripped; None without one."""
        codes = self.CODE_BLOCK.findall(response)
        return codes[-1].strip() if codes else None

    def evaluate_one(self, code: str | None) -> float:
        """reward_sign * the metric of what the code's function returns; -inf without code, when the evaluator rejects
        it or when the metric is not finite."""
        if not code:
            return -math.inf
        result = self.evaluator.evaluate_algorithm(code, self.problem)
        reward = self.reward_sign * result.metrics[self.problem.config["conf_metric_name"]]
        return reward if result.success and math.isfinite(reward) else -math.inf

    def evaluate(self, responses: list[str]) -> list[float]:
        """Evaluates the responses' codes with evaluate_one, concurrency at a time.

        Args:
            responses: list (N) of str

        Returns:
            list (N) of float, each response's reward, higher is better; -inf without code or when it fails
        """
        with concurrent.futures.ThreadPoolExecutor(self.concurrency) as executor:
            return list(executor.map(self.evaluate_one, map(self.extract_code, responses)))


class CirclePacking(ScsTask):
    """26 circles packed in the unit square for the largest sum of radii, validated up to np.isclose."""

    PROBLEM_NAME = "short/geometry/pack_circ26"


class KissingNumber(ScsTask):
    """The most points C in 11 dimensions, 0 ∉ C, whose smallest pairwise distance is at least their largest norm,
    a lower bound of the kissing number."""

    PROBLEM_NAME = "short/geometry/kissing_11d"


class MaxMinDistanceRatio(ScsTask):
    """16 points in the plane for the smallest squared ratio of their largest to their smallest pairwise distance."""

    PROBLEM_NAME = "short/geometry/maxmin_dist_ratio"


class SumDifferenceSets(ScsTask):
    """A set U of integers for the largest lower bound K = 1 + log(|U-U| / |U+U|) / log(2 max(U) + 1) of the sums and
    differences of sets constant C."""

    PROBLEM_NAME = "short/combinatorics/sumdiff_sets"


class Story:
    """A story pulled toward a higher score of REWARD_MODEL_NAME as the reply to USER_PROMPT.

    The prompts are PROMPT_TEMPLATE without references, for the first generation of sequential conditioned sampling
    (SCS), and IN_CONTEXT_PROMPT_TEMPLATE with references and their rewards for its later generations. There is no seed
    story.

    A response's story is its last code block's. A story's reward is REWARD_MODEL_NAME's score of the
    chat of USER_PROMPT and the story as the assistant's reply, in its chat template without a system turn as its model
    card recommends; REWARD_MODEL_NAME is loaded on the first visible GPU at construction and scores concurrency stories
    per forward pass.
    """

    REWARD_MODEL_NAME = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    USER_PROMPT = "Write a story in one sentence."
    PROMPT_TEMPLATE = inspect.cleandoc("""
        Create a story in one sentence.

        The metric is the reward model score; higher is better.

        Write within a ```text block.
    """)
    IN_CONTEXT_PROMPT_TEMPLATE = inspect.cleandoc("""
        {instruction}
        Here are some example solutions from previous attempts, please improve on them:

        {item}

        Please now provide your new, better solution.
    """)
    IN_CONTEXT_ITEM_TEMPLATE = inspect.cleandoc("""
        # Solution {index}:
        ```text
        {story}
        ```
        This solution, solution {index}, got a reward model score of {metric}.
    """)
    STORY_BLOCK = re.compile(r"```(?:text)?\n?(.*?)```", re.DOTALL)

    def __init__(self, concurrency: int):
        """Loads REWARD_MODEL_NAME on the first visible GPU.

        Args:
            concurrency: int, stories per forward pass of REWARD_MODEL_NAME
        """
        self.concurrency = concurrency
        self.tokenizer = AutoTokenizer.from_pretrained(self.REWARD_MODEL_NAME)
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(self.REWARD_MODEL_NAME, dtype=torch.bfloat16, num_labels=1).cuda().eval()

    def prompt(self) -> str:
        """PROMPT_TEMPLATE, the prompt of SCS's first generation."""
        return self.PROMPT_TEMPLATE

    def prompts_with_rewards(self, responses: list[str], rewards: list[float]) -> str:
        """Every response's story as a reference of one IN_CONTEXT_PROMPT_TEMPLATE, each by IN_CONTEXT_ITEM_TEMPLATE
        with its reward, numbered from 1 in the given order, for SCS's later generations.

        Args:
            responses: list (G) of str, G ≥ 1, each holding a code block
            rewards: list (G) of float, each response's evaluate, finite

        Returns:
            str
        """
        in_contexts = []
        for i, (reward, response) in enumerate(zip(rewards, responses), 1):
            in_contexts.append(self.IN_CONTEXT_ITEM_TEMPLATE.format(index=i, metric=reward, story=self.extract_story(response)))
        return self.IN_CONTEXT_PROMPT_TEMPLATE.format(instruction=self.prompt(), item="\n\n\n".join(in_contexts))

    def extract_story(self, response: str) -> str | None:
        """The response's last code block, its text tag dropped, stripped; None without one or when it is empty."""
        stories = self.STORY_BLOCK.findall(response)
        return (stories[-1].strip() or None) if stories else None

    @torch.no_grad()
    def judge(self, stories: list[str]) -> list[float]:
        """REWARD_MODEL_NAME's score of each story as the reply to USER_PROMPT, concurrency stories per forward pass.

        Args:
            stories: list (B) of str, each nonempty

        Returns:
            list (B) of float, unbounded, higher is better
        """
        chats = []
        for story in stories:
            chat = self.tokenizer.apply_chat_template([{"role": "user", "content": self.USER_PROMPT}, {"role": "assistant", "content": story}], tokenize=False)
            chats.append(chat.removeprefix(self.tokenizer.bos_token or ""))  # Tokenizing adds the BOS again.
        rewards = []
        for b in range(0, len(chats), self.concurrency):
            inputs = self.tokenizer(chats[b:b + self.concurrency], return_tensors="pt", padding=True).to(self.reward_model.device)
            rewards += self.reward_model(**inputs).logits[:, 0].float().tolist()
        torch.cuda.empty_cache()
        return rewards

    def evaluate(self, responses: list[str]) -> list[float]:
        """Evaluates the responses' stories, concurrency per forward pass of REWARD_MODEL_NAME.

        Args:
            responses: list (N) of str, each holding a code block

        Returns:
            list (N) of float, REWARD_MODEL_NAME's score of each response's story as the reply to USER_PROMPT,
            unbounded, higher is better; -inf without a story
        """
        stories = [self.extract_story(response) for response in responses]
        stories_valid = [story for story in stories if story]
        rewards = iter(self.judge(stories_valid) if stories_valid else [])
        return [next(rewards) if story else -math.inf for story in stories]


class Docking:

    import tdc.metadata
    import tdc.utils
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    # A rejected molecule is logged by evaluate's caller, and RDKit's own parse errors raise from its Python logger.
    RDLogger.DisableLog("rdApp.error")

    PDB_ID = None
    PROTEIN = None
    RECEPTOR_DIR = "assets/docking"
    N_CPUS = 4
    EXHAUSTIVENESS = 8
    N_POSES = 10
    SEED = 1
    PROMPT_TEMPLATE = inspect.cleandoc("""
        Propose a molecule that binds {protein} as strongly as possible.

        The metric is the AutoDock Vina docking score in kcal/mol; lower (more negative) is better.

        Write the molecule's SMILES within a ```smiles block.
    """)
    IN_CONTEXT_PROMPT_TEMPLATE = inspect.cleandoc("""
        {instruction}
        Here are some example solutions from previous attempts, please improve on them:

        {item}

        Please now provide your new, better solution.
    """)
    IN_CONTEXT_ITEM_TEMPLATE = inspect.cleandoc("""
        # Solution {index}:
        ```smiles
        {molecule}
        ```
        This solution, solution {index}, got a Vina score of {metric}.
    """)
    MOLECULE_BLOCK = re.compile(r"```(?:smiles)?\n?(.*?)```", re.DOTALL)

    def __init__(self, concurrency: int):
        """Loads TDC's receptor and creates a pool whose workers start on evaluation.

        Args:
            concurrency: Maximum simultaneous molecules, each docking on N_CPUS CPUs.
        """
        self.concurrency = concurrency
        self.box = self.tdc.metadata.docking_target_info[self.PDB_ID]
        self.tdc.utils.receptor_load(self.PDB_ID, path=self.RECEPTOR_DIR)
        self.receptor = os.path.join(self.RECEPTOR_DIR, f"{self.PDB_ID}.pdbqt")
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.concurrency, mp_context=multiprocessing.get_context("spawn"), initializer=self.init_worker,
            initargs=(self.receptor, self.box, self.N_CPUS, self.SEED),
        )

    @classmethod
    def init_worker(cls, receptor: str, box: dict, cpu: int, seed: int) -> None:
        """Initializes this worker's Vina and affinity maps before processing any ligands."""
        from vina import Vina

        cls.seed = seed
        cls.vina = Vina(sf_name="vina", cpu=cpu, seed=seed, verbosity=0)
        cls.vina.set_receptor(rigid_pdbqt_filename=receptor)
        cls.vina.compute_vina_maps(center=list(box["center"]), box_size=list(box["size"]))

    def prompt(self) -> str:
        """PROMPT_TEMPLATE, the prompt of SCS's first generation."""
        return self.PROMPT_TEMPLATE.format(protein=self.PROTEIN)

    def prompts_with_rewards(self, responses: list[str], rewards: list[float]) -> str:
        """Every response's molecule as a reference of one IN_CONTEXT_PROMPT_TEMPLATE, each by
        IN_CONTEXT_ITEM_TEMPLATE with its Vina score, -reward, numbered from 1 in the given order,
        for SCS's later generations.

        Args:
            responses: list (G) of str, G ≥ 1, each holding a code block
            rewards: list (G) of float, each response's evaluate, finite

        Returns:
            str
        """
        in_contexts = []
        for i, (reward, response) in enumerate(zip(rewards, responses), 1):
            in_contexts.append(self.IN_CONTEXT_ITEM_TEMPLATE.format(index=i, metric=-reward, molecule=self.extract_molecule(response)))
        return self.IN_CONTEXT_PROMPT_TEMPLATE.format(instruction=self.prompt(), item="\n\n\n".join(in_contexts))

    def extract_molecule(self, response: str) -> str | None:
        """Returns canonical SMILES from the last code block, or None for missing or invalid molecules.

        A SMILES is ASCII, and RDKit's parser logs a byte-sliced window of its input that splits any multi-byte
        character, raising SystemError out of the logger, so a non-ASCII block is rejected before parsing.
        """
        blocks = self.MOLECULE_BLOCK.findall(response)
        if not blocks:
            return None
        smiles = blocks[-1].strip()
        if not smiles.isascii():
            return None
        try:
            molecule = self.Chem.MolFromSmiles(smiles)
            if molecule is None or molecule.GetNumAtoms() == 0:
                return None
            return self.Chem.MolToSmiles(molecule)
        except ValueError:
            return None

    @classmethod
    def evaluate_one(cls, molecule: str | None) -> float:
        """Returns negated best-pose affinity, or -inf when a molecule cannot be prepared or docked."""
        if not molecule:
            return -math.inf
        try:
            mol = cls.Chem.MolFromSmiles(molecule)
            if mol is None or mol.GetNumAtoms() == 0 or len(cls.Chem.GetMolFrags(mol)) != 1:
                return -math.inf
            conformer = cls.Chem.AddHs(mol)
            if cls.AllChem.EmbedMolecule(conformer, randomSeed=cls.seed) != 0:
                return -math.inf
            if cls.AllChem.MMFFOptimizeMolecule(conformer, maxIters=1000) != 0:
                return -math.inf
            pdbqt, success, error = cls.PDBQTWriterLegacy.write_string(cls.MoleculePreparation().prepare(conformer)[0])
            if not success:
                return -math.inf
        except ValueError:
            return -math.inf
        try:
            cls.vina.set_ligand_from_string(pdbqt)
        except TypeError as error:
            if "PDBQT parsing error" not in str(error):
                raise
            return -math.inf
        cls.vina.dock(exhaustiveness=cls.EXHAUSTIVENESS, n_poses=cls.N_POSES)
        energies = cls.vina.energies(n_poses=1)
        if len(energies) == 0:
            return -math.inf
        reward = -float(energies[0][0])
        return reward if math.isfinite(reward) else -math.inf

    def evaluate(self, responses: list[str]) -> list[float]:
        """Evaluates every molecule without caching, reusing the pool and its affinity maps.

        Args:
            responses: list (N) of str, each holding a SMILES code block.

        Returns:
            list (N) of float, negated best-pose affinities in kcal/mol, higher is better;
            -inf without a molecule or when it is rejected. System errors propagate.
        """
        return list(self.executor.map(self.evaluate_one, map(self.extract_molecule, responses)))


class Docking3PBL(Docking):

    PDB_ID = "3pbl"
    PROTEIN = "the dopamine receptor D3 (DRD3)"


class Docking2RGP(Docking):

    PDB_ID = "2rgp"
    PROTEIN = "the epidermal growth factor receptor (EGFR)"


class Docking3EML(Docking):

    PDB_ID = "3eml"
    PROTEIN = "the adenosine A2A receptor"


TASKS_CLS = {
    "openfoam": OpenFOAM,
    "circle-packing": CirclePacking,
    "kissing-number": KissingNumber,
    "maxmin-distance-ratio": MaxMinDistanceRatio,
    "sum-difference-sets": SumDifferenceSets,
    "story": Story,
    "docking-3pbl": Docking3PBL,
    "docking-2rgp": Docking2RGP,
    "docking-3eml": Docking3EML,
}

def get_reward_fn(task_name: str, concurrency: int):
    """
    Args:
        task_name: str, a TASKS_CLS name, e.g. "circle-packing"
        concurrency: int, the task's evaluations at once
    """
    return TASKS_CLS[task_name](concurrency=concurrency)
