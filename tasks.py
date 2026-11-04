import asyncio
import concurrent.futures
import functools
import glob
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
import simpletes
import torch
import trimesh
from simpletes.config import EngineConfig, resolve_eval_python, task_dir_from_evaluator_path
from simpletes.evaluator import EvaluatorWorker
from simpletes.generator import Generator
from simpletes.node import score_from_metrics
from simpletes.utils.code_extract import EvolveBlockContext, extract_code_detailed
from simpletes.utils.task_prep import load_task_requirements
from simpletes.utils.text import DEFAULT_METRICS_ERROR_MAX_CHARS, truncate_error_in_metrics
from transformers import AutoModel, AutoTokenizer


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

    SYSTEM_PROMPT = None
    PROMPT_TEMPLATE = ""

    init_program = inspect.cleandoc("""
        import bpy

        bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    """)

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
        return self.PROMPT_TEMPLATE

    def evaluate(self, responses: list[str]) -> list[float | None]:
        """Evaluates the responses with evaluate_one_async, concurrency at a time, blocking until all are done.

        Ctrl+C cancels every evaluate_one_async, each removing its container and temporary directory, then raises
        KeyboardInterrupt.

        Args:
            responses: list (N) of str

        Returns:
            list (N) of float, each response's C_d, None where it failed
        """
        semaphore = asyncio.Semaphore(self.concurrency)

        async def evaluate_async():
            return await asyncio.gather(*(self.evaluate_one_async(response, semaphore) for response in responses))

        return asyncio.run(evaluate_async())

    async def evaluate_one_async(self, response: str, semaphore: asyncio.Semaphore) -> float | None:
        """Builds the response's car in Blender, simulates it in a docker container and reads its drag coefficient.

        Every file lives in one temporary directory, removed with the container on any exception or cancellation.

        Args:
            response: str, its last python code block a bpy script building the car in metres
            semaphore: asyncio.Semaphore, held from building the car to the end of its simulation

        Returns:
            float, C_d averaged over the last AVERAGE_LAST iterations, its reference area the car's frontal area; None,
            the failure printed to stderr, when the response has no python code block or any step raises
        """
        name = f"openfoam-{uuid.uuid4().hex}"
        async with semaphore:
            with tempfile.TemporaryDirectory() as home:
                try:
                    script = re.findall(r"```python\n(.*?)```", response, re.DOTALL)[-1]
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
                    return None
                finally:
                    subprocess.run(f"docker rm -f {name}".split(), capture_output=True)
        return Cd


class SimpletesTask:
    """A SimpleTES task, its problem statement, seed program, evaluator and generation prompt taken from its directory
    under SimpleTES's datasets; subclasses only pass that directory, the name of its score and which direction of it is
    better. Its generator builds SimpleTES's rpucg prompt.

    A response's code is the seed with its EVOLVE-BLOCK replaced by the response's, merged by SimpleTES's own
    extract_code_detailed, so only the code between the markers counts. Codes are evaluated as SimpleTES's engine
    does: its EvaluatorWorker runs the task's evaluator in a fresh subprocess per code, concurrency at a time, killed
    after EngineConfig's eval_timeout, its error truncated to DEFAULT_METRICS_ERROR_MAX_CHARS.
    """

    PROMPT_TEMPLATE = inspect.cleandoc("""
        Task: {instruction}

        Generation instruction (must follow exactly):
        1) Only the code between `{start_marker_line}` and `{end_marker_line}` is extracted.
        2) The final program is reconstructed as EXACT_PREFIX + evolved_block + EXACT_SUFFIX.
        3) Keep marker lines exactly as written.
        4) Return one {language_name} code block that includes both EVOLVE-BLOCK markers.

        EXACT_PREFIX (kept unchanged):
        ```{code_fence_tag}
        {prefix}
        ```

        EXACT_SUFFIX (kept unchanged):
        ```{code_fence_tag}
        {suffix}
        ```
        {available_packages_text}
        {in_context_text}
    """)

    UNLABELLED_IN_CONTEXT_TEMPLATE = inspect.cleandoc("""
        In-context inspiration:
        ```{code_fence_tag}
        {code}
        ```

        Create an improved program with {better} {metric}. Try diverse approaches to solve the problem. Think outside the box.
    """)

    LABELLED_IN_CONTEXT_TEMPLATE = inspect.cleandoc("""
        In-context inspirations (sorted by {metric}, {better} is better):

        {item}

        Analyze the inspirations above and create an improved program with {better} {metric}. Try diverse approaches to solve the problem. Think outside the box.
    """)

    LABELLED_IN_CONTEXT_ITEM_TEMPLATE = inspect.cleandoc("""
        --- Inspiration {index} ---
        {metric} : {score}
        Code:
        ```{code_fence_tag}
        {code}
        ```
    """)

    def __init__(self, task: str, metric: str, better: str, concurrency: int):
        """
        Args:
            task: str, the task's directory under SimpleTES's datasets, e.g. "circle_packing/circle_packing_26"
            metric: str, the score's name in the prompts, e.g. "sum_radii"
            better: str, "higher" or "lower", the direction of metric that the prompts ask for
            concurrency: int, codes evaluated at once
        """
        self.metric = metric
        self.better = better
        self.concurrency = concurrency

        # Setup paths
        task_dir = os.path.join(os.path.dirname(os.path.dirname(simpletes.__file__)), "datasets", task)
        self.instruction_path = os.path.join(task_dir, f"{os.path.basename(task)}.txt")
        self.evaluator_path = os.path.join(task_dir, "evaluator.py")
        self.init_program_path = glob.glob(os.path.join(task_dir, "init_program.*"))[0]

        # Setup prompt
        with open(self.instruction_path) as f:
            self.instruction = f.read()
        with open(self.init_program_path) as f:
            self.init_program = f.read()
        self.evolve_context = EvolveBlockContext.from_program(self.init_program)
        config = EngineConfig(
            init_program=self.init_program_path,
            evaluator_path=self.evaluator_path,
            instruction_path=self.instruction_path,
            gen_concurrency=0,
            selector="rpucg",
            eval_python=resolve_eval_python(evaluator_path=self.evaluator_path, eval_venv=None),
        )
        self.generator = Generator(config, self.instruction, self.evolve_context, load_task_requirements(task_dir_from_evaluator_path(self.evaluator_path))[1])
        self.preformat_prompt_templates()

        # Setup evaluator
        self.evaluator = EvaluatorWorker(
            self.evaluator_path,
            timeout=config.eval_timeout,
            python_executable=config.eval_python,
            target_filename=f"program{os.path.splitext(self.init_program_path)[1]}",
        )

    def preformat_prompt_templates(self):
        """Replaces each prompt template, on the instance, by a functools.partial of its str.format pre-filling the
        fields that the task fixes, taking the rest: PROMPT_TEMPLATE in_context_text,
        UNLABELLED_IN_CONTEXT_TEMPLATE code, LABELLED_IN_CONTEXT_TEMPLATE item, LABELLED_IN_CONTEXT_ITEM_TEMPLATE index,
        score and code."""
        packages = self.generator._available_packages
        self.PROMPT_TEMPLATE = functools.partial(
            self.PROMPT_TEMPLATE.format,
            instruction=self.instruction,
            start_marker_line=self.evolve_context.start_marker_line,
            end_marker_line=self.evolve_context.end_marker_line,
            language_name=self.generator._language_name,
            code_fence_tag=self.generator._code_fence_tag,
            prefix=self.evolve_context.prefix.rstrip("\n"),
            suffix=self.evolve_context.suffix.rstrip("\n"),
            available_packages_text="\nAvailable packages from requirements.txt (these are the task-local dependencies you can use):\n" + "".join(f"- {package}\n" for package in packages) if packages else "",
        )
        self.UNLABELLED_IN_CONTEXT_TEMPLATE = functools.partial(self.UNLABELLED_IN_CONTEXT_TEMPLATE.format, code_fence_tag=self.generator._code_fence_tag, metric=self.metric, better=self.better)
        self.LABELLED_IN_CONTEXT_TEMPLATE = functools.partial(self.LABELLED_IN_CONTEXT_TEMPLATE.format, metric=self.metric, better=self.better)
        self.LABELLED_IN_CONTEXT_ITEM_TEMPLATE = functools.partial(self.LABELLED_IN_CONTEXT_ITEM_TEMPLATE.format, code_fence_tag=self.generator._code_fence_tag, metric=self.metric)

    def prompts(self, responses: list[str]) -> list[str]:
        """Each response's code as the one in-context inspiration of its own prompt, PROMPT_TEMPLATE's
        in_context_text by UNLABELLED_IN_CONTEXT_TEMPLATE.

        Args:
            responses: list (G) of str, each holding the EVOLVE-BLOCK markers, e.g. [init_program]

        Returns:
            list (G) of str
        """
        prompts = []
        for response in responses:
            in_context_text = self.UNLABELLED_IN_CONTEXT_TEMPLATE(code=self.extract_code(response))
            prompts.append(self.PROMPT_TEMPLATE(in_context_text=in_context_text))
        return prompts

    def score_to_metric(self, score: float) -> float:
        """The metric of a score, the score itself; a subclass whose score is not its metric overrides this.

        Args:
            score: float, finite, from evaluate

        Returns:
            float, shown in the prompts as the metric's value
        """
        return score

    def prompts_with_scores(self, responses: list[str], scores: list[float]) -> str:
        """Every response's code as an in-context inspiration of one prompt, PROMPT_TEMPLATE's in_context_text
        by LABELLED_IN_CONTEXT_TEMPLATE, each by LABELLED_IN_CONTEXT_ITEM_TEMPLATE with its score_to_metric, numbered
        from 1 by score, highest first.

        Args:
            responses: list (G) of str, each holding the EVOLVE-BLOCK markers
            scores: list (G) of float

        Returns:
            str
        """
        inspirations = []
        for i, (score, response) in enumerate(sorted(zip(scores, responses), key=lambda pair: pair[0], reverse=True), 1):
            inspirations.append(self.LABELLED_IN_CONTEXT_ITEM_TEMPLATE(index=i, score=self.score_to_metric(score), code=self.extract_code(response)))
        in_context_text = self.LABELLED_IN_CONTEXT_TEMPLATE(item="\n\n".join(inspirations))
        return self.PROMPT_TEMPLATE(in_context_text=in_context_text)

    def extract_code(self, response: str) -> str | None:
        """The response's code: the seed with its EVOLVE-BLOCK replaced by the response's, stripped; None without an
        EVOLVE-BLOCK."""
        code, _ = extract_code_detailed(response, self.evolve_context)
        return code and code.strip()

    def evaluate(self, responses: list[str]) -> list:
        """Evaluates the responses with evaluate_one_async, concurrency at a time, blocking until all are done.

        Ctrl+C cancels every evaluate_one_async, the EvaluatorWorker killing its subprocess, then raises KeyboardInterrupt.

        Args:
            responses: list (N) of str, each holding the EVOLVE-BLOCK markers, preferably within a code block

        Returns:
            list (N), each response's evaluate_one_async
        """
        semaphore = asyncio.Semaphore(self.concurrency)

        async def evaluate_async():
            return await asyncio.gather(*(self.evaluate_one_async(response, semaphore) for response in responses))

        return asyncio.run(evaluate_async())

    async def evaluate_one_async(self, response: str, semaphore: asyncio.Semaphore) -> float:
        """Extracts the response's code and evaluates it as SimpleTES's engine's _evaluate_code does.

        Args:
            response: str
            semaphore: asyncio.Semaphore, held during the evaluation

        Returns:
            float, the metrics' score_from_metrics: their combined_score, -inf on any error; the metrics are
            {"error": ..., "combined_score": -inf} without an EVOLVE-BLOCK or when the evaluator crashes, times out or
            raises
        """
        code, reason = extract_code_detailed(response, self.evolve_context)

        if code is None:
            metrics = {"error": reason, "combined_score": -math.inf}
        else:
            async with semaphore:
                try:
                    metrics = (await self.evaluator.evaluate(code)).metrics
                except Exception as exception:
                    metrics = {"error": str(exception), "combined_score": -math.inf}
            metrics = truncate_error_in_metrics(metrics, max_chars=DEFAULT_METRICS_ERROR_MAX_CHARS)

        return score_from_metrics(metrics)


class CirclePacking26(SimpletesTask):

    def __init__(self, concurrency: int):
        super().__init__("circle_packing/circle_packing_26", metric="sum_radii", better="higher", concurrency=concurrency)


class CirclePacking32(SimpletesTask):

    def __init__(self, concurrency: int):
        super().__init__("circle_packing/circle_packing_32", metric="sum_radii", better="higher", concurrency=concurrency)


class AutocorrelationSecond(SimpletesTask):

    def __init__(self, concurrency: int):
        super().__init__("autocorrelation/autocorrelation_second", metric="c2", better="higher", concurrency=concurrency)


class Story:
    """A story pulled toward the hidden TARGET_STORY, scored by the cosine similarity of their embeddings.

    A response's story is its last code block's, of any language. A story's embedding is MODEL_NAME's last-layer hidden
    state of its last token, the <|endoftext|> its tokenizer appends, L2-normalised; the story is tokenized without a
    chat template or instruction, and embedded in batches of concurrency stories on the first visible GPU, where
    MODEL_NAME is loaded at construction.
    """

    MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    TARGET_STORY = "A playful puppy and a sleepy bear chased each other through the forest, then shared a pot of yummy honey."
    init_program = inspect.cleandoc("""
        ```text
        A small boat drifted slowly across the quiet lake at dawn.
        ```
    """)
    PLAIN_PROMPT_TEMPLATE = inspect.cleandoc("""
        With reference to the following story, create a new story in one sentence.

        Reference:
        ```text
        {story}
        ```

        Write the new story within a ```text block.
    """)
    LABELLED_PROMPT_TEMPLATE = inspect.cleandoc("""
        With reference to the following story, create a new story in one sentence.

        {item}

        The score is the similarity to a hidden target story. Analyze the references with their scores, then write a new story (different from the references) to improve the score, within a ```text block.
    """)
    LABELLED_IN_CONTEXT_ITEM_TEMPLATE = inspect.cleandoc("""
        Reference {index} (score={score:.3f}):
        ```text
        {story}
        ```
    """)
    STORY_BLOCK = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)

    def __init__(self, concurrency: int):
        """
        Args:
            concurrency: int, stories per forward pass
        """
        self.concurrency = concurrency
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME, dtype=torch.float32).cuda()
        self.embedding_target = self.embeddings([self.TARGET_STORY])[0]

    def prompts(self, responses: list[str]) -> list[str]:
        """Each response's story as the one reference of its own PLAIN_PROMPT_TEMPLATE.

        Args:
            responses: list (G) of str, each holding a code block, e.g. [init_program]

        Returns:
            list (G) of str
        """
        return [self.PLAIN_PROMPT_TEMPLATE.format(story=self.extract_story(response)) for response in responses]

    def prompts_with_scores(self, responses: list[str], scores: list[float]) -> str:
        """Every response's story as a reference of one LABELLED_PROMPT_TEMPLATE, each by
        LABELLED_IN_CONTEXT_ITEM_TEMPLATE with its score, numbered from 1 in the given order.

        Args:
            responses: list (G) of str, each holding a code block
            scores: list (G) of float

        Returns:
            str
        """
        in_contexts = []
        for i, (score, response) in enumerate(zip(scores, responses), 1):
            in_contexts.append(self.LABELLED_IN_CONTEXT_ITEM_TEMPLATE.format(index=i, score=score, story=self.extract_story(response)))
        return self.LABELLED_PROMPT_TEMPLATE.format(item="\n\n".join(in_contexts))

    def extract_story(self, response: str) -> str | None:
        """The response's last code block, of any language, stripped, its closing fence optional at the end; None
        without one or when it is empty."""
        stories = self.STORY_BLOCK.findall(response)
        return (stories[-1].strip() or None) if stories else None

    @torch.no_grad()
    def embeddings(self, stories: list[str]) -> torch.Tensor:
        """MODEL_NAME's L2-normalised last-layer hidden state of each story's last token, the stories right-padded into
        one batch.

        Args:
            stories: list (B) of str, each nonempty

        Returns:
            (B, D) float32, unit norm
        """
        inputs = self.tokenizer(stories, return_tensors="pt", padding=True, padding_side="right").to(self.model.device)
        last_hidden_state = self.model(**inputs).last_hidden_state
        return torch.nn.functional.normalize(last_hidden_state[torch.arange(len(stories)), inputs.attention_mask.sum(1) - 1], dim=-1)

    def evaluate(self, responses: list[str]) -> list[float]:
        """Evaluates the responses' stories, concurrency per forward pass.

        Args:
            responses: list (N) of str, each holding a code block

        Returns:
            list (N) of float, the cosine similarity between each response's story's embedding and TARGET_STORY's, in
            [-1, 1] and 1 for a perfect match; -inf without a story
        """
        stories = [self.extract_story(response) for response in responses]
        stories_valid = [story for story in stories if story]
        similarities = []
        for b in range(0, len(stories_valid), self.concurrency):
            embeddings = self.embeddings(stories_valid[b:b + self.concurrency])
            similarities += (embeddings @ self.embedding_target).tolist()
        similarities = iter(similarities)
        return [next(similarities) if story else -math.inf for story in stories]


class Docking:
    """Scores molecules by negated AutoDock Vina affinity against TDC's receptor and search box.

    Each response's last code block supplies a SMILES, canonicalized before caching. Workers embed it in 3D,
    converge MMFF optimization within 1,000 iterations, prepare it with Meeko, and return minus the best docked
    pose's energy in kcal/mol.
    RDKit and Vina use the same fixed positive SEED. Reproducibility also requires fixed dependency versions.

    A persistent pool prepares and docks up to concurrency molecules at once, each on N_CPUS CPUs. Each worker
    builds its affinity maps once. Errors propagate; a failed batch leaves the score cache unchanged. Call close()
    when discarding the task to release its workers.
    """

    import tdc.metadata
    import tdc.utils
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem
    from rdkit.Chem import AllChem

    PDB_ID = None
    PROTEIN = None
    RECEPTOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "docking")
    N_CPUS = 4
    EXHAUSTIVENESS = 8
    N_POSES = 10
    SEED = 1
    init_program = inspect.cleandoc("""
        ```smiles
        CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1
        ```
    """)
    PLAIN_PROMPT_TEMPLATE = inspect.cleandoc("""
        With reference to the following molecule, propose a new molecule that binds {protein} more strongly.

        Reference:
        ```smiles
        {molecule}
        ```

        Write the new molecule's SMILES within a ```smiles block.
    """)
    LABELLED_PROMPT_TEMPLATE = inspect.cleandoc("""
        With reference to the following molecules, propose a new molecule that binds {protein} more strongly.

        {item}

        The score is minus the AutoDock Vina docking score against {protein}, in kcal/mol, so a higher score is a more strongly bound molecule. Analyze the references with their scores, then write a new molecule (different from the references), either by crossover and mutation of the references or from your own chemical knowledge, within a ```smiles block.
    """)
    LABELLED_IN_CONTEXT_ITEM_TEMPLATE = inspect.cleandoc("""
        Reference {index} (score={score:.2f}):
        ```smiles
        {molecule}
        ```
    """)
    MOLECULE_BLOCK = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)

    def __init__(self, concurrency: int):
        """Loads TDC's receptor and creates a pool that starts workers on the first evaluation.

        Args:
            concurrency: Maximum simultaneous molecules, each docking on N_CPUS CPUs.
        """
        self.concurrency = concurrency
        self.scores = {}
        box = self.tdc.metadata.docking_target_info[self.PDB_ID]
        self.tdc.utils.receptor_load(self.PDB_ID, path=self.RECEPTOR_DIR)
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=concurrency, mp_context=multiprocessing.get_context("spawn"), initializer=self.init_worker,
            initargs=(os.path.join(self.RECEPTOR_DIR, f"{self.PDB_ID}.pdbqt"), box, self.N_CPUS, self.SEED),
        )

    def close(self) -> None:
        """Waits for pending docking jobs and releases the worker processes."""
        self.executor.shutdown()

    @classmethod
    def init_worker(cls, receptor: str, box: dict, cpu: int, seed: int) -> None:
        """Initializes this worker's Vina and affinity maps before processing any ligands."""
        from vina import Vina

        cls.seed = seed
        cls.vina = Vina(sf_name="vina", cpu=cpu, seed=seed, verbosity=0)
        cls.vina.set_receptor(rigid_pdbqt_filename=receptor)
        cls.vina.compute_vina_maps(center=list(box["center"]), box_size=list(box["size"]))

    def prompts(self, responses: list[str]) -> list[str]:
        """Each response's molecule as the one reference of its own PLAIN_PROMPT_TEMPLATE.

        Args:
            responses: list (G) of str, each holding a code block, e.g. [init_program]

        Returns:
            list (G) of str
        """
        return [self.PLAIN_PROMPT_TEMPLATE.format(protein=self.PROTEIN, molecule=self.extract_molecule(response)) for response in responses]

    def prompts_with_scores(self, responses: list[str], scores: list[float]) -> str:
        """Every response's molecule as a reference of one LABELLED_PROMPT_TEMPLATE, each by
        LABELLED_IN_CONTEXT_ITEM_TEMPLATE with its score, numbered from 1 in the given order.

        Args:
            responses: list (G) of str, each holding a code block
            scores: list (G) of float

        Returns:
            str
        """
        in_contexts = []
        for i, (score, response) in enumerate(zip(scores, responses), 1):
            in_contexts.append(self.LABELLED_IN_CONTEXT_ITEM_TEMPLATE.format(index=i, score=score, molecule=self.extract_molecule(response)))
        return self.LABELLED_PROMPT_TEMPLATE.format(protein=self.PROTEIN, item="\n\n".join(in_contexts))

    def extract_molecule(self, response: str) -> str:
        """Returns canonical SMILES from the last code block, raising for missing or invalid molecules."""
        smiles = self.MOLECULE_BLOCK.findall(response)[-1].strip()
        molecule = self.Chem.MolFromSmiles(smiles)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return self.Chem.MolToSmiles(molecule)

    @classmethod
    def dock_one(cls, molecule: str, exhaustiveness: int, n_poses: int) -> float:
        """Prepares a canonical SMILES and returns its negated best-pose affinity in an initialized worker."""
        conformer = cls.Chem.AddHs(cls.Chem.MolFromSmiles(molecule))
        if cls.AllChem.EmbedMolecule(conformer, randomSeed=cls.seed) != 0:
            raise ValueError(f"No 3D conformer for {molecule}")
        status = cls.AllChem.MMFFOptimizeMolecule(conformer, maxIters=1000)
        if status != 0:
            raise ValueError(f"MMFF optimization failed for {molecule}: status {status}")
        pdbqt, success, error = cls.PDBQTWriterLegacy.write_string(cls.MoleculePreparation().prepare(conformer)[0])
        if not success:
            raise ValueError(error)
        cls.vina.set_ligand_from_string(pdbqt)
        cls.vina.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
        score = -float(cls.vina.energies(n_poses=1)[0][0])
        if not math.isfinite(score):
            raise ValueError(f"Non-finite docking score for {molecule}: {score}")
        return score

    def evaluate(self, responses: list[str]) -> list[float]:
        """Scores uncached molecules in parallel and returns rewards in response order.

        Args:
            responses: Responses containing SMILES code blocks.

        Returns:
            Negated best-pose affinities in kcal/mol. Errors propagate without caching the failed batch.
        """
        molecules = [self.extract_molecule(response) for response in responses]
        pending = list(dict.fromkeys(molecule for molecule in molecules if molecule not in self.scores))
        if pending:
            dock = functools.partial(self.dock_one, exhaustiveness=self.EXHAUSTIVENESS, n_poses=self.N_POSES)
            scores = list(self.executor.map(dock, pending))
            self.scores.update(zip(pending, scores))
        return [self.scores[molecule] for molecule in molecules]


class DockingDRD3(Docking):

    PDB_ID = "3pbl"
    PROTEIN = "the dopamine receptor D3 (DRD3)"


class DockingEGFR(Docking):

    PDB_ID = "2rgp"
    PROTEIN = "the epidermal growth factor receptor (EGFR)"


class DockingAdenosineA2A(Docking):

    PDB_ID = "3eml"
    PROTEIN = "the adenosine A2A receptor"


TASKS_CLS = {
    "openfoam": OpenFOAM,
    "circle-packing-26": CirclePacking26,
    "circle-packing-32": CirclePacking32,
    "autocorrelation-second": AutocorrelationSecond,
    "story": Story,
    "docking-drd3": DockingDRD3,
    "docking-egfr": DockingEGFR,
    "docking-adenosine-a2a": DockingAdenosineA2A,
}

def get_reward_fn(task_name: str, concurrency: int):
    """
    Args:
        task_name: str, a TASKS_CLS name, e.g. "circle-packing-26"
        concurrency: int, the task's evaluations at once
    """
    return TASKS_CLS[task_name](concurrency=concurrency)
