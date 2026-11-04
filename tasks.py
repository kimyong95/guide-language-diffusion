import asyncio
import concurrent.futures
import functools
import glob
import inspect
import math
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
    under SimpleTES's datasets; subclasses only pass that directory and the name of its score. Its generator builds
    SimpleTES's rpucg prompt.

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

        Create an improved program with higher {metric}. Try diverse approaches to solve the problem. Think outside the box.
    """)

    LABELLED_IN_CONTEXT_TEMPLATE = inspect.cleandoc("""
        In-context inspirations (sorted by score, higher is better):

        {item}

        Analyze the inspirations above and create an improved program with higher {metric}. Try diverse approaches to solve the problem. Think outside the box.
    """)

    LABELLED_IN_CONTEXT_ITEM_TEMPLATE = inspect.cleandoc("""
        --- Inspiration {index} ---
        Score: {score}
        Code:
        ```{code_fence_tag}
        {code}
        ```
    """)

    def __init__(self, task: str, metric: str, concurrency: int):
        """
        Args:
            task: str, the task's directory under SimpleTES's datasets, e.g. "circle_packing/circle_packing_26"
            metric: str, the score's name in the prompts' closing instruction, e.g. "sum_radii"
            concurrency: int, codes evaluated at once
        """
        self.metric = metric
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
        self.UNLABELLED_IN_CONTEXT_TEMPLATE = functools.partial(self.UNLABELLED_IN_CONTEXT_TEMPLATE.format, code_fence_tag=self.generator._code_fence_tag, metric=self.metric)
        self.LABELLED_IN_CONTEXT_TEMPLATE = functools.partial(self.LABELLED_IN_CONTEXT_TEMPLATE.format, metric=self.metric)
        self.LABELLED_IN_CONTEXT_ITEM_TEMPLATE = functools.partial(self.LABELLED_IN_CONTEXT_ITEM_TEMPLATE.format, code_fence_tag=self.generator._code_fence_tag)

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

    def prompts_with_scores(self, responses: list[str], scores: list[float]) -> str:
        """Every response's code as an in-context inspiration of one prompt, PROMPT_TEMPLATE's in_context_text
        by LABELLED_IN_CONTEXT_TEMPLATE, each by LABELLED_IN_CONTEXT_ITEM_TEMPLATE with its score, numbered from 1 by
        score, highest first.

        Args:
            responses: list (G) of str, each holding the EVOLVE-BLOCK markers
            scores: list (G) of float

        Returns:
            str
        """
        inspirations = []
        for i, (score, response) in enumerate(sorted(zip(scores, responses), key=lambda pair: pair[0], reverse=True), 1):
            inspirations.append(self.LABELLED_IN_CONTEXT_ITEM_TEMPLATE(index=i, score=score, code=self.extract_code(response)))
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
        super().__init__("circle_packing/circle_packing_26", metric="sum_radii", concurrency=concurrency)


class CirclePacking32(SimpletesTask):

    def __init__(self, concurrency: int):
        super().__init__("circle_packing/circle_packing_32", metric="sum_radii", concurrency=concurrency)


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
    """Binding of a molecule to the protein of PDB entry PDB_ID, scored by Uni-Dock under the vina scoring function
    and negated, so that higher is better.

    A response's molecule is the SMILES of its last code block, canonicalised by RDKit. It is docked into TDC's
    receptor for PDB_ID, over TDC's own search box for it, after being embedded in 3D, MMFF-optimised and prepared by
    Meeko, concurrency molecules at a time; its score is minus the best of N_POSES poses' affinity, the search Uni-Dock's
    SEARCH_MODE preset. Every molecule is docked once and kept in scores, docking being by far the task's cost; each
    evaluate's molecules are split over the visible GPUs, one Uni-Dock batch per GPU, all at once, each sized by
    Uni-Dock to its GPU's free memory.

    Uni-Dock runs in pixi's unidock environment, its conda-forge build needing CUDA 12. TDC, RDKit and Meeko are
    imported in the class body, so they are reached through self.
    """

    import tdc.metadata
    import tdc.utils
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem
    from rdkit.Chem import AllChem

    PDB_ID = None
    PROTEIN = None
    RECEPTOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "docking")
    UNIDOCK = ["pixi", "run", "--manifest-path", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyproject.toml"), "--frozen", "--environment", "unidock", "unidock"]
    SEARCH_MODE = "detail"
    N_POSES = 10
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
        """Downloads TDC's receptor for PDB_ID into RECEPTOR_DIR, keeping a local copy for later runs.

        Args:
            concurrency: int, molecules prepared at once
        """
        self.concurrency = concurrency
        self.scores = {}
        self.box = self.tdc.metadata.docking_target_info[self.PDB_ID]
        self.tdc.utils.receptor_load(self.PDB_ID, path=self.RECEPTOR_DIR)

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

    def extract_molecule(self, response: str) -> str | None:
        """The canonical SMILES of the response's last code block, of any language, its closing fence optional at the
        end; None without one or when RDKit cannot parse it."""
        molecules = self.MOLECULE_BLOCK.findall(response)
        if not molecules:
            return None
        molecule = self.Chem.MolFromSmiles(molecules[-1].strip())
        return self.Chem.MolToSmiles(molecule) if molecule is not None else None

    def prepare(self, molecule: str) -> str | None:
        """The molecule's ligand PDBQT, embedded in 3D, MMFF-optimised and prepared by Meeko; None, the failure printed
        to stderr, when any of them fails.

        Args:
            molecule: str, a canonical SMILES
        """
        try:
            conformer = self.Chem.AddHs(self.Chem.MolFromSmiles(molecule))
            if self.AllChem.EmbedMolecule(conformer) != 0:
                raise ValueError(f"no 3D conformer of {molecule}")
            self.AllChem.MMFFOptimizeMolecule(conformer)
            pdbqt, success, error = self.PDBQTWriterLegacy.write_string(self.MoleculePreparation().prepare(conformer)[0])
            if not success:
                raise ValueError(error)
        except Exception as exception:
            print(f"{type(self).__name__}.prepare failed: {exception!r}", file=sys.stderr)
            return None
        return pdbqt

    def dock(self, pdbqts: list[str]) -> list[float]:
        """Docks the ligands into the receptor, split round-robin over the visible GPUs, one dock_one_async batch per GPU, all
        at once.

        Args:
            pdbqts: list (N) of str, each a ligand's PDBQT

        Returns:
            list (N) of float, minus each ligand's best pose's affinity in kcal/mol, positive for a ligand that binds;
            -inf, printed to stderr, for a ligand Uni-Dock gives no pose

        Raises:
            subprocess.CalledProcessError: a Uni-Dock exits non-zero
        """
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(torch.cuda.device_count())))).split(",")
        with tempfile.TemporaryDirectory() as directory:
            ligands = [os.path.join(directory, f"{i}.pdbqt") for i in range(len(pdbqts))]
            for ligand, pdbqt in zip(ligands, pdbqts):
                with open(ligand, "w") as f:
                    f.write(pdbqt)
            batches = [(device, ligands[g::len(devices)]) for g, device in enumerate(devices)]

            async def dock_async():
                return await asyncio.gather(*(self.dock_one_async(device, batch, directory) for device, batch in batches if batch))

            asyncio.run(dock_async())
            return [self.read_score(ligand.removesuffix(".pdbqt") + "_out.pdbqt") for ligand in ligands]

    async def dock_one_async(self, device: str, ligands: list[str], directory: str) -> None:
        """Runs one Uni-Dock over the ligands on one GPU, over the receptor's search box under the vina scoring
        function, writing each ligand's poses beside it.

        Args:
            device: str, the CUDA device index the GPU is visible as
            ligands: list (M) of str, paths of ligand PDBQTs, nonempty
            directory: str, where the poses and the ligand index go

        Raises:
            subprocess.CalledProcessError: Uni-Dock exits non-zero
        """
        ligand_index = os.path.join(directory, f"ligands-{device}.txt")
        with open(ligand_index, "w") as f:
            f.write("\n".join(ligands))
        await subprocess_run([
            *self.UNIDOCK,
            "--receptor", os.path.join(self.RECEPTOR_DIR, f"{self.PDB_ID}.pdbqt"),
            *(option for axis, center, size in zip("xyz", self.box["center"], self.box["size"]) for option in (f"--center_{axis}", str(center), f"--size_{axis}", str(size))),
            "--scoring", "vina",
            "--search_mode", self.SEARCH_MODE,
            "--num_modes", str(self.N_POSES),
            "--verbosity", "0",
            "--ligand_index", ligand_index,
            "--dir", directory,
        ], env=os.environ | {"CUDA_VISIBLE_DEVICES": device}, stdout=subprocess.DEVNULL)

    def read_score(self, path: str) -> float:
        """Minus the best pose's affinity in Uni-Dock's output, the first REMARK VINA RESULT; -inf, printed to stderr,
        when it has none."""
        try:
            with open(path) as f:
                return -float(next(line for line in f if line.startswith("REMARK VINA RESULT:")).split()[3])
        except (OSError, StopIteration) as exception:
            print(f"{type(self).__name__}.read_score failed: {exception!r} for {path}", file=sys.stderr)
            return -math.inf

    def evaluate(self, responses: list[str]) -> list[float]:
        """Prepares the responses' molecules that scores is missing, concurrency at a time, docks them, then reads every
        score from it.

        Args:
            responses: list (N) of str, each holding a code block

        Returns:
            list (N) of float, minus each response's molecule's docking score in kcal/mol, higher for a molecule bound
            more strongly and about 10 for the strongest binders of these proteins; -inf without a molecule RDKit
            parses or when its preparation or docking fails
        """
        molecules = [self.extract_molecule(response) for response in responses]
        molecules_todo = list({molecule for molecule in molecules if molecule and molecule not in self.scores})
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            pdbqts = dict(zip(molecules_todo, executor.map(self.prepare, molecules_todo)))
        molecules_prepared = [molecule for molecule in molecules_todo if pdbqts[molecule]]
        self.scores.update(dict.fromkeys(molecules_todo, -math.inf))
        self.scores.update(zip(molecules_prepared, self.dock([pdbqts[molecule] for molecule in molecules_prepared])))
        return [self.scores[molecule] if molecule else -math.inf for molecule in molecules]


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
