import asyncio
import glob
import inspect
import math
import os
import re
import subprocess
import sys
import tempfile
import uuid
from unittest import mock
import numpy as np
import shapely
import simpletes
import torch
import trimesh
from simpletes.config import EngineConfig, resolve_eval_python, task_dir_from_evaluator_path
from simpletes.evaluator import EvaluatorWorker
from simpletes.generator import Generator
from simpletes.node import Node, score_from_metrics
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

    evaluate_one builds the car by running the response's bpy script in headless Blender and exporting the scene to STL,
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
    N_PARALLEL = 4

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

    def prompt(self) -> str:
        return self.PROMPT_TEMPLATE

    def evaluate(self, responses: list[str]) -> list[float | None]:
        """Evaluates the responses with evaluate_one, N_PARALLEL at a time, blocking until all are done.

        Ctrl+C cancels every evaluate_one, each removing its container and temporary directory, then raises
        KeyboardInterrupt.

        Args:
            responses: list (N) of str

        Returns:
            list (N) of float, each response's C_d, None where it failed
        """
        semaphore = asyncio.Semaphore(self.N_PARALLEL)

        async def evaluate_async():
            return await asyncio.gather(*(self.evaluate_one(response, semaphore) for response in responses))

        return asyncio.run(evaluate_async())

    async def evaluate_one(self, response: str, semaphore: asyncio.Semaphore) -> float | None:
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
                    print(f"OpenFOAM.evaluate_one failed: {exception!r}", file=sys.stderr)
                    return None
                finally:
                    subprocess.run(f"docker rm -f {name}".split(), capture_output=True)
        return Cd


class SimpletesTask:
    """A SimpleTES task, its problem statement, seed program, evaluator and generation prompt taken from its directory
    under SimpleTES's datasets; subclasses only pass that directory.

    A response's code is the seed with its EVOLVE-BLOCK replaced by the response's, merged by SimpleTES's own
    extract_code_detailed, so only the code between the markers counts. Codes are evaluated as SimpleTES's engine
    does: its EvaluatorWorker runs the task's evaluator in a fresh subprocess per code, EngineConfig's eval_concurrency
    at a time, killed after its eval_timeout, its error truncated to DEFAULT_METRICS_ERROR_MAX_CHARS.
    """

    GENERATION_PROMPT_TEMPLATE = inspect.cleandoc("""
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
        {inspirations_text}

        Generate an improved solution with higher score.
    """)

    REFERENCE_TEMPLATE = inspect.cleandoc("""
        {header}:
        ```{code_fence_tag}
        {code}
        ```
    """)

    def __init__(self, task: str):
        """
        Args:
            task: str, the task's directory under SimpleTES's datasets, e.g. "circle_packing/circle_packing_26"
        """
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
            eval_python=resolve_eval_python(evaluator_path=self.evaluator_path, eval_venv=None),
        )
        self.generator = Generator(config, self.instruction, self.evolve_context, load_task_requirements(task_dir_from_evaluator_path(self.evaluator_path))[1])

        # Setup evaluator
        self.evaluator = EvaluatorWorker(
            self.evaluator_path,
            timeout=config.eval_timeout,
            python_executable=config.eval_python,
            target_filename=f"program{os.path.splitext(self.init_program_path)[1]}",
        )
        self.eval_concurrency = config.eval_concurrency

    def prompts(self, responses: list[str]) -> list[str]:
        """Each response's code as the one reference of its own prompt.

        Args:
            responses: list (G) of str, each holding the EVOLVE-BLOCK markers, e.g. [init_program]

        Returns:
            list (G) of str
        """
        return [self.prompt(["Reference"], [response]) for response in responses]

    def prompts_with_experience(self, responses: list[str], scores: list[float]) -> str:
        """Every response's code as a reference of one prompt, numbered from 1 in the given order with its score.

        Args:
            responses: list (G) of str, each holding the EVOLVE-BLOCK markers
            scores: list (G) of float

        Returns:
            str
        """
        return self.prompt([f"Reference {i} (score={score:.6f})" for i, score in enumerate(scores, 1)], responses)

    @mock.patch.multiple("simpletes.generator", GENERATION_PROMPT_TEMPLATE=GENERATION_PROMPT_TEMPLATE, INSPIRATION_TEMPLATE="{code}")
    def prompt(self, headers: list[str], responses: list[str]) -> str:
        """SimpleTES's Generator.build_prompt under GENERATION_PROMPT_TEMPLATE, with no failure patterns, its
        inspirations the responses' codes, each under its header by REFERENCE_TEMPLATE.

        GENERATION_PROMPT_TEMPLATE and INSPIRATION_TEMPLATE are patched into simpletes.generator only during the call,
        so other build_prompt calls keep SimpleTES's.

        Args:
            headers: list (G) of str
            responses: list (G) of str, each holding the EVOLVE-BLOCK markers

        Returns:
            str
        """
        references = "\n\n".join(self.REFERENCE_TEMPLATE.format(header=header, code_fence_tag=self.generator._code_fence_tag, code=self.extract_code(response)) for header, response in zip(headers, responses))
        return self.generator.build_prompt([Node(id="", code=references, score=0.0)])

    def extract_code(self, response: str) -> str | None:
        """The response's code: the seed with its EVOLVE-BLOCK replaced by the response's, stripped; None without an
        EVOLVE-BLOCK."""
        code, _ = extract_code_detailed(response, self.evolve_context)
        return code and code.strip()

    def evaluate(self, responses: list[str]) -> torch.Tensor:
        """Evaluates the responses with evaluate_one, eval_concurrency at a time, blocking until all are done.

        Ctrl+C cancels every evaluate_one, the EvaluatorWorker killing its subprocess, then raises KeyboardInterrupt.

        Args:
            responses: list (N) of str, each holding the EVOLVE-BLOCK markers, preferably within a code block

        Returns:
            (N,) float64, each response's score_from_metrics, higher is better; -inf without an EVOLVE-BLOCK or on any
            evaluator error
        """
        semaphore = asyncio.Semaphore(self.eval_concurrency)

        async def evaluate_async():
            return await asyncio.gather(*(self.evaluate_one(response, semaphore) for response in responses))

        return torch.tensor(asyncio.run(evaluate_async()), dtype=torch.float64)

    async def evaluate_one(self, response: str, semaphore: asyncio.Semaphore) -> float:
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

    def __init__(self):
        super().__init__("circle_packing/circle_packing_26")


class CirclePacking32(SimpletesTask):

    def __init__(self):
        super().__init__("circle_packing/circle_packing_32")


class Story:
    """A story pulled toward the hidden TARGET_STORY, scored by the distance between their hidden states.

    A response's story is its last code block's, of any language. A story's hidden state is MODEL_NAME's last-layer
    hidden state of its last token, the story tokenized without a chat template, computed on CPU.
    """

    MODEL_NAME = "Qwen/Qwen3-0.6B"
    TARGET_STORY = "A playful puppy and a sleepy bear chased each other through the forest, then shared a pot of yummy honey."
    init_program = inspect.cleandoc("""
        ```text
        A small boat drifted slowly across the quiet lake at dawn.
        ```
    """)
    PROMPT_TEMPLATE = inspect.cleandoc("""
        With reference to the following story, create a new story in one sentence.

        {references}

        Write the new story within a text block.
    """)
    REFERENCE_TEMPLATE = inspect.cleandoc("""
        {header}:
        {story}
    """)
    STORY_BLOCK = re.compile(r"```[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME, dtype=torch.float32)
        self.hidden_state_target = self.hidden_state(self.TARGET_STORY)

    def prompts(self, responses: list[str]) -> list[str]:
        """Each response's story as the one reference of its own prompt.

        Args:
            responses: list (G) of str, each holding a code block, e.g. [init_program]

        Returns:
            list (G) of str
        """
        return [self.prompt(["Reference"], [response]) for response in responses]

    def prompts_with_experience(self, responses: list[str], scores: list[float]) -> str:
        """Every response's story as a reference of one prompt, numbered from 1 in the given order with its score.

        Args:
            responses: list (G) of str, each holding a code block
            scores: list (G) of float

        Returns:
            str
        """
        return self.prompt([f"Reference {i} (score={score:.2f})" for i, score in enumerate(scores, 1)], responses)

    def prompt(self, headers: list[str], responses: list[str]) -> str:
        """PROMPT_TEMPLATE of the responses' stories, each under its header by REFERENCE_TEMPLATE.

        Args:
            headers: list (G) of str
            responses: list (G) of str, each holding a code block

        Returns:
            str
        """
        references = "\n\n".join(self.REFERENCE_TEMPLATE.format(header=header, story=self.extract_story(response)) for header, response in zip(headers, responses))
        return self.PROMPT_TEMPLATE.format(references=references)

    def extract_story(self, response: str) -> str | None:
        """The response's last code block, of any language, stripped, its closing fence optional at the end; None
        without one or when it is empty."""
        stories = self.STORY_BLOCK.findall(response)
        return (stories[-1].strip() or None) if stories else None

    @torch.no_grad()
    def hidden_state(self, story: str) -> torch.Tensor:
        """MODEL_NAME's last-layer hidden state of the story's last token.

        Args:
            story: str, nonempty

        Returns:
            (D,) float32
        """
        input_ids = self.tokenizer(story, return_tensors="pt").input_ids
        return self.model(input_ids).last_hidden_state[0, -1]

    def evaluate(self, responses: list[str]) -> torch.Tensor:
        """Evaluates the responses one by one.

        Args:
            responses: list (N) of str, each holding a code block

        Returns:
            (N,) float64, the negative Euclidean distance between each response's story's hidden state and
            TARGET_STORY's, 0 for a perfect match; -inf without a story
        """
        stories = [self.extract_story(response) for response in responses]
        rewards = [-torch.dist(self.hidden_state(story), self.hidden_state_target).item() if story else -math.inf for story in stories]
        return torch.tensor(rewards, dtype=torch.float64)


TASKS_CLS = {
    "openfoam": OpenFOAM,
    "circle-packing-26": CirclePacking26,
    "circle-packing-32": CirclePacking32,
    "story": Story,
}

def get_reward_fn(task_name: str):
    """
    Args:
        task_name: str, a TASKS_CLS name, e.g. "circle-packing-26"
    """
    return TASKS_CLS[task_name]()
