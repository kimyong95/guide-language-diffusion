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
import numpy as np
import shapely
import simpletes
import trimesh
from simpletes.config import EngineConfig, resolve_eval_python, task_dir_from_evaluator_path
from simpletes.evaluator import EvaluatorWorker
from simpletes.generator import Generator
from simpletes.node import Node, score_from_metrics
from simpletes.utils.code_extract import EvolveBlockContext, extract_code_detailed
from simpletes.utils.task_prep import load_task_requirements


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
        """evaluate_async, blocking until every response is evaluated.

        Ctrl+C cancels every evaluate_one_async, each removing its container and temporary directory, then raises
        KeyboardInterrupt.
        """
        return asyncio.run(self.evaluate_async(responses))

    async def evaluate_async(self, responses: list[str]) -> list[float | None]:
        """Evaluates the responses with evaluate_one_async, N_PARALLEL at a time.

        Args:
            responses: list (N) of str

        Returns:
            list (N) of float, each response's C_d, None where it failed
        """
        semaphore = asyncio.Semaphore(self.N_PARALLEL)
        return await asyncio.gather(*(self.evaluate_one_async(response, semaphore) for response in responses))

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
    under SimpleTES's datasets; subclasses only pass that directory.

    A response's code is the seed with its EVOLVE-BLOCK replaced by the response's, merged by SimpleTES's own
    extract_code_detailed, so only the code between the markers counts. Programs are evaluated as SimpleTES's engine
    does: its EvaluatorWorker runs the task's evaluator in a fresh subprocess per program, EngineConfig's
    eval_concurrency at a time, killed after its eval_timeout.
    """

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

    def prompt(self, codes: list[str], metrics: list[dict], reflections: list[str | None] | None = None, failure_patterns: dict[str, float] | None = None) -> str:
        """SimpleTES's own generation prompt, holding the codes as its inspirations.

        Args:
            codes: list (G) of str, each a whole program's code, e.g. [init_program]
            metrics: list (G) of dict, each code's metrics from evaluate, without "code"
            reflections: list (G) of str or None, each code's reflection; None for none
            failure_patterns: dict of str to float, each error summary's frequency, as the selector's select returns;
                None for none

        Returns:
            str, Generator.build_prompt under EngineConfig's default selector, the codes sorted by score_from_metrics,
            stably
        """
        reflections = reflections or [None] * len(codes)
        return self.generator.build_prompt(
            [Node(id="", code=code, metrics=metrics_i, score=score_from_metrics(metrics_i), reflection=reflection) for code, metrics_i, reflection in zip(codes, metrics, reflections)],
            failure_patterns,
        )

    def evaluate(self, responses: list[str]) -> list[dict]:
        """evaluate_async, blocking until every response is evaluated.

        Ctrl+C cancels every evaluate_one_async, the EvaluatorWorker killing its subprocess, then raises
        KeyboardInterrupt.
        """
        return asyncio.run(self.evaluate_async(responses))

    async def evaluate_async(self, responses: list[str]) -> list[dict]:
        """Evaluates the responses with evaluate_one_async, eval_concurrency at a time.

        Args:
            responses: list (N) of str

        Returns:
            list (N) of dict, each response's metrics
        """
        semaphore = asyncio.Semaphore(self.eval_concurrency)
        return await asyncio.gather(*(self.evaluate_one_async(response, semaphore) for response in responses))

    async def evaluate_one_async(self, response: str, semaphore: asyncio.Semaphore) -> dict:
        """Evaluates the response's code with the EvaluatorWorker.

        Args:
            response: str, holding the EVOLVE-BLOCK markers, preferably within a code block
            semaphore: asyncio.Semaphore, held during the evaluation

        Returns:
            dict, the evaluator's metrics as SimpleTES's engine gets them, {"error": ..., "combined_score": -inf} when
            the evaluator crashes or times out; {"error": reason, "combined_score": -inf} when the response has no
            EVOLVE-BLOCK. score_from_metrics gives the score, higher is better, -inf on any error. Its "code" is the
            response's code, None without an EVOLVE-BLOCK; pop it before the metrics reach a Node, since build_prompt
            and the reflection prompt list every metric
        """
        code, reason = extract_code_detailed(response, self.evolve_context)

        if code is None:
            return {"error": reason, "combined_score": -math.inf, "code": None}

        async with semaphore:
            metrics = (await self.evaluator.evaluate(code)).metrics

        return {**metrics, "code": code}


class CirclePacking26(SimpletesTask):

    def __init__(self):
        super().__init__("circle_packing/circle_packing_26")


class CirclePacking32(SimpletesTask):

    def __init__(self):
        super().__init__("circle_packing/circle_packing_32")


TASKS_CLS = {
    "openfoam": OpenFOAM,
    "circle-packing-26": CirclePacking26,
    "circle-packing-32": CirclePacking32,
}

def get_reward_fn(task_name: str):
    """
    Args:
        task_name: str, a TASKS_CLS name, e.g. "circle-packing-26"
    """
    return TASKS_CLS[task_name]()
