import ast
import asyncio
import concurrent.futures
import glob
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
import numpy as np
import shapely
import torch
import trimesh
from arc_agi_benchmarking.prompts.prompt_manager import convert_task_pairs_to_prompt
from arc_agi_benchmarking.utils.parsing import parse_and_validate_json
from arc_agi_benchmarking.utils.task_utils import get_test_pairs_from_task, get_train_pairs_from_task
from baselines.problems.problem_utils import PROBLEMS_DIR, ProblemConfig
from secret_tasks import Docking2RGP, Docking3EML, Docking3PBL
from transformers import AutoModelForSequenceClassification, AutoTokenizer


async def subprocess_run(args, input=None, **kwargs):
    """subprocess.run(args, input=input, check=True, **kwargs) on asyncio, killing the process if cancelled.

    Returns:
        bytes, the process's stdout if kwargs pipe it, else None

    Raises:
        subprocess.CalledProcessError: the process exits non-zero
    """
    process = await asyncio.create_subprocess_exec(*args, **kwargs)
    try:
        stdout, _ = await process.communicate(input)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, args)
    return stdout


class Aerodynamic:
    """Drag coefficient of a car built by a response's bpy script, simulated by a subclass's evaluate_one_async.

    build_car runs the script in headless Blender and exports the scene to STL, BLENDER_SCRIPT as execute-blender.py
    does before rendering. The car is in metres, its length along x and its front facing -x. [ours] It is moved down
    onto the ground and sideways to be symmetric about y = 0, and its wheels do not rotate, being part of the car's one
    surface. The reward is -C_d over the car's frontal reference area.
    """

    PROMPT_TEMPLATE = inspect.cleandoc("""
        Design a car with the lowest possible drag coefficient, evaluated with {simulation}.

        Write a Blender Python script using bpy to build the car, including its wheels, in metres.
        Place its length along x, its front facing -x, and its front axle at x = 0.
        The scene is exported to STL, centered along y and placed on the ground before simulation.
        The metric is the drag coefficient C_d; lower is better.

        Write the script within a ```python block.
    """)
    SIMULATION = None  # Named in PROMPT_TEMPLATE, e.g. "OpenFOAM".
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

    BLENDER_SCRIPT = inspect.cleandoc("""
        import runpy
        import bpy
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        runpy.run_path("car.py", run_name="__main__")
        bpy.ops.wm.stl_export(filepath="car.stl")
    """)

    def __init__(self, concurrency: int):
        """
        Args:
            concurrency: int, cars simulated at once
        """
        self.concurrency = concurrency

    def prompt(self) -> str:
        """Returns the instruction for the first generation."""
        return self.PROMPT_TEMPLATE.format(simulation=self.SIMULATION)

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

        Ctrl+C cancels every evaluate_one_async, each cleaning up its processes and temporary directory, then raises
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

    async def build_car(self, script: str, home: str) -> tuple[trimesh.Trimesh, float]:
        """Builds the script's car in headless Blender, then moves it onto the ground and symmetric about y = 0.

        Args:
            script: str, a bpy script building the car in metres
            home: str, a directory for the script and its exported car.stl

        Returns:
            The car's mesh in metres and its frontal reference area in m², the area of its projection on the y-z plane.

        Raises:
            subprocess.CalledProcessError: Blender fails
        """
        with open(f"{home}/car.py", "w") as f:
            f.write(script)
        await subprocess_run("blender -b --factory-startup --python-exit-code 1 --python-expr".split() + [self.BLENDER_SCRIPT], cwd=home, stdout=subprocess.DEVNULL)
        mesh = trimesh.load_mesh(f"{home}/car.stl")
        mesh.apply_translation([0, -mesh.bounds[:, 1].mean(), -mesh.bounds[0, 2]])  # [ours]
        Aref = shapely.union_all(shapely.polygons(mesh.triangles[mesh.area_faces * np.abs(mesh.face_normals[:, 0]) > 1e-12][:, :, 1:])).area
        return mesh, Aref

    async def evaluate_one_async(self, response: str, semaphore: asyncio.Semaphore) -> float:
        """Builds the response's car with build_car and simulates it.

        Args:
            response: str, its last python code block a bpy script building the car in metres
            semaphore: asyncio.Semaphore, held from building the car to the end of its simulation

        Returns:
            -C_d; -inf without code, for a nonfinite C_d or when any step raises.
        """
        raise NotImplementedError


class OpenFOAM(Aerodynamic):
    """Drag coefficient of a car, simulated with OpenFOAM v14 (Foundation) following its official drivaerFastback case.

    evaluate_one_async runs IMAGE, microfluidica's docker image of the Foundation's own packages, on a copy of the
    image's tutorials/incompressibleFluid/drivaerFastback. OPENFOAM_COMMAND changes only the entries tagged by their
    source: [DrivAerNet++] following Elrefaie et al. (NeurIPS 2024), [ours] otherwise. Everything else is the official
    case run by its own Allrun, MESH_SIZE being one of its mesh sizes: the domain, snappyHexMesh refinement, k-ω SST
    model, numerics, moving ground, 16 m/s inlet and 1000 iterations.

    The car keeps its scale and x position, so the prompt's frame is the official one: front axle at x = 0.
    """

    SIMULATION = "OpenFOAM"
    IMAGE = "microfluidica/openfoam:14"
    AVERAGE_LAST = 100
    N_CPUS = 6
    MESH_SIZE = "S"  # S, M, L, XL
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
        super().__init__(concurrency)

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
                    mesh, Aref = await self.build_car(script, home)
                    mesh.export(f"{home}/car.obj", include_normals=False, include_texture=False)
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


class FluidX3D(Aerodynamic):
    """Drag coefficient of a car, simulated on a GPU with FluidX3D v3.9's lattice Boltzmann method following its Ahmed
    body setup.

    SETUP is the Ahmed body's main_setup with its extensions, DEFINES: FP16C,
    D3Q19, SRT with Smagorinsky-Lilly LES, a box of BOX_SCALE car lengths by BOX_SCALE car widths by (BOX_SCALE - 1) / 2
    car widths above the car's height, sized to MEMORY megabytes, the car's front half of (BOX_SCALE / 2 car lengths
    less one car width) from the inlet, one cell above a stationary no-slip floor, equilibrium boundaries of LBM_U on
    the other faces, the whole fluid starting at LBM_U, air of SI_NU and SI_RHO, and the drag from object_force over
    ½ SI_RHO SI_U² Aref. [ours] The flow is along x, the car's length, instead of y; the inlet speed SI_U is
    OpenFOAM's 16 m/s; each of the car's parts is voxelized on its own, as the setup of the Mercedes F1 car does, so
    overlapping parts are solid; the run lasts FLOW_PASSES passes of the flow through the box, and the drag is averaged
    every FORCE_INTERVAL steps over the last pass; the reference area is build_car's.

    FluidX3D is built once per SETUP and DEFINES into FLUIDX3D_DIR/build, linked to FluidX3D's bundled OpenCL loader.
    Each car is simulated in its own process on the GPU FluidX3D picks, the fastest visible one, without the caller's
    LD_LIBRARY_PATH and with NVIDIA's compute cache disabled.
    """

    SIMULATION = "a lattice Boltzmann CFD simulation"
    FLUIDX3D_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "packages/FluidX3D")
    DEFINES = {
        r"^#define FP16S\b": "//#define FP16S",
        r"^//#define FP16C\b": "#define FP16C",
        r"^#define BENCHMARK\b": "//#define BENCHMARK",
        r"^//#define FORCE_FIELD\b": "#define FORCE_FIELD",
        r"^//#define EQUILIBRIUM_BOUNDARIES\b": "#define EQUILIBRIUM_BOUNDARIES",
        r"^//#define SUBGRID\b": "#define SUBGRID",
    }
    SETUP = inspect.cleandoc("""
        #include "setup.hpp"

        void main_setup() { // car in a wind tunnel, following the Ahmed body setup; parameters from FX3D_* environment variables
            const uint memory = (uint)atoi(getenv("FX3D_MEMORY"));
            const float lbm_u = (float)atof(getenv("FX3D_LBM_U"));
            const float box_scale = (float)atof(getenv("FX3D_BOX_SCALE"));
            const float si_u = (float)atof(getenv("FX3D_SI_U"));
            const float si_nu = (float)atof(getenv("FX3D_SI_NU")), si_rho = (float)atof(getenv("FX3D_SI_RHO"));
            const float si_length = (float)atof(getenv("FX3D_LENGTH")), si_width = (float)atof(getenv("FX3D_WIDTH")), si_height = (float)atof(getenv("FX3D_HEIGHT"));
            const float si_A = (float)atof(getenv("FX3D_AREA"));
            const float flow_passes = (float)atof(getenv("FX3D_FLOW_PASSES"));
            const uint force_interval = (uint)atoi(getenv("FX3D_FORCE_INTERVAL"));
            const uint parts = (uint)atoi(getenv("FX3D_PARTS"));
            const uint3 lbm_N = resolution(float3(box_scale*si_length, box_scale*si_width, 0.5f*(box_scale-1.0f)*si_width+si_height), memory);
            units.set_m_kg_s((float)lbm_N.x, lbm_u, 1.0f, box_scale*si_length, si_u, si_rho);
            const float lbm_nu = units.nu(si_nu);
            const ulong lbm_T = units.t(flow_passes*box_scale*si_length/si_u);
            const ulong lbm_T_average = units.t(box_scale*si_length/si_u);
            print_info("Re = "+to_string(to_uint(units.si_Re(si_width, si_u, si_nu))));
            LBM lbm(lbm_N, lbm_nu);
            const float3 offset = float3(units.x(0.5f*(0.5f*box_scale*si_length-si_width)), 0.5f*(float)lbm_N.y, 1.0f);
            for(uint i=0u; i<parts; i++) {
                Mesh* mesh = read_stl("car"+to_string(i)+".stl", units.x(1.0f), float3x3(1.0f), offset);
                lbm.voxelize_mesh_on_device(mesh, TYPE_S|TYPE_X);
            }
            const uint Nx=lbm.get_Nx(), Ny=lbm.get_Ny(), Nz=lbm.get_Nz(); parallel_for(lbm.get_N(), [&](ulong n) { uint x=0u, y=0u, z=0u; lbm.coordinates(n, x, y, z);
                if(z==0u) lbm.flags[n] = TYPE_S;
                if(lbm.flags[n]!=TYPE_S) lbm.u.x[n] = lbm_u;
                if(x==0u||x==Nx-1u||y==0u||y==Ny-1u||z==Nz-1u) lbm.flags[n] = TYPE_E;
            });
            lbm.run(0u, lbm_T);
            double force = 0.0;
            ulong samples = 0ull;
            while(lbm.get_t()<lbm_T) {
                lbm.run(force_interval, lbm_T);
                if(lbm.get_t()+lbm_T_average>lbm_T) {
                    force += (double)lbm.object_force(TYPE_S|TYPE_X).x;
                    samples++;
                }
            }
            const float Cd = units.si_F((float)(force/(double)samples))/(0.5f*si_rho*sq(si_u)*si_A);
            print_info("C_d = "+to_string(Cd, 6u));
        }
    """)
    MEMORY = 2000
    LBM_U = 0.05
    BOX_SCALE = 6.0
    SI_U = 16.0
    SI_NU = 1.48e-5
    SI_RHO = 1.225
    FLOW_PASSES = 2.0
    FORCE_INTERVAL = 10

    def __init__(self, concurrency: int):
        """Builds FluidX3D with SETUP and DEFINES unless already built.

        Args:
            concurrency: int, cars simulated at once, all on the GPU FluidX3D picks

        Raises:
            subprocess.CalledProcessError: g++ fails
        """
        super().__init__(concurrency)
        self.binary = self.build()

    @classmethod
    def build(cls) -> str:
        """Compiles FluidX3D's sources with SETUP as setup.cpp and DEFINES applied to defines.hpp, unless already built.

        The build happens in a temporary directory moved into place at the end, so concurrent builds do not collide.

        Returns:
            str, the binary's path, FLUIDX3D_DIR/build/{hash of SETUP and DEFINES}/FluidX3D

        Raises:
            subprocess.CalledProcessError: g++ fails
        """
        build_dir = os.path.join(cls.FLUIDX3D_DIR, "build", hashlib.sha256((cls.SETUP + repr(cls.DEFINES)).encode()).hexdigest()[:16])
        if os.path.exists(f"{build_dir}/FluidX3D"):
            return f"{build_dir}/FluidX3D"
        os.makedirs(os.path.dirname(build_dir), exist_ok=True)
        with tempfile.TemporaryDirectory(dir=os.path.dirname(build_dir)) as work:
            shutil.copytree(f"{cls.FLUIDX3D_DIR}/src", f"{work}/src")
            with open(f"{work}/src/setup.cpp", "w") as f:
                f.write(cls.SETUP)
            with open(f"{work}/src/defines.hpp") as f:
                defines = f.read()
            for pattern, replacement in cls.DEFINES.items():
                defines, count = re.subn(pattern, replacement, defines, flags=re.MULTILINE)
                assert count == 1, f"defines.hpp has {count} matches of {pattern!r}"
            with open(f"{work}/src/defines.hpp", "w") as f:
                f.write(defines)
            os.makedirs(f"{work}/build")
            shutil.copy(f"{work}/src/OpenCL/lib/libOpenCL.so", f"{work}/build/libOpenCL.so.1")
            subprocess.run(
                ["g++", *sorted(glob.glob(f"{work}/src/*.cpp")), "-o", f"{work}/build/FluidX3D", "-std=c++17", "-pthread", "-O", "-Wno-comment",
                 f"-I{work}/src/OpenCL/include", f"-L{work}/src/OpenCL/lib", "-lOpenCL", "-Wl,-rpath,$ORIGIN"],
                check=True,
            )
            try:
                os.rename(f"{work}/build", build_dir)
            except OSError:  # Another process built it first.
                pass
        return f"{build_dir}/FluidX3D"

    async def evaluate_one_async(self, response: str, semaphore: asyncio.Semaphore) -> float:
        """Builds the response's car in Blender, simulates it with FluidX3D in a new process and returns its -C_d.

        The temporary directory is removed, and the process killed, on any exception or cancellation.

        Args:
            response: str, its last python code block a bpy script building the car in metres
            semaphore: asyncio.Semaphore, held from building the car to the end of its simulation

        Returns:
            -C_d; -inf without code, for a nonfinite C_d or when any step raises, with exceptions printed to stderr.
        """
        script = self.extract_code(response)
        if not script:
            return -math.inf
        async with semaphore:
            with tempfile.TemporaryDirectory() as home:
                try:
                    mesh, Aref = await self.build_car(script, home)
                    mesh.apply_translation([-mesh.bounds[0, 0], 0, 0])
                    parts = mesh.split(only_watertight=False)
                    for i, part in enumerate(parts):
                        part.export(f"{home}/car{i}.stl")
                    length, width, height = mesh.extents
                    env = {
                        "FX3D_MEMORY": self.MEMORY, "FX3D_LBM_U": self.LBM_U, "FX3D_BOX_SCALE": self.BOX_SCALE, "FX3D_SI_U": self.SI_U,
                        "FX3D_SI_NU": self.SI_NU, "FX3D_SI_RHO": self.SI_RHO, "FX3D_LENGTH": length, "FX3D_WIDTH": width,
                        "FX3D_HEIGHT": height, "FX3D_AREA": Aref, "FX3D_FLOW_PASSES": self.FLOW_PASSES,
                        "FX3D_FORCE_INTERVAL": self.FORCE_INTERVAL, "FX3D_PARTS": len(parts),
                    }
                    # pixi's cuda-compat libraries make NVIDIA's OpenCL compiler emit PTX the driver rejects, and
                    # NVIDIA's compute cache would replay such kernels.
                    env = {**{k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}, "CUDA_CACHE_DISABLE": "1", **{k: str(v) for k, v in env.items()}}
                    stdout = await subprocess_run([self.binary], cwd=home, env=env, stdout=subprocess.PIPE)
                    Cd = float(re.findall(r"C_d = (\S+)", stdout.decode(errors="replace"))[-1])
                except Exception as exception:
                    print(f"FluidX3D.evaluate_one_async failed: {exception!r}", file=sys.stderr)
                    return -math.inf
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


class FunSearchTask:
    """A problem of FunSearch's (Romera-Paredes et al., 2024) notebooks, the priority function of its SKELETON_NAME
    skeleton evolved; a subclass sets the constants left None.

    The prompt is PROMPT_TEMPLATE with the skeleton, its priority the seed, for IID random sampling and the first
    generation of sequential conditioned sampling (SCS); IN_CONTEXT_PROMPT_TEMPLATE adds previous solutions and their
    metrics for SCS's later generations.

    A response's code is its last code block. It runs before the skeleton's other functions, so it cannot replace them,
    in a fresh `python -I` process in a temporary directory, killed after MAX_EXECUTION_TIME. Its reward is the
    skeleton's evaluate of the inputs, higher is better; its metric is REWARD_SIGN * reward.

    Attributes:
        concurrency: int, codes evaluated at once
        inputs: dict, the keyword arguments of the skeleton's evaluate
        skeleton: str, the notebook's skeleton program
    """

    NOTEBOOK = None
    SKELETON_NAME = None  # The skeleton cell's docstring, e.g. "Finds large cap sets."
    INPUTS = {}  # The keyword arguments of the skeleton's evaluate, e.g. {"n": 8}.
    METRIC_NAME = None  # With its article, e.g. "a cap set size".
    REWARD_SIGN = 1
    MAX_EXECUTION_TIME = None
    PROMPT_TEMPLATE = None  # Formatted with skeleton and the inputs.
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
        This solution, solution {index}, got {metric_name} of {metric}.
    """)
    PROGRAM_TEMPLATE = inspect.cleandoc("""
        import json
        import sys
        import numpy as np

        {code}

        {skeleton}

        print(evaluate(**json.load(sys.stdin)))
    """)
    CODE_BLOCK = re.compile(r"```(?:python)?\n?(.*?)```", re.DOTALL)

    def __init__(self, concurrency: int):
        """Loads the SKELETON_NAME skeleton from NOTEBOOK without running it.

        Args:
            concurrency: int, codes evaluated at once
        """
        self.concurrency = concurrency
        self.inputs = self.INPUTS
        self.skeleton = next(cell for cell in self.code_cells() if cell.startswith(f'"""{self.SKELETON_NAME}"""'))

    @classmethod
    def code_cells(cls) -> list[str]:
        """The sources of NOTEBOOK's code cells."""
        with open(cls.NOTEBOOK) as f:
            return ["".join(cell["source"]) for cell in json.load(f)["cells"] if cell["cell_type"] == "code"]

    def prompt(self) -> str:
        """PROMPT_TEMPLATE, the prompt of IID random sampling and of SCS's first generation."""
        return self.PROMPT_TEMPLATE.format(skeleton=self.skeleton, **self.inputs)

    def prompts_with_rewards(self, responses: list[str], rewards: list[float]) -> str:
        """Every response's code as a solution of one IN_CONTEXT_PROMPT_TEMPLATE, each by IN_CONTEXT_ITEM_TEMPLATE
        with its metric, numbered from 1 in the given order, for SCS's later generations.

        Args:
            responses: list (G) of str, G ≥ 1, each holding code
            rewards: list (G) of float, each response's evaluate, finite

        Returns:
            str
        """
        in_contexts = []
        for i, (reward, response) in enumerate(zip(rewards, responses), 1):
            in_contexts.append(self.IN_CONTEXT_ITEM_TEMPLATE.format(index=i, metric_name=self.METRIC_NAME, metric=self.REWARD_SIGN * reward, code=self.extract_code(response)))
        return self.IN_CONTEXT_PROMPT_TEMPLATE.format(instruction=self.prompt(), item="\n\n\n".join(in_contexts))

    def extract_code(self, response: str) -> str | None:
        """The response's last code block, its python tag dropped, stripped; None without one or when it is empty."""
        codes = self.CODE_BLOCK.findall(response)
        return (codes[-1].strip() or None) if codes else None

    def evaluate(self, responses: list[str]) -> list[float]:
        """Evaluates the responses' codes with evaluate_one_async, concurrency at a time, blocking until all are done.

        Args:
            responses: list (N) of str

        Returns:
            list (N) of float, each response's reward, higher is better; -inf without code or when it fails
        """
        semaphore = asyncio.Semaphore(self.concurrency)

        async def evaluate_async():
            return await asyncio.gather(*(self.evaluate_one_async(self.extract_code(response), semaphore) for response in responses))

        return asyncio.run(evaluate_async())

    async def evaluate_one_async(self, code: str | None, semaphore: asyncio.Semaphore) -> float:
        """The skeleton's evaluate of the inputs with the code's priority.

        Args:
            code: str or None, defining priority
            semaphore: asyncio.Semaphore, held while the program runs

        Returns:
            The reward; -inf without code, when the program exits non-zero, prints no number last or exceeds
            MAX_EXECUTION_TIME, or for a nonfinite score.
        """
        if not code:
            return -math.inf
        program = self.PROGRAM_TEMPLATE.format(code=code, skeleton=self.skeleton.partition("# @funsearch.evolve")[0])
        async with semaphore:
            with tempfile.TemporaryDirectory() as home:
                with open(f"{home}/program.py", "w") as f:
                    f.write(program)
                with open(f"{home}/inputs.json", "w") as f:
                    json.dump(self.inputs, f)
                try:
                    with open(f"{home}/inputs.json", "rb") as stdin:
                        stdout = await asyncio.wait_for(subprocess_run(
                            [sys.executable, "-I", "program.py"], cwd=home,
                            stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                        ), self.MAX_EXECUTION_TIME)
                    reward = float(stdout.decode().splitlines()[-1])
                except (subprocess.CalledProcessError, TimeoutError, ValueError, IndexError):
                    return -math.inf
        return reward if math.isfinite(reward) else -math.inf


class BinPacking(FunSearchTask):
    """Online 1D bin packing of FunSearch's DATASET_NAME instances; the reward is the negated average number of bins,
    deterministic for a deterministic priority."""

    NOTEBOOK = "packages/funsearch/bin_packing/bin_packing.ipynb"
    SKELETON_NAME = "Finds heuristics for online 1d binpacking."
    DATASET_NAME = "Weibull 5k"  # Weibull 5k or OR3
    METRIC_NAME = "an average number of bins"
    REWARD_SIGN = -1
    MAX_EXECUTION_TIME = 300.0
    PROMPT_TEMPLATE = inspect.cleandoc("""
        Design a heuristic for online 1D bin packing that uses as few bins as possible.

        Items arrive one at a time, and this program puts each into the bin of the highest priority among those it fits in:
        ```python
        {skeleton}
        ```

        Write a better `priority` function with the imports it needs; the rest of the program stays as it is.
        The metric is the average number of bins used on FunSearch's {dataset_name} dataset, {num_instances} instances of {num_items} items with bins of capacity {capacity}; lower is better.

        Write the function within a ```python block.
    """)

    def __init__(self, concurrency: int):
        """Loads the skeleton and the DATASET_NAME instances from NOTEBOOK, its data parsed without running it.

        Args:
            concurrency: int, codes evaluated at once
        """
        super().__init__(concurrency)
        datasets = {
            ast.literal_eval(node.targets[0].slice): ast.literal_eval(node.value)
            for node in ast.parse(next(cell for cell in self.code_cells() if "datasets = {}" in cell)).body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript)
        }
        self.inputs = {"instances": datasets[self.DATASET_NAME]}

    def prompt(self) -> str:
        """PROMPT_TEMPLATE, the prompt of IID random sampling and of SCS's first generation."""
        instances = self.inputs["instances"]
        instance = next(iter(instances.values()))
        return self.PROMPT_TEMPLATE.format(skeleton=self.skeleton, dataset_name=self.DATASET_NAME, num_instances=len(instances), num_items=instance["num_items"], capacity=instance["capacity"])


class CapSet(FunSearchTask):
    """The largest cap set in n dimensions, built greedily by priority; FunSearch found 512 for n = 8."""

    NOTEBOOK = "packages/funsearch/cap_set/cap_set.ipynb"
    SKELETON_NAME = "Finds large cap sets."
    INPUTS = {"n": 8}
    METRIC_NAME = "a cap set size"
    MAX_EXECUTION_TIME = 30.0
    PROMPT_TEMPLATE = inspect.cleandoc("""
        Design a priority function that builds a cap set as large as possible: a set of vectors in {{0, 1, 2}}^{n}, no three distinct vectors of which sum to zero modulo 3.

        This program builds the cap set greedily, adding the vector of highest priority among those that keep it a cap set:
        ```python
        {skeleton}
        ```

        Write a better `priority` function with the imports it needs; the rest of the program stays as it is.
        The metric is the size of the cap set in n = {n} dimensions; higher is better.

        Write the function within a ```python block.
    """)


class AdmissibleSet(FunSearchTask):
    """The largest constant-weight admissible set A(n, w), built greedily by priority over all vectors of weight w;
    FunSearch found the full size C(n, w) for (12, 7)."""

    NOTEBOOK = "packages/funsearch/admissible_set/admissible_set.ipynb"
    SKELETON_NAME = "Finds large admissible sets."
    INPUTS = {"n": 12, "w": 7}
    METRIC_NAME = "an admissible set size"
    MAX_EXECUTION_TIME = 120.0
    PROMPT_TEMPLATE = inspect.cleandoc("""
        Design a priority function that builds a constant-weight admissible set A({n}, {w}) as large as possible: a set of vectors in {{0, 1, 2}}^{n}, each with {w} nonzero entries and a support no other vector has, such that any three distinct vectors have a coordinate where their values are {{0, 1, 2}}, {{0, 0, 1}} or {{0, 0, 2}}.

        This program builds the admissible set greedily, adding the vector of highest priority among those that keep it admissible:
        ```python
        {skeleton}
        ```

        Write a better `priority` function with the imports it needs; the rest of the program stays as it is.
        The metric is the size of the admissible set for n = {n} and w = {w}; higher is better.

        Write the function within a ```python block.
    """)


class Story:
    """A story pulled toward a higher score of REWARD_MODEL_NAME as the reply to USER_PROMPT.

    The prompts are PROMPT_TEMPLATE without references, for the first generation of sequential conditioned sampling
    (SCS), and IN_CONTEXT_PROMPT_TEMPLATE with references and their rewards for its later generations. There is no seed
    story.

    A response's story is the whole response, stripped. A story's reward is REWARD_MODEL_NAME's score of the
    chat of USER_PROMPT and the story as the assistant's reply, in its chat template without a system turn as its model
    card recommends; REWARD_MODEL_NAME is loaded on the first visible GPU at construction and scores concurrency stories
    per forward pass.
    """

    REWARD_MODEL_NAME = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    USER_PROMPT = "Write a story in one sentence."
    PROMPT_TEMPLATE = inspect.cleandoc("""
        Create a story in one sentence.

        The metric is the reward model score; higher is better.

        Reply with the story only.
    """)
    IN_CONTEXT_PROMPT_TEMPLATE = inspect.cleandoc("""
        {instruction}
        Here are some example solutions from previous attempts, please improve on them:

        {item}

        Please now provide your new, better solution.
    """)
    IN_CONTEXT_ITEM_TEMPLATE = inspect.cleandoc("""
        # Solution {index}:
        {story}
        This solution, solution {index}, got a reward model score of {metric}.
    """)

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
            responses: list (G) of str, G ≥ 1, each nonempty after stripping
            rewards: list (G) of float, each response's evaluate, finite

        Returns:
            str
        """
        in_contexts = []
        for i, (reward, response) in enumerate(zip(rewards, responses), 1):
            in_contexts.append(self.IN_CONTEXT_ITEM_TEMPLATE.format(index=i, metric=reward, story=self.extract_story(response)))
        return self.IN_CONTEXT_PROMPT_TEMPLATE.format(instruction=self.prompt(), item="\n\n\n".join(in_contexts))

    def extract_story(self, response: str) -> str | None:
        """The response, stripped; None when it is empty."""
        return response.strip() or None

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
            responses: list (N) of str

        Returns:
            list (N) of float, REWARD_MODEL_NAME's score of each response's story as the reply to USER_PROMPT,
            unbounded, higher is better; -inf for an empty response
        """
        stories = [self.extract_story(response) for response in responses]
        stories_valid = [story for story in stories if story]
        rewards = iter(self.judge(stories_valid) if stories_valid else [])
        return [next(rewards) if story else -math.inf for story in stories]


class ArcAgi:
    """A test input of an ARC-AGI task in DATA_DIR's training or evaluation set, DATA_DIR set by a subclass.

    The prompt is arcprize/arc-agi-benchmarking's, the official prompt of the ARC Prize leaderboard, for IID random
    sampling and the first generation of sequential conditioned sampling (SCS); IN_CONTEXT_PROMPT_TEMPLATE adds
    previous attempts' grids and rewards for SCS's later generations.

    A response's grid is the official parse_and_validate_json's. Its reward is the cell accuracy of OpenCompass's
    ARCPrizeEvaluator against the test output: the cells equal to the test output's over the cells of the smallest
    rectangle holding both grids, 1 iff the grid is the test output.

    Attributes:
        training_pairs: list of ARCPair, the task's training pairs
        test_pair: ARCPair, the test input and its output
    """

    DATA_DIR = None
    IN_CONTEXT_PROMPT_TEMPLATE = inspect.cleandoc("""
        {instruction}

        The metric is the cell accuracy, the fraction of cells matching the correct output; higher is better, and 1 is correct.
        Here are some example solutions from previous attempts, please improve on them:

        {item}

        Please now provide your new, better solution.
    """)
    IN_CONTEXT_ITEM_TEMPLATE = inspect.cleandoc("""
        # Solution {index}:
        {grid}
        This solution, solution {index}, got a cell accuracy of {metric}.
    """)

    def __init__(self, concurrency: int, task_id: str, pair_index: int | str):
        """Loads the task's training pairs and test pair from DATA_DIR's training or evaluation set.

        Args:
            concurrency: int, unused, a grid's evaluation being a comparison
            task_id: str, a task's file name in DATA_DIR/training or DATA_DIR/evaluation without .json, unique
                across both, e.g. "00576224"
            pair_index: int or its str, the test input's index among the task's test pairs
        """
        self.concurrency = concurrency
        [task_file] = glob.glob(f"{self.DATA_DIR}/*/{task_id}.json")
        data_dir = os.path.dirname(task_file)
        self.training_pairs = get_train_pairs_from_task(data_dir, task_id)
        self.test_pair = get_test_pairs_from_task(data_dir, task_id)[int(pair_index)]

    def prompt(self) -> str:
        """The official prompt of the test input, the prompt of IID random sampling and of SCS's first generation."""
        return convert_task_pairs_to_prompt(self.training_pairs, self.test_pair)

    def prompts_with_rewards(self, responses: list[str], rewards: list[float]) -> str:
        """Every response's grid as a solution of one IN_CONTEXT_PROMPT_TEMPLATE, each by IN_CONTEXT_ITEM_TEMPLATE
        with its cell accuracy, numbered from 1 in the given order, for SCS's later generations. The official prompt
        enters it without its closing "Your response:".

        Args:
            responses: list (G) of str, G ≥ 1, each holding a grid
            rewards: list (G) of float, each response's evaluate, finite

        Returns:
            str
        """
        in_contexts = []
        for i, (reward, response) in enumerate(zip(rewards, responses), 1):
            in_contexts.append(self.IN_CONTEXT_ITEM_TEMPLATE.format(index=i, metric=reward, grid=json.dumps(self.extract_grid(response))))
        instruction = self.prompt().removesuffix("Your response:").rstrip()
        return self.IN_CONTEXT_PROMPT_TEMPLATE.format(instruction=instruction, item="\n\n\n".join(in_contexts))

    def extract_grid(self, response: str) -> list[list] | None:
        """The official parse_and_validate_json's grid: the first \\boxed{} list of lists, else the last list of
        lists; None without one."""
        try:
            return parse_and_validate_json(response)
        except ValueError:
            return None

    def evaluate_one(self, grid: list[list] | None) -> float:
        """The grid's cell accuracy against the test output; -inf without a grid."""
        if grid is None:
            return -math.inf
        output = self.test_pair.output
        rows = max(len(grid), len(output))
        cols = max(map(len, grid + output))
        return sum(cell == cell_output for row, row_output in zip(grid, output) for cell, cell_output in zip(row, row_output)) / (rows * cols)

    def evaluate(self, responses: list[str]) -> list[float]:
        """Evaluates the responses' grids with evaluate_one.

        Args:
            responses: list (N) of str

        Returns:
            list (N) of float, each response's cell accuracy in [0, 1], 1 iff its grid is the test output, higher is
            better; -inf without a grid
        """
        return [self.evaluate_one(self.extract_grid(response)) for response in responses]


class ArcAgi1(ArcAgi):
    """A test input of ARC-AGI-1's public training or evaluation set, of 400 + 400 tasks and 416 + 419 test inputs."""

    DATA_DIR = "packages/ARC-AGI/data"


class ArcAgi2(ArcAgi):
    """A test input of ARC-AGI-2's public training or evaluation set, of 1000 + 120 tasks and 1076 + 167 test inputs."""

    DATA_DIR = "packages/ARC-AGI-2/data"


TASKS_CLS = {
    "openfoam": OpenFOAM,
    "fluidx3d": FluidX3D,
    "circle-packing": CirclePacking,
    "kissing-number": KissingNumber,
    "maxmin-distance-ratio": MaxMinDistanceRatio,
    "sum-difference-sets": SumDifferenceSets,
    "bin-packing": BinPacking,
    "cap-set": CapSet,
    "admissible-set": AdmissibleSet,
    "story": Story,
    "docking-3pbl": Docking3PBL,
    "docking-2rgp": Docking2RGP,
    "docking-3eml": Docking3EML,
    "arc-agi-1": ArcAgi1,
    "arc-agi-2": ArcAgi2,
}

def get_reward_fn(task_name: str, concurrency: int):
    """
    Args:
        task_name: str, a TASKS_CLS name, e.g. "circle-packing", optionally followed by the class's arguments after
            concurrency in brackets, comma-separated and passed positionally as str, e.g. "arc-agi-2[136b0064,0]"
        concurrency: int, the task's evaluations at once
    """
    name, *args = re.findall(r"[\w-]+", task_name)
    return TASKS_CLS[name](concurrency, *args)
