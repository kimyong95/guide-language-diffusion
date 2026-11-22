"""Black-box tasks whose reward comes from an external simulator rather than a model or a metric in Python.

tasks.py imports the task classes defined here and registers them in its TASKS_CLS.
"""

import concurrent.futures
import inspect
import math
import multiprocessing
import os
import re


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
    PROTEIN = "the dopamine type 3 receptor (DRD3)"


class Docking2RGP(Docking):

    PDB_ID = "2rgp"
    PROTEIN = "the epidermal growth factor receptor (EGFR)"


class Docking3EML(Docking):

    PDB_ID = "3eml"
    PROTEIN = "the adenosine A2A receptor"
