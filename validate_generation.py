import numpy as np
import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem, QED, Descriptors
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


# ----------------------------------------------------
# ۱. استخراج توزیع‌ها و مشخصات مجموعه داده آموزش (Train)
# ----------------------------------------------------
def analyze_training_set(train_smiles_list):
    """
    Analyze the training set and extract:
    - allowed elements
    - molecular weight range (1st to 99th percentile)
    - heavy atom count range (1st to 99th percentile)
    - maximum absolute formal charge
    - Morgan fingerprints for similarity search
    - canonical SMILES in both list and set form
    """
    mw_list = []
    heavy_atom_list = []
    max_abs_charge = 0
    allowed_elements = set()

    train_fps = []
    train_canonical_smiles_list = []
    train_canonical_smiles_set = set()

    for smiles in train_smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)

        # برای exact novelty
        train_canonical_smiles_set.add(canonical)

        # برای nearest-neighbor mapping
        train_canonical_smiles_list.append(canonical)

        # molecular properties
        mw_list.append(Descriptors.ExactMolWt(mol))
        heavy_atom_list.append(mol.GetNumHeavyAtoms())

        abs_charge = abs(Chem.GetFormalCharge(mol))
        if abs_charge > max_abs_charge:
            max_abs_charge = abs_charge

        for atom in mol.GetAtoms():
            allowed_elements.add(atom.GetSymbol())

        morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)
        fp = morgan_gen.GetFingerprint(mol)
        train_fps.append(fp)

    mw_min, mw_max = np.percentile(mw_list, 1), np.percentile(mw_list, 99)
    heavy_min, heavy_max = np.percentile(heavy_atom_list, 1), np.percentile(heavy_atom_list, 99)

    return {
        "allowed_elements": allowed_elements,
        "mw_range": (mw_min, mw_max),
        "heavy_atom_range": (heavy_min, heavy_max),
        "max_abs_charge": max_abs_charge,
        "train_fps": train_fps,
        "train_canonical_smiles_list": train_canonical_smiles_list,
        "train_canonical_smiles_set": train_canonical_smiles_set
    }


# ----------------------------------------------------
# ۲. منطق فیلتر و اعتبارسنجی یک مولکول منفرد (مراحل a تا h)
# ----------------------------------------------------
def validate_single_molecule(selfies_str, train_stats):
    """
    یک توالی SELFIES را با ۹ فیلتر تعیین شده در بخش 1.3.3 ارزیابی می‌کند.
    """
    # a. دکد کردن SELFIES به SMILES و بررسی خطاها
    try:
        smiles = sf.decoder(selfies_str)
        if not smiles or smiles.strip() == "":
            return None, "a_failed"
    except Exception:
        return None, "a_failed"

    # b. پارس اولیه بدون Sanitization
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None, "b_failed"

    # c. اعمال Sanitization اصلی (بررسی معتبر بودن RDKit)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None, "c_failed"  # فیلتر RDKit-valid در اینجا رد می‌شود

    # d. بررسی تک‌جزئی بودن (Connected Component)
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) != 1:
        return None, "d_failed"

    # e. بررسی اتم‌های مجاز و عدم وجود اتم‌های ساختگی (Dummy)
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atomic_num = atom.GetAtomicNum()
        # if atomic_num = 0 it's a dummy
        if atomic_num == 0 or symbol not in train_stats["allowed_elements"]:
            return None, "e_failed"

    # f. بررسی عدم وجود رادیکال آزاد
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() != 0:
            return None, "f_failed"

    # g. فیلتر محدوده وزن مولکولی و تعداد اتم سنگین
    mw = AllChem.CalcExactMolWt(mol)
    mw_min, mw_max = train_stats["mw_range"]
    if not (mw_min <= mw <= mw_max):
        return None, "g_failed_mw"

    heavy_count = mol.GetNumHeavyAtoms()
    heavy_min, heavy_max = train_stats["heavy_atom_range"]
    if not (heavy_min <= heavy_count <= heavy_max):
        return None, "g_failed_heavy"

    # h. فیلتر بار رسمی حداکثر
    abs_charge = abs(Chem.GetFormalCharge(mol))
    if abs_charge > train_stats["max_abs_charge"]:
        return None, "h_failed"

    # i. تبدیل به ساختار کانونی ایزومریک استاندارد
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    return canonical_smiles, "passed"


# ----------------------------------------------------
# ۳. ارزیابی کلی و محاسبه معیارهای آماری برای دماهای مختلف
# ----------------------------------------------------
def evaluate_generated_molecules(generated_selfies_dict, train_stats):
    """
    برای هر دما معیارهای زیر را گزارش می‌دهد:
    - RDKit Validity
    - Acceptance Rate
    - Uniqueness
    - Exact Novelty
    - Structural Novelty
    - Mean QED
    """
    results = {}
    morgan_gen = GetMorganGenerator(radius=2, fpSize=2048)

    for temp, selfies_list in generated_selfies_dict.items():
        total = len(selfies_list)
        if total == 0:
            continue

        rdkit_valid_count = 0
        accepted_mols = []
        accepted_canonical_smiles = []

        for s in selfies_list:
            # بررسی اعتبار اولیه RDKit (مراحل a تا c)
            try:
                smiles = sf.decoder(s)
                if smiles:
                    m = Chem.MolFromSmiles(smiles, sanitize=False)
                    if m is not None:
                        try:
                            Chem.SanitizeMol(m)
                            rdkit_valid_count += 1
                        except:
                            pass
            except:
                pass

            # اعتبارسنجی کامل فیلترها
            canonical_smiles, status = validate_single_molecule(s, train_stats)
            if status == "passed":
                accepted_canonical_smiles.append(canonical_smiles)
                accepted_mols.append(Chem.MolFromSmiles(canonical_smiles))

        # fraction that passes RDKit parsing and sanitization.الف
        rdkit_validity = rdkit_valid_count / total
        # how many passed the 9 filters?
        acceptance_rate = len(accepted_mols) / total

        # ب. یکتایی (Uniqueness)
        unique_smiles = list(set(accepted_canonical_smiles))
        uniqueness = len(unique_smiles) / len(accepted_canonical_smiles) if accepted_canonical_smiles else 0.0

        # ج. نوآوری دقیق (Exact Novelty)
        # fraction of unique accepted molecules absent from the training set.
        novel_smiles = [s for s in unique_smiles if s not in train_stats["train_canonical_smiles_set"]]
        exact_novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0.0

        # د. نوآوری ساختاری (Structural Novelty) و امتیاز QED
        structural_novel_count = 0
        qed_scores = []

        for s_smiles in novel_smiles:
            mol = Chem.MolFromSmiles(s_smiles)
            # QED:(Quantitative Estimate of Drug-likeness) عددی بین ۰ تا ۱ است که نشان می‌دهد یک مولکول چقدر خواص فیزیکوشیمیایی شبیه به داروهای تجاری دارد (هرچه به ۱ نزدیک‌تر، بهتر).
            qed_scores.append(QED.qed(mol))
            # Tanimoto Similarity : works on fingerprints (اجتماع / اشتراک ->)
            fp = morgan_gen.GetFingerprint(mol)
            max_sim = 0.0
            for t_fp in train_stats["train_fps"]:
                sim = TanimotoSimilarity(fp, t_fp)
                if sim > max_sim:
                    max_sim = sim

            # شرط نوآوری ساختاری: حداکثر شباهت کمتر از 0.85
            # یعنی مولکول نه تنها جدید است، بلکه از نظر ساختار دارویی هم کپی‌برداری ساده از داده‌های آموزش نیست و یک نوآوری واقعی است.
            if max_sim < 0.85:
                structural_novel_count += 1

        structural_novelty = structural_novel_count / len(novel_smiles) if novel_smiles else 0.0
        mean_qed = np.mean(qed_scores) if qed_scores else 0.0

        results[temp] = {
            "RDKit Validity": rdkit_validity,
            "Acceptance Rate": acceptance_rate,
            "Uniqueness": uniqueness,
            "Exact Novelty": exact_novelty,
            "Structural Novelty": structural_novelty,
            "Mean QED": mean_qed,
            "novel_molecules_smiles": novel_smiles
        }

    return results