import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import SimpleITK as sitk

# ──────────────────────────────────────────────
PACIENTES_DIR = Path(r"D:\tfg_francis\SANOS")
DIXON_REG_SUBDIR = "dixon_coregistration_opcion2/output/Dixon"
# ──────────────────────────────────────────────


def find_subdir(patient_dir: Path, keywords: list) -> Path:
    """Busca la primera subcarpeta que contenga TODOS los keywords (case-insensitive)."""
    for d in sorted(patient_dir.iterdir()):
        if d.is_dir() and all(kw.lower() in d.name.lower() for kw in keywords):
            return d
    raise FileNotFoundError(f"No se encontró carpeta con {keywords} en: {patient_dir}")


def find_nii(folder: Path) -> Path:
    for pattern in ("*.nii.gz", "*.nii"):
        hits = sorted(folder.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"No se encontró ningún .nii/.nii.gz en: {folder}")


def resample_to_reference(moving_path: Path, reference_path: Path) -> np.ndarray:
    """
    Usa SimpleITK para hacer resample de 'moving_path' a la geometría de 'reference_path'.
    Devuelve la matriz de datos como un numpy array.
    """
    moving_img = sitk.ReadImage(str(moving_path))
    ref_img = sitk.ReadImage(str(reference_path))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(sitk.Transform()) # Transformación Identidad
    resampler.SetDefaultPixelValue(0)

    resampled_img = resampler.Execute(moving_img)
    # SimpleITK devuelve array en orden (z, y, x), nibabel es (x, y, z)
    # Transponemos para mantener la misma estructura visual
    data = sitk.GetArrayFromImage(resampled_img).transpose(2, 1, 0)
    return data


def process_slice_data(data: np.ndarray) -> np.ndarray:
    """Extrae el corte medio, lo rota y lo normaliza."""
    if data.ndim == 4:
        data = data[..., 0]
    mid = data.shape[2] // 2
    slc = np.rot90(data[:, :, mid])
    
    # Normalización para visualización
    if slc.max() > 0:
        p2, p98 = np.percentile(slc[slc > 0], (2, 98))
    else:
        p2, p98 = 0, 1
    
    slc = np.clip(slc, p2, p98)
    slc = (slc - slc.min()) / (slc.max() - slc.min() + 1e-8)
    return slc


def load_mid_slice(nii_path: Path):
    """Carga clásica de un NIfTI y procesa su corte central."""
    img = nib.load(str(nii_path))
    data = img.get_fdata()
    return process_slice_data(data)


def make_overlay(base, overlay, alpha=0.45):
    """
    Genera el array RGB mezclando la base en escala de grises con el overlay en color viridis.
    Asume que base y overlay ya tienen la misma forma (shape).
    """
    # Si por alguna razón siguen sin coincidir, levantamos error en lugar de estirar a la fuerza
    if base.shape != overlay.shape:
        raise ValueError(f"Las dimensiones no coinciden para el overlay: Base {base.shape} vs Overlay {overlay.shape}")
        
    base_rgb = np.stack([base, base, base], axis=-1)
    overlay_rgb = cm.viridis(overlay)[..., :3]
    return (1 - alpha) * base_rgb + alpha * overlay_rgb


patient_dirs = []
for patient in PACIENTES_DIR.iterdir():
    if patient.is_dir():
        for sub in patient.iterdir():
            if sub.is_dir() and sub.name == "01":
                patient_dirs.append(sub)
patient_dirs = sorted(patient_dirs)

for patient_dir in patient_dirs:
    try:
        t1_dir        = find_subdir(patient_dir, ["T1W"])
        dixon_ip_dir  = find_subdir(patient_dir, ["imagen_IP"])
        dixon_reg_dir = patient_dir / DIXON_REG_SUBDIR

        t1_path        = find_nii(t1_dir)
        dixon_ip_path  = find_nii(dixon_ip_dir)
        dixon_reg_path = find_nii(dixon_reg_dir)

        print(f"[{patient_dir.name}]")
        print(f"  T1             : {t1_dir.name} → {t1_path.name}")
        print(f"  Dixon IP (orig): {dixon_ip_dir.name} → {dixon_ip_path.name}")
        print(f"  Dixon (reg)    : {dixon_reg_path.name}")

        # 1. Cargamos T1 y Dixon Registrada (que ya comparten el mismo FOV y resolución física por el coregistro)
        t1      = load_mid_slice(t1_path)
        dixon_r = load_mid_slice(dixon_reg_path)

        # 2. El cambio clave: Hacemos resample físico de la Dixon IP Original a la geometría de la Dixon Registrada
        dixon_ip_resampled_data = resample_to_reference(dixon_ip_path, dixon_reg_path)
        dixon_ip_resampled      = process_slice_data(dixon_ip_resampled_data)

        print(f"  Shapes (físicos) → T1:{t1.shape}  Dixon_IP_Resampled:{dixon_ip_resampled.shape}  Dixon_R:{dixon_r.shape}")

        # ── PNG 1: Tres imágenes independientes ──────────────────────
        fig1, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig1.patch.set_facecolor("black")
        for ax, img, title in zip(axes,
                                   [t1, dixon_ip_resampled, dixon_r],
                                   ["T1", "Dixon IP Original (Resampled)", "Dixon IP (Registered)"]):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, color="white", fontsize=13, pad=6)
            ax.axis("off")
        plt.suptitle(f"{patient_dir.name} - Comparativa", color="white", fontsize=14)
        plt.tight_layout(pad=1.0)
        fig1.savefig(patient_dir / "comparativa_tres.png", dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig1)

        # ── PNG 2: Los Overlays ──────────────────────────────────────
        fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig2.patch.set_facecolor("black")
        
        # Ojo: ahora t1, dixon_ip_resampled y dixon_r tienen exactamente la misma matriz matemática
        for ax, ov, title in zip(axes,
                                  [make_overlay(t1, dixon_r),
                                   make_overlay(dixon_ip_resampled, dixon_r)],
                                  ["T1 / Dixon IP Reg",
                                   "Dixon IP Orig (Resampled) / Dixon IP Reg"]):
            ax.imshow(ov, vmin=0, vmax=1)
            ax.set_title(title, color="white", fontsize=13, pad=6)
            ax.axis("off")
        plt.suptitle(f"{patient_dir.name} - Overlays", color="white", fontsize=14)
        plt.tight_layout(pad=1.0)
        fig2.savefig(patient_dir / "overlays_same_resolution.png", dpi=150, bbox_inches="tight", facecolor="black")
        plt.close(fig2)

    except FileNotFoundError as e:
        print(f"Saltando {patient_dir.name}: {e}")
        continue
    except Exception as e:
        print(f"Error procesando {patient_dir.name}: {e}")
        continue