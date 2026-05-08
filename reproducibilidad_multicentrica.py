import os
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import combinations


# ============================================================================
# CONFIGURACIÓN DE CENTROS
# ============================================================================

CENTROS = {
    "UNAV": {
        "mascara_dir": "mascaras_sanos_UNAV",
        "datos_dir": "SANOS",
        "estructura_datos": "UNAV"
    },
    "ITALIA": {
        "mascara_dir": "mascaras_sanos_italia",
        "datos_dir": "RESPECT_CENTER01",
        "estructura_datos": "ITALIA"
    }
}

SESSION_SUFFIX = {1: "v2a", 2: "v2b", 3: "v3"}

# ============================================================================
# FUNCIONES DE CARGA
# ============================================================================

def cargar_mascaras_sesion(pac_id, n_ses, centro, mascara_dir):
    """Carga máscaras de un paciente en una sesión."""
    suf = SESSION_SUFFIX[n_ses]
    
    if centro == "ITALIA":
        # ITALIA: pac_id es un número como 501
        base = os.path.join(mascara_dir, f"{pac_id}_01_{suf}")
    else:  # UNAV
        # UNAV: pac_id es un número como 501
        # Las máscaras están como: mascaras_sanos_UNAV/501_02_v2a.nii.gz
        base = os.path.join(mascara_dir, f"{pac_id}_02_{suf}")

    ruta_rinon = f"{base}.nii.gz"
    ruta_pelvis = f"{base}_pelvis.nii.gz"
    ruta_cysts = f"{base}_cysts.nii.gz"

    if not os.path.exists(ruta_rinon):
        raise FileNotFoundError(f"Máscara de riñón no encontrada: {ruta_rinon}")

    img_rinon = sitk.ReadImage(ruta_rinon)
    arr_rinon = sitk.GetArrayFromImage(img_rinon).astype(np.uint8)

    arr_pelvis = np.zeros_like(arr_rinon)
    if os.path.exists(ruta_pelvis):
        arr_pelvis = (sitk.GetArrayFromImage(sitk.ReadImage(ruta_pelvis)) > 0).astype(np.uint8)

    arr_cysts = np.zeros_like(arr_rinon)
    if os.path.exists(ruta_cysts):
        arr_cysts = (sitk.GetArrayFromImage(sitk.ReadImage(ruta_cysts)) > 0).astype(np.uint8)

    return img_rinon, arr_rinon, arr_pelvis, arr_cysts


def mascara_base_existe(pac_id, n_ses_base, centro, mascara_dir):
    """Verifica si existe la máscara base."""
    suf = SESSION_SUFFIX[n_ses_base]
    
    if centro == "ITALIA":
        return os.path.exists(os.path.join(mascara_dir, f"{pac_id}_01_{suf}.nii.gz"))
    else:  # UNAV
        return os.path.exists(os.path.join(mascara_dir, f"{pac_id}_02_{suf}.nii.gz"))


def buscar_archivo_nii(carpeta, extensiones=("*.nii.gz", "*.nii"), contiene=None):
    """Busca archivo NIfTI en una carpeta."""
    for ext in extensiones:
        archivos = glob.glob(os.path.join(carpeta, ext))
        if archivos:
            if contiene:
                archivos_filtrados = [a for a in archivos if contiene in os.path.basename(a)]
                if archivos_filtrados:
                    return archivos_filtrados[0]
            else:
                return archivos[0]
    
    mensaje = f"No se encontraron archivos NIfTI en: {carpeta}"
    if contiene:
        mensaje = f"No se encontraron archivos NIfTI que contengan '{contiene}' en: {carpeta}"
    raise FileNotFoundError(mensaje)


def obtener_rutas_datos(pac_id, n_ses, centro, datos_dir, mascara_base):
    """Obtiene las rutas de los datos PDFF según el centro."""
    if centro == "ITALIA":
        base_folder = os.path.join(datos_dir, f"{pac_id}_01", f"0{n_ses}")
        
        # Imagen IP (siempre en imagen_IP_mask0{mascara_base})
        ruta_ip = buscar_archivo_nii(
            os.path.join(base_folder, f"imagen_IP_mask0{mascara_base}"))
        
        # Grasa (Dixon_b)
        if mascara_base == n_ses:
            # Sesión base
            ruta_grasa = buscar_archivo_nii(
                os.path.join(base_folder, r"coreg_dixons_t1\output\Dixon_b"))
        else:
            # Sesión registrada
            ruta_grasa = buscar_archivo_nii(
                os.path.join(base_folder, rf"coreg_dixon_con_dixon0{mascara_base}\fat_result"),
                contiene='aligned')
    
    else:  # UNAV
        # SANOS: pacientes con estructura {pac_id}_02, e.g. 501_02
        if isinstance(pac_id, str):
            pac_id_str = pac_id
        else:
            pac_id_str = f"{pac_id}_02"
        
        base_folder = os.path.join(datos_dir, pac_id_str, f"0{n_ses}")
        
        if mascara_base == n_ses:
            # Sesión base: IP y grasa en coreg_dixons_t1
            ruta_ip = buscar_archivo_nii(
                os.path.join(base_folder, r"coreg_dixons_t1\output\Dixon"))
            ruta_grasa = buscar_archivo_nii(
                os.path.join(base_folder, r"coreg_dixons_t1\output\Dixon_b"))
        else:
            # Sesión registrada: IP y grasa en dix_coreg_con_dix_0{N}
            ruta_ip = buscar_archivo_nii(
                os.path.join(base_folder, rf"dix_coreg_con_dix_0{mascara_base}\output\Dixon"))
            ruta_grasa = buscar_archivo_nii(
                os.path.join(base_folder, rf"dix_coreg_con_dix_0{mascara_base}\output\Dixon_b"))
    
    return ruta_ip, ruta_grasa


# ============================================================================
# FUNCIONES DE PROCESAMIENTO
# ============================================================================

def erosion(arr_rinon, img_ref, ruta_guardado, mascara_base):
    """Aplica erosión morfológica a la segmentación."""
    ruta_dir = f"resultados_{ruta_guardado}_mascara_0{mascara_base}/output_rinones"
    os.makedirs(ruta_dir, exist_ok=True)
    ruta_output = os.path.join(ruta_dir, "segmentacion_erosionada.nii.gz")

    data_final = np.zeros_like(arr_rinon)
    for etiqueta in [e for e in np.unique(arr_rinon) if e != 0]:
        binaria = (arr_rinon == etiqueta).astype(np.uint8)
        mask_sitk = sitk.GetImageFromArray(binaria)
        mask_sitk.CopyInformation(img_ref)
        eroded = sitk.BinaryErode(
            mask_sitk, kernelRadius=[2, 2, 0], kernelType=sitk.sitkBall
        )
        data_final[sitk.GetArrayFromImage(eroded) > 0] = etiqueta

    img_out = sitk.GetImageFromArray(data_final)
    img_out.CopyInformation(img_ref)
    sitk.WriteImage(img_out, ruta_output)

    return ruta_output, data_final


def mapa_pdff(ruta_grasa, ruta_ip, ruta_guardado, mascara_base):
    """Calcula el mapa de PDFF."""
    img_f = sitk.ReadImage(ruta_grasa)
    img_ip = sitk.ReadImage(ruta_ip)

    arr_f = sitk.GetArrayFromImage(img_f).astype(float)
    arr_ip = sitk.GetArrayFromImage(img_ip).astype(float)

    pdff_np = np.full_like(arr_f, np.nan)
    mask = arr_ip > 1e-10
    pdff_np[mask] = (arr_f[mask] * 100) / arr_ip[mask]

    pdff_itk = sitk.GetImageFromArray(pdff_np)
    pdff_itk.CopyInformation(img_f)

    ruta_dir = f"resultados_{ruta_guardado}_mascara_0{mascara_base}/output_rinones"
    os.makedirs(ruta_dir, exist_ok=True)
    ruta_output = os.path.join(ruta_dir, "mapa_pdff.nii.gz")
    sitk.WriteImage(pdff_itk, ruta_output)
    return ruta_output, pdff_np


def recortar_pdff_al_organo(erosion_np, ruta_pdff, ruta_guardado, mascara_base):
    """Recorta el PDFF al órgano."""
    ruta_dir = f"resultados_{ruta_guardado}_mascara_0{mascara_base}/output_rinones"
    os.makedirs(ruta_dir, exist_ok=True)

    pdff_total = sitk.GetArrayFromImage(sitk.ReadImage(ruta_pdff)).astype(np.float32)
    pdff_org = np.full_like(pdff_total, np.nan)
    pdff_org[erosion_np > 0] = pdff_total[erosion_np > 0]

    img_out = sitk.GetImageFromArray(pdff_org)
    img_out.CopyInformation(sitk.ReadImage(ruta_pdff))
    sitk.WriteImage(img_out, os.path.join(ruta_dir, "mapa_organo.nii.gz"))
    return pdff_org, img_out


def aislar_parenquima(pdff_org, erosion_np, pelvis_np, cysts_np, img_ref, ruta_guardado, mascara_base):
    """Aísla el parénquima."""
    ruta_dir = f"resultados_{ruta_guardado}_mascara_0{mascara_base}/output_rinones"
    os.makedirs(ruta_dir, exist_ok=True)

    par = np.full_like(pdff_org, np.nan)
    mask = (erosion_np > 0) & (pelvis_np == 0) & (cysts_np == 0)
    par[mask] = pdff_org[mask]

    img = sitk.GetImageFromArray(par)
    img.CopyInformation(img_ref)
    sitk.WriteImage(img, os.path.join(ruta_dir, "mapa_solo_parenquima.nii.gz"))
    return par


def aislar_seno(pdff_total, pelvis_np, img_ref, ruta_guardado, mascara_base):
    """Aísla el seno renal."""
    ruta_dir = f"resultados_{ruta_guardado}_mascara_0{mascara_base}/output_rinones"
    os.makedirs(ruta_dir, exist_ok=True)

    seno = np.full_like(pdff_total, np.nan)
    mask = (pelvis_np > 0)
    seno[mask] = pdff_total[mask]

    img = sitk.GetImageFromArray(seno)
    img.CopyInformation(img_ref)
    sitk.WriteImage(img, os.path.join(ruta_dir, "mapa_solo_seno.nii.gz"))
    return seno


def aislar_quistes(pdff_org, erosion_np, cysts_np, img_ref, ruta_guardado, mascara_base):
    """Aísla los quistes."""
    ruta_dir = f"resultados_{ruta_guardado}_mascara_0{mascara_base}/output_rinones"
    os.makedirs(ruta_dir, exist_ok=True)

    quistes = np.full_like(pdff_org, np.nan)
    mask = (erosion_np > 0) & (cysts_np > 0)
    quistes[mask] = pdff_org[mask]

    img = sitk.GetImageFromArray(quistes)
    img.CopyInformation(img_ref)
    sitk.WriteImage(img, os.path.join(ruta_dir, "mapa_solo_quistes.nii.gz"))
    return quistes


def generar_mascara_validacion(par, seno, quistes, img_ref, ruta_guardado, mascara_base):
    """Genera máscara de validación."""
    ruta_dir = f"resultados_{ruta_guardado}_mascara_0{mascara_base}/output_rinones"
    os.makedirs(ruta_dir, exist_ok=True)

    mask = np.zeros(par.shape, dtype=np.uint8)
    mask[~np.isnan(par)] = 1
    mask[~np.isnan(seno)] = 2
    mask[~np.isnan(quistes)] = 3

    img = sitk.GetImageFromArray(mask)
    img.CopyInformation(img_ref)
    sitk.WriteImage(img, os.path.join(ruta_dir, "mascara_validacion.nii.gz"))
    return img


def _histograma(valores, titulo, ruta_png):
    """Genera histograma."""
    limpios = valores[(valores >= 0) & (valores <= 100)]
    if limpios.size == 0:
        return
    plt.figure(figsize=(9, 5))
    plt.hist(limpios, bins=100, color="skyblue", edgecolor="black", alpha=0.7)
    med = np.median(limpios)
    plt.axvline(med, color="green", linestyle="--", label=f"Mediana: {med:.2f}%")
    plt.title(titulo)
    plt.xlabel("Grasa (%)")
    plt.ylabel("Vóxeles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=150)
    plt.close()


def asignar_lado_por_posicion(shape_array):
    """Asigna lado izquierdo/derecho según posición."""
    z_size, y_size, x_size = shape_array
    centro_x = x_size / 2.0
    
    x_indices = np.arange(x_size)
    x_grid = np.broadcast_to(x_indices[np.newaxis, np.newaxis, :], (z_size, y_size, x_size))
    
    mask_izq = x_grid < centro_x
    mask_der = x_grid >= centro_x
    
    return mask_izq, mask_der


def estadisticas_sesion(par, seno, quistes, erosion_np, img_ref,
                        id_suj, cod_pac, ruta_guardado, mascara_base):
    """Calcula estadísticas por sesión."""
    spacing = img_ref.GetSpacing()
    vol_voxel_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000

    mask_izq, mask_der = asignar_lado_por_posicion(par.shape)
    
    res = {}
    vals_plot = {}

    for nombre, mask_lado in [("IZQUIERDO", mask_izq), ("DERECHO", mask_der)]:
        vp = par[mask_lado & np.isfinite(par)]
        vs = seno[mask_lado & np.isfinite(seno)]
        vq = quistes[mask_lado & np.isfinite(quistes)]
        vox_tot = np.sum(erosion_np[mask_lado] > 0)

        def _stat(arr, p=None):
            if arr.size == 0:
                return 0
            return np.percentile(arr, p) if p is not None else np.mean(arr)

        res[nombre] = {
            "Vol_Total": round(vox_tot * vol_voxel_ml, 4),
            "Vol_Parenquima": round(vp.size * vol_voxel_ml, 4),
            "Vol_Seno": round(vs.size * vol_voxel_ml, 4),
            "Vol_Quistes": round(vq.size * vol_voxel_ml, 4),
            "Media_Parenquima": round(_stat(vp) if vp.size > 0 else 0, 4),
            "Mediana_Parenquima": round(np.median(vp) if vp.size > 0 else 0, 4),
            "P95_Parenquima": round(_stat(vp, 95) if vp.size > 0 else 0, 4),
            "IQR_Parenquima": round((_stat(vp, 75) - _stat(vp, 25)) if vp.size > 0 else 0, 4),
            "Std_Parenquima": round(np.std(vp) if vp.size > 0 else 0, 4),
            "Media_Seno": round(_stat(vs) if vs.size > 0 else 0, 4),
            "Mediana_Seno": round(np.median(vs) if vs.size > 0 else 0, 4),
            "P95_Seno": round(_stat(vs, 95) if vs.size > 0 else 0, 4),
            "IQR_Seno": round((_stat(vs, 75) - _stat(vs, 25)) if vs.size > 0 else 0, 4),
            "Std_Seno": round(np.std(vs) if vs.size > 0 else 0, 4),
            "Media_Quistes": round(_stat(vq) if vq.size > 0 else 0, 4),
            "Mediana_Quistes": round(np.median(vq) if vq.size > 0 else 0, 4),
        }
        vals_plot[nombre] = vp.flatten()

        dir_out = f"resultados_{ruta_guardado}_mascara_0{mascara_base}/output_rinones"
        _histograma(vp, f"PDFF Parénquima — {nombre} ({id_suj})",
                    os.path.join(dir_out, f"hist_{nombre}_parenquima.png"))
        _histograma(vs, f"PDFF Seno — {nombre} ({id_suj})",
                    os.path.join(dir_out, f"hist_{nombre}_seno.png"))
        if vq.size > 0:
            _histograma(vq, f"PDFF Quistes — {nombre} ({id_suj})",
                        os.path.join(dir_out, f"hist_{nombre}_quistes.png"))

    # Boxplot y violinplot
    if "DERECHO" in vals_plot and "IZQUIERDO" in vals_plot:
        vd, vi = vals_plot["DERECHO"], vals_plot["IZQUIERDO"]
        if vd.size > 0 and vi.size > 0:
            df_plot = pd.DataFrame({
                "PDFF (%)": np.concatenate([vd, vi]),
                "Riñón": ["Derecho"] * len(vd) + ["Izquierdo"] * len(vi),
            })
            ruta_dir = f"resultados_{ruta_guardado}_mascara_0{mascara_base}/output_rinones"
            sns.set_theme(style="whitegrid")

            plt.figure(figsize=(8, 5))
            sns.boxplot(x="Riñón", y="PDFF (%)", data=df_plot,
                        hue="Riñón", palette="Set2", legend=False)
            plt.title(f"PDFF Parénquima — {id_suj}")
            plt.tight_layout()
            plt.savefig(f"{ruta_dir}/boxplot.png", bbox_inches="tight", dpi=150)
            plt.close()

            plt.figure(figsize=(8, 5))
            sns.violinplot(x="Riñón", y="PDFF (%)", data=df_plot,
                           hue="Riñón", palette="Pastel1",
                           inner="quartile", legend=False)
            plt.title(f"Densidad PDFF Parénquima — {id_suj}")
            plt.tight_layout()
            plt.savefig(f"{ruta_dir}/violinplot.png", bbox_inches="tight", dpi=150)
            plt.close()

    return res


# ============================================================================
# FUNCIONES DE REPRODUCIBILIDAD
# ============================================================================

def calcular_cvws_dos_sesiones(datos_grupo, ses_a, ses_b):
    """Calcula coeficiente de variación intra-sujeto (CVws)."""
    diferencias = []
    todos_valores = []
    
    for pid, ses_vals in datos_grupo.items():
        val_a = ses_vals.get(ses_a)
        val_b = ses_vals.get(ses_b)
        
        if val_a is not None and val_b is not None:
            diferencias.append((val_a - val_b) ** 2)
            todos_valores.extend([val_a, val_b])
    
    if len(diferencias) == 0:
        return np.nan, 0
    
    n = len(diferencias)
    sum_sq = sum(diferencias)
    sd = np.sqrt(sum_sq / (2 * n))
    media_total = np.mean(todos_valores)
    
    cvws = round(100 * sd / media_total, 2) if media_total != 0 else np.nan
    return cvws, n


def calcular_icc_grupo(datos_grupo):
    """Calcula ICC(2,1) para el grupo."""
    ids = sorted(datos_grupo)
    sess = sorted({s for v in datos_grupo.values() for s in v})
    data = np.array(
        [[datos_grupo[p].get(s, np.nan) for s in sess] for p in ids],
        dtype=float
    )
    validos = [i for i, fila in enumerate(data) if np.sum(~np.isnan(fila)) >= 2]
    if len(validos) < 2:
        return np.nan
    data = data[validos]
    n, k = data.shape
    for j in range(k):
        nan_j = np.isnan(data[:, j])
        if nan_j.any():
            data[nan_j, j] = np.nanmean(data[:, j])
    grand = np.mean(data)
    row_m = np.mean(data, axis=1)
    col_m = np.mean(data, axis=0)
    SSr = k * np.sum((row_m - grand) ** 2)
    SSc = n * np.sum((col_m - grand) ** 2)
    SSt = np.sum((data - grand) ** 2)
    SSe = SSt - SSr - SSc
    MSr = SSr / (n - 1)
    MSc = SSc / (k - 1)
    MSe = SSe / ((n - 1) * (k - 1))
    if MSe == 0:
        return np.nan
    icc = (MSr - MSe) / (MSr + (k - 1) * MSe + k * (MSc - MSe) / n)
    return round(float(np.clip(icc, 0, 1)), 3)


def generar_bland_altman_grupo(datos_grupo, metrica_nombre, ruta_guardado):
    """Genera Bland-Altman plots."""
    sess = sorted({s for v in datos_grupo.values() for s in v})
    pares = list(combinations(sess, 2))
    if not pares or len(datos_grupo) < 2:
        return

    fig, axes = plt.subplots(1, len(pares),
                             figsize=(6 * len(pares), 5),
                             squeeze=False,
                             sharex=True,
                             sharey=True)
    for col, (sA, sB) in enumerate(pares):
        ax = axes[0][col]
        means, diffs, labels = [], [], []
        for pid, sv in datos_grupo.items():
            vA, vB = sv.get(sA), sv.get(sB)
            if vA is not None and vB is not None:
                means.append((vA + vB) / 2)
                diffs.append(vA - vB)
                labels.append(str(pid))
        if len(means) < 2:
            ax.set_title(f"{sA} vs {sB}: datos insuficientes")
            continue
        means = np.array(means)
        diffs = np.array(diffs)
        bias = np.mean(diffs)
        sd_diff = np.std(diffs, ddof=1)
        loa_sup = bias + 1.96 * sd_diff
        loa_inf = bias - 1.96 * sd_diff
        ax.scatter(means, diffs, s=90, alpha=0.75, color="steelblue", zorder=3)
        for m_v, d, lbl in zip(means, diffs, labels):
            ax.annotate(lbl, (m_v, d), textcoords="offset points",
                        xytext=(5, 4), fontsize=8)
        ax.axhline(bias, color="red", lw=2, ls="-",
                   label=f"Bias: {bias:.3f}")
        ax.axhline(loa_sup, color="orange", lw=1.5, ls="--",
                   label=f"+1.96 SD: {loa_sup:.3f}")
        ax.axhline(loa_inf, color="orange", lw=1.5, ls="--",
                   label=f"-1.96 SD: {loa_inf:.3f}")
        ax.axhline(0, color="gray", lw=1, ls=":")
        ax.set_xlabel(f"Promedio ({sA}+{sB})/2", fontsize=11)
        ax.set_ylabel(f"Diferencia ({sA}-{sB})", fontsize=11)
        ax.set_title(f"Bland-Altman: {sA} vs {sB}\n{metrica_nombre}",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    ruta_dir = os.path.join(ruta_guardado, "reproducibilidad")
    os.makedirs(ruta_dir, exist_ok=True)
    plt.savefig(os.path.join(ruta_dir, f"bland_altman_{metrica_nombre}.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================================
# FUNCIONES DE DETECCIÓN DE PACIENTES
# ============================================================================

def detectar_pacientes_automaticamente(centro):
    """Detecta automáticamente los pacientes en un centro."""
    pacientes = []
    
    if centro == "UNAV":
        centro_dir = "SANOS"
        mascara_dir = "mascaras_sanos_UNAV"
    else:  # ITALIA
        centro_dir = "RESPECT_CENTER01"
        mascara_dir = "mascaras_sanos_italia"
    
    try:
        if not os.path.exists(centro_dir):
            print(f"⚠️  Carpeta no encontrada: {centro_dir}")
            return pacientes
        
        for item in os.listdir(centro_dir):
            item_path = os.path.join(centro_dir, item)
            
            if not os.path.isdir(item_path):
                continue
            
            # Verificar si existe máscara para este paciente
            found_mascara = False
            
            if centro == "ITALIA":
                # Para ITALIA: buscar mascaras como "501_01_v2a.nii.gz"
                pac_id_num = item.split("_")[0]  # Extrae "501" de "501_01"
                for suffix in SESSION_SUFFIX.values():
                    mascara_file = os.path.join(mascara_dir, f"{pac_id_num}_01_{suffix}.nii.gz")
                    if os.path.exists(mascara_file):
                        found_mascara = True
                        break
                
                if found_mascara:
                    pacientes.append({
                        "id": int(pac_id_num),
                        "codigo": f"P{pac_id_num}",
                        "nombre": f"Paciente ITALIA {pac_id_num}",
                        "mascara_sesion": 2
                    })
            else:  # UNAV
                # Para UNAV: buscar mascaras como "501_v2a.nii.gz" 
                # items son como "501_02", "502_02", etc.
                pac_id_num = item.split("_")[0]  # Extrae "501" de "501_02"
                for suffix in SESSION_SUFFIX.values():
                    mascara_file = os.path.join(mascara_dir, f"{pac_id_num}_{suffix}.nii.gz")
                    if os.path.exists(mascara_file):
                        found_mascara = True
                        break
                
                if found_mascara:
                    pacientes.append({
                        "id": item,  # "501_02"
                        "codigo": f"P{pac_id_num}",
                        "nombre": f"Paciente UNAV {pac_id_num}",
                        "mascara_sesion": 2
                    })
    
    except Exception as e:
        print(f"Error detectando pacientes en {centro}: {e}")
    
    return sorted(pacientes, key=lambda x: str(x["id"]))


def generar_comparacion_multicentrica(datos_unav, datos_italia, ruta_salida):
    """Genera análisis de reproducibilidad multicéntrica (UNAV vs ITALIA)."""
    print(f"\n{'='*70}")
    print("ANÁLISIS MULTICÉNTRICO (UNAV vs ITALIA)")
    print(f"{'='*70}\n")
    
    os.makedirs(ruta_salida, exist_ok=True)
    
    # Recopilar todas las métricas por centro
    METRICAS = ["Media_Parenquima", "Mediana_Parenquima", "Media_Seno", 
                "Mediana_Seno", "Vol_Parenquima", "Vol_Seno"]
    
    # Comparar media de cada métrica entre centros
    tabla_comparacion = [["METRICA", "LADO", "UNAV_Media", "UNAV_Std", 
                          "ITALIA_Media", "ITALIA_Std", "Diferencia (%)"]]
    
    for metrica in METRICAS:
        for lado in ["DERECHO", "IZQUIERDO"]:
            # Recopilar valores de UNAV
            vals_unav = []
            for pac_data in datos_unav.values():
                for ses_data in pac_data.values():
                    if lado in ses_data and metrica in ses_data[lado]:
                        vals = ses_data[lado][metrica]
                        if vals and not np.isnan(vals):
                            vals_unav.append(vals)
            
            # Recopilar valores de ITALIA
            vals_italia = []
            for pac_data in datos_italia.values():
                for ses_data in pac_data.values():
                    if lado in ses_data and metrica in ses_data[lado]:
                        vals = ses_data[lado][metrica]
                        if vals and not np.isnan(vals):
                            vals_italia.append(vals)
            
            if vals_unav and vals_italia:
                media_unav = np.mean(vals_unav)
                std_unav = np.std(vals_unav)
                media_italia = np.mean(vals_italia)
                std_italia = np.std(vals_italia)
                
                # Diferencia porcentual
                if media_unav != 0:
                    diff_pct = 100 * (media_italia - media_unav) / media_unav
                else:
                    diff_pct = 0
                
                tabla_comparacion.append([
                    metrica, lado,
                    f"{media_unav:.4f}", f"{std_unav:.4f}",
                    f"{media_italia:.4f}", f"{std_italia:.4f}",
                    f"{diff_pct:.2f}%"
                ])
    
    # Guardar a Excel
    df = pd.DataFrame(tabla_comparacion[1:], columns=tabla_comparacion[0])
    df.to_excel(os.path.join(ruta_salida, "comparacion_multicentrica.xlsx"), index=False)
    print("✓ Comparación multicéntrica guardada")


# ============================================================================
# FUNCIONES DE REPORTES
# ============================================================================

def generar_excel_por_paciente(datos_sesiones_pac, paciente_id, ruta_excel):
    """Genera Excel con datos por paciente."""
    METRICAS = [
        "Vol_Total", "Vol_Parenquima", "Vol_Seno", "Vol_Quistes",
        "Media_Parenquima", "Mediana_Parenquima", "P95_Parenquima",
        "IQR_Parenquima", "Std_Parenquima",
        "Media_Seno", "Mediana_Seno", "P95_Seno", "IQR_Seno", "Std_Seno",
        "Media_Quistes", "Mediana_Quistes",
    ]
    sesiones = sorted(datos_sesiones_pac)

    with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:
        for ses in sesiones:
            datos = datos_sesiones_pac[ses]
            tabla = [["METRICA", "RINON DERECHO", "RINON IZQUIERDO"]]
            for m in METRICAS:
                tabla.append([
                    m,
                    datos.get("DERECHO", {}).get(m, ""),
                    datos.get("IZQUIERDO", {}).get(m, ""),
                ])
            pd.DataFrame(tabla).to_excel(
                writer, sheet_name=f"Sesion_{ses}", index=False, header=False)


def generar_excel_grupo(todos_datos, ruta_excel, mascara_base):
    """Genera Excel con reproducibilidad del grupo."""
    METRICAS_DATOS = ["Media_Parenquima", "Mediana_Parenquima",
                      "Vol_Parenquima", "Vol_Total", "Vol_Seno"]
    METRICAS_CVws = ["Media_Parenquima", "Media_Seno", "Mediana_Parenquima", "Mediana_Seno"]
    ids = sorted(todos_datos)
    sesiones = sorted({s for p in todos_datos.values() for s in p})

    with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:
        for lado in ["DERECHO", "IZQUIERDO"]:
            cols = ["Paciente"] + [f"{s}_{m}" for s in sesiones for m in METRICAS_DATOS]
            filas = []
            for pid in ids:
                fila = [pid]
                for s in sesiones:
                    for m in METRICAS_DATOS:
                        fila.append(
                            todos_datos.get(pid, {}).get(s, {})
                                       .get(lado, {}).get(m, ""))
                filas.append(fila)
            pd.DataFrame(filas, columns=cols).to_excel(
                writer, sheet_name=f"Datos_{lado[:3]}", index=False)

        tabla = [["METRICA", "LADO", "CVws S1 vs S2 (%)", "CVws S1 vs S3 (%)", "ICC(2,1)"]]
        n_s1_s2_final = 0
        n_s1_s3_final = 0
        for m in METRICAS_CVws:
            for lado in ["DERECHO", "IZQUIERDO"]:
                grupo = {}
                for pid in ids:
                    sv = {
                        s: todos_datos[pid][s][lado][m]
                        for s in sesiones
                        if lado in todos_datos.get(pid, {}).get(s, {})
                        and m in todos_datos[pid][s][lado]
                    }
                    if len(sv) >= 2:
                        grupo[pid] = sv
                if len(grupo) < 2:
                    continue

                cvws_s1_s2, n_s1_s2 = calcular_cvws_dos_sesiones(grupo, "S1", "S2")
                cvws_s1_s3, n_s1_s3 = calcular_cvws_dos_sesiones(grupo, "S1", "S3")
                n_s1_s2_final = n_s1_s2 if n_s1_s2 > 0 else n_s1_s2_final
                n_s1_s3_final = n_s1_s3 if n_s1_s3 > 0 else n_s1_s3_final

                icc = calcular_icc_grupo(grupo)
                tabla.append([m, lado, cvws_s1_s2, cvws_s1_s3, icc])

        tabla.append(["", "", "", "", ""])
        tabla.append(["Num Pacientes", "", n_s1_s2_final, n_s1_s3_final, ""])

        pd.DataFrame(tabla).to_excel(
            writer, sheet_name="Reproducibilidad_Grupo",
            index=False, header=False)


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def procesar_centro(centro_nombre, pacientes, mascara_base_por_pac=None):
    """Procesa un centro completo."""
    print(f"\n{'='*70}")
    print(f"PROCESANDO CENTRO: {centro_nombre.upper()}")
    print(f"{'='*70}\n")
    
    centro_config = CENTROS[centro_nombre]
    mascara_dir = centro_config["mascara_dir"]
    datos_dir = centro_config["datos_dir"]
    
    todos_datos = {}
    
    for pac in pacientes:
        pac_id = pac["id"]
        cod_pac = pac["codigo"]
        id_sujeto = pac["nombre"]
        mascara_base = pac.get("mascara_sesion", 3)
        
        print(f"Procesando {centro_nombre} - Paciente {pac_id} con máscara S{mascara_base}")
        
        datos_pac = {}
        
        if not mascara_base_existe(pac_id, mascara_base, centro_nombre, mascara_dir):
            print(f"  ⚠️  Sin máscara de la sesión base {mascara_base}")
            continue
        
        try:
            img_ref, arr_rinon_base, arr_pelvis_base, arr_cysts_base = \
                cargar_mascaras_sesion(pac_id, mascara_base, centro_nombre, mascara_dir)
        except FileNotFoundError as e:
            print(f"  ❌ Error al cargar máscara base: {e}")
            continue
        
        for n_ses in [1, 2, 3]:
            try:
                ruta_g = f"{centro_nombre}_{pac_id}_0{n_ses}"
                ruta_ip, ruta_grasa = obtener_rutas_datos(
                    pac_id, n_ses, centro_nombre, datos_dir, mascara_base)
                
                os.makedirs(f"resultados_{ruta_g}_mascara_0{mascara_base}", exist_ok=True)
                
                img_grasa_ref = sitk.ReadImage(ruta_grasa)
                
                # ITALIA: detectar flip automáticamente según orientación de la imagen
                # UNAV: sin flip, usa img_ref (de la máscara) como referencia espacial
                if centro_nombre == "ITALIA":
                    needs_flip = img_grasa_ref.GetDirection()[4] < 0
                    if needs_flip:
                        print(f"  Flipeando máscaras en eje Y (Imagen detectada como INVERTIDA)...")
                        arr_rinon  = np.flip(arr_rinon_base, axis=1)
                        arr_pelvis = np.flip(arr_pelvis_base, axis=1)
                        arr_cysts  = np.flip(arr_cysts_base, axis=1)
                    else:
                        print(f"  Usando máscaras originales (Imagen detectada como DERECHA)...")
                        arr_rinon  = arr_rinon_base
                        arr_pelvis = arr_pelvis_base
                        arr_cysts  = arr_cysts_base
                    img_proc = img_grasa_ref  # referencia espacial: imagen de grasa
                else:  # UNAV
                    arr_rinon  = arr_rinon_base
                    arr_pelvis = arr_pelvis_base
                    arr_cysts  = arr_cysts_base
                    img_proc = img_ref  # referencia espacial: imagen de la máscara
                
                _, erosion_np = erosion(arr_rinon, img_proc, ruta_g, mascara_base)
                ruta_pdff, pdff_total = mapa_pdff(ruta_grasa, ruta_ip, ruta_g, mascara_base)
                pdff_org, _ = recortar_pdff_al_organo(erosion_np, ruta_pdff, ruta_g, mascara_base)
                
                par_np = aislar_parenquima(pdff_org, erosion_np,
                                           arr_pelvis, arr_cysts, img_proc, ruta_g, mascara_base)
                seno_np = aislar_seno(pdff_total, arr_pelvis, img_proc, ruta_g, mascara_base)
                quistes_np = aislar_quistes(pdff_org, erosion_np,
                                            arr_cysts, img_proc, ruta_g, mascara_base)
                generar_mascara_validacion(par_np, seno_np, quistes_np,
                                           img_proc, ruta_g, mascara_base)
                
                res = estadisticas_sesion(
                    par_np, seno_np, quistes_np, erosion_np, img_proc,
                    id_sujeto, cod_pac, ruta_g, mascara_base)
                
                datos_pac[f"S{n_ses}"] = res
                print(f"  ✓ Sesión {n_ses} completada")
            
            except Exception as exc:
                print(f"  ❌ Error en sesión {n_ses}: {exc}")
        
        if datos_pac:
            ruta_dir_pac = f"resultados_{centro_nombre}_{pac_id}_reproducibilidad_mascara_0{mascara_base}"
            os.makedirs(ruta_dir_pac, exist_ok=True)
            generar_excel_por_paciente(
                datos_pac, id_sujeto,
                os.path.join(ruta_dir_pac, f"paciente_{pac_id}_mascara_0{mascara_base}.xlsx"))
            todos_datos[pac_id] = datos_pac
            print(f"  ✓ Paciente {pac_id} completado\n")
    
    # Generar reportes de reproducibilidad por centro
    grupo_valido = {pid: d for pid, d in todos_datos.items() if len(d) >= 2}
    
    if len(grupo_valido) < 2:
        print(f"  ⚠️  Se necesitan al menos 2 pacientes con 2+ sesiones para reproducibilidad.")
    else:
        ruta_grupo = f"resultados_{centro_nombre}_grupo_mascara_0{mascara_base}"
        os.makedirs(ruta_grupo, exist_ok=True)
        
        METRICAS_BA = ["Media_Parenquima", "Mediana_Parenquima", "Media_Seno", "Mediana_Seno"]
        for lado in ["DERECHO", "IZQUIERDO"]:
            for m in METRICAS_BA:
                grupo_m = {}
                for pid, datos_pac in grupo_valido.items():
                    sv = {s: datos_pac[s][lado][m]
                          for s in datos_pac
                          if lado in datos_pac[s] and m in datos_pac[s][lado]}
                    if len(sv) >= 2:
                        grupo_m[pid] = sv
                if len(grupo_m) >= 2:
                    generar_bland_altman_grupo(grupo_m, f"{m}_{lado}", ruta_grupo)
        
        generar_excel_grupo(
            grupo_valido,
            os.path.join(ruta_grupo, f"reproducibilidad_{centro_nombre}_GRUPO.xlsx"),
            mascara_base)
        print(f"  ✓ Excel de reproducibilidad por centro guardado\n")
    
    return todos_datos


if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("ANÁLISIS DE REPRODUCIBILIDAD MULTICÉNTRICA")
    print("="*70)
    print("Centros: UNAV (SANOS) e ITALIA (RESPECT_CENTER01)")
    print()
    
    
    PACIENTES_UNAV = [
        {"id": 501, "codigo": "P501", "nombre": "Paciente 501", "mascara_sesion": 1},
        {"id": 504, "codigo": "P504", "nombre": "Paciente 504", "mascara_sesion": 2},
        {"id": 506, "codigo": "P506", "nombre": "Paciente 506", "mascara_sesion": 3},
        {"id": 507, "codigo": "P507", "nombre": "Paciente 507", "mascara_sesion": 1},
        {"id": 508, "codigo": "P508", "nombre": "Paciente 508", "mascara_sesion": 2},
        {"id": 510, "codigo": "P510", "nombre": "Paciente 510", "mascara_sesion": 1},
        {"id": 511, "codigo": "P511", "nombre": "Paciente 511", "mascara_sesion": 1},
        {"id": 513, "codigo": "P513", "nombre": "Paciente 513", "mascara_sesion": 2},
        {"id": 514, "codigo": "P514", "nombre": "Paciente 514", "mascara_sesion": 3},
        {"id": 516, "codigo": "P516", "nombre": "Paciente 516", "mascara_sesion": 3},
    ]
    
    PACIENTES_ITALIA = [
        {"id": 501, "codigo": "P501", "nombre": "Paciente 501", "mascara_sesion": 2},
        {"id": 506, "codigo": "P506", "nombre": "Paciente 506", "mascara_sesion": 3},
        {"id": 508, "codigo": "P508", "nombre": "Paciente 508", "mascara_sesion": 1},
        {"id": 511, "codigo": "P511", "nombre": "Paciente 511", "mascara_sesion": 2},
        {"id": 512, "codigo": "P512", "nombre": "Paciente 512", "mascara_sesion": 2},
        {"id": 513, "codigo": "P513", "nombre": "Paciente 513", "mascara_sesion": 2},
        {"id": 514, "codigo": "P514", "nombre": "Paciente 514", "mascara_sesion": 2},
        {"id": 516, "codigo": "P516", "nombre": "Paciente 516", "mascara_sesion": 1},
        {"id": 518, "codigo": "P518", "nombre": "Paciente 518", "mascara_sesion": 2},
        {"id": 520, "codigo": "P520", "nombre": "Paciente 520", "mascara_sesion": 2},
        {"id": 522, "codigo": "P522", "nombre": "Paciente 522", "mascara_sesion": 2},
    ]
    
    if not PACIENTES_UNAV and not PACIENTES_ITALIA:
        print("⚠️  No hay pacientes configurados.")
        exit(1)
    
    print(f"  ✓ {len(PACIENTES_UNAV)} pacientes configurados en UNAV")
    print(f"  ✓ {len(PACIENTES_ITALIA)} pacientes configurados en ITALIA")
    
    # ────────────────────────────────────────────────────────────────
    # PROCESAR CADA CENTRO
    # ────────────────────────────────────────────────────────────────
    
    datos_unav = procesar_centro("UNAV", PACIENTES_UNAV) if PACIENTES_UNAV else {}
    datos_italia = procesar_centro("ITALIA", PACIENTES_ITALIA) if PACIENTES_ITALIA else {}
    
    # ────────────────────────────────────────────────────────────────
    # ANÁLISIS MULTICÉNTRICO: comparación básica entre centros
    # ────────────────────────────────────────────────────────────────

    if datos_unav and datos_italia:
        generar_comparacion_multicentrica(datos_unav, datos_italia, "resultados_multicentricos")
    elif datos_italia:
        print("\n✓ Procesados datos de ITALIA sin comparación (UNAV vacío)")
    elif datos_unav:
        print("\n✓ Procesados datos de UNAV sin comparación (ITALIA vacío)")
    else:
        print("\n⚠️  No se procesaron datos")

    # ────────────────────────────────────────────────────────────────
    # REPRODUCIBILIDAD GLOBAL: todos los pacientes de ambos centros
    # como una única cohorte (CVws S1vsS2, S1vsS3, ICC)
    # ────────────────────────────────────────────────────────────────

    todos_combinados = {}
    for pid, datos in datos_unav.items():
        todos_combinados[f"UNAV_{pid}"] = datos
    for pid, datos in datos_italia.items():
        todos_combinados[f"ITALIA_{pid}"] = datos

    grupo_combinado = {pid: d for pid, d in todos_combinados.items() if len(d) >= 2}

    if len(grupo_combinado) >= 2:
        print("\n" + "="*70)
        print("REPRODUCIBILIDAD GLOBAL (UNAV + ITALIA combinados)")
        print("="*70)

        ruta_global = "resultados_cohorte_global"
        os.makedirs(ruta_global, exist_ok=True)

        # Bland-Altman para la cohorte global
        METRICAS_BA = ["Media_Parenquima", "Mediana_Parenquima", "Media_Seno", "Mediana_Seno"]
        for lado in ["DERECHO", "IZQUIERDO"]:
            for m in METRICAS_BA:
                grupo_m = {}
                for pid, datos_pac in grupo_combinado.items():
                    sv = {s: datos_pac[s][lado][m]
                          for s in datos_pac
                          if lado in datos_pac[s] and m in datos_pac[s][lado]}
                    if len(sv) >= 2:
                        grupo_m[pid] = sv
                if len(grupo_m) >= 2:
                    generar_bland_altman_grupo(grupo_m, f"{m}_{lado}", ruta_global)

        # Excel con CVws e ICC globales (misma función que por centro)
        # mascara_base no se usa en generar_excel_grupo, se pasa un valor neutro
        mascara_base_ref = list(grupo_combinado.values())[0]  # cualquier valor válido
        generar_excel_grupo(
            grupo_combinado,
            os.path.join(ruta_global, "reproducibilidad_COHORTE_GLOBAL.xlsx"),
            mascara_base=None)
        print(f"  ✓ Excel de cohorte global guardado en: {ruta_global}/")
    else:
        print("\n⚠️  No hay suficientes pacientes con 2+ sesiones para el análisis global.")

    print("\n" + "="*70)
    print("✓ PROCESO COMPLETADO")
    print("="*70 + "\n")
