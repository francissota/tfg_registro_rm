import os
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import combinations


FRANCIS_DIR    = "Francis"          # carpeta con las máscaras de Francis
SESSION_SUFFIX = {1: "v2a", 2: "v2b", 3: "v3"}


def cargar_mascaras_sesion(pac_id, n_ses):
    
    suf  = SESSION_SUFFIX[n_ses]
    base = os.path.join(FRANCIS_DIR, f"{pac_id}_02_{suf}")

    ruta_rinon  = f"{base}.nii.gz"
    ruta_pelvis = f"{base}_pelvis.nii.gz"
    ruta_cysts  = f"{base}_cysts.nii.gz"

    if not os.path.exists(ruta_rinon):
        raise FileNotFoundError(f"Máscara de riñón no encontrada: {ruta_rinon}")

    img_rinon  = sitk.ReadImage(ruta_rinon)
    arr_rinon  = sitk.GetArrayFromImage(img_rinon).astype(np.uint8)

    arr_pelvis = np.zeros_like(arr_rinon)
    if os.path.exists(ruta_pelvis):
        arr_pelvis = (sitk.GetArrayFromImage(sitk.ReadImage(ruta_pelvis)) > 0).astype(np.uint8)

    arr_cysts = np.zeros_like(arr_rinon)
    if os.path.exists(ruta_cysts):
        arr_cysts = (sitk.GetArrayFromImage(sitk.ReadImage(ruta_cysts)) > 0).astype(np.uint8)

    return img_rinon, arr_rinon, arr_pelvis, arr_cysts


def mascara_base_existe(pac_id, n_ses_base):
    suf = SESSION_SUFFIX[n_ses_base]
    return os.path.exists(
        os.path.join(FRANCIS_DIR, f"{pac_id}_02_{suf}.nii.gz")
    )


def erosion(arr_rinon, img_ref, ruta_guardado):

    ruta_dir = f"resultados_{ruta_guardado}_mascara_0{mascara_base}/output_rinones"
    os.makedirs(ruta_dir, exist_ok=True)
    ruta_output = os.path.join(ruta_dir, "segmentacion_erosionada.nii.gz")

    data_final = np.zeros_like(arr_rinon)
    for etiqueta in [e for e in np.unique(arr_rinon) if e != 0]:
        binaria   = (arr_rinon == etiqueta).astype(np.uint8)
        mask_sitk = sitk.GetImageFromArray(binaria)
        mask_sitk.CopyInformation(img_ref)
        eroded    = sitk.BinaryErode(
            mask_sitk, kernelRadius=[2, 2, 0], kernelType=sitk.sitkBall
        )
        data_final[sitk.GetArrayFromImage(eroded) > 0] = etiqueta

    img_out = sitk.GetImageFromArray(data_final)
    img_out.CopyInformation(img_ref)
    sitk.WriteImage(img_out, ruta_output)

    return ruta_output, data_final


def mapa_pdff(ruta_grasa, ruta_ip, ruta_guardado):
    img_f  = sitk.ReadImage(ruta_grasa)
    img_ip = sitk.ReadImage(ruta_ip)

    arr_f  = sitk.GetArrayFromImage(img_f).astype(float)
    arr_ip = sitk.GetArrayFromImage(img_ip).astype(float)

    pdff_np      = np.full_like(arr_f, np.nan)
    mask         = arr_ip > 1e-10
    pdff_np[mask] = (arr_f[mask] * 100) / arr_ip[mask]

    pdff_itk = sitk.GetImageFromArray(pdff_np)
    pdff_itk.CopyInformation(img_f)

    ruta_dir = f"resultados_{ruta_guardado}_mascara_0{mascara_base}/output_rinones"
    os.makedirs(ruta_dir, exist_ok=True)
    ruta_output = os.path.join(ruta_dir, "mapa_pdff.nii.gz")
    sitk.WriteImage(pdff_itk, ruta_output)
    return ruta_output, pdff_np


def recortar_pdff_al_organo(erosion_np, ruta_pdff, ruta_guardado):

    ruta_dir   = f"resultados_{ruta_guardado}_mascara_0{mascara_base}/output_rinones"
    os.makedirs(ruta_dir, exist_ok=True)

    pdff_total = sitk.GetArrayFromImage(sitk.ReadImage(ruta_pdff)).astype(np.float32)
    pdff_org   = np.full_like(pdff_total, np.nan)
    pdff_org[erosion_np > 0] = pdff_total[erosion_np > 0]

    img_out = sitk.GetImageFromArray(pdff_org)
    img_out.CopyInformation(sitk.ReadImage(ruta_pdff))
    sitk.WriteImage(img_out, os.path.join(ruta_dir, "mapa_organo.nii.gz"))
    return pdff_org, img_out


def aislar_parenquima(pdff_org, erosion_np, pelvis_np, cysts_np, img_ref, ruta_guardado):
    """Parénquima = riñón erosionado − pelvis renal − quistes."""
    ruta_dir = f"resultados_{ruta_guardado}_mascara_0{mascara_base}/output_rinones"
    os.makedirs(ruta_dir, exist_ok=True)

    par  = np.full_like(pdff_org, np.nan)
    mask = (erosion_np > 0) & (pelvis_np == 0) & (cysts_np == 0)
    par[mask] = pdff_org[mask]

    img = sitk.GetImageFromArray(par)
    img.CopyInformation(img_ref)
    sitk.WriteImage(img, os.path.join(ruta_dir, "mapa_solo_parenquima.nii.gz"))
    return par


def aislar_seno(pdff_total, pelvis_np, img_ref, ruta_guardado):
    ruta_dir = f"resultados_{ruta_guardado}_mascara_0{mascara_base}/output_rinones"
    os.makedirs(ruta_dir, exist_ok=True)

    seno = np.full_like(pdff_total, np.nan)
    mask = (pelvis_np > 0)
    seno[mask] = pdff_total[mask]

    img = sitk.GetImageFromArray(seno)
    img.CopyInformation(img_ref)
    sitk.WriteImage(img, os.path.join(ruta_dir, "mapa_solo_seno.nii.gz"))
    return seno


def aislar_quistes(pdff_org, erosion_np, cysts_np, img_ref, ruta_guardado):

    ruta_dir = f"resultados_{ruta_guardado}_mascara_0{mascara_base}/output_rinones"
    os.makedirs(ruta_dir, exist_ok=True)

    quistes = np.full_like(pdff_org, np.nan)
    mask    = (erosion_np > 0) & (cysts_np > 0)
    quistes[mask] = pdff_org[mask]

    img = sitk.GetImageFromArray(quistes)
    img.CopyInformation(img_ref)
    sitk.WriteImage(img, os.path.join(ruta_dir, "mapa_solo_quistes.nii.gz"))
    return quistes


def generar_mascara_validacion(par, seno, quistes, img_ref, ruta_guardado):
    
    ruta_dir = f"resultados_{ruta_guardado}_mascara_0{mascara_base}/output_rinones"
    os.makedirs(ruta_dir, exist_ok=True)

    mask = np.zeros(par.shape, dtype=np.uint8)
    mask[~np.isnan(par)]     = 1
    mask[~np.isnan(seno)]    = 2
    mask[~np.isnan(quistes)] = 3

    img = sitk.GetImageFromArray(mask)
    img.CopyInformation(img_ref)
    sitk.WriteImage(img, os.path.join(ruta_dir, "mascara_validacion.nii.gz"))
    return img


def buscar_archivo_nii(carpeta, extensiones=("*.nii.gz", "*.nii")):
    for ext in extensiones:
        archivos = glob.glob(os.path.join(carpeta, ext))
        if archivos:
            return archivos[0]
    raise FileNotFoundError(f"No se encontraron archivos NIfTI en: {carpeta}")


def _histograma(valores, titulo, ruta_png):
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
    
    z_size, y_size, x_size = shape_array
    centro_x = x_size / 2.0
    
    x_indices = np.arange(x_size)
    x_grid = np.broadcast_to(x_indices[np.newaxis, np.newaxis, :], (z_size, y_size, x_size))
    
    mask_izq = x_grid < centro_x
    mask_der = x_grid >= centro_x
    
    return mask_izq, mask_der


def estadisticas_sesion(par, seno, quistes, erosion_np, img_ref,
                        id_suj, cod_pac, ruta_guardado):

    spacing      = img_ref.GetSpacing()
    vol_voxel_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000

    mask_izq, mask_der = asignar_lado_por_posicion(par.shape)
    
    res       = {}
    vals_plot = {}

    for nombre, mask_lado in [("IZQUIERDO", mask_izq), ("DERECHO", mask_der)]:

        vp = par    [mask_lado & np.isfinite(par)]      
        vs = seno   [mask_lado & np.isfinite(seno)]      
        vq = quistes[mask_lado & np.isfinite(quistes)]  
        vox_tot = np.sum(erosion_np[mask_lado] > 0)    

        def _stat(arr, p=None):
            if arr.size == 0:
                return 0
            return np.percentile(arr, p) if p is not None else np.mean(arr)

        res[nombre] = {
            "Vol_Total"          : round(vox_tot * vol_voxel_ml, 4),
            "Vol_Parenquima"     : round(vp.size  * vol_voxel_ml, 4),
            "Vol_Seno"           : round(vs.size  * vol_voxel_ml, 4),
            "Vol_Quistes"        : round(vq.size  * vol_voxel_ml, 4),
            "Media_Parenquima"   : round(_stat(vp)          if vp.size > 0 else 0, 4),
            "Mediana_Parenquima" : round(np.median(vp)      if vp.size > 0 else 0, 4),
            "P95_Parenquima"     : round(_stat(vp, 95)      if vp.size > 0 else 0, 4),
            "IQR_Parenquima"     : round((_stat(vp,75)-_stat(vp,25)) if vp.size > 0 else 0, 4),
            "Std_Parenquima"     : round(np.std(vp)         if vp.size > 0 else 0, 4),
            "Media_Seno"         : round(_stat(vs)          if vs.size > 0 else 0, 4),
            "Mediana_Seno"       : round(np.median(vs)      if vs.size > 0 else 0, 4),
            "P95_Seno"           : round(_stat(vs, 95)      if vs.size > 0 else 0, 4),
            "IQR_Seno"           : round((_stat(vs,75)-_stat(vs,25)) if vs.size > 0 else 0, 4),
            "Std_Seno"           : round(np.std(vs)         if vs.size > 0 else 0, 4),
            "Media_Quistes"      : round(_stat(vq)          if vq.size > 0 else 0, 4),
            "Mediana_Quistes"    : round(np.median(vq)      if vq.size > 0 else 0, 4),
        }
        vals_plot[nombre] = vp.flatten() #convertimos el array a 1D para poder hacer graficos

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
                "Riñón"   : ["Derecho"] * len(vd) + ["Izquierdo"] * len(vi),
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

"""
def calcular_cvws_grupo(datos_grupo):
    sum_sq_total, n_pares_total = 0.0, 0
    all_vals = []
    for ses_vals in datos_grupo.values():
        vals = [ses_vals[s] for s in sorted(ses_vals)]
        all_vals.extend(vals)
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                sum_sq_total  += (vals[i] - vals[j]) ** 2
                n_pares_total += 1
    if n_pares_total == 0:
        return np.nan
    sd_pooled  = np.sqrt(sum_sq_total / (2 * n_pares_total))
    grand_mean = np.mean(all_vals)
    return round(100 * sd_pooled / grand_mean, 2) if grand_mean != 0 else np.nan
"""

def calcular_cvws_dos_sesiones(datos_grupo, ses_a, ses_b):
    
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
    ids  = sorted(datos_grupo)
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
    SSr   = k * np.sum((row_m - grand) ** 2)
    SSc   = n * np.sum((col_m - grand) ** 2)
    SSt   = np.sum((data - grand) ** 2)
    SSe   = SSt - SSr - SSc
    MSr   = SSr / (n - 1)
    MSc   = SSc / (k - 1)
    MSe   = SSe / ((n - 1) * (k - 1))
    if MSe == 0:
        return np.nan
    icc = (MSr - MSe) / (MSr + (k - 1) * MSe + k * (MSc - MSe) / n)
    return round(float(np.clip(icc, 0, 1)), 3)


def generar_bland_altman_grupo(datos_grupo, metrica_nombre, ruta_guardado):
    sess  = sorted({s for v in datos_grupo.values() for s in v})
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
        means   = np.array(means)
        diffs   = np.array(diffs)
        bias    = np.mean(diffs)
        sd_diff = np.std(diffs, ddof=1)
        loa_sup = bias + 1.96 * sd_diff
        loa_inf = bias - 1.96 * sd_diff
        ax.scatter(means, diffs, s=90, alpha=0.75, color="steelblue", zorder=3)
        for m_v, d, lbl in zip(means, diffs, labels):
            ax.annotate(lbl, (m_v, d), textcoords="offset points",
                        xytext=(5, 4), fontsize=8)
        ax.axhline(bias,    color="red",    lw=2,   ls="-",
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


def generar_excel_por_paciente(datos_sesiones_pac, paciente_id, ruta_excel):
    METRICAS = [
        "Vol_Total", "Vol_Parenquima", "Vol_Seno", "Vol_Quistes",
        "Media_Parenquima", "Mediana_Parenquima", "P95_Parenquima",
        "IQR_Parenquima", "Std_Parenquima",
        "Media_Seno", "Mediana_Seno", "P95_Seno", "IQR_Seno", "Std_Seno",
        "Media_Quistes", "Mediana_Quistes",
    ]
    sesiones = sorted(datos_sesiones_pac)

    with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:

        # Una hoja por sesión
        for ses in sesiones:
            datos = datos_sesiones_pac[ses]
            tabla = [["METRICA", "RINON DERECHO", "RINON IZQUIERDO"]]
            for m in METRICAS:
                tabla.append([
                    m,
                    datos.get("DERECHO",   {}).get(m, ""),
                    datos.get("IZQUIERDO", {}).get(m, ""),
                ])
            pd.DataFrame(tabla).to_excel(
                writer, sheet_name=f"Sesion_{ses}", index=False, header=False)


def media_y_mediana_todos_datos(todos_datos, writer):
    
    METRICAS = ["Media_Seno", "Mediana_Seno", "Media_Parenquima", "Mediana_Parenquima"]
    ids      = sorted(todos_datos)

    for lado in ["DERECHO", "IZQUIERDO"]:
        
        cabecera = ["Paciente"]
        for m in METRICAS:
            cabecera += [f"Media_({m})", f"Mediana_({m})"]
        filas = []
        grupo_vals = {f"Media_({m})": [] for m in METRICAS}
        grupo_vals.update({f"Mediana_({m})": [] for m in METRICAS})

        for pid in ids:
            fila = [pid]
            for m in METRICAS:
               
                vals_sesiones = [
                    todos_datos[pid][s][lado][m]
                    for s in sorted(todos_datos[pid])
                    if lado in todos_datos[pid][s] and m in todos_datos[pid][s][lado]
                ]
                if vals_sesiones:
                    media_pac   = round(float(np.mean(vals_sesiones)),   4)
                    mediana_pac = round(float(np.median(vals_sesiones)), 4)
                else:
                    media_pac = mediana_pac = ""

                fila += [media_pac, mediana_pac]

                if media_pac != "":
                    grupo_vals[f"Media_({m})"].append(media_pac)
                    grupo_vals[f"Mediana_({m})"].append(mediana_pac)

            filas.append(fila)

        filas.append([""] * len(cabecera))

        fila_media_grupo = ["Media grupo"]
        fila_mediana_grupo = ["Mediana grupo"]
        for m in METRICAS:
            vm = grupo_vals[f"Media_({m})"]
            vmd = grupo_vals[f"Mediana_({m})"]
            fila_media_grupo  += [
                round(float(np.mean(vm)),   4) if vm  else "",
                round(float(np.mean(vmd)),  4) if vmd else "",
            ]
            fila_mediana_grupo += [
                round(float(np.median(vm)),   4) if vm  else "",
                round(float(np.median(vmd)),  4) if vmd else "",
            ]
        filas.append(fila_media_grupo)
        filas.append(fila_mediana_grupo)

        pd.DataFrame(filas, columns=cabecera).to_excel(
            writer, sheet_name=f"EstPac_{lado[:3]}", index=False)


def generar_excel_grupo(todos_datos, ruta_excel):
    METRICAS_DATOS = ["Media_Parenquima", "Mediana_Parenquima",
                      "Vol_Parenquima", "Vol_Total", "Vol_Seno"]
    METRICAS_CVws = ["Media_Parenquima", "Media_Seno", "Mediana_Parenquima", "Mediana_Seno"]  # Solo estas para CVws
    ids      = sorted(todos_datos)
    sesiones = sorted({s for p in todos_datos.values() for s in p})

    with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:

        for lado in ["DERECHO", "IZQUIERDO"]:
            cols  = ["Paciente"] + [f"{s}_{m}" for s in sesiones for m in METRICAS_DATOS]
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

        media_y_mediana_todos_datos(todos_datos, writer)


if __name__ == "__main__":

    PACIENTES = [
        {"id": 501, "codigo": "P501", "nombre": "Paciente 501", "mascara_sesion": 1}, 
        {"id": 504, "codigo": "P504", "nombre": "Paciente 504", "mascara_sesion": 2}, 
        {"id": 505, "codigo": "P505", "nombre": "Paciente 505", "mascara_sesion": 2}, #outlier
        {"id": 506, "codigo": "P506", "nombre": "Paciente 506", "mascara_sesion": 3}, 
        {"id": 507, "codigo": "P507", "nombre": "Paciente 507", "mascara_sesion": 1},
        {"id": 508, "codigo": "P508", "nombre": "Paciente 508", "mascara_sesion": 2}, 
        {"id": 510, "codigo": "P510", "nombre": "Paciente 510", "mascara_sesion": 1},
        {"id": 511, "codigo": "P511", "nombre": "Paciente 511", "mascara_sesion": 1},
        {"id": 513, "codigo": "P513", "nombre": "Paciente 513", "mascara_sesion": 2}, 
        {"id": 514, "codigo": "P514", "nombre": "Paciente 514", "mascara_sesion": 3},
        {"id": 516, "codigo": "P516", "nombre": "Paciente 516", "mascara_sesion": 3},
        {"id": 517, "codigo": "P517", "nombre": "Paciente 517", "mascara_sesion": 3} #outlier
    ]

    todos_datos = {}

    for pac in PACIENTES:
        pac_id       = pac["id"]
        cod_pac      = pac["codigo"]
        id_sujeto    = pac["nombre"]
        mascara_base = pac.get("mascara_sesion", 3)  
        print(f"\nProcesando paciente {pac_id} con máscara de sesión {mascara_base}")

        datos_pac = {}

        if not mascara_base_existe(pac_id, mascara_base):
            print(f"  Sin máscara de la sesión base {mascara_base}")
            continue

        try:
            img_ref, arr_rinon_base, arr_pelvis_base, arr_cysts_base = \
                cargar_mascaras_sesion(pac_id, mascara_base)
        except FileNotFoundError as e:
            print(f"  Error al cargar máscara base: {e}")
            continue

        for n_ses in [1, 2, 3]:
            ruta_g      = f"{pac_id}_0{n_ses}"
            base_folder = f"SANOS/{pac_id}_02/0{n_ses}"
            os.makedirs(f"resultados_{ruta_g}_mascara_0{mascara_base}", exist_ok=True)
            try:
                if mascara_base == n_ses:
                    ruta_ip    = buscar_archivo_nii(
                        os.path.join(base_folder, r"coreg_dixons_t1\output\Dixon"))
                    ruta_grasa = buscar_archivo_nii(
                        os.path.join(base_folder, r"coreg_dixons_t1\output\Dixon_b"))
                else:
                    ruta_ip    = buscar_archivo_nii(
                        os.path.join(base_folder, rf"dix_coreg_con_dix_0{mascara_base}\output\Dixon"))
                    ruta_grasa = buscar_archivo_nii(
                        os.path.join(base_folder, rf"dix_coreg_con_dix_0{mascara_base}\output\Dixon_b"))

                arr_rinon  = arr_rinon_base
                arr_pelvis = arr_pelvis_base
                arr_cysts  = arr_cysts_base

                _, erosion_np   = erosion(arr_rinon, img_ref, ruta_g)
                ruta_pdff, pdff_total = mapa_pdff(ruta_grasa, ruta_ip, ruta_g)
                pdff_org, _     = recortar_pdff_al_organo(erosion_np, ruta_pdff, ruta_g)

                par_np      = aislar_parenquima(pdff_org, erosion_np,
                                                arr_pelvis, arr_cysts,
                                                img_ref, ruta_g)
                seno_np     = aislar_seno(pdff_total, arr_pelvis, img_ref, ruta_g)
                quistes_np  = aislar_quistes(pdff_org, erosion_np,
                                             arr_cysts, img_ref, ruta_g)
                generar_mascara_validacion(par_np, seno_np, quistes_np,
                                           img_ref, ruta_g)

                res = estadisticas_sesion(
                    par_np, seno_np, quistes_np, erosion_np, img_ref,
                    id_sujeto, cod_pac, ruta_g)

                datos_pac[f"S{n_ses}"] = res
                print(f"Sesión {n_ses} completada")

            except Exception as exc:
                print(f"  Error en sesión {n_ses}: {exc}")

        if datos_pac:
            ruta_dir_pac = f"resultados_{pac_id}_reproducibilidad_mascara_0{mascara_base}"
            os.makedirs(ruta_dir_pac, exist_ok=True)
            generar_excel_por_paciente(
                datos_pac, id_sujeto,
                os.path.join(ruta_dir_pac, f"paciente_{pac_id}_mascara_0{mascara_base}.xlsx"))
            todos_datos[pac_id] = datos_pac
            print(f"Paciente {pac_id} completado")

    grupo_valido = {pid: d for pid, d in todos_datos.items() if len(d) >= 2}

    if len(grupo_valido) < 2:
        print("Se necesitan al menos 2 pacientes con 2+ sesiones.")
    else:
        ruta_grupo = f"resultados_grupo_mascara_0{mascara_base}"
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
                    generar_bland_altman_grupo(
                        grupo_m, f"{m}_{lado}", ruta_grupo)

        generar_excel_grupo(
            grupo_valido,
            os.path.join(ruta_grupo, "reproducibilidad_GRUPO.xlsx"))
        print(f"Excel de grupo guardado")
