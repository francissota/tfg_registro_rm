import sys
import os
import numpy as np
from totalsegmentator.python_api import totalsegmentator
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import subprocess

def abrir_en_itksnap(ruta_imagen, ruta_seg=None):
    itk_exe = r"C:\Program Files\ITK-SNAP 4.0\bin\ITK-SNAP.exe"
    if not os.path.exists(itk_exe):
        print("Error: No se encuentra el ejecutable de ITK-SNAP en la ruta especificada.")
        return
    comando = [itk_exe, "-g", ruta_imagen]
    if ruta_seg and os.path.exists(ruta_seg):
        comando.extend(["-s", ruta_seg])
    subprocess.Popen(comando)

def total_segmentator(ruta, imagen_agua, out_dir):
    ruta_input = os.path.join(ruta, imagen_agua)
    ruta_output = os.path.join(out_dir, "segmentacion_combinada.nii.gz")
    os.makedirs(out_dir, exist_ok=True) 

    totalsegmentator(
        input = ruta_input,
        output = ruta_output,
        task = "total_mr",
        fast = False,
        roi_subset = ['kidney_right', 'kidney_left'],
        ml = True, 
        device = "gpu"
    )
    
    resultado = sitk.ReadImage(ruta_output)
    data = sitk.GetArrayFromImage(resultado)
    data_final = np.zeros_like(data)
    
    list_etiquetas = np.unique(data)
    list_etiquetas = [e for e in list_etiquetas if e != 0]
    
    for etiqueta in list_etiquetas:
        rinon_indiv = (data == etiqueta).astype(np.uint8)
        mask_sitk = sitk.GetImageFromArray(rinon_indiv)
        mask_sitk.CopyInformation(resultado)
        mask_ajustada = sitk.BinaryDilate(mask_sitk, [1,1,0], sitk.sitkCross)
        mask_sitk = sitk.BinaryMorphologicalClosing(mask_ajustada, [3,3,1])
        mask_rellena = sitk.BinaryFillhole(mask_sitk)
        mask_rellena = sitk.GetArrayFromImage(mask_rellena)
        data_final[mask_rellena > 0] = etiqueta
    
    img_final = sitk.GetImageFromArray(data_final)
    img_final.CopyInformation(resultado)
    sitk.WriteImage(img_final, ruta_output)
    return ruta_output

def erosion(segmentacion, out_dir):
    ruta_output = os.path.join(out_dir, "segmentacion_erosionada.nii.gz")
    mapa_segmentacion = sitk.ReadImage(segmentacion)
    data = sitk.GetArrayFromImage(mapa_segmentacion)  
    data_final = np.zeros_like(data) 
    
    list_etiquetas = np.unique(data)
    list_etiquetas = [e for e in list_etiquetas if e != 0]
    
    for etiqueta in list_etiquetas:        
        rinon_indiv = (data == etiqueta).astype(np.uint8)
        mask_sitk = sitk.GetImageFromArray(rinon_indiv)
        mask_sitk.CopyInformation(mapa_segmentacion)
            
        mask_eroded = sitk.BinaryErode(
            mask_sitk,
            kernelRadius=[1,1,0],
            kernelType=sitk.sitkCross
        )
            
        mask_final_np = sitk.GetArrayFromImage(mask_eroded)
        data_final[mask_final_np > 0] = etiqueta
        
    rinones_erosionados = sitk.GetImageFromArray(data_final)
    rinones_erosionados.CopyInformation(mapa_segmentacion)
    sitk.WriteImage(rinones_erosionados, ruta_output)
    
    pixeles_original = np.sum(data > 0)
    pixeles_final = np.sum(data_final > 0)
    print(f"Erosión: {((pixeles_original-pixeles_final)/pixeles_original)*100:.2f}% reducido")
    return ruta_output, data_final

def mapa_pdff(imgF, imgW, out_dir):
    img_f = sitk.ReadImage(imgF)
    img_w = sitk.ReadImage(imgW)
    img_f_np = sitk.GetArrayFromImage(img_f).astype(float)
    img_w_np = sitk.GetArrayFromImage(img_w).astype(float)
    
    denominador = img_f_np + img_w_np 
    pdff_np = np.full_like(img_f_np, np.nan)
    mask_valida = denominador > 1e-10
    pdff_np[mask_valida] = (img_f_np[mask_valida] * 100) / denominador[mask_valida]
    
    pdff = sitk.GetImageFromArray(pdff_np)
    pdff.CopyInformation(img_f)
    
    ruta_output = os.path.join(out_dir, "mapa_pdff.nii.gz")
    sitk.WriteImage(pdff, ruta_output)
    return ruta_output, pdff_np

def pdff_organo(erosion_np, ruta_pdff, out_dir):
    ruta_output = os.path.join(out_dir, "mapa_organo.nii.gz")
    mapa_total_pdff = sitk.ReadImage(ruta_pdff)
    pdff_total_np = sitk.GetArrayFromImage(mapa_total_pdff).astype(np.float32)

    pdff_organo_np = np.full_like(pdff_total_np, np.nan)
    pdff_organo_np[erosion_np > 0] = pdff_total_np[erosion_np > 0]

    pdff_organo = sitk.GetImageFromArray(pdff_organo_np)
    pdff_organo.CopyInformation(mapa_total_pdff)
    sitk.WriteImage(pdff_organo, ruta_output)
    return pdff_organo_np, pdff_organo

def histograma_pdff(valores_pdff, nombre_rinon, out_dir):
    plt.figure(figsize=(10, 6))
    valores_limpios = valores_pdff[(valores_pdff >= 0) & (valores_pdff <= 100)]
    plt.hist(valores_limpios, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.median(valores_limpios), color='green', linestyle='dashed', label=f'Mediana: {np.median(valores_limpios):.2f}')
    
    plt.title(f'Distribución PDFF - Riñón {nombre_rinon}')
    plt.xlabel('Grasa (%)')
    plt.ylabel('Píxeles')
    plt.legend()
    
    ruta_hist = os.path.join(out_dir, f"histograma_{nombre_rinon}.png")
    plt.savefig(ruta_hist)
    plt.close() 

def aislar_parenquima(pdff_organo_np, erosion_np, img_ref, out_dir, umbral):
    parenquima_np = np.full_like(pdff_organo_np, np.nan)
    list_etiquetas = [e for e in np.unique(erosion_np) if e != 0]
    ruta_output = os.path.join(out_dir, "mapa_solo_parenquima_otsu.nii.gz")
    
    for etiqueta in list_etiquetas:
        mask_parenquima = (erosion_np == etiqueta) & (pdff_organo_np <= umbral)
        parenquima_np[mask_parenquima] = pdff_organo_np[mask_parenquima]

    img_final = sitk.GetImageFromArray(parenquima_np)
    img_final.CopyInformation(img_ref)
    sitk.WriteImage(img_final, ruta_output)
    return parenquima_np

def aislar_seno(pdff_organo_np, erosion_np, img_ref, out_dir, umbral):
    seno_np = np.full_like(pdff_organo_np, np.nan)
    list_etiquetas = [e for e in np.unique(erosion_np) if e != 0]
    ruta_output = os.path.join(out_dir, "mapa_solo_seno_otsu.nii.gz")

    for etiqueta in list_etiquetas:
        mask_seno = (erosion_np == etiqueta) & (pdff_organo_np > umbral)
        seno_np[mask_seno] = pdff_organo_np[mask_seno]

    img_final = sitk.GetImageFromArray(seno_np)
    img_final.CopyInformation(img_ref)
    sitk.WriteImage(img_final, ruta_output)
    return seno_np

def generar_mascara_validacion(parenquima_np, seno_np, img_ref):

    mask_np = np.zeros(parenquima_np.shape, dtype=np.uint8)
    

    mask_np[np.nan_to_num(parenquima_np) > 0] = 1
    mask_np[np.nan_to_num(seno_np) > 0] = 2
    

    mask_itk = sitk.GetImageFromArray(mask_np)
    mask_itk.CopyInformation(img_ref)
    
    ruta_output = os.path.join(out_dir, "pdff_binario.nii.gz")
    sitk.WriteImage(mask_itk, ruta_output)
    
    return mask_itk

def estadisticas_pdff(pdff_parenquima, pdff_seno, erosion, seg_original, img_ref, id_suj, cod_pac, out_dir, umbral):
    space = img_ref.GetSpacing()
    vol_voxel = space[0] * space[1] * space[2]  
    
    list_etiquetas = np.unique(erosion)
    list_etiquetas = [e for e in list_etiquetas if e != 0]
    
    res = {}
    valores_para_graficar = {}
    
    for etiqueta in list_etiquetas:
        valores_parenquima = pdff_parenquima[(erosion == etiqueta) & (~np.isnan(pdff_parenquima))]
        valores_seno = pdff_seno[(erosion == etiqueta) & (~np.isnan(pdff_seno))]
        vox_tot = np.sum(erosion == etiqueta)
        
        vox_parenquima = valores_parenquima.size
        vol_parenquima_ml = (vox_parenquima * vol_voxel) / 1000
        vox_seno = valores_seno.size
        vol_seno_ml = (vox_seno * vol_voxel) / 1000
        
        # Evitar errores si no hay valores
        if len(valores_parenquima) == 0: continue
        
        media_pdff = np.mean(valores_parenquima)
        p95_pdff = np.percentile(valores_parenquima, 95) 
        iqr_pdff = np.percentile(valores_parenquima, 75) - np.percentile(valores_parenquima, 25)
        
        media_pdff_seno = np.mean(valores_seno) if len(valores_seno) > 0 else 0
        p95_pdff_seno = np.percentile(valores_seno, 95) if len(valores_seno) > 0 else 0
        iqr_pdff_seno = np.percentile(valores_seno, 75) - np.percentile(valores_seno, 25) if len(valores_seno) > 0 else 0
        
        nombre_rinon = "DERECHO" if etiqueta == min(list_etiquetas) else "IZQUIERDO"
        valores_para_graficar[nombre_rinon] = valores_parenquima.flatten() 
        res[nombre_rinon] = {
            "Vol_Total": round((vox_tot * vol_voxel) / 1000, 4),
            "Vol_Parenquima": round(vol_parenquima_ml, 4),  
            "Vol_Seno": round(vol_seno_ml, 4),            
            "Media": round(media_pdff, 4),
            "Mediana": round(np.median(valores_parenquima), 4),
            "P95": round(p95_pdff, 4),
            "IQR": round(iqr_pdff, 4),
            "Std": round(np.std(valores_parenquima), 4),
            "Media_Seno": round(media_pdff_seno, 4),
            "Mediana_Seno": round(np.median(valores_seno) if len(valores_seno) > 0 else 0, 4),
            "P95_seno": round(p95_pdff_seno, 4),
            "IQR_Seno": round(iqr_pdff_seno, 4),
            "Std_Seno": round(np.std(valores_seno) if len(valores_seno) > 0 else 0, 4)
        }
        
        histograma_pdff(valores_parenquima, f"{nombre_rinon}_PARENQUIMA", out_dir)
        histograma_pdff(valores_seno, f"{nombre_rinon}_SENO", out_dir)
        
    v_der = valores_para_graficar.get("DERECHO", np.array([]))
    v_izq = valores_para_graficar.get("IZQUIERDO", np.array([]))
            
    if v_der.size > 0 and v_izq.size > 0:
        df_plot = pd.DataFrame({
            'PDFF (%)': np.concatenate([v_der, v_izq]),
            'Riñón': ['Derecho'] * len(v_der) + ['Izquierdo'] * len(v_izq)
        })
        sns.set_theme(style="whitegrid")
        
        plt.figure(figsize=(8, 6))
        ax = sns.boxplot(x='Riñón', y='PDFF (%)', data=df_plot, hue='Riñón', palette="Set2", legend=False)
        plt.axhline(umbral, ls='--', color='red', label=f'Umbral ({umbral}%)')
        plt.title(f'Distribución PDFF Parénquima - {id_suj}')
        plt.legend() 
        plt.savefig(os.path.join(out_dir, "boxplot_final.png"), bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.violinplot(x='Riñón', y='PDFF (%)', data=df_plot, hue='Riñón', palette="Pastel1", inner="quartile", legend=False)
        plt.axhline(umbral, ls='--', color='red')
        plt.title(f'Densidad de Grasa en Parénquima - {id_suj}')
        plt.savefig(os.path.join(out_dir, "violinplot_final.png"), bbox_inches='tight')
        plt.close()
    
    # PROTECCIÓN: Si falta algún riñón, ponemos la comparativa a 0 para que no salte error
    medidas = ["Media", "Mediana", "Vol_Parenquima"]
    rep_data = {}
    
    if "DERECHO" in res and "IZQUIERDO" in res:
        for m in medidas:
            val_d = res["DERECHO"][m]
            val_i = res["IZQUIERDO"][m]
            dif_abs = abs(val_d - val_i)
            promedio = np.mean([val_d, val_i])
            err_rel = (dif_abs / promedio) * 100 if promedio != 0 else 0
            cv = (np.std([val_d, val_i]) / promedio) * 100 if promedio != 0 else 0
            rep_data[m] = {"err": round(err_rel, 2), "cv": round(cv, 2), "abs": round(dif_abs, 3)}
    else:
        for m in medidas:
            rep_data[m] = {"err": "N/A", "cv": "N/A", "abs": "N/A"}
        
    tabla_vertical = [
        ["METADATOS", "VALOR", ""],
        ["ID_Sujeto", id_suj, ""],
        ["Código del paciente", cod_pac, ""],
        ["UMBRAL PDFF", f"{umbral}%", ""],
        ["", "", ""],
        ["MÉTRICA", "RIÑÓN DERECHO", "RIÑÓN IZQUIERDO"],
        ["Volumen Total Órgano (ml)", res["DERECHO"]["Vol_Total"], res["IZQUIERDO"]["Vol_Total"]],
        ["Volumen Parénquima (ml)", res["DERECHO"]["Vol_Parenquima"], res["IZQUIERDO"]["Vol_Parenquima"]],
        ["Volumen Seno (ml)", res["DERECHO"]["Vol_Seno"], res["IZQUIERDO"]["Vol_Seno"]],
        ["Ratio Seno/Total", round(res["DERECHO"]["Vol_Seno"]/res["DERECHO"]["Vol_Total"], 3), round(res["IZQUIERDO"]["Vol_Seno"]/res["IZQUIERDO"]["Vol_Total"], 3)],
        ["PDFF Media Parénquima (%)", res["DERECHO"]["Media"], res["IZQUIERDO"]["Media"]],
        ["PDFF Mediana Parénquima (%)", res["DERECHO"]["Mediana"], res["IZQUIERDO"]["Mediana"]],
        ["PDFF P95 Parénquima (%)", res["DERECHO"]["P95"], res["IZQUIERDO"]["P95"]],
        ["", "", ""],
        ["ANÁLISIS DE REPRODUCIBILIDAD", "MEDIA PDFF", "MEDIANA PDFF", "VOL. PARENQUIMA"],
        ["ERROR RELATIVO (%)", rep_data["Media"]["err"], rep_data["Mediana"]["err"], rep_data["Vol_Parenquima"]["err"]],
        ["COEF. VARIACIÓN (%)", rep_data["Media"]["cv"], rep_data["Mediana"]["cv"], rep_data["Vol_Parenquima"]["cv"]],
        ["DIFERENCIA ABSOLUTA", rep_data["Media"]["abs"], rep_data["Mediana"]["abs"], rep_data["Vol_Parenquima"]["abs"]],
    ]
        
    df_excel = pd.DataFrame(tabla_vertical) 
    ruta_final = os.path.join(out_dir, f"resultados_finales_rinones_{umbral}.xlsx")
    df_excel.to_excel(ruta_final, index=False, header=False)   
    return res 

def guardar_captura(img_np, auto_np, umbral, out_dir):
    corte_idx = np.argmax(np.sum(auto_np > 0, axis=(1, 2)))
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_np[corte_idx, :, :], cmap='gray')
    plt.title("Imagen Original (Agua)")
    plt.axis('off')
        
    plt.subplot(1, 2, 2)
    plt.imshow(img_np[corte_idx, :, :], cmap='gray')
    plt.imshow(auto_np[corte_idx, :, :], alpha=0.5, cmap='jet')
    plt.title(f"Automática (Umbral {umbral}%)")
    plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"comparativa_{umbral}.png"))
    plt.close()     

def diagnosticar_media_lineal(pdff_organo_np, erosion_np, out_dir):
    etiquetas = [e for e in np.unique(erosion_np) if e != 0]
    if not etiquetas: return
    etiqueta = etiquetas[0] 
    
    mask_rinon = (erosion_np == etiqueta) & (~np.isnan(pdff_organo_np))
    valores_rinon = pdff_organo_np[mask_rinon]
    
    tabla_vertical = [["Umbral", "N voxels", "Media", "Mediana", "Nuevos"]]
    n_previo = 0
    
    for umb in range(0, 80):
        mask_umb = mask_rinon & (pdff_organo_np <= umb)
        vals_umb = pdff_organo_np[mask_umb]
        n_actual = len(vals_umb)
        n_nuevos = n_actual - n_previo
        media = np.mean(vals_umb) if n_actual > 0 else 0
        mediana = np.median(vals_umb) if n_actual > 0 else 0
        
        n_previo = n_actual
        tabla_vertical.append([umb, n_actual, media, mediana, n_nuevos])    
    
    df_excel = pd.DataFrame(tabla_vertical) 
    df_excel.to_excel(os.path.join(out_dir, "incorporaciones.xlsx"), index=False, header=False)

def buscar_archivo_nii(carpeta):
    if not os.path.exists(carpeta):
        return None
    archivos = [f for f in os.listdir(carpeta) if f.endswith('.nii') or f.endswith('.nii.gz')]
    return os.path.join(carpeta, archivos[0]) if archivos else None

# --- BUCLE PRINCIPAL PARA 10 PACIENTES ---
if __name__ == "__main__":
    
    root_dir = "PACIENTES"
    # Carpeta maestra para todos los resultados
    carpeta_resultados_global = "RESULTADOS_TFG"
    os.makedirs(carpeta_resultados_global, exist_ok=True)
    
    resumen_global = []
    umbral_elegido = 18 # Tengo que hablar con María para ver el umbral, no ejecturar hasta no encontrar umbral óptimo

    for i in range(1, 21):
        paciente_id = f"5{i:02d}_02" 
        print(f"\n" + "="*50)
        print(f"PROCESANDO PACIENTE: {paciente_id}")
        print("="*50)
        
        path_paciente = os.path.join(root_dir, paciente_id)
        
        # Carpeta única para guardar todos los archivos de este paciente
        out_dir = os.path.join(carpeta_resultados_global, paciente_id)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(path_paciente):
            print(f"La carpeta {path_paciente} no existe. Saltando...")
            continue

        try:
            # 1. Localizar archivos Dixon
            subcarpetas = os.listdir(path_paciente)
            folder_w = next((f for f in subcarpetas if "_W" in f), None)
            folder_f = next((f for f in subcarpetas if "_F" in f), None)

            if not folder_w or not folder_f:
                print(f"Error: No se encuentran carpetas Dixon para {paciente_id}")
                continue

            ruta_w = buscar_archivo_nii(os.path.join(path_paciente, folder_w))
            ruta_f = buscar_archivo_nii(os.path.join(path_paciente, folder_f))

            # 2. Pipeline (pasando out_dir a todo)
            segmentacion = total_segmentator(os.path.join(path_paciente, folder_w), os.path.basename(ruta_w), out_dir)
            
            img_ref = sitk.ReadImage(ruta_w)
            seg_original_np = sitk.GetArrayFromImage(sitk.ReadImage(segmentacion))
            
            seg_erosionada, erosion_np = erosion(segmentacion, out_dir)

            ruta_pdff, pdff_np2 = mapa_pdff(ruta_f, ruta_w, out_dir)
            pdff_organo_np, pdff_organo_img = pdff_organo(erosion_np, ruta_pdff, out_dir)

            parenquima_np = aislar_parenquima(pdff_organo_np, erosion_np, img_ref, out_dir, umbral_elegido)
            seno_np = aislar_seno(pdff_organo_np, erosion_np, img_ref, out_dir, umbral_elegido)
            
            mask_auto = generar_mascara_validacion(parenquima_np, seno_np, img_ref)
            
            res = estadisticas_pdff(parenquima_np, seno_np, erosion_np, seg_original_np, img_ref, paciente_id, paciente_id, out_dir, umbral_elegido)
            
            guardar_captura(sitk.GetArrayFromImage(img_ref), erosion_np, umbral_elegido, out_dir)
            diagnosticar_media_lineal(pdff_organo_np, erosion_np, out_dir)

            # 3. Guardar datos para el Resumen Global
            for lado in ["DERECHO", "IZQUIERDO"]:
                if lado in res:
                    resumen_global.append({
                        "Paciente": paciente_id,
                        "Riñón": lado,
                        "Vol_Total_ml": res[lado].get("Vol_Total", 0),
                        "Vol_Parenquima_ml": res[lado].get("Vol_Parenquima", 0),
                        "PDFF_Media_Parénquima": res[lado].get("Media", 0),
                        "PDFF_Mediana_Parénquima": res[lado].get("Mediana", 0),
                        "PDFF_Std_Parénquima": res[lado].get("Std", 0),
                        "PDFF_P95_Parénquima": res[lado].get("P95", 0),
                        "PDFF_IQR_Parénquima": res[lado].get("IQR", 0),
                        "Vol_Seno_ml": res[lado].get("Vol_Seno", 0),
                        "PDFF_Media_Seno": res[lado].get("Media_Seno", 0),
                        "PDFF_Mediana_Seno": res[lado].get("Mediana_Seno", 0),
                        "PDFF_Std_Seno": res[lado].get("Std_Seno", 0),
                        "PDFF_P95_Seno": res[lado].get("P95_seno", 0),
                        "PDFF_IQR_Seno": res[lado].get("IQR_Seno", 0),
                        "Ratio Seno/Parenquima": res[lado].get("Vol_Seno", 0) / res[lado].get("Vol_Parenquima", 1) if res[lado].get("Vol_Parenquima", 0) != 0 else 0
                    })

            print(f"Paciente {paciente_id} finalizado con éxito.")

        except Exception as e:
            print(f"Error procesando {paciente_id}: {str(e)}")

    # 4. EXCEL COMPARATIVO FINAL
    if resumen_global:
        df_comparativo = pd.DataFrame(resumen_global)
        ruta_comparativa = os.path.join(carpeta_resultados_global, "RESUMEN_PACIENTES.xlsx")
        df_comparativo.to_excel(ruta_comparativa, index=False)
        print(f"\n" + "fin")