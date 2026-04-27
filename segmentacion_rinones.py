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


def total_segmentator(ruta, imagen_agua):
    ruta_input = os.path.join(ruta, imagen_agua)
    ruta_output = os.path.join(f"output_rinones", "segmentacion_combinada.nii.gz")

    os.makedirs(f"output_rinones", exist_ok=True) #es para asegurarse de que la carpeta existe y sino crearla dentro de esta misma carpeta

    totalsegmentator(
        input = ruta_input,
        output = ruta_output,
        task = "total",
        fast = False,
        roi_subset = ['kidney_right', 'kidney_left'],
        ml = True, #me combina los dos riñones en el mismo NIfTI al tenerlo a True
        device = "gpu"
    )
    
    resultado = sitk.ReadImage(ruta_output)
    data = sitk.GetArrayFromImage(resultado)
    data_final = np.zeros_like(data)
    
    #creamos la laista con el numero de etiquetas que tienen nuestros dos riñones y excluimos el 0 para no contar el fondo
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
    
    abrir_en_itksnap(ruta_input, ruta_output)
    
    return ruta_output

def erosion(segmentacion):
    ruta_output = os.path.join(f"output_rinones", "segmentacion_erosionada.nii.gz")
    mapa_segmentacion = sitk.ReadImage(segmentacion)
    data = sitk.GetArrayFromImage(mapa_segmentacion) #extraemos los numeros de la matriz  
    data_final = np.zeros_like(data) #creamos una matriz con todo 0s con el tamaño de data
    
    #creamos la laista con el numero de etiquetas que tienen nuestros dos riñones y excluimos el 0 para no contar el fondo
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
    print(((pixeles_original-pixeles_final)/pixeles_original)*100)
    
    abrir_en_itksnap(ruta_completa_agua, ruta_output)
    
    return ruta_output, data_final

def mapa_pdff(imgF, imgW):
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
    
    ruta_output = os.path.join(f"output_rinones", "mapa_pdff.nii.gz")
    sitk.WriteImage(pdff, ruta_output)
    
    abrir_en_itksnap(ruta_output, None)
    
    return ruta_output, pdff_np

def pdff_organo(erosion_np, ruta_pdff):

    
    ruta_output = os.path.join(f"output_rinones", "mapa_organo.nii.gz")
    
    mapa_total_pdff = sitk.ReadImage(ruta_pdff)
    pdff_total_np = sitk.GetArrayFromImage(mapa_total_pdff).astype(np.float32)

    pdff_organo_np = np.full_like(pdff_total_np, np.nan)
    pdff_organo_np[erosion_np > 0] = pdff_total_np[erosion_np > 0]

    pdff_organo = sitk.GetImageFromArray(pdff_organo_np)
    pdff_organo.CopyInformation(mapa_total_pdff)
    sitk.WriteImage(pdff_organo, ruta_output)

    abrir_en_itksnap(ruta_output, None)
    
    return pdff_organo_np, pdff_organo

def histograma_pdff(valores_pdff, nombre_rinon):
    plt.figure(figsize=(10, 6))
    valores_limpios = valores_pdff[(valores_pdff >= 0) & (valores_pdff <= 100)]
    
    plt.hist(valores_limpios, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.median(valores_limpios), color='green', linestyle='dashed', label=f'Mediana: {np.median(valores_limpios):.2f}')
    
    plt.title(f'Distribución PDFF - Riñón {nombre_rinon}')
    plt.xlabel('Grasa (%)')
    plt.ylabel('Píxeles')
    plt.legend()
    
    ruta_hist = os.path.join(f"output_rinones_{umbral}", f"histograma_{nombre_rinon}.png")
    plt.savefig(ruta_hist)
    plt.close() 

def aislar_parenquima(pdff_organo_np, erosion_np, img_ref):
    os.makedirs(f"output_rinones_{umbral}", exist_ok=True)

    parenquima_np = np.full_like(pdff_organo_np, np.nan)
    list_etiquetas = [e for e in np.unique(erosion_np) if e != 0]

    ruta_output = os.path.join(f"output_rinones_{umbral}", "mapa_solo_parenquima_otsu.nii.gz")
    
    for etiqueta in list_etiquetas:
        mask_parenquima = (erosion_np == etiqueta) & (pdff_organo_np <= umbral)
        parenquima_np[mask_parenquima] = pdff_organo_np[mask_parenquima]

    img_final = sitk.GetImageFromArray(parenquima_np)
    img_final.CopyInformation(img_ref)
    sitk.WriteImage(img_final, ruta_output)
    
    #abrir_en_itksnap(ruta_output, None)
    
    return parenquima_np

def aislar_seno(pdff_organo_np, erosion_np, img_ref):

    seno_np = np.full_like(pdff_organo_np, np.nan)
    list_etiquetas = [e for e in np.unique(erosion_np) if e != 0]

    ruta_output = os.path.join(f"output_rinones_{umbral}", "mapa_solo_seno_otsu.nii.gz")

    for etiqueta in list_etiquetas:
        mask_seno = (erosion_np == etiqueta) & (pdff_organo_np > umbral)
        seno_np[mask_seno] = pdff_organo_np[mask_seno]

    img_final = sitk.GetImageFromArray(seno_np)
    img_final.CopyInformation(img_ref)
    sitk.WriteImage(img_final, ruta_output)
    
    #abrir_en_itksnap(ruta_output, None)
    
    return seno_np

def generar_mascara_validacion(parenquima_np, seno_np, img_ref):

    mask_np = np.zeros(parenquima_np.shape, dtype=np.uint8)
    

    mask_np[~np.isnan(parenquima_np)] = 1
    mask_np[~np.isnan(seno_np)] = 2
    

    mask_itk = sitk.GetImageFromArray(mask_np)
    mask_itk.CopyInformation(img_ref)
    
  
    ruta_guardado = os.path.join(f"output_rinones_{umbral}", f"pdff_binario_{umbral}.nii.gz")
    sitk.WriteImage(mask_itk, ruta_guardado)
    
    return mask_itk
    
    

def estadisticas_pdff(pdff_parenquima, pdff_seno, erosion, seg_original, img_ref, id_suj, cod_pac):
      
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
        
        media_pdff = np.mean(valores_parenquima)
        p95_pdff = np.percentile(valores_parenquima, 95) # El valor del 5% más graso
        iqr_pdff = np.percentile(valores_parenquima, 75) - np.percentile(valores_parenquima, 25)
        
        media_pdff_seno = np.mean(valores_seno)
        p95_pdff_seno = np.percentile(valores_seno, 95) # El valor del 5% más graso
        iqr_pdff_seno = np.percentile(valores_seno, 75) - np.percentile(valores_seno, 25)
        
    

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
            "Mediana_Seno": round(np.median(valores_seno), 4),
            "P95_seno": round(p95_pdff_seno, 4),
            "IQR_Seno": round(iqr_pdff_seno, 4),
            "Std_Seno": round(np.std(valores_seno), 4)
        }
        

        histograma_pdff(valores_parenquima, f"{nombre_rinon}_PARENQUIMA")
        histograma_pdff(valores_seno, f"{nombre_rinon}_SENO")
        
    v_der = valores_para_graficar["DERECHO"]
    v_izq = valores_para_graficar["IZQUIERDO"]
            
    if v_der.size > 0 and v_izq.size > 0:
        
        df_plot = pd.DataFrame({
            'PDFF (%)': np.concatenate([v_der, v_izq]),
            'Riñón': ['Derecho'] * len(v_der) + ['Izquierdo'] * len(v_izq)
        })

        sns.set_theme(style="whitegrid")

        
        plt.figure(figsize=(8, 6))
        ax = sns.boxplot(x='Riñón', y='PDFF (%)', data=df_plot, hue='Riñón', palette="Set2", legend=False)
        plt.axhline(14, ls='--', color='red', label='Umbral Seguridad (14%)')
        plt.title(f'Distribución PDFF Parénquima - {id_suj}')
        plt.legend() 
        plt.savefig(f"output_rinones_{umbral}/boxplot_final.png", bbox_inches='tight')
        plt.close()


        plt.figure(figsize=(8, 6))
        sns.violinplot(x='Riñón', y='PDFF (%)', data=df_plot, hue='Riñón', palette="Pastel1", inner="quartile", legend=False)
        plt.axhline(14, ls='--', color='red')
        plt.title(f'Densidad de Grasa en Parénquima - {id_suj}')
        plt.savefig(f"output_rinones_{umbral}/violinplot_final.png", bbox_inches='tight')
        plt.close()
    
    medidas = ["Media", "Mediana", "Vol_Parenquima"]
    rep_data = {}

    for m in medidas:
        val_d = res["DERECHO"][m]
        val_i = res["IZQUIERDO"][m]
        
        dif_abs = abs(val_d - val_i)
        
        promedio = np.mean([val_d, val_i])
        
        err_rel = (dif_abs / promedio) * 100 if promedio != 0 else 0
        
        cv = (np.std([val_d, val_i]) / promedio) * 100 if promedio != 0 else 0
        
        rep_data[m] = {
            "err": round(err_rel, 2),
            "cv": round(cv, 2),
            "abs": round(dif_abs, 3)
        }
        
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
    ruta_final = f"output_rinones_{umbral}/resultados_finales_rinones_{umbral}.xlsx"
    df_excel.to_excel(ruta_final, index=False, header=False)   
    return res 

def calculate_dice_score(pred_mask, gt_mask, label):

    try:
        label_int = int(label)
        
        if (pred_mask.GetSize() != gt_mask.GetSize() or 
            pred_mask.GetSpacing() != gt_mask.GetSpacing() or
            pred_mask.GetOrigin() != gt_mask.GetOrigin()):
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(gt_mask)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            pred_mask = resampler.Execute(pred_mask)
        
        pred_binary = sitk.BinaryThreshold(pred_mask, lowerThreshold=label_int, 
                                          upperThreshold=label_int, insideValue=1, outsideValue=0)
        gt_binary = sitk.BinaryThreshold(gt_mask, lowerThreshold=label_int, 
                                        upperThreshold=label_int, insideValue=1, outsideValue=0)
        
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(pred_binary)
        p1 = stats.GetNumberOfPixels(1) if stats.HasLabel(1) else 0
        stats.Execute(gt_binary)
        p2 = stats.GetNumberOfPixels(1) if stats.HasLabel(1) else 0
        
        if p1 == 0 and p2 == 0:
            return 1.0  

        overlap = sitk.LabelOverlapMeasuresImageFilter()
        overlap.Execute(pred_binary, gt_binary)
        return float(overlap.GetDiceCoefficient())
        
    except Exception as e:
        print(f"Error calculando Dice para etiqueta {label}: {e}")
        return 0.0      
    
def guardar_captura(img_np, manual_np, auto_np, umbral):
  
    corte_idx = np.argmax(np.sum(auto_np > 0, axis=(1, 2)))
        
    plt.figure(figsize=(15, 5))
        
 
    plt.subplot(1, 3, 1)
    plt.imshow(img_np[corte_idx, :, :], cmap='gray')
    plt.title("Imagen Original (Agua)")
    plt.axis('off')
        
   
    plt.subplot(1, 3, 2)
    plt.imshow(img_np[corte_idx, :, :], cmap='gray')
    plt.imshow(manual_np[corte_idx, :, :], alpha=0.5, cmap='jet')
    plt.title("Segmentación Manual")
    plt.axis('off')
        
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_np[corte_idx, :, :], cmap='gray')
    plt.imshow(auto_np[corte_idx, :, :], alpha=0.5, cmap='jet')
    plt.title(f"Automática (Umbral {umbral}%)")
    plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(f"output_rinones_{umbral}", f"comparativa.png"))
    plt.close()     
    
def crear_grafica_comparativa(resultados, id_sujeto):
    df = pd.DataFrame(resultados)
    
    fig, ax1 = plt.subplots(figsize = (10,6))
    
    ax1.set_xlabel("Umbral", fontsize = 12, fontweight = 'bold')
    ax1.set_ylabel('Dice Score (Precisión)', fontsize=12, color='tab:blue')
    
    ax1.plot(df['Umbral'], df['Dice_P'], marker='o', label='Dice Parénquima', color='lightblue', linestyle='--')
    ax1.plot(df['Umbral'], df['Dice_S'], marker='o', label='Dice Seno', color='orange', linestyle='--')
    ax1.plot(df['Umbral'], df['Dice_Medio'], marker='s', label='Dice Medio (Global)', color='tab:blue', linewidth=3)
    
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 1) # El Dice siempre va de 0 a 1
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')


    ax2 = ax1.twinx() 
    ax2.set_ylabel('Volumen Seno Renal (ml)', fontsize=12, color='tab:red')
    
    vol_medio = (df['Vol_Seno_DERECHO'] + df['Vol_Seno_IZQUIERDO']) / 2
    ax2.plot(df['Umbral'], vol_medio, color='tab:red', marker='x', label='Vol. Seno Medio', alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    fig.tight_layout()
    plt.savefig(f"output_rinones/grafica_tendencia_{id_sujeto}.png", dpi=300)
    plt.close()
    
def diagnosticar_media_lineal(pdff_organo_np, erosion_np):

    etiqueta = [e for e in np.unique(erosion_np) if e != 0][0] 
    
    mask_rinon = (erosion_np == etiqueta) & (~np.isnan(pdff_organo_np))
    valores_rinon = pdff_organo_np[mask_rinon]
    
    print(f"\nTOTAL voxels riñón: {len(valores_rinon)}")
    print(f"PDFF medio TOTAL: {np.mean(valores_rinon):.2f}%")
    
    print(f"\n{'Umbral':<10} {'N voxels':<12} {'Media':<10} {'Mediana':<10} {'Nuevos':<10}")
    print("-"*60)
    
    tabla_vertical = [["Umbral", "N voxels", "Media", "Mediana", "Nuevos"]]
    
    n_previo = 0
    for umb in range(0, 80):
        mask_umb = mask_rinon & (pdff_organo_np <= umb)
        vals_umb = pdff_organo_np[mask_umb]
        
        n_actual = len(vals_umb)
        n_nuevos = n_actual - n_previo
        
        media = np.mean(vals_umb)
        mediana = np.median(vals_umb)
        
        print(f"{umb:<10} {n_actual:<12} {media:<10.2f} {mediana:<10.2f} {n_nuevos:<10}")
        
        n_previo = n_actual
        
        tabla_vertical.append([umb, n_actual, media, mediana, n_nuevos])    
    
    df_excel = pd.DataFrame(tabla_vertical) 
    ruta_final = f"output_rinones/incorporaciones.xlsx"
    df_excel.to_excel(ruta_final, index=False, header=False)
    
    print("="*70)
    

def grafica_con_desaceleracion(resultados, id_sujeto):
 
    df = pd.DataFrame(resultados)
    

    df['Incremento'] = df['Media_DERECHO'].diff()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(df['Umbral'], df['Media_DERECHO'], 
             marker='o', linewidth=2, markersize=6, color='blue')
    ax1.axvline(30, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Umbral PDFF (%)', fontsize=12)
    ax1.set_ylabel('Media PDFF Parénquima (%)', fontsize=12, color='blue')
    ax1.set_title(f'Curva PDFF Parénquima - {id_sujeto}', 
                  fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    ax2.bar(df['Umbral'][1:], df['Incremento'][1:], 
            color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axhline(0.05, color='red', linestyle='--')
    ax2.set_xlabel('Umbral PDFF (%)', fontsize=12)
    ax2.set_ylabel('Incremento Media PDFF (%)', fontsize=12)
    ax2.set_title('Tasa de Cambio', 
                  fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"output_rinones/desaceleracion_{id_sujeto}.png", dpi=300)
    plt.close()
    

if __name__ == "__main__":
    
    resultados = []
    
    base_folder = "PACIENTES/516_02"

    partes = "H"
    id_sujeto = "H"
    cod_paciente = "H"
    

    ruta_completa_agua = os.path.join(base_folder, "DIXON_BH_W_0031/00010101_000000ANONYMIZEDs031a1001.nii")
    ruta_completa_grasa = os.path.join(base_folder, "DIXON_BH_F_0030/00010101_000000ANONYMIZEDs030a1001.nii")
    
    pdff_manual = sitk.ReadImage("seg_501.nii.gz")

    abrir_en_itksnap(ruta_completa_agua)
    abrir_en_itksnap(ruta_completa_grasa)
    
    segmentacion = total_segmentator(os.path.join(base_folder, "DIXON_BH_W_0031"), "00010101_000000ANONYMIZEDs031a1001.nii")
        
    img_ref = sitk.ReadImage(ruta_completa_agua)
    seg_original_np = sitk.GetArrayFromImage(sitk.ReadImage(segmentacion))

    seg_erosionada, erosion_np = erosion(segmentacion)

    ruta_pdff, pdff_np2 = mapa_pdff(ruta_completa_grasa, ruta_completa_agua)

    pdff_organo_np, pdff_organo_img = pdff_organo(erosion_np, ruta_pdff)
    
    
    list_etiquetas = np.unique(erosion_np)
    list_etiquetas = [e for e in list_etiquetas if e != 0] 
    
    
    for umbral in range(0, 80):
  
        parenquima_np = aislar_parenquima(pdff_organo_np, erosion_np, img_ref)
        seno_np = aislar_seno(pdff_organo_np, erosion_np, img_ref)
        
        mask_auto = generar_mascara_validacion(parenquima_np, seno_np, img_ref)
        
        d_p = calculate_dice_score(mask_auto, pdff_manual, label=1)
        d_s = calculate_dice_score(mask_auto, pdff_manual, label=2)
        d_medio = (d_p + d_s) / 2
        
        fila = {
            "Umbral": umbral,
            "Dice_P": d_p,
            "Dice_S": d_s,
            "Dice_Medio": d_medio
        }

        res = estadisticas_pdff(parenquima_np, seno_np, erosion_np, seg_original_np, img_ref, id_sujeto, cod_paciente)
        
        for lado in ["DERECHO","IZQUIERDO"]:
            for metrica, valor in res[lado].items():
                fila[f"{metrica}_{lado}"] = valor
        resultados.append(fila)
        
        guardar_captura(sitk.GetArrayFromImage(img_ref), sitk.GetArrayFromImage(pdff_manual), sitk.GetArrayFromImage(mask_auto), umbral)
        


    if resultados:
        df_final = pd.DataFrame(resultados)
        
        columnas_principales = ["Umbral", "Dice_Medio", "Dice_P", "Dice_S"]
        otras_columnas = [c for c in df_final.columns if c not in columnas_principales]
        df_final = df_final[columnas_principales + otras_columnas]

        ruta_resumen = os.path.join("output_rinones", f"RESUMEN_COMPLETO_{id_sujeto}.xlsx")
        df_final.to_excel(ruta_resumen, index=False, header=True)
        
    crear_grafica_comparativa(resultados, 'sujeto_1')
    
    diagnosticar_media_lineal(pdff_organo_np, erosion_np)
    
    grafica_con_desaceleracion(resultados, id_sujeto)
        
    
           
        

        
        
        

  
            

        
    
    

    

    
    
    
