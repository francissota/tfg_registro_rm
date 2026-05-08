import os
import glob
import shutil
import subprocess
import json
import pandas as pd
import numpy as np
import pydicom
import nibabel as nib
import SimpleITK as sitk

# ─────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────

def get_z_direction(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith('.dcm') or f.endswith('.IMA')])
    if len(files) < 2:
        return None
    ds0 = pydicom.dcmread(os.path.join(folder, files[0]), force=True)
    dsN = pydicom.dcmread(os.path.join(folder, files[-1]), force=True)
    z0 = float(ds0[(0x0020, 0x0032)].value[2])
    zN = float(dsN[(0x0020, 0x0032)].value[2])
    return 1 if zN > z0 else -1


def dicom_folder_to_nifti(folder, output_nii_path):
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(folder)

    if not series_IDs:
        raise RuntimeError(f"No series en {folder}")

    max_files = []
    for sid in series_IDs:
        files = reader.GetGDCMSeriesFileNames(folder, sid)
        if len(files) > len(max_files):
            max_files = files

    reader.SetFileNames(max_files)
    image = reader.Execute()
    sitk.WriteImage(image, output_nii_path)

    print(f"  OK NIfTI: {os.path.basename(output_nii_path)}")


def limpiar_transforms(output_dir):
    """Limpia rutas en los ficheros de transform de elastix.
    Arregla dos casos:
      1) rutas relativas tipo 'output/transform...' (generadas dentro del contenedor)
      2) rutas absolutas /app/output/... o /app/data/... que dejan de ser válidas
         cuando el montaje cambia entre ejecuciones.
    Deja el nombre de fichero sin directorio, lo que permite que transformix
    lo resuelva correctamente cuando se le pasa -tp <fichero> desde su carpeta.
    """
    txt_folder = os.path.join(output_dir, "output")
    for f in glob.glob(os.path.join(txt_folder, "*.txt")):
        with open(f, 'r') as file:
            content = file.read()
        modified = content
        # Caso 1: "output/transform... -> "transform...
        if '"output/' in modified:
            modified = modified.replace('"output/', '"')
        # Caso 2: "/app/output/transform... -> "transform...
        import re
        modified = re.sub(r'"(/app/output/|/app/data/[^"]+/)([^/"]+\.txt)"',
                          r'"\2"', modified)
        if modified != content:
            with open(f, 'w') as file:
                file.write(modified)
            print(f"  OK transform limpio: {os.path.basename(f)}")


def flip_nifti_z(input_nii_path, output_nii_path):
    img = nib.load(input_nii_path)
    data = img.get_fdata()
    affine = img.affine.copy()

    data_flipped = np.flip(data, axis=2)
    affine_flipped = affine.copy()
    affine_flipped[:, 2] = -affine[:, 2]
    affine_flipped[:3, 3] = affine[:3, 3] + affine[:3, 2] * (data.shape[2] - 1)

    nib.save(nib.Nifti1Image(data_flipped, affine_flipped, img.header), output_nii_path)
    print(f"  OK flip: {os.path.basename(output_nii_path)}")


def preparar_nifti(fixed_folder, moving_folder, tmp_dir, nombre):
    os.makedirs(tmp_dir, exist_ok=True)

    nii_path = os.path.join(tmp_dir, f"{nombre}.nii.gz")
    nii_flip = os.path.join(tmp_dir, f"{nombre}_flip.nii.gz")

    dicom_folder_to_nifti(moving_folder, nii_path)

    dir_fixed  = get_z_direction(fixed_folder)
    dir_moving = get_z_direction(moving_folder)

    # LOG: útil para saber si el flip se activa y diagnosticar problemas
    print(f"  z-dir fixed={dir_fixed}, moving={dir_moving} -> {'FLIP aplicado' if dir_fixed != dir_moving else 'sin flip'}")

    if dir_fixed != dir_moving:
        flip_nifti_z(nii_path, nii_flip)
        os.remove(nii_path)
        return nii_flip
    else:
        return nii_path


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

base_path = r"D:\tfg_francis\RESPECT_CENTER01"
sesion_fija = "02"
sesiones_movil = ["03"]

metricas = []

# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

for paciente_id in ["518_01"]:

    print(f"\n{'='*60}")
    print(f"PACIENTE: {paciente_id}")
    print(f"{'='*60}")

    try:
        patient_folder = os.path.join(base_path, paciente_id)
        if not os.path.exists(patient_folder):
            print(f"AVISO: No existe la carpeta del paciente {paciente_id}")
            continue

        folder_fija = os.path.join(patient_folder, sesion_fija)
        if not os.path.exists(folder_fija):
            print(f"AVISO: No existe la sesión fija {sesion_fija} para {paciente_id}")
            continue

        # FIXED WATER (Dixon original de la sesión fija)
        # Buscamos la carpeta original (no la de coreg)
        fixed_candidates = [f for f in glob.glob(os.path.join(folder_fija, '*WATER*')) 
                            if os.path.isdir(f) and 'coreg' not in f.lower() and 'dixon_like' not in f.lower()]
        if not fixed_candidates:
            print(f"AVISO: No se encuentra carpeta FIXED WATER (Dixon original) en {folder_fija}")
            continue
            
        fixed_path = fixed_candidates[0]
        fixed_name = os.path.basename(fixed_path)

        for sesion in sesiones_movil:
            try:
                print(f"\n--- SESION {sesion} ---")
                folder_movil = os.path.join(patient_folder, sesion)
                if not os.path.exists(folder_movil):
                    print(f"AVISO: No existe la sesión móvil {sesion}")
                    continue

                # ── BUSCAR WATER / FAT ──────────────────────────
                exclude = ['coreg', 'dixon_like', 'en_t1', 'dicom_', 'anonym', 'imagen_ip']
                w_cands = [f for f in glob.glob(os.path.join(folder_movil, '*WATER_D*')) 
                           if os.path.isdir(f) and not any(k in f.lower() for k in exclude)]
                f_cands = [f for f in glob.glob(os.path.join(folder_movil, '*FAT_D*')) 
                           if os.path.isdir(f) and not any(k in f.lower() for k in exclude)]
                
                if not w_cands or not f_cands:
                    print(f"AVISO: No se han encontrado carpetas DICOM originales en {folder_movil}")
                    continue
                    
                water_path = w_cands[0]
                fat_path   = f_cands[0]
                print(f"   ---> WATER: {os.path.basename(water_path)}")
                print(f"   ---> FAT:   {os.path.basename(fat_path)}")

                tmp_dir = os.path.join(folder_movil, "_tmp")

                # ── NIFTI + FLIP ──
                water_nii = preparar_nifti(fixed_path, water_path, tmp_dir, "water")
                fat_nii   = preparar_nifti(fixed_path, fat_path,   tmp_dir, "fat")

                rel_water = os.path.relpath(water_nii, patient_folder).replace("\\", "/")
                rel_fat   = os.path.relpath(fat_nii,   patient_folder).replace("\\", "/")

                # ── PASO 1: REGISTRO ──
                output_dir = os.path.join(folder_movil, f"coreg_dixon_con_dixon{sesion_fija}")
                shutil.rmtree(output_dir, ignore_errors=True)
                os.makedirs(output_dir)

                # Verificación de carpeta fija no vacía (evita error list index out of range en el contenedor)
                if not os.listdir(fixed_path):
                    print(f"AVISO: La carpeta fija está vacía: {fixed_path}")
                    continue

                cont = f"regg_{paciente_id}_{sesion}"
                subprocess.run(['docker','rm','-f',cont], capture_output=True)

                cmd = (
                    f'docker run --name {cont} '
                    f'-v "{os.path.abspath(patient_folder)}":/app/data '
                    f'siria_pipeline '
                    f'"data/{sesion_fija}/{fixed_name}/, '
                    f'data/{rel_water}, '
                    f'tra, '
                    f'data/{sesion}/anonym/parametermaps/par_water_water.txt, '
                    f'data/{sesion}/anonym/parametermaps/config.json, TransformNii" '
                    f'"output/"'
                )

                print("\n[REGISTRO]")
                p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True)

                metric = "N/A"
                for l in p.stdout:
                    print(l, end="")
                    if "Final metric value" in l:
                        try: metric = float(l.split("=")[-1])
                        except: pass

                p.wait()
                if p.returncode != 0:
                    print(f"ERROR: Falló el registro para {paciente_id} sesión {sesion}")
                    continue

                os.system(f'docker cp {cont}:/app/output "{output_dir}"')
                subprocess.run(['docker','rm','-f',cont], capture_output=True)
                
                limpiar_transforms(output_dir)

                # ── TRANSFORM SELECTION ──
                tp_candidates = glob.glob(os.path.join(output_dir, "output", "transform_TransformNii_*.txt"))
                if not tp_candidates:
                    print("AVISO: No se encontraron archivos de transform.")
                    continue
                
                tp = tp_candidates[0]
                for suffix in ['_a.txt', '_r.txt', '_t.txt']:
                    match = [f for f in tp_candidates if suffix in f]
                    if match: tp = match[0]; break
                
                rel_tp = os.path.relpath(tp, patient_folder).replace("\\", "/")

                # ── PASO 2: APPLY FAT (ITK Directo) ──
                print("\n[APPLY FAT]")
                out_fat_final = os.path.join(output_dir, "fat_result")
                os.makedirs(out_fat_final, exist_ok=True)
                
                py_cmd = (
                    f"import itk; import numpy as np; import os; "
                    f"tp_file = '/app/data/{rel_tp}'; "
                    f"os.chdir(os.path.dirname(tp_file)); "
                    f"params = itk.ParameterObject.New(); "
                    f"params.ReadParameterFile(os.path.basename(tp_file)); "
                    f"params.SetParameter('FinalBSplineInterpolationOrder', '1'); "
                    f"img = itk.imread('/app/data/{rel_fat}', itk.F); "
                    f"res = itk.transformix_filter(img, params); "
                    f"arr = itk.GetArrayViewFromImage(res); "
                    f"arr[arr < 0] = 0; "
                    f"itk.imwrite(res, '/app/data/fat_tmp_res.nii.gz')"
                )
                cmd_fat = f'docker run --rm -v "{os.path.abspath(patient_folder)}":/app/data --entrypoint python3 siria_pipeline -c "{py_cmd}"'
                p2 = subprocess.run(cmd_fat, shell=True, capture_output=True, text=True)
                
                if p2.returncode == 0:
                    res_path_local = os.path.join(patient_folder, "fat_tmp_res.nii.gz")
                    if os.path.exists(res_path_local):
                        shutil.move(res_path_local, os.path.join(out_fat_final, "fat_coreg.nii.gz"))
                        print("   ---> OK FAT registrado.")

                # ── PASO 4: ALIGN TO T1 (Aplicar el registro Dixon->T1 de la sesión fija) ──
                dixon_t1_dir = os.path.join(folder_fija, "coreg_dixons_t1", "output")
                if os.path.exists(dixon_t1_dir):
                    print("\n[ALIGN TO T1]")
                    limpiar_transforms(os.path.dirname(dixon_t1_dir))
                    
                    t1_tx_cands = glob.glob(os.path.join(dixon_t1_dir, "transform_Dixon_*.txt"))
                    if t1_tx_cands:
                        t1_tx = t1_tx_cands[0]
                        for sfx in ['_a.txt', '_r.txt', '_t.txt']:
                            m = [f for f in t1_tx_cands if sfx in f]
                            if m: t1_tx = m[0]; break
                        
                        rel_t1_tx = os.path.relpath(t1_tx, patient_folder).replace("\\", "/")
                        print(f"   ---> Aplicando transform Dixon-T1: {os.path.basename(t1_tx)}")
                        
                        res_water_reg = glob.glob(os.path.join(output_dir, "output", "TransformNii", "*.nii.gz"))
                        if not res_water_reg: res_water_reg = glob.glob(os.path.join(output_dir, "output", "*.nii.gz"))
                        
                        imgs_to_t1 = []
                        if res_water_reg: imgs_to_t1.append(("water", res_water_reg[0]))
                        imgs_to_t1.append(("fat", os.path.join(out_fat_final, "fat_coreg.nii.gz")))
                        
                        for label, img_path in imgs_to_t1:
                            if not os.path.exists(img_path): continue
                            rel_img = os.path.relpath(img_path, patient_folder).replace("\\", "/")
                            out_name = f"{label}_aligned_to_t1.nii.gz"
                            
                            tp_container  = f"/app/data/{rel_t1_tx}"
                            img_container = f"/app/data/{rel_img}"
                            out_container = f"/app/data/{out_name}"
                            py_t1 = (
                                f"import itk, numpy as np, os; "
                                f"tp='{tp_container}'; "
                                f"os.chdir(os.path.dirname(tp)); "
                                f"p=itk.ParameterObject.New(); "
                                f"p.ReadParameterFile(os.path.basename(tp)); "
                                f"p.SetParameter('FinalBSplineInterpolationOrder', '1'); "
                                f"img=itk.imread('{img_container}', itk.F); "
                                f"res=itk.transformix_filter(img, p); "
                                f"arr=itk.GetArrayViewFromImage(res); "
                                f"arr[arr < 0] = 0; "
                                f"itk.imwrite(res, '{out_container}')"
                            )
                            cmd_t1 = (
                                f'docker run --rm '
                                f'-v "{os.path.abspath(patient_folder)}":/app/data '
                                f'--entrypoint python3 siria_pipeline '
                                f'-c "{py_t1}"'
                            )
                            r = subprocess.run(cmd_t1, shell=True, capture_output=True, text=True)
                            if r.returncode != 0:
                                print(f"   WARN ITK stderr: {r.stderr[-400:]}")
                            
                            res_t1_local = os.path.join(patient_folder, out_name)
                            if os.path.exists(res_t1_local):
                                final_dest = os.path.join(os.path.dirname(img_path), out_name)
                                shutil.move(res_t1_local, final_dest)
                                print(f"   ---> OK {label} alineado a T1.")
                            else:
                                print(f"   WARN: no se creó {out_name} para {label}")

                # ── FIX HEADERS ──────────────────────────────────────────────
                # Corregimos metadatos de cabecera asegurando que el espacio físico
                # sea exacto al de la referencia correspondiente.
                print("\n[FIX HEADERS]")
                try:
                    # Referencia Dixon (Sesión Fija)
                    dixon_t1_dir = os.path.join(folder_fija, "coreg_dixons_t1", "output")
                    ref_dixon_cands = (
                        glob.glob(os.path.join(dixon_t1_dir, "Dixon_b", "*.nii.gz")) +
                        glob.glob(os.path.join(dixon_t1_dir, "Dixon_b", "*.nii"))
                    )
                    
                    # Referencia T1 (Sesión Fija)
                    ref_t1_path = os.path.join(dixon_t1_dir, "Moving_img", "fixed.nii.gz")
                    
                    targets = [
                        ("fat_t1", os.path.join(out_fat_final, "fat_aligned_to_t1.nii.gz")),
                        ("water_t1", os.path.join(output_dir, "output", "TransformNii", "water_aligned_to_t1.nii.gz")),
                        ("fat_dix", os.path.join(out_fat_final, "fat_coreg.nii.gz")),
                    ]

                    for label, img_path in targets:
                        if not os.path.exists(img_path):
                            continue
                        
                        img_fix = sitk.ReadImage(img_path)
                        
                        # Elegir la referencia correcta según el espacio (T1 o Dixon)
                        if "_t1" in label:
                            if os.path.exists(ref_t1_path):
                                img_ref = sitk.ReadImage(ref_t1_path)
                                if img_fix.GetSize() == img_ref.GetSize():
                                    img_fix.CopyInformation(img_ref)
                                    sitk.WriteImage(img_fix, img_path)
                                    print(f"   ---> Header corregido (Espacio T1): {os.path.basename(img_path)}")
                                else:
                                    print(f"   WARN: Size mismatch T1 ({img_fix.GetSize()} vs {img_ref.GetSize()}) para {os.path.basename(img_path)}")
                            else:
                                print(f"   WARN: No se encontró referencia T1 en {ref_t1_path}")
                        else:
                            if ref_dixon_cands:
                                img_ref = sitk.ReadImage(ref_dixon_cands[0])
                                if img_fix.GetSize() == img_ref.GetSize():
                                    img_fix.CopyInformation(img_ref)
                                    sitk.WriteImage(img_fix, img_path)
                                    print(f"   ---> Header corregido (Espacio Dixon): {os.path.basename(img_path)}")
                                else:
                                    print(f"   WARN: Size mismatch Dixon ({img_fix.GetSize()} vs {img_ref.GetSize()}) para {os.path.basename(img_path)}")
                            else:
                                print(f"   WARN: No se encontró referencia Dixon en Dixon_b")
                                
                except Exception as e_hdr:
                    print(f"   WARN fix headers: {e_hdr}")

                # ── PASO 5: CONVERSIÓN A DICOM ──
                print("\n[CONVERSIÓN DICOM]")
                import sys
                main_repo = r"D:\tfg_francis\RESPECT_Co-Registration_Module-main"
                if main_repo not in sys.path: sys.path.append(main_repo)
                
                import nifti2dicom
                
                # Agua
                water_final = os.path.join(output_dir, "output", "TransformNii", "water_aligned_to_t1.nii.gz")
                if not os.path.exists(water_final):
                    cands = glob.glob(os.path.join(output_dir, "output", "TransformNii", "*.nii.gz"))
                    if not cands: cands = glob.glob(os.path.join(output_dir, "output", "*.nii.gz"))
                    water_final = cands[0] if cands else None
                
                if water_final:
                    nifti2dicom.nifti_to_dicom(water_final, fixed_path, water_path, os.path.dirname(water_final))
                    print(f"   ---> OK WATER DICOM en: {os.path.basename(os.path.dirname(water_final))}")

                # Grasa
                fat_final = os.path.join(out_fat_final, "fat_aligned_to_t1.nii.gz")
                if not os.path.exists(fat_final):
                    fat_final = os.path.join(out_fat_final, "fat_coreg.nii.gz")
                
                if os.path.exists(fat_final):
                    nifti2dicom.nifti_to_dicom(fat_final, fixed_path, fat_path, os.path.dirname(fat_final))
                    print(f"   ---> OK FAT DICOM en: {os.path.basename(os.path.dirname(fat_final))}")

                shutil.rmtree(tmp_dir, ignore_errors=True)
                metricas.append({"Paciente": paciente_id, "Sesion": sesion, "Metric": metric})

            except Exception as e:
                print(f"ERROR en {paciente_id} Sesion {sesion}: {e}")
                continue

    except Exception as e:
        print(f"ERROR CRÍTICO en paciente {paciente_id}: {e}")
        continue


# ── EXPORT ──

df = pd.DataFrame(metricas)
ruta = os.path.join(base_path, "resultados.xlsx")
df.to_excel(ruta, index=False)

print("\n✔ TERMINADO")
print(f"Excel: {ruta}")