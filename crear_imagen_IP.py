import os
import numpy as np
import SimpleITK as sitk
import pydicom
import glob

def buscar_nii(folder, patron="*aligned_to_t1.nii*"):
    cands = glob.glob(os.path.join(folder, patron))
    if not cands:
        cands = glob.glob(os.path.join(folder, "*.nii.gz")) + \
                glob.glob(os.path.join(folder, "*.nii"))
    return cands[0] if cands else None

if __name__ == "__main__":

    root_dir = "RESPECT_CENTER01"
    pacientes_folders = ["518_01"]

    for pac_folder in pacientes_folders:
        pac_path = os.path.join(root_dir, pac_folder)

        print(f"PROCESANDO PACIENTE: {pac_folder}")



        sesion_fija = "02"
        
        if not sesion_fija:
            print(f"No se detectó sesión fija (base) para {pac_folder}")
            continue

        for ses in ["01", "02", "03"]:
            path_sesion = os.path.join(pac_path, ses)
            if not os.path.exists(path_sesion): continue
            
            print(f"\n SESION {ses} (base={sesion_fija}) ---")

            if ses == sesion_fija:
                folder_w = os.path.join(path_sesion, "coreg_dixons_t1", "output", "Dixon")
                folder_f = os.path.join(path_sesion, "coreg_dixons_t1", "output", "Dixon_b")
            else:
                folder_w = os.path.join(path_sesion, f"coreg_dixon_con_dixon{sesion_fija}", "output", "TransformNii")
                folder_f = os.path.join(path_sesion, f"coreg_dixon_con_dixon{sesion_fija}", "fat_result")

            ruta_w = buscar_nii(folder_w)
            ruta_f = buscar_nii(folder_f)

            if not ruta_w or not ruta_f:
                print(f"No se encontraron NIfTIs en {folder_w} o {folder_f}")
                continue

            print(f"Water: {os.path.basename(ruta_w)}")
            print(f"Fat:   {os.path.basename(ruta_f)}")


            try:
                w_img = sitk.ReadImage(ruta_w)
                f_img = sitk.ReadImage(ruta_f)
                
                w_arr = sitk.GetArrayFromImage(w_img).astype(float)
                f_arr = sitk.GetArrayFromImage(f_img).astype(float)
                ip_arr = w_arr + f_arr
                
                ip_img = sitk.GetImageFromArray(ip_arr)
                ip_img.CopyInformation(w_img)
                
                out_nii_dir = os.path.join(path_sesion, f"imagen_IP_mask{sesion_fija}")
                os.makedirs(out_nii_dir, exist_ok=True)
                
                ruta_salida = os.path.join(out_nii_dir, "imagen_IP.nii.gz")
                sitk.WriteImage(ip_img, ruta_salida)
                print(f"OK NIfTI IP: {ruta_salida}")
            except Exception as e:
                print(f"Error NIfTI: {e}")

            dcm_w_cands = [d for d in os.listdir(folder_w) if d.endswith(('.dcm', '.IMA'))]
            dcm_f_cands = [d for d in os.listdir(folder_f) if d.endswith(('.dcm', '.IMA'))]
            
            if dcm_w_cands and dcm_f_cands:
                try:
                    out_dcm_dir = os.path.join(path_sesion, f"imagen_IP_mask{sesion_fija}", "DICOM_IP")
                    os.makedirs(out_dcm_dir, exist_ok=True)
                    
                    for dw, df in zip(sorted(dcm_w_cands), sorted(dcm_f_cands)):
                        ds_w = pydicom.dcmread(os.path.join(folder_w, dw))
                        ds_f = pydicom.dcmread(os.path.join(folder_f, df))
                        
                        arr_ip = ds_w.pixel_array.astype(float) + ds_f.pixel_array.astype(float)
                        arr_ip = np.clip(arr_ip, 0, 65535).astype(ds_w.pixel_array.dtype)
                        
                        ds_w.PixelData = arr_ip.tobytes()
                        ds_w.SeriesDescription = "InPhase_Generated"
                        
                        ds_w.save_as(os.path.join(out_dcm_dir, dw.replace("_W", "_IP")))
                    print(f"OK DICOMs IP: {len(dcm_w_cands)} archivos")
                except Exception as e:
                    print(f"Error DICOM: {e}")
