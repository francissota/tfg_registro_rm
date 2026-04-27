import sys
import os
import numpy as np
import SimpleITK as sitk
import pydicom

def buscar_archivo_nii(carpeta):
    if not os.path.exists(carpeta):
        return None
    archivos = [f for f in os.listdir(carpeta) if f.endswith('.nii') or f.endswith('.nii.gz')]
    return os.path.join(carpeta, archivos[0]) if archivos else None

if __name__ == "__main__":

    for i in ["v1a", "v2a", "v3a"]:
        pacientes = [501, 502, 503, 504, 505, 506, 507, 508, 509, 510]
        root_dir = "RESPECT_CENTER003"
        for j in pacientes:
            paciente_id = f"{j}_03"
            print(f"\n" + "="*50)
            print(f"PROCESANDO PACIENTE: {paciente_id}")
            print("="*50)
            
            path_paciente = os.path.join(root_dir, paciente_id, f"{j}-03_{i}")
            
            if not os.path.exists(path_paciente):
                print(f"La carpeta {path_paciente} no existe. Saltando...")
                continue
            
            out_dir = os.path.join(path_paciente, "imagen_IP")
            os.makedirs(out_dir, exist_ok=True)

            subcarpetas = os.listdir(path_paciente)
            folder_w = next((f for f in subcarpetas if "WATER" in f), None)
            folder_f = next((f for f in subcarpetas if "FAT" in f), None)

            if not folder_w or not folder_f:
                print(f"Error: No se encuentran carpetas Dixon para {paciente_id}")
                continue

            ruta_w_nii = buscar_archivo_nii(os.path.join(path_paciente, folder_w))
            ruta_f_nii = buscar_archivo_nii(os.path.join(path_paciente, folder_f))
            
            if not ruta_w_nii:
                print(f"Error: No se encontró archivo .nii en {folder_w}")
                continue
            if not ruta_f_nii:
                print(f"Error: No se encontró archivo .nii en {folder_f}")
                continue
            
            try:
                w = sitk.ReadImage(ruta_w_nii)
                f = sitk.ReadImage(ruta_f_nii)
                w_np = sitk.GetArrayFromImage(w)
                f_np = sitk.GetArrayFromImage(f)
                ip_np = w_np + f_np
                ip = sitk.GetImageFromArray(ip_np)
                ip.CopyInformation(w)
                
                ruta_salida_nii = os.path.join(out_dir, "imagen_IP.nii.gz")
                sitk.WriteImage(ip, ruta_salida_nii)
                print(f"Imagen IP guardada en: {ruta_salida_nii}")
            except Exception as e:
                print(f"Error al procesar imágenes: {e}")
                continue
    
            
            path_w_dicom = os.path.join(path_paciente, folder_w)
            path_f_dicom = os.path.join(path_paciente, folder_f)
            
            archivos_w = sorted([f for f in os.listdir(path_w_dicom) if f.endswith('.IMA') or f.endswith('.dcm')])
            archivos_f = sorted([f for f in os.listdir(path_f_dicom) if f.endswith('.IMA') or f.endswith('.dcm')])

            if not archivos_w or not archivos_f:
                print(f"Error: No se encuentran archivos DICOM en las carpetas Dixon")
                continue

            try:
                for aw, af in zip(archivos_w, archivos_f):
                    ds_w = pydicom.dcmread(os.path.join(path_w_dicom, aw))
                    ds_f = pydicom.dcmread(os.path.join(path_f_dicom, af))
                    
                    # Extraemos las matrices 2D de píxeles
                    arr_w = ds_w.pixel_array.astype(np.float32)
                    arr_f = ds_f.pixel_array.astype(np.float32)
                    
                    arr_ip = arr_w + arr_f
                    
                    arr_ip = np.clip(arr_ip, 0, 65535).astype(ds_w.pixel_array.dtype)
                    
                    ds_w.PixelData = arr_ip.tobytes()
                    
                    ds_w.SeriesDescription = "InPhase_Generated"
                    
                    nombre_salida = aw.replace('_W', '_IP') if '_W' in aw else f"IP_{aw}"
                    ds_w.save_as(os.path.join(out_dir, nombre_salida))
                
                print(f"Se procesaron {len(archivos_w)} archivos DICOM correctamente")
            except Exception as e:
                print(f"Error al procesar archivos DICOM: {e}")
                continue
                