import pydicom
import os
import glob

def get_direction(folder):
    # Buscar todos los archivos dicom en la carpeta
    files = [f for f in os.listdir(folder) if f.endswith('.dcm') or f.endswith('.IMA')]
    
    if not files:
        print("  ⚠️ No se encontraron archivos .dcm o .IMA en esta carpeta.")
        return

    # Ordenar los archivos alfabéticamente para coger el primero y el último corte
    files.sort()
    
    # Leer el primer y el último archivo
    ds_first = pydicom.dcmread(os.path.join(folder, files[0]), force=True)
    ds_last = pydicom.dcmread(os.path.join(folder, files[-1]), force=True)

    iop = ds_first.get((0x0020, 0x0037), None)
    pos_first = ds_first.get((0x0020, 0x0032), None)
    pos_last = ds_last.get((0x0020, 0x0032), None)

    print(f"  Carpeta analizada: {os.path.basename(folder)}")
    print(f"  ImageOrientationPatient (Cosenos): {iop.value if iop else 'N/A'}")
    
    if pos_first and pos_last:
        # Extraemos la tercera coordenada (Z)
        z_first = float(pos_first.value[2])
        z_last = float(pos_last.value[2])
        
        print(f"  Z Inicial (Corte 1):    {z_first:.3f}")
        print(f"  Z Final   (Corte N):    {z_last:.3f}")
        
        # Comparamos para ver la dirección
        if z_last > z_first:
            print("  -> DIRECCIÓN: SUBIENDO (Inferior a Superior)")
        elif z_last < z_first:
            print("  -> DIRECCIÓN: BAJANDO (Superior a Inferior)")
        else:
            print("  -> DIRECCIÓN: PLANA (No hay cambio en Z, revisar serie)")
    else:
        print("  ImagePositionPatient: N/A")

# ── RUTAS AUTOMÁTICAS CON GLOB ──
base_path = r"D:\tfg_francis\RESPECT_CENTER01\501_01"

try:
    water_ses1 = [f for f in glob.glob(os.path.join(base_path, '01', '*WATER*')) if os.path.isdir(f)][0]
    water_ses2 = [f for f in glob.glob(os.path.join(base_path, '02/water_coreg_con_water_01', '*Dixon*')) if os.path.isdir(f)][0]
except IndexError:
    print("⚠️ Error: No se encontró la carpeta WATER en 01 o 02.")
    exit()

print("\n=== SESIÓN 1 ===")
get_direction(water_ses1)

print("\n=== SESIÓN 2 ===")
get_direction(water_ses2)