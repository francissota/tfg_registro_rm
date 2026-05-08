import os
import glob
import shutil
import subprocess
import json
import pandas as pd

metricas = []

for i in [6, 16]:
    
    main_repo_path = r"D:\tfg_francis\RESPECT_Co-Registration_Module-main"
    paciente_id = f"5{i:02d}_01"
    folder = fr"D:\tfg_francis\RESPECT_CENTER01\{paciente_id}"
    session_name = "03"
    folder_01 = os.path.join(folder, session_name)

    destination_dir = os.path.join(folder_01, 'anonym')
    dest_params = os.path.join(destination_dir, 'parametermaps')
    os.makedirs(dest_params, exist_ok=True)

    config_content = {
        "translation": {
            "MaximumNumberOfIterations": ["500"],
            "AutomaticTransformInitialization": ["true"],
            "AutomaticTransformInitializationMethod": ["GeometricalCenter"],
            "NumberOfHistogramBins": ["64"],
            "NumberOfSpatialSamples": ["4096"]
        },
        "Dixon": {
            "NumberOfHistogramBins": ["64"],
            "NumberOfSpatialSamples": ["4096"]
        }
    }

    config_file = os.path.join(dest_params, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_content, f, indent=4)

    output_coreg = os.path.join(folder_01, 'coreg_dixons_t1')
    if os.path.exists(output_coreg):
        shutil.rmtree(output_coreg)
    os.makedirs(output_coreg, exist_ok=True)

    container_name = f"coreg_sin_mascara_{paciente_id}_sesion01_t"
    subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True)

    try:
        fixed_path = [f for f in glob.glob(os.path.join(folder_01, '*_T1w*')) if os.path.isdir(f)][0]
        water_path = [f for f in glob.glob(os.path.join(folder_01, "*WATER*")) if os.path.isdir(f)][0]
        fat_path   = [f for f in glob.glob(os.path.join(folder_01, "*FAT*"))  if os.path.isdir(f)][0]
    except IndexError:
        print(f"ERROR: Faltan carpetas para el paciente {paciente_id}")
        continue

    fixed_name = os.path.basename(fixed_path)
    water_name = os.path.basename(water_path)
    fat_name   = os.path.basename(fat_path)

    print(f"\nProcesando paciente: {paciente_id} ---")
    print(f"T1 fijo: {fixed_name}")
    print(f"IP (móvil): {water_name}")
    print(f"Grasa (móvil): {fat_name}")

    multistage = "t"

    command = (
        f'docker run --name {container_name} --entrypoint /bin/sh '
        f'-v "{os.path.abspath(folder)}":/app/data '
        f'-v "{main_repo_path}\\Coregistration_multistage_v3.py":/app/Coregistration_multistage_v3.py '
        f'-v "{main_repo_path}\\nifti2dicom.py":/app/nifti2dicom.py '
        f'siria_pipeline '
        f'-c "mkdir -p output/ && python Coregistration_multistage_v3.py '
        f'\'data/{session_name}/{fixed_name}/, data/{session_name}/{water_name}/, data/{session_name}/{fat_name}/, '
        f'{multistage}, '
        f'data/{session_name}/anonym/parametermaps/config.json, '
        f'data/{session_name}/anonym/parametermaps/config.json, Dixon\' '
        f'\'output/\'"'
    )

    metric_val = "N/A"
    proceso = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    for linea in proceso.stdout:
        print(linea, end='')
        if 'Final metric value' in linea:
            valor_texto = linea.split('=')[-1].strip()
            metric_val = float(valor_texto)

    proceso.wait()
    exit_code = proceso.returncode

    if exit_code == 0:
        os.system(f'docker cp {container_name}:/app/output "{os.path.abspath(output_coreg)}"')
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True)
        metricas.append({
            "Patient": paciente_id,
            "Metric": metric_val,
        })
    else:
        print(f"\nERROR: El contenedor {container_name} falló")

if metricas:
    df = pd.DataFrame(metricas)
    ruta_excel = os.path.join(main_repo_path, "Resultados_SinMascara_Pacientes.xlsx")
    df.to_excel(ruta_excel, index=False)
else:
    print("\nFalló")