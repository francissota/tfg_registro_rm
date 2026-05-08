import os
import glob
import shutil
import subprocess
import json
import re


for i in [8]: 
    paciente_id = f"5{i:02d}_01"
    sesion_fija = "02"
    
    main_repo_path = r"D:\tfg_francis\RESPECT_Co-Registration_Module-main"

    for sesion in ["01", "03"]:
        patient_folder = fr"D:\tfg_francis\RESPECT_CENTER01\{paciente_id}"
        folder_fixed = os.path.join(patient_folder, sesion_fija)
        folder_moving = os.path.join(patient_folder, sesion)

        # ── 1. Parámetros Elastix Exclusivos para T1 -> T1 ──
        param_content = """// T1 to T1 REGISTRATION PARAMETERS
        (FixedInternalImagePixelType "float")
        (MovingInternalImagePixelType "float")
        (FixedImageDimension 3)
        (MovingImageDimension 3)
        (UseDirectionCosines "true")

        (Registration "MultiResolutionRegistration") // CORREGIDO: Sin MultiMetric
        (Interpolator "BSplineInterpolator")
        (ResampleInterpolator "FinalBSplineInterpolator")
        (Resampler "DefaultResampler")

        (Optimizer "AdaptiveStochasticGradientDescent")
        (Transform "AffineTransform")
        (HowToCombineTransforms "Compose")

        (Metric "AdvancedMattesMutualInformation")
        (NumberOfHistogramBins 64)
        (UseFastAndLowMemoryVersion "true")

        (NumberOfResolutions 3)
        (MaximumNumberOfIterations 1500)
        (MaximumStepLength 0.5)

        (AutomaticParameterEstimation "true")
        (AutomaticTransformInitialization "true")
        (AutomaticTransformInitializationMethod "GeometricalCenter")
        (ASGDParameterEstimationMethod "Original")

        (ImageSampler "RandomCoordinate")
        (NumberOfSpatialSamples 4096)
        (NewSamplesEveryIteration "true")

        (ErodeMask "false")
        (DefaultPixelValue 0)
        (WriteResultImage "true")
        (ResultImagePixelType "float")
        (ResultImageFormat "nii.gz")
        (CompressResultImage "true")
        """

        destination_dir = os.path.join(folder_moving, 'anonym')
        dest_params = os.path.join(destination_dir, 'parametermaps')
        os.makedirs(dest_params, exist_ok=True)

        param_file = os.path.join(dest_params, 'par_t1_t1.txt')
        with open(param_file, 'w', encoding='utf-8') as f:
            f.write(param_content)

        config_content = {
            "translation": {
                "MaximumNumberOfIterations": ["1000"],
                "AutomaticTransformInitialization": ["true"],
                "AutomaticTransformInitializationMethod": ["GeometricalCenter"],
                "NumberOfHistogramBins": ["64"],
                "NumberOfSpatialSamples": ["4096"]
            },
            "rigid": {
                "MaximumNumberOfIterations": ["1000"],
                "NumberOfHistogramBins": ["64"],
                "NumberOfSpatialSamples": ["4096"]
            },
            "affine": {
                "MaximumNumberOfIterations": ["1500"],
                "NumberOfHistogramBins": ["64"],
                "NumberOfSpatialSamples": ["4096"],
                "MaximumStepLength": ["0.5"]
            }
        }

        config_file = os.path.join(dest_params, 'config_t1.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_content, f, indent=4)

        # ── 2. Buscar las imágenes T1 ──
        try:
            t1_fixed_path = [f for f in glob.glob(os.path.join(folder_fixed, '*_T1w*')) if os.path.isdir(f)][0]
            t1_moving_path = [f for f in glob.glob(os.path.join(folder_moving, '*_T1w*')) if os.path.isdir(f)][0]
            t1_water_path = [f for f in glob.glob(os.path.join(folder_moving, "coreg_dixons_t1/output/Dixon")) if os.path.isdir(f)][0]

        except IndexError:
            print(f"⚠️ Aviso: Faltan carpetas T1 para el paciente {paciente_id} sesión {sesion}")
            continue

        t1_fixed_name = os.path.basename(t1_fixed_path)
        t1_moving_name = os.path.basename(t1_moving_path)
        t1_water_name = os.path.basename(t1_water_path)

        print(f"\n--- Registrando T1: Paciente {paciente_id} | Móvil: {sesion} -> Fija: {sesion_fija} ---")

        # ── 3. Ejecutar Contenedor T1 -> T1 ──
        output_coreg = os.path.join(folder_moving, f't1_coreg_con_t1_{sesion_fija}')
        if os.path.exists(output_coreg):
            shutil.rmtree(output_coreg)
        os.makedirs(output_coreg, exist_ok=True)

        container_name = f"coreg_t1_{paciente_id}_{sesion}"
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True)

        command = (
            f'docker run --name {container_name} '
            f'-v "{os.path.abspath(patient_folder)}":/app/data '
            f'siria_pipeline '
            f'"data/{sesion_fija}/{t1_fixed_name}/, '
            f'data/{sesion}/{t1_moving_name}/, data/{sesion}/coreg_dixons_t1/output/Dixon, '
            f'tra, '
            f'data/{sesion}/anonym/parametermaps/par_t1_t1.txt, '
            f'data/{sesion}/anonym/parametermaps/config_t1.json, Dixon" ' # CORREGIDO: Palabra clave 'Dixon'
            f'"output/"'
        )

        print(">> Ejecutando pipeline Afín (Translación -> Rígido -> Afín)...")
        proceso = subprocess.Popen(
            command, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace'
        )
        
        for linea in proceso.stdout:
            print(linea, end='') # CORREGIDO: Log activado para ver todo en vivo

        proceso.wait()

        if proceso.returncode == 0:
            os.system(f'docker cp {container_name}:/app/output "{output_coreg}"')
            subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True)
            print(f"OK: T1 registrada guardada en: {output_coreg}")
        else:
            print(f"ERROR: El contenedor falló para {paciente_id} sesion {sesion}")
            subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True)