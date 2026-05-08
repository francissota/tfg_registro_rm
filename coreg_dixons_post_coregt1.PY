import os
import glob
import shutil
import subprocess
import json
import pandas as pd

metricas = []

for i in [6, 16, 22]:
    
    main_repo_path = r"D:\tfg_francis\RESPECT_Co-Registration_Module-main"
    paciente_id = f"5{i:02d}_01"
    sesion_fija = "03"
    for sesion in ["02", "01"]:
        patient_folder = fr"D:\tfg_francis\RESPECT_CENTER01\{paciente_id}"
        folder_fixed = os.path.join(patient_folder, sesion_fija)  
        folder_moving = os.path.join(patient_folder, sesion)     
        
        param_content = """// DIXON to DIXON PARAMETERS

        // ImageTypes ***
        (FixedInternalImagePixelType "float")
        (MovingInternalImagePixelType "float")
        (FixedImageDimension 3)
        (MovingImageDimension 3)
        (UseDirectionCosines "true")

        // Components 
        (Registration "MultiMetricMultiResolutionRegistration")
        (Interpolator "BSplineInterpolator")
        (ResampleInterpolator "FinalBSplineInterpolator")
        (Resampler "DefaultResampler")
        (BSplineInterpolationOrder 1)
        (FinalBSplineInterpolationOrder 1)
        (FixedImagePyramid "FixedSmoothingImagePyramid")
        (MovingImagePyramid "MovingSmoothingImagePyramid")
        (Optimizer "AdaptiveStochasticGradientDescent")
        (Transform "BSplineTransform")
        (HowToCombineTransforms "Compose")

        // Metrics: MI for similarity + BendingEnergy for regularization
        (Metric "AdvancedMattesMutualInformation" "TransformBendingEnergyPenalty")
        (Metric0Weight 1.0)

        // HIGH AND CONSTANT PENALTY: Prohibits internal deformations
        (Metric1Weight 500.0)
        (NumberOfHistogramBins 64)
        (UseFastAndLowMemoryVersion "true")

        // LARGE MESH: 25mm. The kidney will move almost as a solid block
        (FinalGridSpacingInPhysicalUnits 25.0)

        // Optimizer settings 
        (NumberOfResolutions 3) // Lowered to 3 to avoid overfitting
        (MaximumNumberOfIterations 2000)
        (MaximumStepLength 0.25)
        (MinimumGradientMagnitude 1e-8)
        (MinimumStepLength 0.001)

        (AutomaticParameterEstimation "true")
        (AutomaticTransformInitialization "false")
        (ASGDParameterEstimationMethod "Original")

        // Pyramid settings (3 levels) 
        (ImagePyramidSchedule 4 4 2  2 2 1  1 1 1)

        // parameters 
        (NumberOfSpatialSamples 4096)
        (NewSamplesEveryIteration "true")
        (ImageSampler "RandomCoordinate")
        (CheckNumberOfSamples "true")
        (MaximumNumberOfSamplingAttempts 10)

        // Mask settings 
        (ErodeMask "false")
        (ErodeFixedMask "false")

        // Output settings
        (DefaultPixelValue 0)
        (WriteResultImage "true")
        (ResultImagePixelType "float")
        (ResultImageFormat "nii.gz")
        (CompressResultImage "true")
        """

        destination_dir = os.path.join(folder_moving, 'anonym')
        dest_params = os.path.join(destination_dir, 'parametermaps')
        os.makedirs(dest_params, exist_ok=True)

        param_file = os.path.join(dest_params, 'par_pairwise_RESPECT.txt')
        with open(param_file, 'w', encoding='utf-8') as f:
            f.write(param_content)

        config_content = {
            "translation": {
                "MaximumNumberOfIterations": ["2000"],
                "AutomaticTransformInitialization": ["true"],
                "AutomaticTransformInitializationMethod": ["GeometricalCenter"],
                "NumberOfHistogramBins": ["64"],
                "NumberOfSpatialSamples": ["8192"]
            },
            "rigid": {
                "MaximumNumberOfIterations": ["2000"],
                "NumberOfHistogramBins": ["64"],
                "NumberOfSpatialSamples": ["8192"]
            },
            "affine": {
                "MaximumNumberOfIterations": ["2000"],
                "NumberOfHistogramBins": ["64"],
                "NumberOfSpatialSamples": ["8192"],
                "MaximumStepLength": ["0.5"]
            },
            "Dixon": {
                "NumberOfHistogramBins": ["64"],
                "NumberOfSpatialSamples": ["8192"]
            }
        }

        config_file = os.path.join(dest_params, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_content, f, indent=4)

        output_coreg = os.path.join(folder_moving, f'dix_coreg_con_dix_{sesion_fija}')
        if os.path.exists(output_coreg):
            shutil.rmtree(output_coreg)
        os.makedirs(output_coreg, exist_ok=True)

        container_name = f"coreg_dixon_temporal_{paciente_id}_{sesion}"
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True)

        try:
            fixed_path = os.path.join(folder_fixed, "coreg_dixons_t1", "output", "Dixon")
            if not os.path.exists(fixed_path):
                raise IndexError(f"No existe la carpeta Fija en {fixed_path}")

            water_path = [f for f in glob.glob(os.path.join(folder_moving, '*_IP*')) if os.path.isdir(f) and any(glob.glob(os.path.join(f, '*.IMA')) or glob.glob(os.path.join(f, '*.dcm')))][0]
            fat_path = [f for f in glob.glob(os.path.join(folder_moving, '*FAT*')) if os.path.isdir(f) and any(glob.glob(os.path.join(f, '*.IMA')) or glob.glob(os.path.join(f, '*.dcm')))][0]
        except IndexError as e:
            print(f"ERROR: Faltan carpetas para el paciente {paciente_id}")
            continue

        water_name = os.path.basename(water_path)
        fat_name = os.path.basename(fat_path)

        print(f"\nProcesando paciente: {paciente_id} ---")
        print(f"Agua (móvil): {water_name}")
        print(f"Grasa (móvil): {fat_name}")

        multistage = "trad"

        command = (
            f'docker run --name {container_name} '
            f'-v "{os.path.abspath(patient_folder)}":/app/data '
            f'siria_pipeline '
            f'"data/{sesion_fija}/coreg_dixons_t1/output/Dixon/, data/{sesion}/{water_name}/, data/{sesion}/{fat_name}/, '
            f'{multistage}, '
            f'data/{sesion}/anonym/parametermaps/par_pairwise_RESPECT.txt, '
            f'data/{sesion}/anonym/parametermaps/config.json, Dixon" '
            f'"output/"'
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
        nombre_excel = "Resultados_IP_Temporal_Pacientes.xlsx"
        ruta_excel = os.path.join(main_repo_path, nombre_excel)
        df.to_excel(ruta_excel, index=False)
    else:
        print("\nFalló")