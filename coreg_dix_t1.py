import os
import glob
import shutil
import subprocess
import json
import pandas as pd

metricas = []

for i in [1]:
    
    main_repo_path = r"D:\tfg_francis\RESPECT_Co-Registration_Module-main"
    paciente_id = f"5{i:02d}_01"
    folder = fr"D:\tfg_francis\RESPECT_CENTER01\{paciente_id}"
    folder_01 = os.path.join(folder, f"5{i:02d}-01_v2a")
    
    param_content = """// DIXON to T1 PARAMETERS

    // Image Types
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

    // Metrics: Mutual Information for similarity + Bending Energy for regularization
    (Metric "AdvancedMattesMutualInformation" "TransformBendingEnergyPenalty")
    (Metric0Weight 1.0)

    // High penalty to prohibit internal deformations
    (Metric1Weight 50.0)
    (NumberOfHistogramBins 64)
    (UseFastAndLowMemoryVersion "true")

    (FinalGridSpacingInPhysicalUnits 25.0 25.0 35.0)

    // Optimizer settings
    (NumberOfResolutions 3)
    (MaximumNumberOfIterations 2000)
    (MaximumStepLength 0.25)
    (MinimumGradientMagnitude 1e-8)
    (MinimumStepLength 0.001)

    (AutomaticParameterEstimation "true")
    (AutomaticTransformInitialization "false")
    (ASGDParameterEstimationMethod "Original")

    (ImagePyramidSchedule 4 4 1  2 2 1  1 1 1)

    // Sampler parameters
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

    destination_dir = os.path.join(folder_01, 'anonym')
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

    output_coreg = os.path.join(folder_01, 'coreg_dixons_t1')
    if os.path.exists(output_coreg):
        shutil.rmtree(output_coreg)
    os.makedirs(output_coreg, exist_ok=True)

    container_name = f"coreg_sin_mascara_{paciente_id}_sesion01"
    subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True)

    try:
        fixed_path = [f for f in glob.glob(os.path.join(folder_01, '*_T1w*')) if os.path.isdir(f)][0]
        water_path = [f for f in glob.glob(os.path.join(folder_01, "*_IP*")) if os.path.isdir(f)][0]
        fat_path = [f for f in glob.glob(os.path.join(folder_01, "*FAT*")) if os.path.isdir(f)][0]
    except IndexError:
        print(f"ERROR: Faltan carpetas para el paciente {paciente_id}")
        continue

    fixed_name = os.path.basename(fixed_path)
    water_name = os.path.basename(water_path)
    fat_name = os.path.basename(fat_path)

    print(f"\nProcesando paciente: {paciente_id} ---")
    print(f"T1 fijo: {fixed_name}")
    print(f"Agua (móvil): {water_name}")
    print(f"Grasa (móvil): {fat_name}")

    multistage = "trad"
    session_name = f"5{i:02d}-01_v2a"

    command = (
        f'docker run --name {container_name} '
        f'-v "{os.path.abspath(folder)}":/app/data '
        f'siria_pipeline '
        f'"data/{session_name}/{fixed_name}/, data/{session_name}/{water_name}/, data/{session_name}/{fat_name}/, '
        f'{multistage}, '
        f'data/{session_name}/anonym/parametermaps/par_pairwise_RESPECT.txt, '
        f'data/{session_name}/anonym/parametermaps/config.json, Dixon" '
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
    nombre_excel = "Resultados_SinMascara_Pacientes.xlsx"
    ruta_excel = os.path.join(main_repo_path, nombre_excel)
    df.to_excel(ruta_excel, index=False)
else:
    print("\nFalló")
