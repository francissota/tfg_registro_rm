import os
import glob
import shutil
import subprocess
import json
import pandas as pd

main_repo_path = r"D:\tfg_francis\RESPECT_Co-Registration_Module-main"
metricas = []


param_content_t1t2 = """// T1 (fixed) to T2 (moving) - OPTIMIZED PARAMETERS v3
// Same resolution and matrix size: 1.5 x 1.5 x 5 mm | 256 x 256 x 17

// *** ImageTypes ***
(FixedInternalImagePixelType "float")
(MovingInternalImagePixelType "float")
(FixedImageDimension 3)
(MovingImageDimension 3)
(UseDirectionCosines "true")

// *** Components ***
(Registration "MultiMetricMultiResolutionRegistration")
(Interpolator "BSplineInterpolator")
(ResampleInterpolator "FinalBSplineInterpolator")
(Resampler "DefaultResampler")
(BSplineInterpolationOrder 1)
(FinalBSplineInterpolationOrder 3)
(FixedImagePyramid "FixedSmoothingImagePyramid")
(MovingImagePyramid "MovingSmoothingImagePyramid")
(Optimizer "AdaptiveStochasticGradientDescent")
(Transform "BSplineTransform")
(HowToCombineTransforms "Compose")

// *** Metrics: MI for similarity + BendingEnergy for regularity ***
(Metric "AdvancedMattesMutualInformation" "TransformBendingEnergyPenalty")
(Metric0Weight 1.0)

// ---> ¡CORRECCIÓN 1! Reducimos la penalización de 700 a 50 para permitir que la malla elástica alinee el riñón <---
(Metric1Weight 50.0)
(NumberOfHistogramBins 64)
(UseFastAndLowMemoryVersion "true")

// ---> ¡CORRECCIÓN 2! Malla ajustada a 18mm para capturar mejor la escala anatómica de los riñones <---
(FinalGridSpacingInPhysicalUnits 18.0)

// *** Optimizer settings ***
(NumberOfResolutions 3)
(MaximumNumberOfIterations 1000)
(MaximumStepLength 0.15)
(MinimumGradientMagnitude 1e-8)
(MinimumStepLength 0.001)

(AutomaticParameterEstimation "true")
(AutomaticTransformInitialization "true")
(AutomaticTransformInitializationMethod "GeometricalCenter")
(ASGDParameterEstimationMethod "Original")

// *** Pyramid settings (3 levels) ***
// Z has only 17 slices -> do not downsample in Z beyond x1
(ImagePyramidSchedule 4 4 1  2 2 1  1 1 1)

// *** Sampler parameters ***
(NumberOfSpatialSamples 8192)
(NewSamplesEveryIteration "true")
(ImageSampler "RandomCoordinate")
(CheckNumberOfSamples "true")
(MaximumNumberOfSamplingAttempts 10)

// *** Mask settings ***
(ErodeMask "false")
(ErodeFixedMask "false")

// *** Output settings ***
(DefaultPixelValue 0)
(WriteResultImage "true")
(ResultImagePixelType "float")
(ResultImageFormat "nii.gz")
(CompressResultImage "true")
"""

# ---------------------------------------------------------------------------
# Configuración de las etapas (La etapa afín no se usará)
# ---------------------------------------------------------------------------
config_content = {
    "translation": {
        "MaximumNumberOfIterations": ["500"],
        "AutomaticTransformInitialization": ["true"],
        "AutomaticTransformInitializationMethod": ["GeometricalCenter"],
        "NumberOfHistogramBins": ["64"],
        "NumberOfSpatialSamples": ["8192"]
    },
    "rigid": {
        "MaximumNumberOfIterations": ["500"],
        "NumberOfHistogramBins": ["64"],
        "NumberOfSpatialSamples": ["8192"]
    },
    "affine": {
        "MaximumNumberOfIterations": ["500"],
        "NumberOfHistogramBins": ["64"],
        "NumberOfSpatialSamples": ["8192"],
        "MaximumStepLength": ["0.3"]
    }
}

# ---------------------------------------------------------------------------
# Bucle principal: pacientes + sesiones
# NOTA: Está en range(3, 4) para probar solo al paciente 503. 
# Recuerda cambiar a range(1, 21) para correr todo tu dataset.
# ---------------------------------------------------------------------------
for i in range(3, 4):

    paciente_id = f"5{i:02d}_02"
    paciente_base = fr"D:\tfg_francis\SANOS\{paciente_id}"

    if not os.path.exists(paciente_base):
        print(f"ERROR: Carpeta {paciente_base} no existe")
        continue

    for sesion in ["01", "02", "03"]:

        folder = os.path.join(paciente_base, sesion)

        if not os.path.exists(folder):
            print(f"ADVERTENCIA: {folder} no existe, saltando...")
            continue

        # ── Archivos de parámetros ──────────────────────────────────────────
        destination_dir = os.path.join(folder, 'anonym')
        dest_params = os.path.join(destination_dir, 'parametermaps')
        os.makedirs(dest_params, exist_ok=True)

        param_file = os.path.join(dest_params, 'par_pairwise_t1_t2.txt')
        with open(param_file, 'w', encoding='utf-8') as f:
            f.write(param_content_t1t2)

        config_file = os.path.join(dest_params, 'config.json')
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_content, f, indent=4)

        # ── Carpeta de salida ───────────────────────────────────────────────
        output_coreg = os.path.join(folder, 't1t2_coregistration')
        if os.path.exists(output_coreg):
            shutil.rmtree(output_coreg)
        os.makedirs(output_coreg, exist_ok=True)

        # ── Contenedor Docker ───────────────────────────────────────────────
        container_name = f"coreg_t1t2_{paciente_id}_s{sesion}"
        subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True)

        # ── Localizar carpetas T1 y T2 ─────────────────────────────────────
        try:
            t1_path = [f for f in glob.glob(os.path.join(folder, '*T1W*')) if os.path.isdir(f)][0]
            t2_path = [f for f in glob.glob(os.path.join(folder, '*T2W*')) if os.path.isdir(f)][0]
        except IndexError:
            print(f"ERROR: Faltan carpetas T1W o T2W en {paciente_id}/sesion {sesion}")
            continue

        t1_name = os.path.basename(t1_path)
        t2_name = os.path.basename(t2_path)

        print(f"\nProcesando {paciente_id} [sesion {sesion}]")
        print(f"  T1 (fija):   {t1_name}")
        print(f"  T2 (móvil):  {t2_name}\n")

        # ── Comando Docker ──────────────────────────────────────────────────
        # ---> ¡CORRECCIÓN 3! Quitamos la 'a' (affine). El pipeline es: Traslación(t) -> Rígido(r) -> Deformable(d) <---
        multistage = "trd"

        command = (
            f'docker run --name {container_name} '
            f'-v "{os.path.abspath(paciente_base)}":/app/data '
            f'siria_pipeline '
            f'"data/{sesion}/{t1_name}/, data/{sesion}/{t2_name}/, {multistage}, '
            f'data/{sesion}/anonym/parametermaps/par_pairwise_t1_t2.txt, '
            f'data/{sesion}/anonym/parametermaps/config.json, T2w" '
            f'"output/"'
        )

        print(f"Running registration in container: {container_name} ...\n")

        # ── Ejecución y captura de salida ───────────────────────────────────
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
                try:
                    metric_val = float(valor_texto)
                except ValueError:
                    metric_val = "N/A"

        proceso.wait()
        exit_code = proceso.returncode

        if exit_code == 0:
            os.system(f'docker cp {container_name}:/app/output "{os.path.abspath(output_coreg)}"')
            subprocess.run(['docker', 'rm', '-f', container_name], capture_output=True)
            print(f"  ---> Métrica final: {metric_val}")

            metricas.append({
                "Paciente": paciente_id,
                "Sesion": sesion,
                "Metrica": metric_val,
            })

        else:
            print(f"\nERROR: El contenedor {container_name} falló (exit code {exit_code})")

# ---------------------------------------------------------------------------
# Guardar resultados en Excel
# ---------------------------------------------------------------------------
if metricas:
    df = pd.DataFrame(metricas)
    nombre_excel = "Resultados_T1_T2_Coregistration.xlsx"
    ruta_excel = os.path.join(main_repo_path, nombre_excel)
    df.to_excel(ruta_excel, index=False)
    print(f"\n¡Proceso terminado! Excel guardado en: {ruta_excel}")
else:
    print("\nNo se pudo procesar ningún paciente. No se generó Excel.")