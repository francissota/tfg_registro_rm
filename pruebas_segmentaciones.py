from totalsegmentator.python_api import totalsegmentator
import SimpleITK as sitk
import numpy as np

RUTA_INPUT  = r"Francis\501_02_v2a_pelvis.nii.gz"

if __name__ == "__main__":
    #calcular el volumen de la ruta d eentrada que es una mascara sin asumir voxel size
    img = sitk.ReadImage(RUTA_INPUT)
    arr = sitk.GetArrayFromImage(img)
    print("Numero de voxeles segmentados: ", np.sum(arr > 0))
    print("Volumen segmentado (mm³): ", np.sum(arr > 0) * img.GetSpacing()[0] * img.GetSpacing()[1] * img.GetSpacing()[2])
    
    