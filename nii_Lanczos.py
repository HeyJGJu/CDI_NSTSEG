import os
import cv2
import numpy as np
import pydicom
import SimpleITK as sitk

def lanczos_interpolation(img, new_size):
    return cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4)

input_folder = 'path_to_your_input_folder'
output_folder_dcm = 'path_to_your_output_folder_dcm'
output_folder_nii = 'path_to_your_output_folder_nii'

if not os.path.exists(output_folder_dcm):
    os.makedirs(output_folder_dcm)

if not os.path.exists(output_folder_nii):
    os.makedirs(output_folder_nii)

for filename in os.listdir(input_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img = cv2.imread(os.path.join(input_folder, filename), cv2.IMREAD_GRAYSCALE)
        new_size = (512, 512)
        img_resized = lanczos_interpolation(img, new_size)
        dcm_filename = os.path.splitext(filename)[0] + '.dcm'
        dcm_file_path = os.path.join(output_folder_dcm, dcm_filename)
        dicom = pydicom.Dataset()
        dicom.file_meta = pydicom.FileMetaDataset()
        dicom.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
        dicom.PixelData = img_resized.tobytes()
        dicom.Rows, dicom.Columns = img_resized.shape
        dicom.ImageType = ['DERIVED', 'PRIMARY', 'AXIAL']
        dicom.save_as(dcm_file_path)
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(output_folder_dcm)
reader.SetFileNames(dicom_names)
image = reader.Execute()
output_file_path_nii = os.path.join(output_folder_nii, 'output.nii')
sitk.WriteImage(image, output_file_path_nii)
