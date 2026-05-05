def build_metadata_text(file_name, width, height, bit_depth, file_format,
                        modality="N/A", patient_name="N/A",
                        patient_age="N/A", body_part="N/A"):

    info_text = f"""
FILE INFORMATION
================

File Name    : {file_name}
Format       : {file_format}
Width        : {width} pixels
Height       : {height} pixels
Bit Depth    : {bit_depth} bits per pixel
Total Pixels : {width * height:,}

"""

    if file_format == "DICOM":
        info_text += f"""
DICOM MEDICAL TAGS
==================

Modality     : {modality}
Patient Name : {patient_name}
Patient Age  : {patient_age}
Body Part    : {body_part}
"""

    return info_text