def average_heart_cycles(input_dict: dict, task:str):
    # works only with regression files

    first_key = next(iter(input_dict))
    EF_key = list(input_dict[first_key].keys())[0]

    patient_dicoms = {}

    for heart_cycle_id, values in input_dict.items():

        id_parts = heart_cycle_id.split('__')
        patient_dicom_id = "{}__{}".format(id_parts[0], id_parts[1])

        if patient_dicom_id not in patient_dicoms:
            patient_dicoms[patient_dicom_id] = [float(values[EF_key])]
        else:
            patient_dicoms[patient_dicom_id].append(float(values[EF_key]))

    output_dict = {}

    for patient_dicom_id in patient_dicoms:
        averaged_heart_cycle = sum(patient_dicoms[patient_dicom_id]) / len(patient_dicoms[patient_dicom_id])
        if task=="classification":
            averaged_heart_cycle +=0.00001 # round to 1 (normal EF)
            averaged_heart_cycle = round(averaged_heart_cycle)
            
        output_dict[patient_dicom_id] = {EF_key: averaged_heart_cycle}

    return output_dict