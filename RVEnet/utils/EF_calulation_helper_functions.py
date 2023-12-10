def quantize_EF(EF_value: float, nbr_of_bins: int=10, EF_min: int=10, EF_max: int=80):

    EF_range = EF_max - EF_min + 1
    EF_step = EF_range / nbr_of_bins
    EF_class = int((EF_value - EF_min) / EF_step)

    EF_class = max(0,EF_class)
    EF_class = min(EF_class,nbr_of_bins-1)

    return EF_class