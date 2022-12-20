from pyts.approximation import PiecewiseAggregateApproximation

def get_disc_data(samples, paa_window):
    paa = PiecewiseAggregateApproximation(window_size=paa_window)
    paa_out = []
    for sample in samples:
        disc_sample = paa.transform(sample.T)
        paa_out.append(disc_sample.T)
    return paa_out
