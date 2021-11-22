def get_data_scaler(centred):
  """Data normalizer. Assume data are always in [0, 1]."""
  if centred:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(centred):
  """Inverse data normalizer."""
  if centred:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x