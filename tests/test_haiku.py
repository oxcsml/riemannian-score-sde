import haiku as hk

class Pass(hk.Module):
  def __call__(self, x):
    return x

print(hk.transform(Pass))