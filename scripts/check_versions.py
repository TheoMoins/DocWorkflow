import torch, torchvision

tv = torchvision.__version__
t = torch.__version__
tv_minor = int(tv.split(".")[1])
t_minor = int(t.split(".")[1])
assert t_minor == tv_minor - 15, f"torchvision {tv} incompatible avec torch {t}"
print(f"torch {t} + torchvision {tv} : OK")
