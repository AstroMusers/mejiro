import sys

sys.path.append('/grad/bwedig/mejiro')

from mejiro.instruments.hwo import HWO

hwo = HWO()
print(hwo.derived_bandpass)
