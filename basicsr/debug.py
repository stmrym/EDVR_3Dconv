from torchinfo import summary
from basicsr.archs.edvr_arch import EDVR
model = EDVR()

with open('summary.txt', 'w') as f:
    f.write(repr(summary(model=model, input_size=(2, 5, 3, 256, 256))))
# summary(
#     model,
#     input_size=(2, 5, 3, 256, 256),
#     )