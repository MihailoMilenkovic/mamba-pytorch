import torch
import torch.nn as nn
import torch.onnx
from einops import rearrange

class RearrangeModule(nn.Module):
    def __init__(self):
        super(RearrangeModule, self).__init__()

    def forward(self, a):
        b=rearrange(a, 'm n -> n m 1')
        c=rearrange(b,'n m p -> (n m p)')
        return c

# Create an instance of the RearrangeModule
rearrange_model = RearrangeModule()

# Export the module
torch.onnx.export(rearrange_model,  # model being run
                  (torch.zeros((2, 3), dtype=torch.float16),),  # input tensor(s)
                  "rearrange.onnx",  # where to save the model (can be a file or file-like object)
                  verbose=True  # print out a lot of information about what the exporter is doing
                  )

# Call the module's forward method
a = torch.zeros((2, 3), dtype=torch.float16)
output = rearrange_model(a)
print(output)
