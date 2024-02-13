import torch

debug_info_gpu=torch.load("../benchmarks/debug_info_mamba_gpu.pth")
debug_info_cpu=torch.load("../../mamba-cpu/debug_info_mamba_cpu.pth")

print("GPU KEYS",debug_info_gpu.keys())
print("CPU KEYS",debug_info_cpu.keys())
assert(all([x in debug_info_cpu.keys() for x in debug_info_gpu.keys()]))
assert(all([x in debug_info_gpu.keys() for x in debug_info_cpu.keys()]))

def check_abs_error(t1:torch.Tensor,t2:torch.Tensor):
  t1=t1.to("cpu")
  t2=t2.to("cpu")
  absolute_error = torch.abs(t1 - t2)
  max_abs_err=torch.max(absolute_error)
  abs_thresh=1e-4
  if max_abs_err>abs_thresh:
    print("MISMATCH!!!")
    print("ABS ERROR:",max_abs_err)
    print("T1:",t1)
    print("T2:",t2)
    exit(1)

def compare_tensors(gpu_info=debug_info_gpu,cpu_info=debug_info_cpu,tensor_name=""):
  gpu_tensor=gpu_info[tensor_name]
  cpu_tensor=cpu_info[tensor_name]
  print(f"COMPARING TENSOR {tensor_name}")
  check_abs_error(gpu_tensor,cpu_tensor)
  print("VALUES MATCH")


compare_tensors(debug_info_gpu,debug_info_cpu,"input_ids")
compare_tensors(debug_info_gpu,debug_info_cpu,"embedding_layer_states")
compare_tensors(debug_info_gpu,debug_info_cpu,"hidden_states_first_layer_before_mixer")
compare_tensors(debug_info_gpu,debug_info_cpu,"first_layer_out_states")

for i in range(debug_info_gpu["curr_step"]):
  curr_info_gpu=debug_info_gpu["steps"][i]
  curr_info_cpu=debug_info_cpu["steps"][i]
  print("INFERENCE STEP ID IS",i)
  compare_tensors(curr_info_gpu,curr_info_cpu,"logits")
