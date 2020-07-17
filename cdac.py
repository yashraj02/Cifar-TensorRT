import mxnet as mx
import time
import os
import cv2
import numpy as np

print(1)

batch_shape = (1, 3, 256, 256)
# resnet18 = vision.resnet18_v2(pretrained=True)
# resnet18.hybridize()
# resnet18.forward(mx.nd.zeros(batch_shape))
# resnet18.export('resnet18_v2')
sym, arg_params, aux_params = mx.model.load_checkpoint('R50', 0)

print(2)


 
# Execute with MXNet
os.environ['MXNET_USE_TENSORRT'] = '0'
executor = sym.simple_bind(ctx=mx.gpu(0), data=batch_shape, grad_req='null', force_rebind=True)
executor.copy_params_from(arg_params, aux_params)

print(3)

# Create sample input
input = []
# Preprocessing
def processing(folder = "images"):
  print(4)
  for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_UNCHANGED)
    if img is not None:
      res = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
      res = res[np.newaxis, ...]
      res = np.rollaxis(res, 3, 1)
      input.append(res)
    else:
        print("no image found")
  print(5)
processing()
print(len(input),"Images Fed")

 
 
# Execute with MXNet
os.environ['MXNET_USE_TENSORRT'] = '0'
executor = sym.simple_bind(ctx=mx.gpu(0), data=batch_shape, grad_req='null', force_rebind=True)
executor.copy_params_from(arg_params, aux_params)
 
# Warmup
print('Warming up MXNet')
for i in range(0, 10):
    y_gen = executor.forward(is_train=False, data=input[i])
    y_gen[0].wait_to_read()
 
# Timing
print('Starting MXNet timed run')
start = time.process_time()
for i in range(0, len(input)):
    y_gen = executor.forward(is_train=False, data=input[i])
    y_gen[0].wait_to_read()
end = time.time()
print(time.process_time() - start)


# Execute with TensorRT FP32
print('Building TensorRT engine FP32')
trt_sym = sym.get_backend_symbol('TensorRT')
arg_params, aux_params = mx.contrib.tensorrt.init_tensorrt_params(trt_sym, arg_params, aux_params)
executor = trt_sym.simple_bind(ctx=mx.gpu(), data=batch_shape,
                               grad_req='null', force_rebind=True)
executor.copy_params_from(arg_params, aux_params)
# Warmup
print('Warming up TensorRT FP32')
for i in range(0, 10):
    y_gen = executor.forward(is_train=False, data=input[i])
    y_gen[0].wait_to_read()
 
# Timing
print('Starting TensorRT FP32 timed run')
start = time.process_time()
for i in range(0, len(input)):
    y_gen = executor.forward(is_train=False, data=input[i])
    y_gen[0].wait_to_read()
end = time.time()
print(time.process_time() - start)

tensorRt_atchitecture = executor.get_optimized_symbol()
mx.symbol.Symbol.save(tensorRt_atchitecture, 'trt_retinaFace_fp32.json')

                                             
                                             
# Execute with TensorRT FP16
print('Building TensorRT engine FP16')
trt_sym = sym.get_backend_symbol('TensorRT')
arg_params, aux_params = mx.contrib.tensorrt.init_tensorrt_params(trt_sym, arg_params, aux_params)
mx.contrib.tensorrt.set_use_fp16(True)
executor = trt_sym.simple_bind(ctx=mx.gpu(), data=batch_shape,
                               grad_req='null', force_rebind=True)
executor.copy_params_from(arg_params, aux_params)
                                             
                                             
                                           
# Warmup
print('Warming up TensorRT FP16')
for i in range(0, 10):
    y_gen = executor.forward(is_train=False, data=input[i])
    y_gen[0].wait_to_read()
 
# Timing
print('Starting TensorRT FP16 timed run')
start = time.process_time()
for i in range(0, len(input)):
    y_gen = executor.forward(is_train=False, data=input[i])
    y_gen[0].wait_to_read()
end = time.time()
print(time.process_time() - start)

tensorRt_atchitecture = executor.get_optimized_symbol()
mx.symbol.Symbol.save(tensorRt_atchitecture, 'trt_retinaFace_fp16.json')