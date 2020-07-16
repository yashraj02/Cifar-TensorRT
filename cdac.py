import mxnet as mx
import time
import os

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


# Execute with TensorRT
print('Building TensorRT engine')
os.environ['MXNET_USE_TENSORRT'] = '1'
arg_params.update(aux_params)
all_params = dict([(k, v.as_in_context(mx.gpu(0))) for k, v in arg_params.items()])
executor = mx.contrib.tensorrt.tensorrt_bind(sym, ctx=mx.gpu(0), all_params=all_params,
                                             data=batch_shape, grad_req='null', force_rebind=True)
# Warmup
print('Warming up TensorRT')
for i in range(0, 10):
    y_gen = executor.forward(is_train=False, data=input[i])
    y_gen[0].wait_to_read()
 
# Timing
print('Starting TensorRT timed run')
start = time.process_time()
for i in range(0, len(input)):
    y_gen = executor.forward(is_train=False, data=input[i])
    y_gen[0].wait_to_read()
end = time.time()
print(time.process_time() - start)