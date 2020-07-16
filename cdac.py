import mxnet as mx
import time
import os

batch_shape = (1, 3, 256, 256)
# resnet18 = vision.resnet18_v2(pretrained=True)
# resnet18.hybridize()
# resnet18.forward(mx.nd.zeros(batch_shape))
# resnet18.export('resnet18_v2')
sym, arg_params, aux_params = mx.model.load_checkpoint('/R50', 0)

# Create sample input
input = mx.nd.zeros(batch_shape)
 
# Execute with MXNet
os.environ['MXNET_USE_TENSORRT'] = '0'
executor = sym.simple_bind(ctx=mx.gpu(0), data=batch_shape, grad_req='null', force_rebind=True)
executor.copy_params_from(arg_params, aux_params)


# Create sample input
input = mx.nd.zeros(batch_shape)
 
# Execute with MXNet
os.environ['MXNET_USE_TENSORRT'] = '0'
executor = sym.simple_bind(ctx=mx.gpu(0), data=batch_shape, grad_req='null', force_rebind=True)
executor.copy_params_from(arg_params, aux_params)
 
# Warmup
print('Warming up MXNet')
for i in range(0, 10):
    y_gen = executor.forward(is_train=False, data=input)
    y_gen[0].wait_to_read()
 
# Timing
print('Starting MXNet timed run')
start = time.process_time()
for i in range(0, 10000):
    y_gen = executor.forward(is_train=False, data=input)
    y_gen[0].wait_to_read()
end = time.time()
print(time.process_time() - start)