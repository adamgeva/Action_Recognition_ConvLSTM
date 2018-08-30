import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


latest_ckp = '/home/ubadmin/summer_interns/Adam/action_recognition_v1/experiments/2018-08-30 14:51:23/checkpoint/-1618'
print_tensors_in_checkpoint_file(latest_ckp, all_tensors=False, tensor_name='MobilenetV2/expanded_conv_7/project/weights', all_tensor_names=False)
print('*************************************************************************')


latest_ckp = '/home/ADAMGE/action_recognition/models/mobile_net_imagenet/mobilenet_v2_1.0_224.ckpt'
print_tensors_in_checkpoint_file(latest_ckp, all_tensors=False, tensor_name='mobile_net/MobilenetV2/expanded_conv_7/project/weights', all_tensor_names=False)