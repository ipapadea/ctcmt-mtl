_base_ = ['./yolox_x_mtl_v2_8xb4-24e_cityscapes.py']

optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
test_cfg = dict(type='TestLoop', fp16=True)
