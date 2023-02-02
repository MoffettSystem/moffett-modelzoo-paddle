import numpy as np
import sys
import os
import time
import struct
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../build"))

import spu_backend

def get_memory_info():
    os.system("mf-smi -i 0 --memory-used>>memory.txt")
    file = open('memory.txt','r')
    content = file.readlines()
    memory_used=int(content[0])
    file.close()
    os.system("rm memory.txt")
    return int(memory_used/1024/1024)    #单位转为M 

def np_float2np_bf16(arr):
    ''' Convert a numpy array of float to a numpy array
    of bf16 in uint16'''
    orig = arr.view('<u4')
    bias = np.bitwise_and(np.right_shift(orig, 16), 1) + 0x7FFF
    new_arr = np.right_shift(orig + bias, 16).astype('uint16')
    if isinstance(new_arr, np.uint16):
        return new_arr
    else:
        return np.where(new_arr == 32768, 0, new_arr)

# def s4_preprocess(input_data):
#     input_data = np.transpose(input_data,(0,2,3,1))
#     input_data_pad = np.pad(input_data,((0,0),(0,0),(0,0),(0,5)),mode="constant",constant_values=(0))
#     new_input_data = np_float2np_bf16(input_data_pad)
#     new_input_data = np.ascontiguousarray(new_input_data, np.uint16)
#     return new_input_data

def s4_preprocess(input_data,s4_input_shape):
    input_data = np.transpose(input_data,(0,2,3,1))
    s4_h,s4_w,s4_c = s4_input_shape
    pad_h,pad_w,pad_c = s4_h-input_data.shape[1],s4_w-input_data.shape[2],s4_c-input_data.shape[3]
    # import pdb;pdb.set_trace()
    input_data_pad = np.pad(input_data,((0,0),(0,pad_h),(0,pad_w),(0,pad_c)),mode="constant",constant_values=(0))
    new_input_data = np_float2np_bf16(input_data_pad)
    new_input_data = np.ascontiguousarray(new_input_data, np.uint16)
    return new_input_data



def bin2float(num):
    return struct.unpack('f', struct.pack('I', num))[0]


def s4_postprocess(output):
    shape = output.shape
    output = [bin2float(i << 16) for i in output.flatten().tolist()]
    output = np.array(output).reshape(shape)
    output = output.astype(np.float32)
    new_output = output[:,0:1000]
    return new_output

def mf_s4_infer(model_path,input_tensor):

    # input_tensor = pre_process(input_data)

    model_batch_size = 1
    # input_size = model_batch_size*224*224*8*2   #resnet50
    input_shape = input_tensor.shape
    # import pdb;pdb.set_trace()
    input_size = model_batch_size*input_shape[1]*input_shape[2]*input_shape[3]*2   #bf16 
    output_size = model_batch_size*1024*2  #bf16 

    buf_depth = 1
    ringbuffer = 1
    data_nums = input_tensor.shape[0]
    print("model_path:",model_path)
    print("input_size:",input_size)
    print("output_size:",output_size)
    spu_backend.spu_backend_init(0, model_path, False, model_batch_size,
                            [input_size], [output_size], False, buf_depth, ringbuffer)
    group_size = spu_backend.spu_backend_get_group_size(data_nums)
    # print("group_size: ", group_size)
    batch_size = model_batch_size * group_size
    print("batch_size: ", batch_size)
    # copy_batch = batch_size // model_batch_size
    # print("copy_batch: ", copy_batch)

    input_list = [np.vstack([input_tensor])]
    output_list = []
    output_data = np.empty([batch_size, 1024], dtype=np.uint16)
    output_list.append(output_data)

    start_inference = time.time()
    for i in range(5):
        #print("Inference: Iter = ", i)
        spu_backend.spu_backend_inference(batch_size, input_list, output_list)
    memory_used = get_memory_info()
    spu_backend.spu_backend_destroy()
    end_inference = time.time()
    inference_time = end_inference - start_inference
    print(f"model infer cost time: {inference_time * 1000} ms, fps: {data_nums / inference_time *5}")
    perf = data_nums / inference_time * 5
    return output_list[0],perf,memory_used



if __name__ == '__main__':
    # input_ids
    input_path = '/home/moffett/work/zhongming/project/paddle_compile_output/rn50_ci/paddle_resnet50#N1#224_sparseX1_cores_1/chip_output_core1_f29002a/tensors/sub1_0_E0_inputs_pad_memcpy'

    # compare_output
    compare_output = '/home/moffett/work/zhongming/project/paddle_compile_output/rn50_ci/paddle_resnet50#N1#224_sparseX1_cores_1/chip_output_core1_f29002a/tensors/E0_510-_save'

    model_path = '/home/moffett/work/zhongming/project/paddle_compile_output/rn50_ci/paddle_resnet50#N1#224_sparseX1_cores_1/chip_output_core1_f29002a/model.bin'

    model_batch_size = 1
    # numpy
    input_tensor = np.fromfile(input_path, dtype=np.uint16).reshape(model_batch_size, 224, 224, 8)
    compare_data = np.fromfile(compare_output, dtype=np.uint16).reshape(model_batch_size, 1024)

    copy_batch = 4000
    group_size = 4000
    # import pdb;pdb.set_trace()
    input_tensor = np.vstack([input_tensor] * copy_batch)
    start_inference = time.time()
    import pdb;pdb.set_trace()
    output = mf_s4_infer(model_path,input_tensor)
    end_inference = time.time()
    inference_time = end_inference - start_inference
    print(f"model infer cost time: {inference_time * 1000} ms, fps: {group_size / inference_time}")

   
    # group_size = spu_backend.spu_backend_get_group_size(data_nums)
    # print("group_size: ", group_size)
    # batch_size = model_batch_size * group_size
    # print("batch_size: ", batch_size)
    # copy_batch = batch_size // model_batch_size
    # print("copy_batch: ", copy_batch)

    # input_list = [np.vstack([input_tensor] * copy_batch)]
    # output_list = []
    # output_data = np.empty([batch_size, 1024], dtype=np.uint16)
    # output_list.append(output_data)

    # start_inference = time.time()
    # for i in range(1):
    #     spu_backend.spu_backend_inference(batch_size, input_list, output_list)
    # end_inference = time.time()
    # inference_time = end_inference - start_inference
    # print(f"model infer cost time: {inference_time * 1000} ms, fps: {batch_size / inference_time}")


    # spu_backend.spu_backend_destroy()

    compare_data = np.vstack([compare_data] * copy_batch)
    if np.array_equal(compare_data, output):
        print('result pass!')
    else:
        print('result failed!')
