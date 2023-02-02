import numpy as np
import threading
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

def pad_input(input_data):
    # input_data = input_data.astype(np.float32)
    # input_data = np.transpose(input_data,(0,2,3,1))
    input_shape = input_data.shape
    model_batch_size = 8
    least_batch_size = model_batch_size * 4
    if input_shape[0]<least_batch_size:
        input_data_pad = np.pad(input_data,((0,least_batch_size-input_shape[0]),(0,128-input_shape[1])),mode="constant",constant_values=(0))
    else:
        input_data_pad = np.pad(input_data,((0,0),(0,128-input_shape[1])),mode="constant",constant_values=(0))
    # new_input_data = np_float2np_bf16(input_data_pad)
    # new_input_data = np.ascontiguousarray(new_input_data, np.uint16)
    return input_data_pad

def s4_preprocess_4core(input_ids_tensor,token_type_ids_tensor):
    # input_ids_data = input_ids_tensor.astype(np.float32)
    # input_ids_data_2 = input_ids_tensor.astype(np.float32)
    # token_type_ids_data = token_type_ids_tensor.astype(np.float32)
    input_ids_data_pad = pad_input(input_ids_tensor)
    token_type_ids_data_pad = pad_input(token_type_ids_tensor)
    new_input_ids_data = np.ascontiguousarray(input_ids_data_pad, np.uint32)
    new_token_type_ids_data = np.ascontiguousarray(token_type_ids_data_pad, np.uint32)

    input_ids_data_2 = input_ids_data_pad.astype(np.float32)
    new_input_ids_data_2 = np_float2np_bf16(input_ids_data_2)
    new_input_ids_data_2 = new_input_ids_data_2.reshape(-1,1,4,1,1,32)
    new_input_ids_data_2 = np.ascontiguousarray(new_input_ids_data_2, np.uint16)
    return new_input_ids_data,new_token_type_ids_data,new_input_ids_data_2 


def bin2float(num):
    return struct.unpack('f', struct.pack('I', num))[0]


def uint16tofloat32(data):
    data = np.left_shift(data.astype(np.int32), 16).view(np.float32)
    return data

def s4_postprocess(output):
    last_shpae = output.shape[-1]
    output = output.reshape(-1,256)
    new_output = output[:,0:2]
    new_output = uint16tofloat32(new_output)
    # new_output = output.astype(np.float32)
    return new_output


# def mf_s4_infer_4core(model_path,input_ids_tensor,token_type_ids_tensor,input_ids_2_tensor):
def mf_s4_infer_4core(model_path,input_ids_2_tensor,input_ids_tensor,token_type_ids_tensor):

    model_batch_size = 8
    # numpy
    input_ids_size = model_batch_size*128*4
    token_type_ids_size = model_batch_size*128*4
    # input_ids_2_size = model_batch_size*4*32*4
    input_ids_2_size = model_batch_size*4*32*2
    output_size = model_batch_size*256*2

    buf_depth = 1
    ringbuffer = 1
    # data_nums = 256
    data_nums = input_ids_tensor.shape[0]

    print("input_ids_tensor shape after preprocess:",input_ids_tensor.shape)
    print("token_type_ids_tensor shape after preprocess:",token_type_ids_tensor.shape)
    print("input_ids_2_tensor shape after preprocess:",input_ids_2_tensor.shape)


    # spu_backend.spu_backend_init(0, model_path, True, model_batch_size,
    #                             [input_ids_size, token_type_ids_size, input_ids_2_size], [output_size], False,
    #                             buf_depth, ringbuffer)
    spu_backend.spu_backend_init(0, model_path, True, model_batch_size,
                                [input_ids_2_size, input_ids_size, token_type_ids_size], [output_size], False,
                                buf_depth, ringbuffer)
    
    group_size = spu_backend.spu_backend_get_group_size(data_nums)
    # print("group_size: ", group_size)
    batch_size = model_batch_size * group_size
    print("batch_size: ", batch_size)
  
   
    # input_list = [np.vstack([input_ids_tensor]), np.vstack(token_type_ids_tensor), np.vstack([input_ids_2_tensor])]
    input_list = [np.vstack([input_ids_2_tensor]),np.vstack([input_ids_tensor]), np.vstack(token_type_ids_tensor)]
    output_list = []
    output_data = np.empty([batch_size,256], dtype=np.uint16)
    output_list.append(output_data)
    start_inference = time.time()
    for i in range(10):
        spu_backend.spu_backend_inference(batch_size, input_list, output_list)
    end_inference = time.time()
    inference_time = end_inference - start_inference
    # print(f"model infer cost time: {inference_time * 1000} ms, fps: {data_nums / inference_time}")
    perf = data_nums / inference_time
    memory_used = get_memory_info()
    print(f"fps: {int(perf*10)}")
    print(f"memory_used: {memory_used}M")
    spu_backend.spu_backend_destroy()
    
    return output_list[0]





if __name__ == '__main__':

    # model_path = '/home/moffett/work/zhongming/project/paddle_compile_output/bert_base_uncased/compiler_output_1219/model.bin'
    model_path = '/home/moffett/work/zhongming/project/paddle_compile_output/bert_base_uncased/compiler_output_change_placeholder_1221/backend_output/model.bin'
    # input_data = np.ones(shape=[32, 44], dtype=np.int64)
    # print("input_shape:",input_data.shape)
    # input_data = s4_preprocess(input_data)

    input_ids_tensor = np.ones(shape=[32, 128], dtype=np.int32)
    token_type_ids_tensor = np.ones(shape=[32, 128], dtype=np.int32)
    input_ids_2_tensor = input_ids_tensor.reshape(-1,1,4,1,1,32)
    print("input_ids_tensor shape after preprocess:",input_ids_tensor.shape)
    print("token_type_ids_tensor shape after preprocess:",token_type_ids_tensor.shape)
    print("input_ids_2_tensor shape after preprocess:",input_ids_2_tensor.shape)
    # import pdb;pdb.set_trace()
    output = mf_s4_infer_4core(model_path,input_ids_tensor,token_type_ids_tensor,input_ids_2_tensor)
    print("output_shape:",output.shape)
    # output = s4_postprocess(output)
    # print("output_shape after postprocess:",output.shape)
    print("infer complete!")


