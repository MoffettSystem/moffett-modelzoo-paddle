##@example compile_facenet.py
import argparse
import os
import logging
import sys
import onnxruntime as ort
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../moffett_software"))
# from nn_compiler.common.constants import  TransltMode, VerifyMode
from nn_compiler.common.utils.log_utils import get_logger
from sparse_compile.common import ModelDtype, dump_module_infos,save_model_bin
from dataset.dataset import FacenetDataset,ImageNetDataset
from torch.utils.data import DataLoader as DataLoaderX
from sparse_compile.builder import Builder
from sparse_compile.parser import OnnxParser
from sparse_compile.profiler import Profiler
from sparse_compile.verifier import Verifier
# from build_model import t_or_f

SPARSE_LOGGER = get_logger("SparseCompile",level = logging.INFO)

def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
       return True
    elif 'FALSE'.startswith(ua):
       return False
    else:
       pass  

def get_input_dict(onnx_model_path):
    onnx_session = ort.InferenceSession(onnx_model_path)
    input_infos = {}
    for node in onnx_session.get_inputs():
        # input_names.append(node.name)
        assert len(node.shape) == 4
        input_infos[node.name] = node.shape
    return input_infos

# def build_engine_onnx(model_file,data_root,verify = False):
def build_engine_onnx(model_file,args,compile_ouput_path=None):
    builder = Builder(SPARSE_LOGGER)
    network = builder.create_network()

    #init config
    config = builder.create_builder_config()
    # # config.dump_dir = os.path.dirname(model_file) + "out"
    # if compile_ouput_path:
    #     config.dump_dir = compile_ouput_path # for bath compile
    # else:
    #     config.dump_dir = args.dump_dir if args.dump_dir else os.path.dirname(model_file)
    # batch_size = 8
    input_shape_dict = get_input_dict(model_file)
    input_name = list(input_shape_dict)[0]
    # input_shape = input_dict[input_name]
    # input_shape[0] = args.batch_size
    input_shape_dict[input_name][0] = args.batch_size
    
    # config.input_dict = {"x":"x"}# dataset item name to model input
    config.input_dict = {input_name:input_name}# dataset item name to model input
    #quantize config,
    # config.model_dtype = ModelDtype.MixInt8Bf16 # or ModelDtype.Bf16
    if args.dtype == "bf16":
        config.model_dtype = ModelDtype.Bf16 # or ModelDtype.Bf16
    elif args.dtype == "MixInt8bf16":
        config.model_dtype = ModelDtype.MixInt8Bf16
    else:
        config.model_dtype = ModelDtype.Int8 

    # config.fuse_actlut = True #fuse act op into Conv/Matmul op
    # config.fuse_actlut = True #fuse act op into Conv/Matmul op
    config.do_kl = True 

    #provide dataload for quant calibration
    if args.dtype == "int8":
        data_root = args.data_root
        # dataset = FacenetDataset(data_root)
        # dataloader = DataLoaderX(dataset ,batch_size = args.batch_size)
        # config.dataloader = dataloader
        dataset = ImageNetDataset(data_root,transform_file="../moffett_software/dataset/configs/mxnet_imagenet_trans_320.json")
        dataloader = DataLoaderX(dataset ,batch_size = args.batch_size)
        config.dataloader = dataloader

    config.opt_level = 5
    config.batch_size = args.batch_size
    config.total_cores = args.cores
    #parser
    # input_shape_dict = {"actual_input":(8,3,160,160)}
    # input_shape_dict = {"x":(args.batch_size,3,299,299)}
    parser = OnnxParser(network,input_shape_dict,SPARSE_LOGGER )

    with open(model_file, "rb") as model:
        if not parser.parse(model):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    if args.verify: 
        engine = builder.build_debug_engine(network, config)
        verifier = Verifier(engine,config)
        verifier.verify()
        verifier.dump_verify_infos(config.dump_dir)
    else:
        engine = builder.build_engine(network, config)
    return engine

def main(args):
    onnx_path = args.path
    # data_root = args.data_root
    # engine = build_engine_onnx(onnx_path,data_root,verify=args.verify)
    engine = build_engine_onnx(onnx_path,args)
    save_dir = args.dump_dir if args.dump_dir else os.path.dirname(onnx_path) 
    if save_dir:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_model_bin(engine,save_dir)
        profiler = Profiler(engine)
        profiler.dump_perf_log(save_dir)
        engine.dump(save_dir)
        dump_module_infos(engine,save_dir)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="the path of onnx model")
    parser.add_argument("--data_root", type=str,help="the dir of calibrate dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
    parser.add_argument("--cores", type=int, default=1, help="batch_size")
    parser.add_argument("--dtype", type=str, default="bf16", help="int8,bf16 or MixInt8Bf16")
    # parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify", type = t_or_f, default = False)
    # parser.add_argument("-o", "--out_dir")
    parser.add_argument("-o","--dump_dir", type=str, help="the dir of dump output")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)


