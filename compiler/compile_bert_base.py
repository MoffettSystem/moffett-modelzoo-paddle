import argparse
import os
import logging
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../moffett_software"))
from torch.utils.data import DataLoader
# from nn_compiler.common.constants import OpType, TransltMode
from nn_compiler.common.utils.log_utils import get_logger
# from sparse_compile.builder_config import TransltEdgeMode
from sparse_compile.common import ModelDtype, dump_module_infos, save_model_bin
from dataset.dataset import MZJBertDataset
from sparse_compile.builder import Builder
from sparse_compile.parser import OnnxParser
from sparse_compile.profiler import Profiler
from sparse_compile.verifier import Verifier
#from build_model import t_or_f

SPARSE_LOGGER = get_logger("SparseCompile",level = logging.INFO)

def convert_paddle_model(path):
    model_dir = os.path.dirname(args.model_path)
    file_name = model_dir.split("/")[-1]+".onnx"
    save_file_path = os.path.join(os.path.dirname(model_dir),file_name)
    model_filename = args.model_path.split("/")[-1]
    params_filename = args.model_filename.replace("pdmodel","pdiparams")
    cmd = f'paddle2onnx --model_dir {model_dir} --model_filename {model_filename} --params_filename {params_filename} \
        --save_file {save_file_path} --opset_version=11--enable_onnx_checker=True'
    os.system(cmd)
    return save_file_path

def build_engine_onnx(model_file,data_root,args):
    # bert-base and albert
    # batch_size = 12
    input_shape_dict = {
        "input_ids":(args.batch_size,128), 
        # "attention_mask":(args.batch_size,128),
        "token_type_ids":(args.batch_size,128)
        }
    input_dict = {
        "input_ids": "input_ids",
        # "attention_mask": "attention_mask",
        "token_type_ids": "token_type_ids"
    }# dataset item name to model input

    verify = args.verify
    builder = Builder(SPARSE_LOGGER)
    network = builder.create_network()

    #init config
    config = builder.create_builder_config()
    # if args.dump_dir:
    #     config.dump_dir = args.dump_dir
    config.dump_dir = args.dump_dir if args.dump_dir else os.path.dirname(model_file)
    # config.input_dict 需要设置
    config.input_dict = input_dict
    #quantize config,

    # config.model_dtype = ModelDtype.MixInt8Bf16 # or ModelDtype.Bf16
    config.model_dtype = args.dtype
    config.opt_level = 5
    if args.dtype == "int8" or args.dtype == "MixInt8Bf16":
        #provide dataload for quant calibration
        calibrate_data_path = data_root
        dataset = MZJBertDataset(data_path=calibrate_data_path, input_info=input_dict)
        dataloader = DataLoader(dataset ,batch_size = args.batch_size, shuffle = False, num_workers = 4)
        config.dataloader = dataloader 

    # # fixed config for conformer
    # config.translt_edge_mode = TransltEdgeMode.EdgeTransV1
    #parser
    parser = OnnxParser(network,input_shape_dict = input_shape_dict,logger = SPARSE_LOGGER)
  
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, "rb") as model:
        if not parser.parse(model):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    if args.verify: 
        engine = builder.build_debug_engine(network,config)
        verifier = Verifier(engine,config)
        verifier.verify()
        verifier.dump_verify_infos(config.dump_dir)
    else:
        engine = builder.build_engine(network, config)
    return engine

def main(args):
    onnx_path = convert_paddle_model(args.path)
    data_root = args.data_root
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
    engine = build_engine_onnx(onnx_path,data_root,args)
    save_dir = args.dump_dir if args.dump_dir else os.path.dirname(onnx_path) 
    if save_dir:
        save_model_bin(engine,save_dir)
        profiler = Profiler(engine)
        profiler.dump_perf_log(save_dir)
        engine.dump(save_dir)
        dump_module_infos(engine,save_dir)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="the path of paddle model")
    parser.add_argument("--data_root", type=str, help="the path of dataset")
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
    parser.add_argument("--dtype", type=str, default="bf16", help="int8,bf16 or MixInt8Bf16")
    # parser.add_argument("--verify", action="store_true")
   #  parser.add_argument("--verify", type = t_or_f, default = False)
    parser.add_argument("--verify", type = str, default = False)
    parser.add_argument("-o","--dump_dir", type=str, help="the dir of dump output")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    main(args)


