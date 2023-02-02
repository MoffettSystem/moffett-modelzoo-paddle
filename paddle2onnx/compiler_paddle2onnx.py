import argparse
import os

def main():
    model_dir = os.path.dirname(args.model_path)
    file_name = model_dir.split("/")[-1]+".onnx"
    if args.save_dir:
        save_file_path = os.path.join(args.save_dir,file_name)
    else:
        save_file_path = os.path.join(os.path.dirname(model_dir),file_name)
    args.model_filename = args.model_path.split("/")[-1]
    args.params_filename = args.model_filename.replace("pdmodel","pdiparams")
    cmd = f'paddle2onnx --model_dir {model_dir} --model_filename {args.model_filename} --params_filename {args.params_filename} \
        --save_file {save_file_path} --opset_version={args.opset_version} --enable_onnx_checker={args.enable_onnx_checker}'
    os.system(cmd)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--model_path", type=str, help="the path of paddle model")
    parser.add_argument("-m","--model_filename", type=str, default="inference.pdmodel",help="paddle model name")
    parser.add_argument("-p","--params_filename", type=str, default="inference.pdiparams",help="params file name")
    parser.add_argument("-o","--save_dir", type=str, help="onnx output_name")
    parser.add_argument("-ov","--opset_version", type=int, default=11, help="onnx opt level")
    parser.add_argument("-ch","--enable_onnx_checker", type=str, default=True, help="check onnx")
    return parser.parse_args()
if __name__ == "__main__":
    args = get_args()
    main()