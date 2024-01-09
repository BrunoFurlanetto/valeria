import argparse
import torch

from model import LSTMWW


def trace(model):
    model.eval()
    x = torch.rand(80, 1, 40)
    traced = torch.jit.trace(model, x)
    return traced


def main(args):
    print("Loading model", args.model_checkpoint)
    checkpoint = torch.load(args.model_checkpoint, map_location=torch.device('cpu'))
    model = LSTMWW(**checkpoint['model_params'], device='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    print("Tracking model...")
    traced_model = trace(model)
    print("Saving in", args.save_path)
    traced_model.save(args.save_path)
    print("Ready!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing the wake word")
    parser.add_argument('--model_checkpoint', type=str, default=None, required=True,
                        help='Model checkpoint to optimize')
    parser.add_argument('--save_path', type=str, default=None, required=True,
                        help='Path to save the optimized model')

    args = parser.parse_args()
    main(args)
