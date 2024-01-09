import argparse
import torch

from model import LSTMWW


def trace(model):
    model.eval()
    x = torch.rand(80, 1, 40)
    traced = torch.jit.trace(model, x)
    return traced


def main(args):
    print("carregando modelo de", args.modelo_checkpoint)
    checkpoint = torch.load(args.modelo_checkpoint, map_location=torch.device('cpu'))
    model = LSTMWW(**checkpoint['model_params'], device='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    print("modelo de rastreamento...")
    traced_model = trace(model)
    print("salvando em", args.save_path)
    traced_model.save(args.save_path)
    print("Pronto!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="testando o wakeword")
    parser.add_argument('--modelo_checkpoint', type=str, default=None, required=True,
                        help='Checkpoint de modelo para otimizar')
    parser.add_argument('--save_path', type=str, default=None, required=True,
                        help='caminho para salvar o modelo otimizado')

    args = parser.parse_args()
    main(args)