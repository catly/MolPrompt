import argparse
def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--device", default="1", type=str, )

    parser.add_argument("--output_path", default='save_model/prompt_gp_v1_pretrain_200k', type=str, )
    parser.add_argument("--lr", default=5e-5, type=float, )  # 4
    parser.add_argument("--warmup", default=0.2, type=float, )
    parser.add_argument("--total_steps", default=250000, type=int, )
    parser.add_argument("--batch_size", default=64, type=int, )
    parser.add_argument("--epoch", default=50, type=int, )
    parser.add_argument("--seed", default=42, type=int, )  # 73 99 108
    parser.add_argument("--text_max_len", default=128, type=int, )
    parser.add_argument("--prompt_max_len", default=20, type=int, )
    parser.add_argument("--margin", default=0.2, type=int, )

    parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
    parser.add_argument('--graph_hidden_dim', type=int, default=768, help='')
    parser.add_argument('--drop_ratio', default=0.1, type=float)
    # Bert
    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
    parser.add_argument('--projection_dim', type=int, default=256)
    # optimization
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='optimizer weight decay')

    args = parser.parse_args()
    return args