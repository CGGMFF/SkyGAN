import click
import os
import json
import dnnlib
from train import parse_comma_separated_list


@click.command()

# Required.
@click.option('--outdir',       help='Where to save the results', metavar='DIR',                required=True)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]',            type=parse_comma_separated_list, default='fid50k_full', show_default=True)
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.

    assert len(opts.metrics) == 1, "Can only decide based on one metric"
    metric = opts.metrics[0]

    minimum = float('inf')
    keep_snapshots = []
    print("Going through metric values and only keeping snapshots from the cumulative minimum")
    with open(os.path.join(opts.outdir, f'metric-{metric}.jsonl'), 'r') as f:
        for line in f.readlines():
            line = json.loads(line)
            value = float(line['results'][metric])
            if value < minimum:
                minimum = value
                keep_snapshots.append(line['snapshot_pkl'])
    
    for file in os.listdir(opts.outdir):
        if ('snapshot' in file
             and file.endswith('.pkl')
             and file not in keep_snapshots):
            os.remove(os.path.join(opts.outdir, file))

    print(f"Keeping {len(keep_snapshots)} snapshots")

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter