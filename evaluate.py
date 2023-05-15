# encoding: utf-8


import os
from pytorch_lightning import Trainer

from trainer import BertNerTagger # start 0111

def evaluate(ckpt, hparams_file):
	"""main"""

	trainer = Trainer(distributed_backend="dp")

	model = BertNerTagger.load_from_checkpoint(
		checkpoint_path=ckpt,
		hparams_file=hparams_file,
		map_location=None,
		batch_size=1,
		max_length=128,
	)
	trainer.test(model=model)


if __name__ == '__main__':

	CHECKPOINTS = 'inference\events.out.tfevents.1646915817.b4d3ba25ffd4.652.0'
	HPARAMS = 'inference\hparams.yaml'
	evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)
