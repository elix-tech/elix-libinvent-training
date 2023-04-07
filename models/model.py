"""
Model class.
"""

import torch
import torch.nn as tnn
from datetime import timedelta

import models.decorator as mdec
from running_modes.enums import GenerativeModelRegimeEnum


class DecoratorModel:
    def __init__(
        self,
        vocabulary,
        decorator,
        max_sequence_length=256,
        no_cuda=False,
        mode="train",
        distributed=False,
        world_size=1,
        rank=0,
    ):
        """
        Implements the likelihood and scaffold_decorating functions of the decorator model.
        :param vocabulary: A DecoratorVocabulary instance with the vocabularies of both the encoder and decoder.
        :param network_params: A dict with parameters for the encoder and decoder networks.
        :param decorator: An decorator network instance.
        :param max_sequence_length: Maximium number of tokens allowed to sample.
        :param no_cuda: Forces the model not to use CUDA, even if it is available.
        :param mode: Mode in which the model should be initialized.
        :return:
        """
        self.vocabulary = vocabulary
        self.max_sequence_length = max_sequence_length
        self.network = decorator
        self.distributed = distributed
        self.rank = rank

        self._model_modes = GenerativeModelRegimeEnum()

        self.device = torch.device(
            f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu"
        )

        if torch.cuda.is_available() and not no_cuda:
            # self.network.cuda()
            self.network = self.network.to(self.device)

        if distributed:
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size,
                timeout=timedelta(hours=1),
            )
            torch.cuda.set_device(rank)
            self.network = tnn.parallel.DistributedDataParallel(
                self.network, device_ids=[rank], output_device=rank
            )

        self._nll_loss = tnn.NLLLoss(reduction="none", ignore_index=0)
        self.set_mode(mode)

    @classmethod
    def load_from_file(
        cls, path, mode="train", distributed=False, world_size=1, rank=0
    ):
        """
        Loads a model from a single file
        :param path: Path to the saved model.
        :param mode: Mode in which the model should be initialized.
        :return: An instance of the RNN.
        """
        data = torch.load(path)

        decorator = mdec.Decorator(**data["decorator"]["params"])
        decorator.load_state_dict(data["decorator"]["state"])

        model = DecoratorModel(
            decorator=decorator,
            mode=mode,
            distributed=distributed,
            world_size=world_size,
            rank=rank,
            **data["model"],
        )

        return model

    def save(self, path):
        """
        Saves the model to a file.
        :param path: Path to the file which the model will be saved to.
        """
        save_dict = {
            "model": {
                "vocabulary": self.vocabulary,
                "max_sequence_length": self.max_sequence_length,
            },
            "decorator": {
                "params": self.network.get_params()
                if not self.distributed
                else self.network.module.get_params(),
                "state": self.network.state_dict()
                if not self.distributed
                else self.network.module.state_dict(),
            },
        }
        torch.save(save_dict, path)

    def set_mode(self, mode):
        """
        Changes the mode of the RNN to training or eval.
        :param mode: Mode to change to (training, eval)
        :return: The model instance.
        """
        if mode == self._model_modes.INFERENCE:
            self.network.eval()
        else:
            self.network.train()
        return self

    def likelihood(
        self,
        scaffold_seqs,
        scaffold_seq_lengths,
        decoration_seqs,
        decoration_seq_lengths,
    ):
        """
        Retrieves the likelihood of a scaffold and its respective decorations.
        :param scaffold_seqs: (batch, seq) A batch of padded scaffold sequences.
        :param scaffold_seq_lengths: The length of the scaffold sequences (for packing purposes).
        :param decoration_seqs: (batch, seq) A batch of decorator sequences.
        :param decoration_seq_lengths: The length of the decorator sequences (for packing purposes).
        :return:  (batch) Log likelihood for each item in the batch.
        """
        # NOTE: the decoration_seq_lengths have a - 1 to prevent the end token to be forward-passed.
        logits = self.network(
            scaffold_seqs.to(self.device),
            scaffold_seq_lengths,
            decoration_seqs.to(self.device),
            decoration_seq_lengths - 1,
        )  # (batch, seq - 1, voc)
        log_probs = logits.log_softmax(dim=2).transpose(
            1, 2
        )  # (batch, voc, seq - 1)
        return self._nll_loss(log_probs, decoration_seqs[:, 1:]).sum(
            dim=1
        )  # (batch)

    @torch.no_grad()
    def sample_decorations(self, scaffold_seqs, scaffold_seq_lengths):
        # TODO: fix the return type to be SampledSequencesDTO
        """
        Samples as many decorations as scaffolds in the tensor.
        :param scaffold_seqs: A tensor with the scaffolds to sample already encoded and padded.
        :param scaffold_seq_lengths: A tensor with the length of the scaffolds.
        :return: An iterator with (scaffold_smi, decoration_smi, nll) triplets.
        """
        batch_size = scaffold_seqs.size(0)

        input_vector = torch.full(
            (batch_size, 1),
            self.vocabulary.decoration_vocabulary["^"],
            dtype=torch.long,
        ).cuda()  # (batch, 1)
        # print(f"input_vector: {input_vector}")
        seq_lengths = torch.ones(batch_size)  # (batch)
        if self.distributed:
            (
                encoder_padded_seqs,
                hidden_states,
            ) = self.network.module.forward_encoder(
                scaffold_seqs, scaffold_seq_lengths
            )
        else:
            encoder_padded_seqs, hidden_states = self.network.forward_encoder(
                scaffold_seqs, scaffold_seq_lengths
            )
        nlls = torch.zeros(batch_size).cuda()
        not_finished = torch.ones(batch_size, 1, dtype=torch.long).cuda()
        sequences = []
        for _ in range(self.max_sequence_length - 1):
            if self.distributed:
                logits, hidden_states, _ = self.network.module.forward_decoder(
                    input_vector,
                    seq_lengths,
                    encoder_padded_seqs,
                    hidden_states,
                )  # (batch, 1, voc)
            else:
                logits, hidden_states, _ = self.network.forward_decoder(
                    input_vector,
                    seq_lengths,
                    encoder_padded_seqs,
                    hidden_states,
                )  # (batch, 1, voc)
            probs = logits.softmax(dim=2).squeeze()  # (batch, voc)
            log_probs = logits.log_softmax(dim=2).squeeze()  # (batch, voc)
            input_vector = (
                torch.multinomial(probs, 1) * not_finished
            )  # (batch, 1)
            sequences.append(input_vector)
            nlls += self._nll_loss(log_probs, input_vector.squeeze())
            not_finished = (input_vector > 1).type(
                torch.long
            )  # 0 is padding, 1 is end token
            if not_finished.sum() == 0:
                break

        decoration_smiles = [
            self.vocabulary.decode_decoration(seq)
            for seq in torch.cat(sequences, 1).data.cpu().numpy()
        ]
        scaffold_smiles = [
            self.vocabulary.decode_scaffold(seq)
            for seq in scaffold_seqs.data.cpu().numpy()
        ]
        return zip(
            scaffold_smiles,
            decoration_smiles,
            nlls.data.cpu().numpy().tolist(),
        )
