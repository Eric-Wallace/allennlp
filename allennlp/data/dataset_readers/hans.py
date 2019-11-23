from typing import Dict
import json
import logging

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer

import re 

logger = logging.getLogger(__name__)


@DatasetReader.register("hans")
class HansReader(DatasetReader):
    """
    Reads a file from the Heuristic Analysis for NLI Systems (HANS) dataset.  This data is
    formatted as text, one instance per line.  The keys in the data are "gold_label",
    "sentence1", and "sentence2". We convert these keys into fields named "label", "premise"
    and "hypothesis", along with a metadata field containing the tokenized strings of the
    premise and hypothesis.
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``SpacyTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as hans_file:
            logger.info("Reading HANS instances from text dataset at: %s", file_path)
            for i, line in enumerate(hans_file):
                if i != 0:
                  match = re.match(r"^(non-entailment|entailment)\s([(].*[)])\s((?:(?:\w|,)*\s)*[.])\s((?:(?:\w|,)*\s)*[.])\s(\w*)\s(\w*)\s((?:\w|/)*)\s(\w*)$", line)
                  label = match.group(1)
                  premise = match.group(3)
                  hypothesis = match.group(4)
                  heuristic = match.group(6)

                  yield self.text_to_instance(premise, hypothesis, heuristic, label)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        premise: str,
        hypothesis: str,
        heuristic: str,
        label: str = None
    ) -> Instance:

        fields: Dict[str, Field] = {}
        premise_tokens = self._tokenizer.tokenize(premise)
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)
        fields["premise"] = TextField(premise_tokens, self._token_indexers)
        fields["hypothesis"] = TextField(hypothesis_tokens, self._token_indexers)
        fields["heuristic"] = LabelField(heuristic)
        if label:
            fields["label"] = LabelField(label)

        metadata = {
            "premise_tokens": [x.text for x in premise_tokens],
            "hypothesis_tokens": [x.text for x in hypothesis_tokens],
        }
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)