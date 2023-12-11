import spacy
import torch
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration


class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        A tokenizer can support different preprocessing options or use different methods
        for determining word breaks.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
        """
        # TODO: Save arguments that are needed as fields of this class
        self.lowercase = lowercase
        self.max_mwe_len = 0
        self.multiword_expressions = defaultdict(set)
        if multiword_expressions is not None:
            for exp in multiword_expressions:
                exp = exp.split()
                self.multiword_expressions[exp[0].lower()].add(exp)
                self.max_mwe_len = max(self.max_mwe_len, len(exp))

    def find_and_replace_mwes(self, input_tokens: list[str]) -> list[str]:
        """
        For the given sequence of tokens, finds any recognized multi-word expressions in the sequence
        and replaces that subsequence with a single token containing the multi-word expression.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens containing processed multi-word expressions
        """
        raise NotImplemented("MWE is not supported")

    def postprocess(self, input_tokens: list[str]) -> list[str]:
        """
        Performs any set of optional operations to modify the tokenized list of words such as
        lower-casing and returns the modified list of tokens.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens processed by lower-casing depending on the given condition
        """
        # TODO: Add support for lower-casing
        return [token.lower() for token in input_tokens]

    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        raise NotImplementedError(
            'tokenize() is not implemented in the base class; please use a subclass')


class SplitTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        super().__init__(lowercase, multiword_expressions)

    def tokenize(self, text: str) -> list[str]:
        tokens = text.split()
        idx = 0
        results = []
        while idx < len(tokens):
            curr_token = tokens[idx]
            match = False
            is_exception = False
            if curr_token[0].lower() not in self.multiword_expressions:
                is_exception = True
            else:
                curr_len = self.max_mwe_len
                if idx + curr_len > len(tokens):
                    curr_len = len(tokens) - idx
                while curr_len > 0:
                    if ' '.join(tokens[idx:idx+curr_len]).lower() in self.multiword_expressions[curr_token[0].lower()]:
                        match = True
                        break
                    curr_len -= 1
                    if match:
                        break
            if match:
                if not is_exception:
                    curr_token = ' '.join(tokens[idx:idx+curr_len])
                idx += curr_len
                results.append(curr_token)
            else:
                results.append(curr_token)
                idx += 1
        return self.postprocess(results)


class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses NLTK's RegexpTokenizer to tokenize a given string.

        Args:
            token_regex: Use the following default regular expression pattern: '\\w+'
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
        """
        super().__init__(lowercase, multiword_expressions)
        # TODO: Save a new argument that is needed as a field of this class
        # TODO: Initialize the NLTK's RegexpTokenizer
        self.tokenizer = RegexpTokenizer(token_regex)

    def tokenize(self, text: str) -> list[str]:
        """Uses NLTK's RegexTokenizer and a regular expression pattern to tokenize a string.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # TODO: Tokenize the given text and perform postprocessing on the list of tokens
        #       using the postprocess function
        return self.postprocess(self.tokenizer.tokenize(text))


class SpaCyTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        super().__init__(lowercase, multiword_expressions)
        self.nlp = spacy.load('en_core_web_sm')
        self.punctuations = ['.', ',', ';', ':', '!', '?', '(', ')', '"']

    def tokenize(self, text: str) -> list[str]:
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        idx = 0
        results = []
        while idx < len(tokens):
            curr_token = tokens[idx]
            match = False
            is_exception = False
            if curr_token[0].lower() not in self.multiword_expressions:
                match = True
                is_exception = True
            else:
                for exp in self.multiword_expressions[curr_token[0].lower()]:
                    pos = 0
                    length = 0
                    total_length = len(exp)
                    if tokens[idx] in self.punctuations:
                        break
                    while True:
                        if idx + pos >= len(tokens):
                            break
                        if tokens[idx + pos] in exp:
                            length += len(tokens[idx + pos])
                            if length == total_length:
                                match = True
                                break
                            if '\'' not in tokens[idx + pos]:
                                length += 1
                            pos += 1
                        else:
                            break
                    if match:
                        break
            if match:
                if not is_exception:
                    curr_token = exp
                idx = pos + 1
                results.append(curr_token)
            else:
                results.append(curr_token)
                idx += 1
        return self.postprocess(results)


# TODO: Take in a doc2query model and generate queries from a piece of text
# Note: For downstream tasks such as index augmentation with the queries, use doc2query.csv
class Doc2QueryAugmenter:
    """
    This class is responsible for generating queries for a document.
    These queries can augment the document before indexing.

    MUST READ: https://huggingface.co/doc2query/msmarco-t5-base-v1

    OPTIONAL reading
        1. Document Expansion by Query Prediction (Nogueira et al.): https://arxiv.org/pdf/1904.08375.pdf
    """

    def __init__(self, doc2query_model_name: str = 'doc2query/msmarco-t5-base-v1') -> None:
        """
        Creates the T5 model object and the corresponding dense tokenizer.

        Args:
            doc2query_model_name: The name of the T5 model architecture used for generating queries
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # TODO: Create the dense tokenizer and query generation model using HuggingFace transformers
        self.tokenizer = T5Tokenizer.from_pretrained(doc2query_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            doc2query_model_name).to(self.device)

    def get_queries(self, document: str, n_queries: int = 5, prefix_prompt: str = '') -> list[str]:
        """
        Steps
            1. Use the dense tokenizer/encoder to create the dense document vector.
            2. Use the T5 model to generate the dense query vectors (you should have a list of vectors).
            3. Decode the query vector using the tokenizer/decode to get the appropriate queries.
            4. Return the queries.

        TODO:
            Neural models are best performing when batched to the GPU.
            Try writing a separate function which can deal with batches of documents.

        Args:
            document: The text from which queries are to be generated
            n_queries: The total number of queries to be generated
            prefix_prompt: An optional parameter that gets added before the text.
                Some models like flan-t5 are not fine-tuned to generate queries.
                So we need to add a prompt to instruct the model to generate queries.
                This string enables us to create a prefixed prompt to generate queries for the models.
                See the PDF for what you need to do for this part.
                Prompt-engineering: https://en.wikipedia.org/wiki/Prompt_engineering

        Returns:
            A list of query strings generated from the text
        """
        # Note: Change these values to experiment
        document_max_token_length = 400  # as used in OPTIONAL Reading 1
        top_p = 0.85

        # NOTE: See https://huggingface.co/doc2query/msmarco-t5-base-v1 for details

        # TODO: For the given model, generate a list of queries that might reasonably be issued to search
        #       for that document
        # NOTE: Do not forget edge cases
        input_ids = self.tokenizer.encode(
            prefix_prompt + document,
            max_length=document_max_token_length,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=64,
            do_sample=True,
            top_p=top_p,
            num_return_sequences=n_queries
        )
        queries = []
        for output in outputs:
            queries.append(self.tokenizer.decode(
                output, skip_special_tokens=True))
        return queries


if __name__ == '__main__':
    pass
