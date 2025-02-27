import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from haystack.nodes.file_converter import BaseConverter
from haystack.schema import Document

logger = logging.getLogger(__name__)


class CsvTextConverter(BaseConverter):
    """
    Converts a CSV file containing FAQs to text Documents. The CSV file must have two columns: 'question' and 'answer'. Use this node for FAQ-style question answering.
    """

    outgoing_edges = 1

    def convert(
        self,
        file_path: Union[Path, List[Path], str, List[str], List[Union[Path, str]]],
        meta: Optional[Dict[str, Any]],
        remove_numeric_tables: Optional[bool] = None,
        valid_languages: Optional[List[str]] = None,
        encoding: Optional[str] = "UTF-8",
        id_hash_keys: Optional[List[str]] = None,
    ) -> List[Document]:
        """
                Load a CSV file containing question-answer pairs and convert it to Documents.

                :param file_path: Path to the CSV file you want to convert. The file must have two columns called 'question' and 'answer'.
                    The first will be interpreted as a question, the second as content.
                :param meta: A dictionary of metadata key-value pairs that you want to append to the returned document. It's optional.
                :param encoding: Specifies the file encoding. It's optional. The default value is `UTF-8`.
                :param id_hash_keys: Generates the document ID from a custom list of strings that refer to the document's
        attributes. To ensure you don't have duplicate documents in your DocumentStore when texts are
        not unique, modify the metadata and pass, for example, "meta" to this field (example: ["content", "meta"]).
        Then the ID is generated by using the content and the metadata you defined.
                :param remove_numeric_tables: unused
                :param valid_languages: unused
                :returns: List of document, 1 document per line in the CSV.
        """
        if not isinstance(file_path, list):
            file_path = [file_path]

        docs: List[DocumentType] = []
        for path in file_path:
            df = pd.read_csv(path, encoding=encoding)

            if len(df.columns) != 2 or df.columns[0] != "question" or df.columns[1] != "answer":
                raise ValueError("The CSV must contain two columns named 'question' and 'answer'")

            df.fillna(value="", inplace=True)
            df["question"] = df["question"].apply(lambda x: x.strip())

            df = df.rename(columns={"question": "content"})
            docs_dicts = df.to_dict(orient="records")

            for dictionary in docs_dicts:
                if meta:
                    dictionary["meta"] = meta
                if id_hash_keys:
                    dictionary["id_hash_keys"] = id_hash_keys
                docs.append(DocumentType.from_dict(dictionary))

        return docs
