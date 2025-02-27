import json
import logging
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional

from haystack.nodes.file_converter import BaseConverter
from haystack.schema import Document

logger = logging.getLogger(__name__)


class JsonConverter(BaseConverter):
    """Extracts text from JSON files and casts it into Document objects."""

    outgoing_edges = 1

    def convert(
        self,
        file_path: Path,
        meta: Optional[Dict[str, Any]] = None,
        remove_numeric_tables: Optional[bool] = None,
        valid_languages: Optional[List[str]] = None,
        encoding: Optional[str] = "UTF-8",
        id_hash_keys: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Reads a JSON file and converts it into a list of Documents.

        It's a wrapper around `Document.from_dict()` and, as such, acts as the inverse of `Document.to_dict()`.

        It expects one of these formats:
        - A JSON file with a list of Document dicts.
        - A JSONL file with every line containing either a Document dict or a list of dicts.

        :param file_path: Path to the JSON file you want to convert.
        :param meta: Optional dictionary with metadata you want to attach to all resulting documents.
                     Can be any custom keys and values.
                     The result will have a union of metadata specified here and already present in the json.
                     In case of same keys being used, the one passed here takes precedence/overwrites the one from the json.
        :param remove_numeric_tables: Uses heuristics to remove numeric rows from the tables.
                     Note: Not currently used in this Converter.
        :param valid_languages: Validates languages from a list of languages specified in the [ISO 639-1]
                     Note: Not currently used in this Converter.
        :param encoding: Encoding used when opening the json file.
        :param id_hash_keys: Generate the document id from a custom list of strings that refer to the document's
            attributes. To ensure you don't have duplicate documents in your DocumentStore if texts are
            not unique, modify the metadata and pass, for example, `"meta"` to this field (example: [`"content"`, `"meta"`]).
            The id is then generated by using the content and the defined metadata.
            If specified here or during initialization of the JsonConverter, it will overwrite any `id_hash_keys` present in the json file.
        """

        if id_hash_keys is None:
            id_hash_keys = self.id_hash_keys

        docs: List[Document] = []

        with open(file_path, mode="r", encoding=encoding, errors="ignore") as f:
            data: List[Dict] = []
            try:
                for line in f:
                    line_obj = json.loads(line)
                    if isinstance(line_obj, list):
                        data.extend(line_obj)
                    elif isinstance(line_obj, dict):
                        data.append(line_obj)
            except JSONDecodeError:
                try:
                    f.seek(0)
                    # Assume it's a full json file
                    data = json.load(f)
                except JSONDecodeError as e:
                    msg = (
                        f"Couldn't decode the json file provided: {file_path}. "
                        "Please check if it's a valid json or jsonl file."
                    )
                    error_with_custom_message = JSONDecodeError(msg, e.doc, e.pos)
                    raise error_with_custom_message from e

            for doc_dict in data:
                # Overwrite the id_hash_keys if specified
                # Else we let it be whatever it is in the doc_dict
                if id_hash_keys is not None:
                    doc_dict["id_hash_keys"] = id_hash_keys

                if meta is not None:
                    existing_meta = doc_dict.get("meta", {})
                    # In case of duplicate keys, the newly specified vals (in `meta`) take precedence
                    doc_dict["meta"] = {**existing_meta, **meta}

                docs.append(Document.from_dict(doc_dict))

        return docs
