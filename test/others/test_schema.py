from haystack.schema import Document, Label, Answer, Span, MultiLabel, TableCell
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def text_labels():
    return [
        Label(
            query="some",
            answer=Answer(
                answer="an answer",
                type="extractive",
                score=0.1,
                document_ids=["123"],
                offsets_in_document=[Span(start=1, end=3)],
            ),
            document=Document(content="some text", content_type="text"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="user-feedback",
        ),
        Label(
            query="some",
            answer=Answer(answer="annother answer", type="extractive", score=0.1, document_ids=["123"]),
            document=Document(content="some text", content_type="text"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="user-feedback",
        ),
        Label(
            query="some",
            answer=Answer(
                answer="an answer",
                type="extractive",
                score=0.1,
                document_ids=["123"],
                offsets_in_document=[Span(start=1, end=3)],
            ),
            document=Document(content="some text", content_type="text"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="user-feedback",
        ),
    ]


@pytest.fixture
def text_answer():
    return Answer(
        answer="an answer",
        type="extractive",
        score=0.1,
        context="abc",
        offsets_in_document=[Span(start=1, end=10)],
        offsets_in_context=[Span(start=3, end=5)],
        document_ids=["123"],
    )


@pytest.fixture
def table_doc():
    data = {
        "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
        "age": [58, 47, 60],
        "number of movies": [87, 53, 69],
        "date of birth": ["18 december 1963", "11 november 1974", "6 may 1961"],
    }
    return Document(content=pd.DataFrame(data), content_type="table", id="doc1")


@pytest.fixture
def table_doc_with_embedding():
    data = {
        "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
        "age": [58, 47, 60],
        "number of movies": [87, 53, 69],
        "date of birth": ["18 december 1963", "11 november 1974", "6 may 1961"],
    }
    return Document(
        content=pd.DataFrame(data), content_type="table", id="doc2", embedding=np.random.rand(768).astype(np.float32)
    )


def test_no_answer_label():
    labels = [
        Label(
            query="question",
            answer=Answer(answer=""),
            is_correct_answer=True,
            is_correct_document=True,
            document=Document(content="some", id="777"),
            origin="gold-label",
        ),
        Label(
            query="question",
            answer=Answer(answer=""),
            is_correct_answer=True,
            is_correct_document=True,
            document=Document(content="some", id="777"),
            origin="gold-label",
        ),
        Label(
            query="question",
            answer=Answer(answer="some"),
            is_correct_answer=True,
            is_correct_document=True,
            document=Document(content="some", id="777"),
            origin="gold-label",
        ),
        Label(
            query="question",
            answer=Answer(answer="some"),
            is_correct_answer=True,
            is_correct_document=True,
            document=Document(content="some", id="777"),
            origin="gold-label",
        ),
    ]

    assert labels[0].no_answer == True
    assert labels[1].no_answer == True
    assert labels[2].no_answer == False
    assert labels[3].no_answer == False


def test_equal_label(text_labels):
    assert text_labels[2] == text_labels[0]
    assert text_labels[1] != text_labels[0]


def test_label_to_json(text_labels):
    j0 = text_labels[0].to_json()
    l_new = Label.from_json(j0)
    assert l_new == text_labels[0]
    assert l_new.answer.offsets_in_document[0].start == 1


def test_label_to_dict(text_labels):
    j0 = text_labels[0].to_dict()
    l_new = Label.from_dict(j0)
    assert l_new == text_labels[0]
    assert l_new.answer.offsets_in_document[0].start == 1


def test_answer_to_json(text_answer):
    a = text_answer
    j = a.to_json()
    assert type(j) == str
    assert len(j) > 30
    a_new = Answer.from_json(j)
    assert type(a_new.offsets_in_document[0]) == Span
    assert a_new == a


def test_answer_to_dict(text_answer):
    a = text_answer
    j = a.to_dict()
    assert type(j) == dict
    a_new = Answer.from_dict(j)
    assert type(a_new.offsets_in_document[0]) == Span
    assert a_new == a


def test_document_from_dict():
    doc = Document(
        content="this is the content of the document", meta={"some": "meta"}, id_hash_keys=["content", "meta"]
    )
    assert doc == Document.from_dict(doc.to_dict())


def test_table_document_from_dict(table_doc):
    assert table_doc == Document.from_dict(table_doc.to_dict())


def test_doc_to_json():
    # With embedding
    d = Document(
        content="some text",
        content_type="text",
        id_hash_keys=["meta"],
        score=0.99988,
        meta={"name": "doc1"},
        embedding=np.random.rand(768).astype(np.float32),
    )
    j0 = d.to_json()
    d_new = Document.from_json(j0)
    assert d == d_new

    # No embedding
    d = Document(
        content="some text",
        content_type="text",
        score=0.99988,
        meta={"name": "doc1"},
        id_hash_keys=["meta"],
        embedding=None,
    )
    j0 = d.to_json()
    d_new = Document.from_json(j0)
    assert d == d_new


def test_table_doc_to_json(table_doc, table_doc_with_embedding):
    # With embedding
    j0 = table_doc_with_embedding.to_json()
    d_new = Document.from_json(j0)
    assert table_doc_with_embedding == d_new

    # No embedding
    j0 = table_doc.to_json()
    d_new = Document.from_json(j0)
    assert table_doc == d_new


def test_answer_postinit():
    a = Answer(answer="test", offsets_in_document=[{"start": 10, "end": 20}])
    assert a.meta == {}
    assert isinstance(a.offsets_in_document[0], Span)


def test_table_answer_postinit():
    a = Answer(answer="test", offsets_in_document=[{"row": 1, "col": 2}])
    assert a.meta == {}
    assert isinstance(a.offsets_in_document[0], TableCell)


def test_generate_doc_id_using_text():
    text1 = "text1"
    text2 = "text2"
    doc1_text1 = Document(content=text1, meta={"name": "doc1"})
    doc2_text1 = Document(content=text1, meta={"name": "doc2"})
    doc3_text2 = Document(content=text2, meta={"name": "doc3"})

    assert doc1_text1.id == doc2_text1.id
    assert doc1_text1.id != doc3_text2.id


def test_generate_doc_id_using_custom_list():
    text1 = "text1"
    text2 = "text2"

    doc1_meta1_id_by_content = Document(content=text1, meta={"name": "doc1"}, id_hash_keys=["content"])
    doc1_meta2_id_by_content = Document(content=text1, meta={"name": "doc2"}, id_hash_keys=["content"])
    assert doc1_meta1_id_by_content.id == doc1_meta2_id_by_content.id

    doc1_meta1_id_by_content_and_meta = Document(content=text1, meta={"name": "doc1"}, id_hash_keys=["content", "meta"])
    doc1_meta2_id_by_content_and_meta = Document(content=text1, meta={"name": "doc2"}, id_hash_keys=["content", "meta"])
    assert doc1_meta1_id_by_content_and_meta.id != doc1_meta2_id_by_content_and_meta.id

    doc1_text1 = Document(content=text1, meta={"name": "doc1"}, id_hash_keys=["content"])
    doc3_text2 = Document(content=text2, meta={"name": "doc3"}, id_hash_keys=["content"])
    assert doc1_text1.id != doc3_text2.id

    with pytest.raises(ValueError):
        _ = Document(content=text1, meta={"name": "doc1"}, id_hash_keys=["content", "non_existing_field"])


def test_generate_doc_id_custom_list_meta():
    text1 = "text1"
    text2 = "text2"

    doc1_text1 = Document(
        content=text1, meta={"name": "doc1", "url": "https://deepset.ai"}, id_hash_keys=["content", "meta.url"]
    )
    doc2_text1 = Document(
        content=text1, meta={"name": "doc2", "url": "https://deepset.ai"}, id_hash_keys=["content", "meta.url"]
    )
    assert doc1_text1.id == doc2_text1.id

    doc1_text1 = Document(content=text1, meta={"name": "doc1", "url": "https://deepset.ai"}, id_hash_keys=["meta.url"])
    doc2_text2 = Document(content=text2, meta={"name": "doc2", "url": "https://deepset.ai"}, id_hash_keys=["meta.url"])
    assert doc1_text1.id == doc2_text2.id

    doc1_text1 = Document(content=text1, meta={"name": "doc1", "url": "https://deepset.ai"}, id_hash_keys=["meta.url"])
    doc2_text2 = Document(
        content=text2, meta={"name": "doc2", "url": "https://deepset.ai"}, id_hash_keys=["meta.url", "meta.name"]
    )
    assert doc1_text1.id != doc2_text2.id


def test_aggregate_labels_with_labels():
    label1_with_filter1 = Label(
        query="question",
        answer=Answer(answer="1"),
        is_correct_answer=True,
        is_correct_document=True,
        document=Document(content="some", id="777"),
        origin="gold-label",
        filters={"name": ["filename1"]},
    )
    label2_with_filter1 = Label(
        query="question",
        answer=Answer(answer="2"),
        is_correct_answer=True,
        is_correct_document=True,
        document=Document(content="some", id="777"),
        origin="gold-label",
        filters={"name": ["filename1"]},
    )
    label3_with_filter2 = Label(
        query="question",
        answer=Answer(answer="2"),
        is_correct_answer=True,
        is_correct_document=True,
        document=Document(content="some", id="777"),
        origin="gold-label",
        filters={"name": ["filename2"]},
    )
    label = MultiLabel(labels=[label1_with_filter1, label2_with_filter1])
    assert label.filters == {"name": ["filename1"]}
    with pytest.raises(ValueError):
        label = MultiLabel(labels=[label1_with_filter1, label3_with_filter2])


def test_multilabel_preserve_order():
    labels = [
        Label(
            id="0",
            query="question",
            answer=Answer(answer="answer1", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some", id="123"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        Label(
            id="1",
            query="question",
            answer=Answer(answer="answer2", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some", id="123"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        Label(
            id="2",
            query="question",
            answer=Answer(answer="answer3", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some other", id="333"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        Label(
            id="3",
            query="question",
            answer=Answer(answer="", offsets_in_document=[Span(start=0, end=0)]),
            document=Document(content="some", id="777"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        Label(
            id="4",
            query="question",
            answer=Answer(answer="answer5", offsets_in_document=[Span(start=12, end=18)]),
            document=Document(content="some", id="123"),
            is_correct_answer=False,
            is_correct_document=True,
            origin="gold-label",
        ),
    ]

    multilabel = MultiLabel(labels=labels)

    for i in range(0, 5):
        assert multilabel.labels[i].id == str(i)


def test_multilabel_preserve_order_w_duplicates():
    labels = [
        Label(
            id="0",
            query="question",
            answer=Answer(
                answer="answer1",
                offsets_in_document=[Span(start=12, end=18)],
                offsets_in_context=[Span(start=1, end=7)],
            ),
            document=Document(content="some", id="123"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        Label(
            id="1",
            query="question",
            answer=Answer(
                answer="answer2",
                offsets_in_document=[Span(start=10, end=16)],
                offsets_in_context=[Span(start=0, end=6)],
            ),
            document=Document(content="some", id="123"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        Label(
            id="2",
            query="question",
            answer=Answer(
                answer="answer3",
                offsets_in_document=[Span(start=14, end=20)],
                offsets_in_context=[Span(start=2, end=8)],
            ),
            document=Document(content="some other", id="333"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        Label(
            id="0",
            query="question",
            answer=Answer(
                answer="answer1",
                offsets_in_document=[Span(start=12, end=18)],
                offsets_in_context=[Span(start=1, end=7)],
            ),
            document=Document(content="some", id="123"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
        Label(
            id="2",
            query="question",
            answer=Answer(
                answer="answer3",
                offsets_in_document=[Span(start=14, end=20)],
                offsets_in_context=[Span(start=2, end=8)],
            ),
            document=Document(content="some other", id="333"),
            is_correct_answer=True,
            is_correct_document=True,
            origin="gold-label",
        ),
    ]

    multilabel = MultiLabel(labels=labels)

    assert multilabel.query == "question"
    assert multilabel.answers == ["answer1", "answer2", "answer3"]
    assert multilabel.document_ids == ["123", "123", "333"]
    assert multilabel.contexts == ["some", "some", "some other"]
    assert multilabel.offsets_in_documents == [
        {"start": 12, "end": 18},
        {"start": 10, "end": 16},
        {"start": 14, "end": 20},
    ]
    assert multilabel.offsets_in_contexts == [{"start": 1, "end": 7}, {"start": 0, "end": 6}, {"start": 2, "end": 8}]

    for i in range(0, 3):
        assert multilabel.labels[i].id == str(i)


def test_multilabel_id():
    query1 = "question 1"
    query2 = "question 2"
    document1 = Document(content="something", id="1")
    answer1 = Answer(answer="answer 1")
    filter1 = {"name": ["name 1"]}
    filter2 = {"name": ["name 1"], "author": ["author 1"]}
    label1 = Label(
        query=query1,
        document=document1,
        is_correct_answer=True,
        is_correct_document=True,
        origin="gold-label",
        answer=answer1,
        filters=filter1,
    )
    label2 = Label(
        query=query2,
        document=document1,
        is_correct_answer=True,
        is_correct_document=True,
        origin="gold-label",
        answer=answer1,
        filters=filter2,
    )
    label3 = Label(
        query=query1,
        document=document1,
        is_correct_answer=True,
        is_correct_document=True,
        origin="gold-label",
        answer=answer1,
        filters=filter2,
    )

    assert MultiLabel(labels=[label1]).id == "33a3e58e13b16e9d6ec682ffe59ccc89"
    assert MultiLabel(labels=[label2]).id == "1b3ad38b629db7b0e869373b01bc32b1"
    assert MultiLabel(labels=[label3]).id == "531445fa3bdf98b8598a3bea032bd605"


def test_multilabel_with_doc_containing_dataframes():
    table = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    table_doc = Document(content=table, content_type="table", id="table1")
    label = Label(
        query="A question",
        document=table_doc,
        is_correct_answer=True,
        is_correct_document=True,
        origin="gold-label",
        answer=Answer(
            answer="1",
            context=table,
            offsets_in_document=[TableCell(0, 0)],
            offsets_in_context=[TableCell(0, 0)],
            document_ids=[table_doc.id],
        ),
    )
    multilabel = MultiLabel(labels=[label])
    assert multilabel.query == "A question"
    assert multilabel.contexts == ["   col1  col2\n0     1     3\n1     2     4"]
    assert multilabel.answers == ["1"]
    assert multilabel.document_ids == ["table1"]
    assert multilabel.offsets_in_documents == [{"row": 0, "col": 0}]
    assert multilabel.offsets_in_contexts == [{"row": 0, "col": 0}]


def test_multilabel_serialization():
    label_dict = {
        "id": "011079cf-c93f-49e6-83bb-42cd850dce12",
        "query": "When was the final season first shown on TV?",
        "document": {
            "content": "\n\n\n\n\nThe eighth and final season of the fantasy drama television series ''Game of Thrones'', produced by HBO, premiered on April 14, 2019, and concluded on May 19, 2019. Unlike the first six seasons, which consisted of ten episodes each, and the seventh season, which consisted of seven episodes, the eighth season consists of only six episodes.\n\nThe final season depicts the culmination of the series' two primary conflicts: the G",
            "content_type": "text",
            "id": "9c82c97c9dc8ba6895893a53aafa610f",
            "meta": {},
            "score": None,
            "embedding": None,
        },
        "is_correct_answer": True,
        "is_correct_document": True,
        "origin": "user-feedback",
        "answer": {
            "answer": "April 14",
            "type": "extractive",
            "score": None,
            "context": "\n\n\n\n\nThe eighth and final season of the fantasy drama television series ''Game of Thrones'', produced by HBO, premiered on April 14, 2019, and concluded on May 19, 2019. Unlike the first six seasons, which consisted of ten episodes each, and the seventh season, which consisted of seven episodes, the eighth season consists of only six episodes.\n\nThe final season depicts the culmination of the series' two primary conflicts: the G",
            "offsets_in_document": [{"start": 124, "end": 132}],
            "offsets_in_context": None,
            "document_ids": None,
            "meta": {},
        },
        "no_answer": False,
        "pipeline_id": None,
        "created_at": "2022-07-22T13:29:33.699781+00:00",
        "updated_at": "2022-07-22T13:29:33.784895+00:00",
        "meta": {"answer_id": "374394", "document_id": "604995", "question_id": "345530"},
        "filters": None,
    }

    label = Label.from_dict(label_dict)
    original_multilabel = MultiLabel([label])

    deserialized_multilabel = MultiLabel.from_dict(original_multilabel.to_dict())
    assert deserialized_multilabel == original_multilabel
    assert deserialized_multilabel.labels[0] == label

    json_deserialized_multilabel = MultiLabel.from_json(original_multilabel.to_json())
    assert json_deserialized_multilabel == original_multilabel
    assert json_deserialized_multilabel.labels[0] == label


def test_span_in():
    assert 10 in Span(5, 15)
    assert not 20 in Span(1, 15)


def test_span_in_edges():
    assert 5 in Span(5, 15)
    assert not 15 in Span(5, 15)


def test_span_in_other_values():
    assert 10.0 in Span(5, 15)
    assert "10" in Span(5, 15)
    with pytest.raises(ValueError):
        "hello" in Span(5, 15)


def test_assert_span_vs_span():
    assert Span(10, 11) in Span(5, 15)
    assert Span(5, 10) in Span(5, 15)
    assert not Span(10, 15) in Span(5, 15)
    assert not Span(5, 15) in Span(5, 15)
    assert Span(5, 14) in Span(5, 15)

    assert not Span(0, 1) in Span(5, 15)
    assert not Span(0, 10) in Span(5, 15)
    assert not Span(10, 20) in Span(5, 15)


def test_id_hash_keys_not_ignored():
    # Test that two documents with the same content but different metadata get assigned different ids if and only if
    # id_hash_keys is set to 'meta'
    doc1 = Document(content="hello world", meta={"doc_id": "1"}, id_hash_keys=["meta"])
    doc2 = Document(content="hello world", meta={"doc_id": "2"}, id_hash_keys=["meta"])
    assert doc1.id != doc2.id
    doc3 = Document(content="hello world", meta={"doc_id": "3"})
    doc4 = Document(content="hello world", meta={"doc_id": "4"})
    assert doc3.id == doc4.id


def test_legacy_answer_document_id():
    legacy_label = {
        "id": "123",
        "query": "Who made the PDF specification?",
        "document": {
            "content": "Some content",
            "content_type": "text",
            "score": None,
            "id": "fc18c987a8312e72a47fb1524f230bb0",
            "meta": {},
            "embedding": [0.1, 0.2, 0.3],
        },
        "answer": {
            "answer": "Adobe Systems",
            "type": "extractive",
            "context": "Some content",
            "offsets_in_context": [{"start": 60, "end": 73}],
            "offsets_in_document": [{"start": 60, "end": 73}],
            # legacy document_id answer
            "document_id": "fc18c987a8312e72a47fb1524f230bb0",
            "meta": {},
            "score": None,
        },
        "is_correct_answer": True,
        "is_correct_document": True,
        "origin": "user-feedback",
        "pipeline_id": "some-123",
    }

    answer = Answer.from_dict(legacy_label["answer"])
    assert answer.document_ids == ["fc18c987a8312e72a47fb1524f230bb0"]

    label = Label.from_dict(legacy_label)
    assert label.answer.document_ids == ["fc18c987a8312e72a47fb1524f230bb0"]


def test_legacy_answer_document_id_is_none():
    legacy_label = {
        "id": "123",
        "query": "Who made the PDF specification?",
        "document": {
            "content": "Some content",
            "content_type": "text",
            "score": None,
            "id": "fc18c987a8312e72a47fb1524f230bb0",
            "meta": {},
            "embedding": [0.1, 0.2, 0.3],
        },
        "answer": {
            "answer": "Adobe Systems",
            "type": "extractive",
            "context": "Some content",
            "offsets_in_context": [{"start": 60, "end": 73}],
            "offsets_in_document": [{"start": 60, "end": 73}],
            # legacy document_id answer
            "document_id": None,
            "meta": {},
            "score": None,
        },
        "is_correct_answer": True,
        "is_correct_document": True,
        "origin": "user-feedback",
        "pipeline_id": "some-123",
    }

    answer = Answer.from_dict(legacy_label["answer"])
    assert answer.document_ids is None

    label = Label.from_dict(legacy_label)
    assert label.answer.document_ids is None
