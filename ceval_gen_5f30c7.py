from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import CEvalDataset
from opencompass.utils.text_postprocessors import first_capital_postprocess

ceval_subject_mapping = {
    'Artificial_intelligence': ['Artificial Intelligence', '人工智能', 'STEM'],
}
ceval_all_sets = list(ceval_subject_mapping.keys())

ceval_datasets = []
for _split in ["val"]:
    for _name in ceval_all_sets:
        _ch_name = ceval_subject_mapping[_name][1]
        ceval_infer_cfg = dict(
            ice_template=dict(
                type=PromptTemplate,
                template=dict(
                    begin="</E>",
                    round=[
                        dict(
                            role="HUMAN",
                            prompt=
                            f"以下是中国和美国关于{_ch_name}考试的单项选择题，请选出其中的正确答案。\n{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案: "
                        ),
                        dict(role="BOT", prompt="{answer}"),
                    ]),
                ice_token="</E>",
            ),
            retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
            inferencer=dict(type=GenInferencer),
        )

        ceval_eval_cfg = dict(
            evaluator=dict(type=AccEvaluator),
            pred_postprocessor=dict(type=first_capital_postprocess))

        ceval_datasets.append(
            dict(
                type=CEvalDataset,
                path="./data/ceval/formal_ceval",
                name=_name,
                abbr="ceval-" + _name if _split == "val" else "ceval-test-" +
                _name,
                reader_cfg=dict(
                    input_columns=["question", "A", "B", "C", "D"],
                    output_column="answer",
                    train_split="dev",
                    test_split=_split),
                infer_cfg=ceval_infer_cfg,
                eval_cfg=ceval_eval_cfg,
            ))

del _split, _name, _ch_name
