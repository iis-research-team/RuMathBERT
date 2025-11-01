# RuMathBERT
MathBert for Russian language

Dataset for training is available here: https://drive.google.com/file/d/11tSWlA_TU0eESvAxo31-Ir5yi3LPbVLy/view?usp=sharing

WordPiece tokenizer is available here: https://drive.google.com/drive/folders/1wwHMSLUsdf5ZwHPxXLVY8F0VaXfGD2Q_?usp=sharing

The `OPTs.json` file containing 98226 formula operator trees is available here: https://drive.google.com/file/d/1jIwokyfs1kLdqBTux2lqZytIuOrwynqJ/view?usp=sharing

The file structure is as follows:

```
[
  {
    "nodes": [
      {
        "id": <node id>,
        "type": "<node type: function, variable, number, etc>",
        "value": "<node value",
        "description": "<additional description for better readability>"
      },
      ...
    ],
    "edges": [
      [<source id>, <target id>],
      ...
    ],
    "latex_string": "<latex expression>"
  },
  ...
]
```
