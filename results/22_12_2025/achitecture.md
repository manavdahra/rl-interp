# Model architecture

```
EgoAttentionNetwork(
  (ego_embedding): MultiLayerPerceptron(
    (layers): ModuleList(
      (0): Linear(in_features=7, out_features=64, bias=True)
      (1): Linear(in_features=64, out_features=64, bias=True)
    )
  )
  (others_embedding): MultiLayerPerceptron(
    (layers): ModuleList(
      (0): Linear(in_features=7, out_features=64, bias=True)
      (1): Linear(in_features=64, out_features=64, bias=True)
    )
  )
  (attention_layer): EgoAttention(
    (value_all): Linear(in_features=64, out_features=64, bias=False)
    (key_all): Linear(in_features=64, out_features=64, bias=False)
    (query_ego): Linear(in_features=64, out_features=64, bias=False)
    (attention_combine): Linear(in_features=64, out_features=64, bias=False)
  )
  (output_layer): MultiLayerPerceptron(
    (layers): ModuleList(
      (0-1): 2 x Linear(in_features=64, out_features=64, bias=True)
    )
    (predict): Linear(in_features=64, out_features=3, bias=True)
  )
)
```
