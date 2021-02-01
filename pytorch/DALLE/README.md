## Pytorch Implementation of DALLE: The OpenAI Text to Image Transformer

Implementation of OpenAI <a href="https://openai.com/blog/dall-e/">DALLE</a> with <a href="https://openai.com/blog/clip/">CLIP</a> for ranking the generation

<a href="https://www.youtube.com/watch?v=j4xgkjWlfL4">Yannic Kilcher's Blog post explanation of the model</a>

Before we replicate DALLE, we can check out <a href="https://github.com/lucidrains/deep-daze">Deep-Daze</a> or <a href="https://github.com/lucidrains/big-sleep">Big Sleep</a>

## Scaling Depth

In the blog post, they used 64 layers to achieve their results. I added reversible networks, from the <a href="https://github.com/lucidrains/reformer-pytorch">Reformer</a> paper, in order for users to attempt to scale depth at the cost of compute. Reversible networks allow you to scale to any depth at no memory cost, but a little over 2x compute cost (each layer is rerun on the backward pass).

Simply set the `reverible` keyword to `True` for the `DALLE` class

```python
dalle = DALLE(
    dim = 1024,
    vae = vae,
    num_text_tokens = 10000,
    text_seq_len = 256,
    depth = 64,
    heads = 16,
    reversible = True  # <-- reversible networks https://arxiv.org/abs/2001.04451
)
```

## Sparse Attention

We can also train with Microsoft Deepspeed <a href="https://www.deepspeed.ai/news/2020/09/08/sparse-attention.html">Deep Speed</a>
