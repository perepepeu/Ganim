# Ganim

Uma biblioteca Python para treinar e usar Redes Adversariais Generativas (GANs) para geracao de imagens.

## Principais Recursos

- API simplificada com `fit()`, `sample()` e `show()`.
- Configuracao centralizada com `setup()`.
- Visualizacao integrada de progresso.
- Plot de perdas com `plot()`.
- Salvamento e carregamento de modelos com `save()` e `load()`.

## Instalacao

Dependencias:

- `torch` e `torchvision`
- `numpy`
- `matplotlib`
- `opencv-python`
- `tqdm`

```bash
pip install torch torchvision numpy matplotlib opencv-python tqdm
```

## Guia Rapido

Estrutura recomendada de pastas:

```bash
seu_projeto/
├── data/
│   └── training_images/
│       ├── imagem1.jpg
│       └── imagem2.png
└── seu_script.py
```

Exemplo de uso:

```python
import ganim

# 1) Treinamento
ganim.setup(
    epochs=500,
    sample_interval=50,
    preview_image_count=25,
    preview_window_size=768,
)

generator, history = ganim.fit(data="./data")

if generator and history:
    ganim.plot(history)
    ganim.save(generator, path="./meu_modelo_ganim.pth")

# 2) Geracao
meu_gerador = ganim.load(path="./meu_modelo_ganim.pth")
novas_imagens = ganim.sample(model=meu_gerador, count=16)

ganim.show(
    novas_imagens,
    window_title="Arte Gerada por Ganim",
    window_size=1024,
)
```

## Referencia da API

`ganim.setup(**kwargs)`

Configura os hiperparametros globais. Deve ser chamada antes de `fit()`.

`ganim.fit(data)`

Inicia o loop de treinamento. Retorna o modelo gerador treinado e um dicionario de historico.

`ganim.sample(model, count=1)`

Gera novas imagens a partir de um modelo treinado.

`ganim.show(images_tensor, window_title="Ganim", window_size=None)`

Exibe um tensor de imagens em uma janela.

`ganim.plot(history)`

Plota os graficos de perda do treinamento.

`ganim.save(model, path="ganim_model.pth")`

Salva os pesos e metadados de configuracao do modelo (como `latent_dim`, `channels` e `image_size`).

`ganim.load(path)`

Carrega um modelo salvo usando os metadados do arquivo quando disponiveis.
Arquivos antigos (apenas `state_dict`) continuam sendo suportados.

## Parametros de `setup`

| Parametro              | Padrao | Descricao |
| ---------------------- | ------ | --------- |
| `image_size`           | 64     | Resolucao da imagem (pixels). |
| `channels`             | 3      | Canais de cor (3 para RGB, 1 para cinza). |
| `latent_dim`           | 100    | Dimensao do vetor de ruido. |
| `epochs`               | 5000   | Numero total de epocas de treinamento. |
| `batch_size`           | 64     | Quantidade de imagens por lote. |
| `learning_rate`        | 0.0002 | Taxa de aprendizado dos modelos. |
| `d_updates_per_g`      | 1      | Quantas atualizacoes do D para cada atualizacao do G (>= 1). |
| `sample_interval`      | 100    | Frequencia (em epocas) para gerar previas. |
| `preview_image_count`  | 16     | Quantidade de imagens por previa. |
| `preview_window_size`  | 512    | Tamanho da janela de previa em pixels. |

Compatibilidade retroativa: os nomes antigos em camelCase (como `imageSize` e `batchSize`) continuam aceitos.

Validacoes de seguranca importantes:
- `channels` aceita apenas `1` (cinza) ou `3` (RGB).
- `sample_interval` e `d_updates_per_g` devem ser `>= 1`.
- `batch_size`, `epochs` e `preview_image_count` devem ser `>= 1`.
