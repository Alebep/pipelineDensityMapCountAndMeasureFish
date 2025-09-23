# Fish Pipeline

Este projeto implementa duas pipelines gémeas para contagem e medição de peixes a partir de imagens RGB-D.
A única diferença entre elas está na etapa final de medição: ponta-a-ponta vs skeleton.
O projeto foi construído com foco em organização, baixo acoplamento e testes unitários.

## Estrutura

```
src/
  fish_pipeline/
    data/            # Entidades, DTOs e validadores
    models/          # Modelos de densidade e segmentação (implementações dummy)
    steps/           # Etapas das pipelines (pré-processamento, QC, etc.)
    utils/           # Funções auxiliares de geometria e calibração
    pipelines/       # Pipeline de treino e pipeline de inferência
```

## Dependências

- Python 3.10+
- PyTest (opcional para testes)

Instalação (modo dev):

```bash
pip install -e .[dev]
```

## Testes

```bash
pytest
```

## Uso

Consulte `fish_pipeline/pipelines/inference.py` para um exemplo de execução das pipelines de inferência
(`EndToEndMeasurementPipeline` e `SkeletonMeasurementPipeline`).
Qualquer pipeline aceita instâncias personalizadas de modelos de densidade ou segmentadores compatíveis – basta
passá-las no construtor em vez das implementações padrão CSRNet/SAM-HQ. O módulo
`fish_pipeline/pipelines/training.py` demonstra a pipeline de treino do modelo de densidade.

## Dataset real e carregamento automático

Para treinar o CSRNet com dados reais basta preparar o diretório de acordo com a estrutura abaixo. O carregador nativo
da pipeline (`FileSystemDensityDataset`) lê automaticamente os ficheiros quando `TrainingConfig.dataset_root` ou os
diretórios específicos (`train_dir`/`val_dir`) são definidos.

```
dataset_root/
  train/
    rgb/
      0001.png
      0002.png
      ...
    density/
      0001.npy        # mapa de densidade no mesmo referencial da imagem
      0002.png        # também suportado em formato imagem (float/tiff/png)
  val/
    rgb/
      0005.png
    density/
      0005.npy
  prompts/            # opcional, máscaras GT para avaliar a segmentação
    0001_mask.png
    0002_mask.png
```

Cada imagem em `rgb/` deve possuir um mapa de densidade com o mesmo nome base. Os mapas aceitam `npy` (float32) ou
imagens (`png`, `tiff`, `jpg`, `bmp`). O carregador precisa das dependências opcionais `numpy` e `Pillow`. Quando um
conjunto de validação dedicado não está disponível, a pipeline usa `validation_split` para separar automaticamente uma
fração do conjunto de treino.

## Configuração de treino e checkpoints

`TrainingConfig` foi estendido com novos campos:

* `dataset_root`, `train_dir` e `val_dir`: apontam para o dataset real na estrutura mostrada acima.
* `load_weights_path`: inicializa o CSRNet a partir de um checkpoint existente (útil para retomar treino).
* `save_weights_dir`: diretório onde a pipeline grava checkpoints por época (`best` e `last`).
* `best_checkpoint_name` / `last_checkpoint_name`: nomes dos ficheiros gerados dentro de `save_weights_dir`.
* `save_weights_path`: caminho opcional para gravar o modelo final (por exemplo, copiando o melhor checkpoint).

Durante a execução de `DensityModelTrainingPipeline.run()` o CSRNet guarda um checkpoint “last” a cada época e
atualiza o “best” sempre que o MAE da validação melhora. Qualquer implementação de `BaseDensityModel` continua
compatível: o pipeline invoca `save_weights`/`load_weights` apenas quando estes métodos estão disponíveis.

Após o treino, o caminho indicado em `save_weights_path` recebe automaticamente o melhor modelo disponível ou, caso não
exista checkpoint, o estado atual do modelo.

## Reutilização em inferência

`InferenceConfig` inclui o campo `density_weights_path`, permitindo carregar o CSRNet treinado antes de processar
imagens de produção:

```python
from fish_pipeline.pipelines.inference import EndToEndMeasurementPipeline, InferenceConfig

config = InferenceConfig(density_weights_path="checkpoints/csrnet_best.pt")
pipeline = EndToEndMeasurementPipeline(config=config)
output = pipeline.run(request)
```

O segmento de densidade (`DensityMapGenerator`) passa a operar com os pesos restaurados automaticamente.

## Integração com SAM-HQ

`SAMHQSegmenter` agora encapsula o pacote oficial `segment-anything-hq`. A classe aceita os parâmetros principais do
modelo (`checkpoint_path`, `model_type`, `device`, `mask_threshold`, `multimask_output`, `use_hq_token_only`) e executa
o predictor nativo quando PyTorch, NumPy e SAM-HQ estão instalados.

Quando as dependências não estão disponíveis a classe degrada para um modo *fallback* leve, garantindo que a pipeline
permaneça funcional para testes automatizados. Para obter segmentações reais, instale as dependências opcionais:

```bash
pip install numpy Pillow torch segment-anything-hq
```

Forneça o caminho do checkpoint oficial via `SAMHQSegmenter(checkpoint_path="/caminho/para/sam_hq_vit_b.pth")` e
injete a instância desejada na pipeline de inferência.
