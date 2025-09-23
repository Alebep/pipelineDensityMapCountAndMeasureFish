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

## Dataset

Para treinar o CSRNet e alimentar o SAM-HQ é necessário dispor de imagens RGB alinhadas com mapas de densidade
e máscaras opcionais para validação. A estrutura recomendada para o diretório de dados é a seguinte:

```
dataset_root/
  train/
    rgb/
      0001.png
      0002.png
      ...
    density/
      0001.npy        # mapa de densidade no mesmo referencial da imagem
      0002.npy
  val/
    rgb/
      0005.png
    density/
      0005.npy
  prompts/            # opcional, máscaras GT para avaliar a segmentação
    0001_mask.png
    0002_mask.png
```

Cada imagem em `rgb/` deve possuir um mapa de densidade com o mesmo nome base. Os mapas podem estar em formato
`npy` (matriz float32) ou `png`/`tiff` normalizados. As máscaras em `prompts/` são opcionais e servem para avaliação
ou calibração manual da segmentação SAM-HQ.

## Treino com CSRNet

Após instalar o pacote (`pip install -e .[dev]`) basta preparar uma lista de amostras `(imagem, densidade)` onde
cada elemento é uma matriz `float` normalizada entre 0 e 1. Um exemplo mínimo de treino é apresentado abaixo:

```python
from fish_pipeline.models.density import CSRNetDensityModel
from fish_pipeline.pipelines.training import DensityModelTrainingPipeline, TrainingConfig

config = TrainingConfig(num_samples=0)  # substituído pelo dataset real
model = CSRNetDensityModel()

# Converter o dataset real para a forma List[Tuple[Matrix, Matrix]]
dataset = load_real_dataset("dataset_root/train")

pipeline = DensityModelTrainingPipeline(model=model, config=config)
pipeline.model.train(dataset)
```

Durante o treino real substitua `load_real_dataset` por um carregador que leia as imagens e mapas de densidade do
diretório estruturado conforme descrito acima. O relatório produzido pela pipeline (`TrainingReport`) contém o MAE e
RMSE de contagem, além de metadados do modelo CSRNet. Depois do treino basta carregar o modelo numa pipeline de
inferência (`EndToEndMeasurementPipeline` ou `SkeletonMeasurementPipeline`) para utilizar o CSRNet e o SAM-HQ na
estimativa de contagem e na segmentação.

> **Nota:** O método `DensityModelTrainingPipeline.run()` continua a gerar um conjunto sintético para testes
> automatizados. Para treinos reais utilize a interface direta do modelo (`model.train(dataset)`) ou estenda a classe
> `SyntheticDensityDataset` para consumir os ficheiros do diretório `dataset_root/`.
> Para treinar com PyTorch instale manualmente a dependência (`pip install torch --extra-index-url https://download.pytorch.org/whl/cpu`) ou utilize a variante GPU antes de executar o código acima.
